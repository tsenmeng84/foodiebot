import os
import json
import re
import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
from collections import deque, defaultdict
import asyncio
import requests
from bs4 import BeautifulSoup
import logging
import aiofiles
import stat
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_KEY')

if not TOKEN or not OPENAI_KEY:
    raise ValueError("Missing required environment variables: DISCORD_TOKEN and/or OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# History per channel
MAX_HISTORY = 10
history_file = "history.json"
channel_histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

def stars_from_rating(rating):
    try:
        rating = int(rating)
        return "★" * rating + "☆" * (10 - rating)
    except:
        return "N/A"
    
# Load history with error handling
try:
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for channel_id, messages in raw_data.items():
                channel_histories[channel_id] = deque(messages, maxlen=MAX_HISTORY)
except (json.JSONDecodeError, IOError) as e:
    logger.error(f"Error loading history file: {e}")
    channel_histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

def save_history():
    """Save conversation history to file with error handling"""
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({k: list(v) for k, v in channel_histories.items()}, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error saving history: {e}")

# Recommendations storage
recommendations_file = "recommendations.json"
recommendations = defaultdict(list)

# Load recommendations with error handling
try:
    if os.path.exists(recommendations_file):
        with open(recommendations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for cuisine, entries in data.items():
                recommendations[cuisine] = entries
except (json.JSONDecodeError, IOError) as e:
    logger.error(f"Error loading recommendations file: {e}")
    recommendations = defaultdict(list)

def save_recommendations():
    """Save recommendations to file with error handling"""
    try:
        with open(recommendations_file, 'w', encoding='utf-8') as f:
            json.dump(dict(recommendations), f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error saving recommendations: {e}")

CUISINE_OPTIONS = [
    "steak", "fish n chips", "japanese", "local", "western",
    "italian", "thai", "indian", "others"
]

user_states = {}
url_pattern = re.compile(r'https?://\S+')

# Rate limiting
user_cooldowns = defaultdict(float)
COOLDOWN_SECONDS = 5

def extract_title_from_url(url: str) -> str:
    """Fetch the <title> of the given URL, ignoring generic or useless titles"""
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
                # Skip generic titles
                bad_titles = {"google maps", "facebook", "instagram", "twitter", "login", "signin"}
                if title.lower() in bad_titles:
                    return "Untitled"
                return title
    except Exception as e:
        logger.warning(f"Failed to extract title from {url}: {e}")
    return "Untitled"


def get_all_recommendations_summary():
    """Get a summary of all recommendations for AI context"""
    summary = []
    for cuisine, entries in recommendations.items():
        if entries:
            for e in entries:
                title = e.get('title', 'No Title')
                user = e.get('user', 'unknown')
                rating = e.get('rating', 'N/A')
                url = e.get('url', 'No URL')
                summary.append(f"{cuisine.title()}: {title} (by {user}, rated {rating}/10) - {url}")
    return "\n".join(summary[:20])  # Limit to first 20 to avoid token overload

def validate_url(url: str) -> bool:
    """Basic URL validation"""
    return bool(url_pattern.match(url.strip()))

def validate_rating(rating_str: str) -> Optional[int]:
    """Validate and convert rating string to integer"""
    try:
        rating = int(rating_str.strip())
        return rating if 0 <= rating <= 10 else None
    except ValueError:
        return None

def is_user_on_cooldown(user_id: int) -> bool:
    """Check if user is on cooldown for API requests"""
    import time
    current_time = time.time()
    if current_time - user_cooldowns[user_id] < COOLDOWN_SECONDS:
        return True
    return False

def set_user_cooldown(user_id: int):
    """Set cooldown for user"""
    import time
    user_cooldowns[user_id] = time.time()

def parse_cuisine_input(user_input: str) -> Optional[str]:
    """Return valid cuisine name from user input (name or number)"""
    input_clean = user_input.strip().lower()
    if input_clean.isdigit():
        index = int(input_clean) - 1
        if 0 <= index < len(CUISINE_OPTIONS):
            return CUISINE_OPTIONS[index]
    elif input_clean in CUISINE_OPTIONS:
        return input_clean
    return None

async def generate_html_with_openai_yelp_style():
    client = OpenAI(api_key=OPENAI_KEY)

    content_lines = []
    for cuisine in CUISINE_OPTIONS:
        entries = recommendations.get(cuisine, [])
        if entries:
            content_lines.append(f"{cuisine.title()} Recommendations:")
            for e in entries:
                title = e.get("title", "No Title")
                url = e.get("url", "#")
                user = e.get("user", "unknown")
                rating = e.get("rating", "N/A")
                stars = stars_from_rating(rating)
                content_lines.append(f"- {title} ({url}) — by {user}, rated {rating}/10 {stars}")
            content_lines.append("")

    prompt = (
        "Convert the following food recommendations into a Yelp-style HTML page.\n"
        "Each section should be grouped by cuisine with a header.\n"
        "Each entry should appear as a modern card with:\n"
        "- Restaurant title as a link\n"
        "- User who recommended it\n"
        "- Rating out of 10 + visual stars (e.g. ★★★★★★★☆☆☆)\n"
        "- Optional image (use https://via.placeholder.com/150 if needed)\n"
        "Use light styling: mobile-friendly, hover effects, subtle colors, and borders.\n\n"
        "Here is the recommendation data:\n\n" + "\n".join(content_lines)
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.6,
    )

    html_code = response.choices[0].message.content.strip()

    async with aiofiles.open("recommendations.html", "w", encoding="utf-8") as f:
        await f.write(html_code)

    return "recommendations.html"


@bot.event
async def on_ready():
    print(f'✅ Bot is online as {bot.user} (ID: {bot.user.id})')
    logger.info(f'Bot connected as {bot.user}')

@bot.command(name='ask')
async def ask_command(ctx, *, question: str):
    """Ask the bot a question with AI response"""
    if is_user_on_cooldown(ctx.author.id):
        await ctx.send("Walao eh, slow down lah! Wait a few seconds before asking again.")
        return
    
    set_user_cooldown(ctx.author.id)
    
    await ctx.typing()
    channel_id = str(ctx.channel.id)
    history = channel_histories[channel_id]
    history.append({"role": "user", "content": question})

    # Get current recommendations for context
    recommendation_context = get_all_recommendations_summary()
    
    messages = [
        {"role": "system", "content": (
            "You are a Discord bot in a channel full of foodies. "
            "You are a Malaysian and hates Singaporean food. "
            "If the user asks to compare Malaysian food with others, you will get sensitive and respond defensively. "
            "If the user asks anything related to food—like makan recommendations, cuisines, or places to eat—you respond helpfully and enthusiastically, like a fellow food lover. "
            "BUT hor, if the question is NOT about food, you answer in a grumpy, and slightly sarcastic tone. "
            "Use local Malaysian english expressions like 'aiyo', 'lah', 'leh', 'walao', 'can or not', and 'why you ask me this' when appropriate."
        )}
    ]
    
    # Add recommendations context if available
    if recommendation_context:
        messages.append({
            "role": "system",
            "content": f"Here are current food recommendations:\n{recommendation_context}"
        })
    
    messages.extend(history)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        save_history()
        await ctx.send(reply)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        await ctx.send("⚠️ OpenAI API error. Try again later lah.")

@bot.command(name='viewpage')
async def deploy_with_openai_and_publish(ctx):
    """Generate Yelp-style HTML via OpenAI and deploy to GitHub Pages"""
    await ctx.send("🧠 Using AI to generate makan site...")

    try:
        # Step 1: Generate HTML with OpenAI
        html_path = await generate_html_with_openai_yelp_style()
        await ctx.send("✅ HTML generated! Deploying now...")

        # Step 2: Push to GitHub
        GITHUB_REPO = "https://github.com/tsenmeng84/foodiebot.git"
        GITHUB_BRANCH = "main"
        LOCAL_REPO_DIR = "temp_github_repo"
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        if not GITHUB_TOKEN:
            await ctx.send("❌ GITHUB_TOKEN is missing! Cannot deploy to GitHub.")
            logger.error("Missing GITHUB_TOKEN in environment variables.")
            return


        import subprocess
        import shutil

        def run_shell(cmd, cwd=None):
            result = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            return result.stdout.strip()

        def remove_readonly(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        if os.path.exists(LOCAL_REPO_DIR):
            shutil.rmtree(LOCAL_REPO_DIR, onerror=remove_readonly)

        authed_repo = GITHUB_REPO.replace("https://", f"https://{GITHUB_TOKEN}@")
        run_shell(f"git clone {authed_repo} {LOCAL_REPO_DIR}")
        shutil.copy(html_path, os.path.join(LOCAL_REPO_DIR, "index.html"))

        run_shell("git config user.name 'makan-bot'", cwd=LOCAL_REPO_DIR)
        run_shell("git config user.email 'bot@example.com'", cwd=LOCAL_REPO_DIR)
        run_shell("git add .", cwd=LOCAL_REPO_DIR)
        run_shell('git commit -m "Auto deploy AI-generated site"', cwd=LOCAL_REPO_DIR)
        run_shell("git push origin main", cwd=LOCAL_REPO_DIR)

        github_pages_url = "https://tsenmeng84.github.io/foodiebot"
        await ctx.send(f"🚀 Done! Your makan site is live:\n{github_pages_url}")

    except Exception as e:
        logger.error(f"AI deploy error: {e}")
        await ctx.send("❌ Failed to generate or publish AI-powered page.")


@bot.command(name='recommend')
async def recommend(ctx):
    """Start the recommendation adding process"""
    if ctx.author.id in user_states:
        await ctx.send("Eh, you already got one process running lah. Finish that first.")
        return
        
    user_states[ctx.author.id] = {"step": "awaiting_url", "user": str(ctx.author)}
    await ctx.send("Sure can! Please drop the URL of your makan recommendation.")

    def url_check(m):
        return m.author == ctx.author and m.channel == ctx.channel and validate_url(m.content)

    try:
        url_msg = await bot.wait_for('message', timeout=30.0, check=url_check)
        url = url_msg.content.strip()

        user_states[ctx.author.id].update({"step": "awaiting_title", "url": url})
        await ctx.send("Trying to auto-grab the title from the link...")

        title = extract_title_from_url(url)
        await ctx.send(f"I got this title: **{title}**\nIf you want to change it, type the new title. Otherwise, type `ok` to keep it.")

        def title_check(m):
            return m.author == ctx.author and m.channel == ctx.channel and len(m.content.strip()) > 0

        title_msg = await bot.wait_for('message', timeout=20.0, check=title_check)
        user_title = title_msg.content.strip()
        if user_title.lower() != "ok":
            title = user_title


        user_states[ctx.author.id].update({"step": "awaiting_cuisine", "title": title})
        cuisine_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(CUISINE_OPTIONS)])
        await ctx.send(f"Shiok! Last one — what type of cuisine is this?\n{cuisine_list}")

        def cuisine_check(m):
            return m.author == ctx.author and m.channel == ctx.channel and parse_cuisine_input(m.content) is not None



        cuisine_msg = await bot.wait_for('message', timeout=20.0, check=cuisine_check)
        cuisine = parse_cuisine_input(cuisine_msg.content)


        await ctx.send("Last question ah! How would you rate this place out of 10?")

        def rating_check(m):
            return m.author == ctx.author and m.channel == ctx.channel and validate_rating(m.content) is not None

        rating_msg = await bot.wait_for('message', timeout=20.0, check=rating_check)
        rating = validate_rating(rating_msg.content)

        recommendations[cuisine].append({
            "title": title,
            "url": url,
            "user": str(ctx.author),
            "rating": rating
        })
        save_recommendations()

        await ctx.send(f"Solid lah! I've added your `{title}` recommendation under `{cuisine}` cuisine with a rating of {rating}/10.")

    except asyncio.TimeoutError:
        await ctx.send("Walao eh, I waited too long and still nothing. Try `!recommend` again when you're ready.")
    except Exception as e:
        logger.error(f"Error in recommend command: {e}")
        await ctx.send("Something went wrong lah. Try again later.")
    finally:
        user_states.pop(ctx.author.id, None)



@bot.command(name='addcuisine')
async def add_cuisine(ctx, *, new_cuisine: str):
    """Add a new cuisine type to the options"""
    cuisine = new_cuisine.strip().lower()
    if not cuisine:
        await ctx.send("Cannot add empty cuisine lah.")
        return
        
    if cuisine in CUISINE_OPTIONS:
        await ctx.send(f"Aiyo... `{cuisine}` already got in the list lah.")
    else:
        CUISINE_OPTIONS.append(cuisine)
        await ctx.send(f"Wah okay! I added `{cuisine}` to the cuisine list. Can use it in your recommendations liao.")

@bot.command(name='viewall')
async def view_all(ctx):
    """View all recommendations grouped by cuisine"""
    if not any(recommendations.values()):
        await ctx.send("No makan recommendations at all leh. Start adding with `!recommend`.")
        return

    all_entries = []
    for cuisine in CUISINE_OPTIONS:
        entries = recommendations.get(cuisine, [])
        if entries:
            formatted = "\n".join([
                f"{i+1}. {e.get('title', 'No Title')} - {e.get('url', 'No URL')} (by {e.get('user', 'unknown')}, rated {e.get('rating', 'N/A')}/10)"
                for i, e in enumerate(entries)
            ])
            all_entries.append(f"🍽️ *{cuisine.capitalize()}*:\n{formatted}")

    if all_entries:
        # Split into chunks if too long
        full_text = "\n\n".join(all_entries)
        if len(full_text) > 2000:
            for entry in all_entries:
                await ctx.send(entry)
        else:
            await ctx.send(full_text)
    else:
        await ctx.send("Still empty lah. Go recommend something first.")

@bot.command(name='view')
async def view_recommend(ctx):
    """View recommendations by cuisine type"""
    if not any(recommendations.values()):
        await ctx.send("Aiyo... no makan recommendations yet leh. Add some first using `!recommend`.")
        return

    cuisine_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(CUISINE_OPTIONS)])
    user_states[ctx.author.id] = {"step": "awaiting_cuisine_view"}
    await ctx.send(f"Shiok ah! What type of cuisine you interested in?\n{cuisine_list}\n(Type a cuisine number or name within 20 seconds!)")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    try:
        msg = await bot.wait_for('message', timeout=20.0, check=check)
        cuisine = parse_cuisine_input(msg.content)
        if not cuisine:
            await ctx.send("Pick properly lah. Use a number or name from the cuisine list.")
            return

        entries = recommendations.get(cuisine, [])
        if not entries:
            await ctx.send("Hmm... no recommendation for that cuisine yet leh.")
        else:
            formatted = "\n".join([
                f"{i+1}. {e.get('title', 'No Title')} - {e.get('url', 'No URL')} (by {e.get('user', 'unknown')}, rated {e.get('rating', 'N/A')}/10)"
                for i, e in enumerate(entries)
            ])
            await ctx.send(f"Here you go, makan places for *{cuisine}*:\n{formatted}")
    except asyncio.TimeoutError:
        await ctx.send("Aiyo... too slow leh. I waited 20 seconds and gave up. Try `!view` again if you still want to see.")
    except Exception as e:
        logger.error(f"Error in view command: {e}")
        await ctx.send("Something went wrong lah. Try again later.")
    finally:
        user_states.pop(ctx.author.id, None)

@bot.command(name='delete')
async def delete_recommendation(ctx):
    """Delete a recommendation"""
    if ctx.author.id in user_states:
        await ctx.send("Eh, you already got one process running lah. Finish that first.")
        return

    if not any(recommendations.values()):
        await ctx.send("Nothing to delete lah. No recommendations added yet.")
        return

    cuisine_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(CUISINE_OPTIONS)])
    user_states[ctx.author.id] = {"step": "awaiting_cuisine_delete"}
    await ctx.send(f"Which cuisine's recommendation you want to delete?\n{cuisine_list}\n(Type a cuisine number or name within 20 seconds!)")

    def check_cuisine(m):
        return m.author == ctx.author and m.channel == ctx.channel

    try:
        cuisine_msg = await bot.wait_for('message', timeout=20.0, check=check_cuisine)
        cuisine = parse_cuisine_input(cuisine_msg.content)
        if not cuisine:
            await ctx.send("Pick properly lah. Use a number or name from the cuisine list.")
            return

        entries = recommendations.get(cuisine, [])
        if not entries:
            await ctx.send("No recommendations in that cuisine leh.")
            return

        formatted = "\n".join([
            f"{i+1}. {e['title']} - {e['url']} (by {e['user']}, rated {e['rating']}/10)"
            for i, e in enumerate(entries)
        ])
        await ctx.send(f"Here are the recommendations for *{cuisine}*:\n{formatted}\nType the number of the recommendation you want to delete within 20 seconds.")

        def check_number(m):
            return (m.author == ctx.author and m.channel == ctx.channel and 
                   m.content.isdigit() and 1 <= int(m.content) <= len(entries))

        num_msg = await bot.wait_for('message', timeout=20.0, check=check_number)
        idx = int(num_msg.content) - 1

        deleted = entries.pop(idx)
        save_recommendations()
        await ctx.send(f"Deleted recommendation `{deleted['title']}` from `{cuisine}` cuisine.")

    except asyncio.TimeoutError:
        await ctx.send("Waited too long lah. Cancelled delete operation.")
    except Exception as e:
        logger.error(f"Error in delete command: {e}")
        await ctx.send("Something went wrong lah. Try again later.")
    finally:
        user_states.pop(ctx.author.id, None)

@bot.command(name='help')
async def help_command(ctx):
    """Show help message"""
    help_text = (
        "- I'm your OpenAI-powered bot, mainly here to store and manage all your makan recommendations 🍜\n"
        "- But if you wanna ask other random questions also can... just don't expect me to be damn happy about it ah. 🙄\n\n"
        
        "**Commands list:**\n"
        "`!ask <question>` - Ask anything, food or otherwise.\n"
        "`!recommend` - Add a new makan recommendation.\n"
        "`!view` - View recommendations by cuisine.\n"
        "`!viewall` - View all recommendations grouped by cuisine.\n"
        "`!addcuisine <cuisine>` - Add a new cuisine category.\n"
        "`!delete` - Delete a recommendation.\n"
        "`!help` - Show this help message."
    )
    await ctx.send(help_text)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    url_pattern = re.compile(r'https?://\S+')
    if (not message.content.startswith('!') and 
        url_pattern.search(message.content) and 
        message.author.id not in user_states):

        url_match = url_pattern.search(message.content)
        url = url_match.group(0)

        def confirm_check(m):
            return (m.author == message.author and m.channel == message.channel and 
                    m.content.lower().strip() in ['yes', 'no', 'y', 'n'])

        try:
            await message.channel.send(
                f"Wah, this link looks like a makan place: {url}\nWant to add it as a recommendation? (yes/no)"
            )
            reply = await bot.wait_for('message', timeout=20.0, check=confirm_check)
            response = reply.content.lower().strip()

            if response in ['yes', 'y']:
                user_states[message.author.id] = {
                    "step": "awaiting_title_from_url",
                    "url": url,
                    "user": str(message.author)
                }
                await message.channel.send("Trying to auto-grab the title from the link...")

                title = extract_title_from_url(url)
                await message.channel.send(f"I got this title: **{title}**\nIf you want to change it, type the new title. Otherwise, type `ok` to keep it.")

                def title_check(m):
                    return m.author == message.author and m.channel == message.channel and len(m.content.strip()) > 0

                title_msg = await bot.wait_for('message', timeout=20.0, check=title_check)
                user_title = title_msg.content.strip()
                if user_title.lower() != "ok":
                    title = user_title

                user_states[message.author.id].update({"step": "awaiting_cuisine", "title": title})
                cuisine_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(CUISINE_OPTIONS)])
                await message.channel.send(f"Shiok! What type of cuisine is this?\n{cuisine_list}")

                def cuisine_check(m):
                    return m.author == message.author and m.channel == message.channel and parse_cuisine_input(m.content) is not None

                cuisine_msg = await bot.wait_for('message', timeout=20.0, check=cuisine_check)
                cuisine = parse_cuisine_input(cuisine_msg.content)

                await message.channel.send("Last one — how would you rate this place out of 10? Rate 0 if you've not tried")

                def rating_check(m):
                    return m.author == message.author and m.channel == message.channel and validate_rating(m.content) is not None

                rating_msg = await bot.wait_for('message', timeout=20.0, check=rating_check)
                rating = validate_rating(rating_msg.content)

                recommendations[cuisine].append({
                    "title": title,
                    "url": url,
                    "user": str(message.author),
                    "rating": rating
                })
                save_recommendations()

                await message.channel.send(
                    f"Solid lah! I've added your `{title}` recommendation under `{cuisine}` cuisine with a rating of {rating}/10."
                )

            else:
                await message.channel.send("Aiyo okay lor, not adding it then.")

        except asyncio.TimeoutError:
            await message.channel.send("Too slow lah... I waited 20 seconds and gave up. Try again if you want to add the link.")
        except Exception as e:
            logger.error(f"Error in URL handling: {e}")
            await message.channel.send("Something went wrong while processing the URL lah.")
        finally:
            user_states.pop(message.author.id, None)

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Eh, that command I don't know leh. Use `!help` to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Walao, you missing some info lah. Check `!help` for proper usage.")
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Slow down lah! Try again in {error.retry_after:.1f} seconds.")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send("Something went wrong lah. Try again later.")

if __name__ == "__main__":
    if not TOKEN:
        print("❌ DISCORD_TOKEN not found in environment variables!")
        exit(1)
    if not OPENAI_KEY:
        print("❌ OPENAI_KEY not found in environment variables!")
        exit(1)
    
    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"❌ Bot failed to start: {e}")
