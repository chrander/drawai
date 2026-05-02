import streamlit as st
import anthropic
import openai
import base64
import json
import io
import hashlib
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


# ── helpers ──────────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def analyze_drawing(image: Image.Image) -> dict:
    """Ask Claude to describe the drawing and craft a DALL-E prompt."""
    client = anthropic.Anthropic()
    img_b64 = image_to_base64(image)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a magical art helper at an elementary school science fair! "
            "Your superpower is finding AMAZING things in children's drawings — "
            "even random scribbles become masterpieces in your eyes. "
            "Always be wildly enthusiastic, positive, and encouraging. "
            "Every single drawing is a wonder of creativity!"
        ),
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "A child drew this! Please do two things:\n\n"
                        "1. Write a fun, enthusiastic description of what you see "
                        "or imagine this drawing could be (3-4 sentences, for kids "
                        "ages 6-12). Use exciting words like 'WOW!', 'Look at that!', "
                        "'Amazing!' — even if it looks like random lines, find "
                        "something creative and exciting in it!\n\n"
                        "2. Write a detailed DALL-E 3 prompt to generate a beautiful, "
                        "colorful, finished version of what you imagine this drawing "
                        "represents. Style: whimsical children's book illustration, "
                        "bright colors, friendly and fun.\n\n"
                        "3. Give a very short name (1-3 words) for what the drawing might be.\n\n"
                        "Respond ONLY with raw JSON — no markdown, no code fences:\n"
                        '{"description": "...", "image_prompt": "...", "subject": "..."}'
                    ),
                },
            ],
        }],
    )

    text = response.content[0].text.strip()
    # Strip accidental markdown fences
    if "```" in text:
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else parts[0]
    return json.loads(text)


IMAGE_MODELS = {
    "dall-e-3":    {"qualities": ["standard", "hd"],        "mode": "generate"},
    "gpt-image-2": {"qualities": ["low", "medium", "high"], "mode": "edit"},
}


def generate_image(
    prompt: str,
    model: str,
    quality: str,
    original: Image.Image | None = None,
) -> Image.Image:
    """Edit or generate a finished drawing with the chosen OpenAI image model."""
    import requests

    client = openai.OpenAI()
    mode = IMAGE_MODELS[model]["mode"]

    if mode == "edit":
        # Convert original to PNG bytes — required by the edit endpoint
        buf = io.BytesIO()
        original.convert("RGBA").save(buf, format="PNG")
        buf.seek(0)

        edit_prompt = (
            "Keep every original drawn line exactly as it is. "
            f"The drawing appears to show: {prompt}. "
            "Add vibrant colors, fun details, and a whimsical background to bring "
            "this drawing to life as a children's book illustration. "
            "Enhance and complete what is drawn — do not remove or replace any "
            "original lines. Kid-friendly, bright colors, no text."
        )
        response = client.images.edit(
            model=model,
            image=("drawing.png", buf, "image/png"),
            prompt=edit_prompt,
            size="1024x1024",
            quality=quality,
            n=1,
        )
        img_bytes = base64.b64decode(response.data[0].b64_json)

    else:
        full_prompt = (
            "Whimsical children's book illustration, colorful, bright, friendly, cute. "
            f"{prompt} "
            "No text. No words. No letters. Bright vivid colors. Kid-friendly."
        )
        response = client.images.generate(
            model=model,
            prompt=full_prompt,
            size="1024x1024",
            quality=quality,
            n=1,
        )
        img_bytes = requests.get(response.data[0].url, timeout=30).content

    return Image.open(io.BytesIO(img_bytes))


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Magic Drawing Maker!",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background: linear-gradient(160deg, #FFF8E7 0%, #EEF4FF 100%); }

  .step-box {
    background: white;
    border-radius: 16px;
    padding: 16px 18px;
    border-left: 7px solid;
    margin: 6px 0;
    font-size: 1.05rem;
    color: #222;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.07);
  }
  .description-card {
    background: linear-gradient(135deg, #FFFDE7, #FCE4EC, #E3F2FD);
    border: 4px solid #FF7043;
    border-radius: 20px;
    padding: 24px 28px;
    font-size: 1.35rem;
    line-height: 1.9;
    color: #222;
    box-shadow: 4px 4px 14px rgba(0,0,0,0.08);
    margin-top: 10px;
  }
  .img-label {
    text-align: center;
    font-size: 1.15rem;
    font-weight: bold;
    color: #E53935;
    margin: 8px 0 4px;
  }
  .placeholder {
    text-align: center;
    padding: 60px 20px;
    background: white;
    border-radius: 20px;
    border: 3px dashed #ccc;
    margin-top: 20px;
  }
 // [data-testid="stCameraInput"] label {
 //   color: #222 !important;
 // }
 // [data-testid="stCameraInput"] > div,
 // [data-testid="stCameraInput"] > div > div {
 //   overflow: visible !important;
 // }
 // [data-testid="stCameraInput"] button {
 //   margin-top: 0;
 // }
</style>
""", unsafe_allow_html=True)


# ── sidebar config ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    image_model = st.selectbox(
        "Image model",
        options=list(IMAGE_MODELS.keys()),
        index=0,
    )
    image_quality = st.selectbox(
        "Quality",
        options=IMAGE_MODELS[image_model]["qualities"],
        index=0,
    )


# ── header ────────────────────────────────────────────────────────────────────

st.markdown(
    '<p style="text-align:center;font-size:5rem;color:#E53935;'
    'text-shadow:3px 3px 0 #FB8C00;margin:0;padding:10px 0 4px">'
    "🎨 Magic Drawing Maker! ✨</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center;font-size:1.4rem;color:#6A1B9A;margin-bottom:18px">'
    "Hold up your drawing — watch AI bring it to life!</p>",
    unsafe_allow_html=True,
)

# ── step instructions ─────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="step-box" style="border-color:#E53935">'
        "✏️ <b>Step 1</b><br>Draw anything on paper — a person, animal, "
        "spaceship, or even fun squiggles!</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="step-box" style="border-color:#1E88E5">'
        "📸 <b>Step 2</b><br>Hold your drawing up to the camera and "
        "press the button to take a picture!</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="step-box" style="border-color:#43A047">'
        "✨ <b>Step 3</b><br>AI reads your drawing, colors it in beautifully, "
        "and writes your story!</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ── camera ────────────────────────────────────────────────────────────────────

picture = st.camera_input(
    "📸 Show the camera your drawing!",
    help="Hold your drawing up to the camera, then click the button.",
)


# ── process ───────────────────────────────────────────────────────────────────

if picture:
    img_bytes = picture.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    if st.session_state.get("img_hash") != img_hash:
        st.session_state.img_hash = img_hash
        st.session_state.result = None

    original_image = Image.open(io.BytesIO(img_bytes))

    if st.session_state.get("result") is None:
        try:
            status = st.status("🪄 AI magic in progress…", expanded=True)
            with status:
                st.write("🔍 Reading your drawing…")
                analysis = analyze_drawing(original_image)

                st.write(f"✨ I see: **{analysis['subject']}**! Painting a beautiful version…")
                finished = generate_image(
                    analysis["image_prompt"], image_model, image_quality, original_image
                )

                st.session_state.result = {
                    "analysis": analysis,
                    "original": original_image,
                    "finished": finished,
                }

            status.update(label="✅ Magic complete!", state="complete")
            st.session_state.show_balloons = True
            st.rerun()

        except json.JSONDecodeError:
            st.error("Hmm, the AI got a bit confused. Try taking the picture again!")
        except Exception as e:
            st.error(f"Oops! Something went wrong: {e}")


# ── display ───────────────────────────────────────────────────────────────────

display_result = st.session_state.get("result") if picture else None

if display_result:
    if st.session_state.get("show_balloons"):
        st.balloons()
        st.session_state.show_balloons = False

    st.markdown(
        '<p style="font-size:2rem;font-weight:bold;color:#E53935;margin:10px 0">'
        "🌟 Your Amazing Artwork!</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<p class="img-label">📷 Your Drawing</p>', unsafe_allow_html=True)
        st.image(display_result["original"], width='stretch')
    with right:
        st.markdown('<p class="img-label">🎨 AI Finished Version</p>', unsafe_allow_html=True)
        st.image(display_result["finished"], width='stretch')

    st.markdown(
        '<p style="font-size:2rem;font-weight:bold;color:#E53935;margin:16px 0 4px">'
        "📖 The Story of Your Drawing!</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="description-card">🌟 {display_result["analysis"]["description"]}</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div class="placeholder">'
        '<p style="font-size:4rem;margin:0">🖼️</p>'
        '<p style="font-size:1.3rem;color:#999;margin-top:12px">'
        "Draw something on paper, then take a picture above!"
        "</p></div>",
        unsafe_allow_html=True,
    )
