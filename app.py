import streamlit as st
import anthropic
import openai
import base64
import json
import io
import hashlib
import requests
import numpy as np
from PIL import Image, ImageEnhance
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


def generate_image(prompt: str) -> Image.Image:
    """Generate a finished drawing via DALL-E 3."""
    client = openai.OpenAI()
    full_prompt = (
        "Whimsical children's book illustration, colorful, bright, friendly, cute. "
        f"{prompt} "
        "No text. No words. No letters. Bright vivid colors. Kid-friendly."
    )
    response = client.images.generate(
        model="dall-e-3",
        prompt=full_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    img_bytes = requests.get(response.data[0].url, timeout=30).content
    return Image.open(io.BytesIO(img_bytes))


def overlay_lines(original: Image.Image, generated: Image.Image) -> Image.Image:
    """Overlay the original drawing's dark lines on top of the AI-generated image."""
    orig = original.resize(generated.size, Image.LANCZOS)

    # Boost contrast so drawn lines become very dark
    orig = ImageEnhance.Contrast(orig).enhance(2.0)
    gray = np.array(orig.convert("L"), dtype=np.float32)

    # Normalize brightness range
    lo, hi = gray.min(), gray.max()
    if hi > lo:
        gray = (gray - lo) / (hi - lo) * 255

    # Dark pixels = drawn lines → high alpha; light pixels = paper → transparent
    alpha = np.clip((255 - gray) * 0.70, 0, 255).astype(np.uint8)
    alpha[alpha < 55] = 0  # drop noise / light grey paper texture

    h, w = alpha.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)  # solid black with variable alpha
    overlay[:, :, 3] = alpha

    base = generated.convert("RGBA")
    composite = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))
    return composite.convert("RGB")


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

  .main-title {
    text-align: center;
    font-size: 3.4rem;
    color: #E53935;
    text-shadow: 3px 3px 0 #FB8C00;
    margin: 0;
    padding: 10px 0 4px;
  }
  .subtitle {
    text-align: center;
    font-size: 1.4rem;
    color: #6A1B9A;
    margin-bottom: 18px;
  }
  .step-box {
    background: white;
    border-radius: 16px;
    padding: 16px 18px;
    border-left: 7px solid;
    margin: 6px 0;
    font-size: 1.05rem;
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
</style>
""", unsafe_allow_html=True)


# ── header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🎨 Magic Drawing Maker! ✨</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Hold up your drawing — watch AI bring it to life!</p>',
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


# ── process & display ─────────────────────────────────────────────────────────

if picture:
    img_bytes = picture.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    # Reset results when a new photo is taken
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
                finished = generate_image(analysis["image_prompt"])

                st.write("🖊️ Adding your original lines…")
                combined = overlay_lines(original_image, finished)

                st.session_state.result = {
                    "analysis": analysis,
                    "original": original_image,
                    "finished": finished,
                    "combined": combined,
                }
            status.update(label="✅ Magic complete!", state="complete")
            st.balloons()

        except json.JSONDecodeError:
            st.error("Hmm, the AI got a bit confused. Try taking the picture again!")
        except Exception as e:
            st.error(f"Oops! Something went wrong: {e}")

    result = st.session_state.get("result")
    if result:
        st.markdown("## 🌟 Your Amazing Artwork!")

        left, mid, right = st.columns(3)
        with left:
            st.markdown('<p class="img-label">📷 Your Drawing</p>', unsafe_allow_html=True)
            st.image(result["original"], use_column_width=True)
        with mid:
            st.markdown('<p class="img-label">🎨 AI Finished Version</p>', unsafe_allow_html=True)
            st.image(result["finished"], use_column_width=True)
        with right:
            st.markdown('<p class="img-label">✏️ + ✨ Your Lines + AI Colors!</p>', unsafe_allow_html=True)
            st.image(result["combined"], use_column_width=True)

        st.markdown("## 📖 The Story of Your Drawing!")
        st.markdown(
            f'<div class="description-card">🌟 {result["analysis"]["description"]}</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("🔄 Try a New Drawing!", type="primary", use_container_width=True):
            st.session_state.result = None
            st.session_state.img_hash = None
            st.rerun()

else:
    st.markdown(
        '<div class="placeholder">'
        '<p style="font-size:4rem;margin:0">🖼️</p>'
        '<p style="font-size:1.3rem;color:#999;margin-top:12px">'
        "Draw something on paper, then take a picture above!"
        "</p></div>",
        unsafe_allow_html=True,
    )
