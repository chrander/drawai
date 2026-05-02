# Magic Drawing Maker

An AI-powered Streamlit app for elementary school science demonstrations. A child draws anything on paper, holds it up to a webcam, and the app uses Claude to analyze the drawing, DALL-E 3 to generate a colorful finished version, and composites the original lines on top — plus writes a fun, enthusiastic story about the artwork.

## How it works

1. Child draws on paper (anything — animals, spaceships, scribbles)
2. Holds the drawing up to the webcam and takes a picture
3. **Claude** (claude-sonnet-4-6) reads the image and writes a kid-friendly description + crafts a DALL-E prompt
4. **DALL-E 3** generates a polished, colorful children's-book-style illustration
5. The original drawn lines are overlaid on the generated image so the child's work stays visible
6. All three images (original, AI version, combined) and the story are displayed side by side

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install via Homebrew (`brew install uv`) or the official installer
- An [Anthropic API key](https://console.anthropic.com/)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd drawai

# 2. Install dependencies
uv sync

# 3. Configure API keys
cp .env.example .env
# Open .env and fill in your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
```

## Running the app

```bash
uv run streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`.

## Cost per demo

Each "magic" button press makes one Claude vision call (~$0.005) and one DALL-E 3 image generation (~$0.04). A demo session with 30 kids costs roughly **$1.35** in API credits.
