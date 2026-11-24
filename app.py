"""
FormFriend 🤝 - AI application-form guide

Upload (or photograph) a form and FormFriend explains it in simple
English, Tamil or Hindi. Groq is tried first, Gemini is the backup.

Run with:   streamlit run app.py
"""

import base64
import hashlib
import hmac
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from html import escape

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat


# ==========================================
# 1. SETTINGS
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model names can be changed in your .env file without touching the code:
#   GROQ_MODEL=...   (must be a VISION model on Groq)
#   GEMINI_MODEL=...
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

MAX_IMAGES = 5          # Groq accepts a limited number of images per request
MAX_IMAGE_SIDE = 1600   # pixels - bigger photos are shrunk before sending
REQUEST_TIMEOUT = 60    # seconds to wait for one AI call
RETRIES = 1             # extra tries on the same API before switching
BLUR_THRESHOLD = 100    # lower = more sensitive blur warning (rough check)


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Planned free analyses per day, for EACH api key (Groq and Gemini).
# Change it in your .env / Streamlit secrets with DAILY_LIMIT=...
DAILY_LIMIT = _env_int("DAILY_LIMIT", 35)
USAGE_FILE = os.getenv("USAGE_FILE", "usage_counter.json")

# Secret password that unlocks Debug mode (open the app with ?admin=1).
# If it is not set, Debug mode is hidden from everyone.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# language name -> browser speech code (used for "Read aloud")
LANGUAGES = {
    "English": "en-IN",
    "Tamil": "ta-IN",
    "Hindi": "hi-IN",
}

# language name -> name written in its own script (helps the AI stay on track)
LANGUAGE_NATIVE = {
    "English": "English",
    "Tamil": "Tamil (தமிழ்)",
    "Hindi": "Hindi (हिन्दी)",
}


# ==========================================
# 2. SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
# SYSTEM IDENTITY: FORMFriend AI

You are FormFriend, an advanced, intelligent,
multilingual AI-powered form-filling assistant.

Your mission is to make complicated application forms
simple, understandable and accessible to everyone.

You are not just an AI chatbot. You are a personal
application guide who helps people understand forms,
identify required information, prepare documents and
complete applications correctly.

Your primary goal is to eliminate confusion, reduce
mistakes and make the application process accessible
regardless of education, language or technical knowledge.


## 1. CORE RESPONSIBILITIES

Whenever a user uploads a form image, screenshot or document:

1. Carefully examine the uploaded document.
2. Identify the form's name and purpose.
3. Explain who should apply for it.
4. Identify every visible field and explain its meaning.
5. Explain what information the user needs to provide.
6. Identify required supporting documents.
7. Explain the application procedure step by step.
8. Highlight important instructions, conditions and
   deadlines when visible.
9. Identify confusing or potentially misleading sections.
10. Help the user understand how to complete the form
    without making mistakes.

Never provide a generic explanation when you can provide
a specific explanation based on the uploaded form.


## 2. ADVANCED IMAGE AND DOCUMENT UNDERSTANDING

Act as an intelligent document analysis system.

When analysing an uploaded image:

- Read visible text carefully.
- Recognize headings, labels, tables, checkboxes
  and instructions.
- Understand the relationship between different fields.
- Identify mandatory and optional fields when indicated.
- Detect blank, unclear, cropped or unreadable sections.
- Interpret abbreviations and technical terminology
  when their meaning is clear.
- Preserve the original meaning of official instructions.

If image quality is poor, ask the user to upload
a clearer image.

Never pretend to read information that is not visible.

If several images are attached, treat them as consecutive
pages of the same form.


## 3. INTELLIGENT FORM EXPLANATION

For every identified field, explain using this structure:

FIELD NAME:
What does it mean?
Why is this information required?
What should the user enter?
Provide a simple example where appropriate.
Mention common mistakes to avoid.

Examples must be clearly marked as examples.

Never present sample personal information as the
user's actual information.

Explain complicated terms in simple everyday language.


## 4. MULTILINGUAL INTELLIGENCE

Support multilingual communication.

Primary languages:

- English
- Tamil
- Hindi

Additional languages should be supported when the
AI model can reliably respond in them.

Language rules:

- Automatically identify the user's preferred language
  from their message.
- Allow the user to explicitly request a language.
- If no language is specified, use simple English.
- Explain the entire form in the selected language.
- Preserve official field names alongside translated meanings.
- Do not translate important legal or administrative
  terminology inaccurately.
- Use simple, natural language rather than complicated
  translations.

For Tamil responses, use clear and natural Tamil
that ordinary people can understand.


## 5. REQUIRED DOCUMENT ANALYSIS

Identify documents required for the application.

For each document, explain:

- Document name.
- Why it is needed.
- Whether it is explicitly mentioned in the uploaded form.
- Whether the requirement needs official verification.
- Any visible format or validity requirements.

IMPORTANT:

Never invent document requirements.

Clearly distinguish between:

A. Requirements explicitly mentioned in the uploaded form.
B. General suggestions.
C. Requirements that need official verification.


## 6. STEP-BY-STEP APPLICATION GUIDE

Convert complicated application procedures into simple
numbered steps.

Where supported by the document, explain:

STEP 1: Understand the application purpose.

STEP 2: Check whether you are eligible.

STEP 3: Prepare the necessary documents.

STEP 4: Fill in the personal information.

STEP 5: Complete the remaining sections.

STEP 6: Review the information for mistakes.

STEP 7: Follow the official submission instructions.

Only include steps supported by available information.

Do not invent submission methods, fees, websites or deadlines.


## 7. ELIGIBILITY AND REQUIREMENT ANALYSIS

When the form contains eligibility conditions:

- Extract the visible eligibility criteria.
- Explain each condition simply.
- Help users compare their provided information
  against those criteria.
- Identify missing information.
- Clearly state when eligibility cannot be determined.

Never guarantee approval or claim someone is eligible
without sufficient evidence.


## 8. PERSONALIZED ASSISTANCE

Understand the user's actual question before answering.

If the user asks:

"What should I write here?"
Explain the specific field.

"Am I eligible?"
Examine the available eligibility conditions and
ask for missing information.

"Which documents do I need?"
List the verified requirements.

"How do I apply?"
Provide a step-by-step process.

"Explain this in Tamil."
Translate and explain in simple Tamil.

"What does this word mean?"
Explain the term with an example.

"Did I make a mistake?"
Review the information provided and identify
possible errors.

Do not overwhelm users with unnecessary information.

Answer the exact question first, then provide
additional useful guidance.


## 9. SECURITY AND PRIVACY

User privacy is extremely important.

- Never request passwords.
- Never request OTPs.
- Never request ATM PINs or banking passwords.
- Never ask users to disclose unnecessary sensitive information.
- Warn users against sharing confidential personal information.
- Do not claim to store, delete or securely process information
  unless the application actually supports those capabilities.

Treat uploaded documents as private user information.


## 10. ACCURACY AND RELIABILITY

Accuracy is more important than giving a confident answer.

Follow these rules:

- Never fabricate government rules.
- Never invent eligibility requirements.
- Never invent document requirements.
- Never create fake official website links.
- Never assume application fees or deadlines.
- Clearly identify uncertain information.
- Distinguish official information from general guidance.
- Recommend checking the relevant official government
  website for information that cannot be verified
  from the uploaded document.

If you cannot understand something, say so honestly.


## 11. RESPONSE FORMAT

Organize complete form analysis using this structure:

# Form Overview

- Form name:
- Purpose:
- Who should apply:
- Application category:

# Understanding the Form

Explain the form in simple language.

# Field-by-Field Explanation

Explain each visible field, what it means and
what the user should enter.

# Required Documents

Separate explicitly mentioned documents
from unverified suggestions.

# Eligibility Criteria

Explain the visible eligibility conditions.

# Application Procedure

Provide clear, numbered steps.

# Important Instructions

Highlight deadlines, conditions, warnings and
common mistakes when supported by the document.

# What Should You Do Next?

Give the user one practical next step.

For short questions, do not force this entire structure.
Give a direct and helpful answer instead.


## 12. PERSONALITY AND COMMUNICATION

Your personality must be:

- Friendly.
- Patient.
- Intelligent.
- Respectful.
- Professional.
- Encouraging.
- Easy to understand.

Imagine explaining a complicated government form
to someone who has never used an online application before.

Never make users feel embarrassed for asking basic questions.

Avoid unnecessary technical terminology.

Use headings, numbered steps and examples
to make information easy to understand.


## 13. FINAL OBJECTIVE

Your ultimate mission is:

"Make every form understandable.
Make every application easier.
Make information accessible to everyone."

Do not behave like a generic chatbot.

Behave like an intelligent, reliable and multilingual
application assistant.

Every response must help the user understand what to do next.

You are FormFriend.
Your purpose is to turn confusion into clarity.


## 14. APP DISPLAY RULES

The app shows your full analysis in tabs, one tab per
main section.

- Use a single "#" ONLY for the main section headings
  from the response format above.
- Inside a section, use "##", "###", bullet points or
  numbered lists, never a single "#".
- Translate the main section headings into the selected
  language.
- For follow-up questions from the user, answer directly
  and briefly. Do not repeat the whole analysis.
"""


def language_label(language):
    """Language name with its own script, e.g. 'Tamil (தமிழ்)'."""
    return LANGUAGE_NATIVE.get(language, language)


def build_system_prompt(language, image_attached=True):
    """System prompt + the language chosen in the sidebar."""
    label = language_label(language)
    # The language rule is placed BEFORE and AFTER the long prompt so
    # the model cannot forget it or copy the language of the form.
    rule = (
        f"LANGUAGE RULE (highest priority): write the ENTIRE answer in "
        f"{label} only. Do not use any other language, even if the form "
        f"itself is written in another language such as Hindi or English. "
        f"Official field names may stay in their original language inside "
        f"brackets next to the {language} meaning."
    )
    prompt = f"{rule}\n\n{SYSTEM_PROMPT}\n\n## 15. SESSION LANGUAGE\n\n{rule}\n"

    if not image_attached:
        prompt += (
            "\n## 16. FOLLOW-UP QUESTIONS\n\n"
            "The form image is NOT attached again, to save resources. "
            "Answer from your earlier analysis in this conversation. "
            "If the question needs details you cannot find there, say so "
            "and ask the user to tick 'Include the form image with my "
            "question' and ask again. Keep the answer short and direct.\n"
        )
    return prompt


# ==========================================
# 3. IMAGE HELPERS
# ==========================================

@st.cache_data(show_spinner=False)
def prepare_image(raw_bytes):
    """Fix phone rotation, shrink big photos and return JPEG bytes."""
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def quality_warnings(img_bytes):
    """Very rough check for low-resolution or blurry photos."""
    warnings = []
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    if min(img.size) < 600:
        warnings.append(
            "The image resolution is low. Text may be hard to read."
        )

    small = img.copy()
    small.thumbnail((800, 800))
    sharpness = ImageStat.Stat(small.filter(ImageFilter.FIND_EDGES)).var[0]
    if sharpness < BLUR_THRESHOLD:
        warnings.append(
            "This photo looks blurry. A clearer, well-lit photo "
            "will give better results."
        )
    return warnings


@st.cache_data(show_spinner=False)
def make_sample_form():
    """Draw a dummy application form so people can try the app."""
    img = Image.new("RGB", (900, 1300), "white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.load_default(size=40)
        font = ImageFont.load_default(size=28)
    except TypeError:  # older Pillow versions
        title_font = font = ImageFont.load_default()

    draw.text((60, 50), "SAMPLE APPLICATION FORM", fill="black", font=title_font)
    draw.text((60, 110), "(Dummy form for demo only)", fill="gray", font=font)

    fields = [
        "1. Full Name (as per ID proof)",
        "2. Father's / Mother's Name",
        "3. Date of Birth (DD/MM/YYYY)",
        "4. Gender:  [ ] Male   [ ] Female   [ ] Other",
        "5. Residential Address",
        "6. Mobile Number",
        "7. Annual Family Income",
        "8. Category:  [ ] General  [ ] OBC  [ ] SC  [ ] ST",
    ]

    y = 200
    for field in fields:
        draw.text((60, y), field, fill="black", font=font)
        draw.rectangle((60, y + 40, 840, y + 90), outline="black", width=2)
        y += 120

    draw.text(
        (60, y + 10),
        "Attach: ID proof, address proof, photograph",
        fill="black",
        font=font,
    )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def show_image(img_bytes, caption):
    """st.image with a fallback for different Streamlit versions."""
    try:
        st.image(img_bytes, caption=caption, width="stretch")
    except Exception:
        st.image(img_bytes, caption=caption, use_container_width=True)


# ==========================================
# 4. DAILY USAGE COUNTER + AI HELPERS (Groq first, Gemini backup)
# ==========================================

IST = timezone(timedelta(hours=5, minutes=30))
PROVIDERS = ("Groq", "Gemini")


class UsageTracker:
    """Counts today's analyses for each provider. Shared by every visitor."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.day = self._today()
        self.counts = {name: 0 for name in PROVIDERS}
        self._load()

    @staticmethod
    def _today():
        return datetime.now(IST).strftime("%Y-%m-%d")

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as file:
                data = json.load(file)
            if data.get("day") == self.day:
                saved = data.get("counts", {})
                for name in PROVIDERS:
                    self.counts[name] = int(saved.get(name, 0))
        except Exception:
            pass  # no file yet (or unreadable): start from zero

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as file:
                json.dump({"day": self.day, "counts": self.counts}, file)
        except Exception:
            pass  # saving is a bonus, the in-memory counter still works

    def _roll_over(self):
        today = self._today()
        if today != self.day:  # a new day started: reset the counters
            self.day = today
            self.counts = {name: 0 for name in PROVIDERS}

    def snapshot(self):
        with self.lock:
            self._roll_over()
            return dict(self.counts)

    def add(self, provider):
        with self.lock:
            self._roll_over()
            if provider in self.counts:
                self.counts[provider] += 1
                self._save()


@st.cache_resource
def get_usage_tracker():
    return UsageTracker(USAGE_FILE)


def configured_providers():
    keys = {"Groq": GROQ_API_KEY, "Gemini": GEMINI_API_KEY}
    return [name for name in PROVIDERS if keys[name]]


def usage_summary():
    """Numbers for the sidebar: used / capacity / left for today."""
    counts = get_usage_tracker().snapshot()
    names = configured_providers()
    used = sum(min(counts[name], DAILY_LIMIT) for name in names)
    capacity = DAILY_LIMIT * len(names)
    return {
        "used": used,
        "capacity": capacity,
        "left": capacity - used,
        "counts": counts,
    }


class AIError(Exception):
    """Raised when no answer could be produced."""

    def __init__(self, errors, rate_limited=False, daily_limit=False):
        super().__init__("; ".join(errors) or "AI error")
        self.errors = errors
        self.rate_limited = rate_limited  # the AI service said "too many requests"
        self.daily_limit = daily_limit    # our own daily counter is used up


def is_rate_limit(error):
    """True when an API error means 'too many requests / quota used up'."""
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if code == 429:
        return True
    text = str(error).lower()
    return any(
        word in text
        for word in ("429", "rate limit", "rate_limit", "resource_exhausted", "quota")
    )


class ThinkFilter:
    """Removes <think>...</think> blocks from streamed text, on the fly."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.buffer = ""
        self.in_think = False
        self.started = False

    def feed(self, text):
        self.buffer += text
        out = []
        while True:
            if self.in_think:
                end = self.buffer.find(self.CLOSE)
                if end == -1:
                    # keep only a possible half-written closing tag
                    self.buffer = self.buffer[-(len(self.CLOSE) - 1):]
                    break
                self.buffer = self.buffer[end + len(self.CLOSE):]
                self.in_think = False
            else:
                start = self.buffer.find(self.OPEN)
                if start == -1:
                    keep = len(self.OPEN) - 1  # maybe a half-written <think>
                    if len(self.buffer) > keep:
                        out.append(self.buffer[:-keep])
                        self.buffer = self.buffer[-keep:]
                    break
                out.append(self.buffer[:start])
                self.buffer = self.buffer[start + len(self.OPEN):]
                self.in_think = True
        return self._clean("".join(out))

    def flush(self):
        rest = "" if self.in_think else self.buffer
        self.buffer = ""
        return self._clean(rest)

    def _clean(self, text):
        if not self.started:
            text = text.lstrip()
            if text:
                self.started = True
        return text


def call_groq(system_prompt, history, images):
    """Yields the answer piece by piece."""
    client = Groq(
        api_key=GROQ_API_KEY,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )

    messages = [{"role": "system", "content": system_prompt}]

    for index, message in enumerate(history):
        # The form images travel with the very first user message.
        if index == 0 and images and message["role"] == "user":
            content = [{"type": "text", "text": message["content"]}]
            for img in images:
                b64 = base64.b64encode(img).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append(
                {"role": message["role"], "content": message["content"]}
            )

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece


def call_gemini(system_prompt, history, images):
    """Yields the answer piece by piece."""
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
    )

    contents = []
    for index, message in enumerate(history):
        role = "model" if message["role"] == "assistant" else "user"
        parts = [types.Part.from_text(text=message["content"])]

        if index == 0 and images and role == "user":
            for img in images:
                parts.append(
                    types.Part.from_bytes(data=img, mime_type="image/jpeg")
                )

        contents.append(types.Content(role=role, parts=parts))

    stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    for chunk in stream:
        try:
            piece = chunk.text
        except Exception:
            piece = None
        if piece:
            yield piece


def stream_ai(system_prompt, history, images=None, info=None, check_limit=False):
    """
    Generator: tries Groq, then Gemini, and yields the answer piece by piece.

    info (a dict you pass in) is filled with:
      info["provider"] - who is answering
      info["partial"]  - True if the answer broke off half way

    check_limit=True skips a provider whose daily limit is already used up.
    Raises AIError if nothing could be produced.
    """
    info = info if info is not None else {}
    functions = {"Groq": call_groq, "Gemini": call_gemini}

    providers = configured_providers()
    if not providers:
        raise AIError(["No API keys found. Add them to your .env file."])

    if check_limit:
        counts = get_usage_tracker().snapshot()
        providers = [name for name in providers if counts[name] < DAILY_LIMIT]
        if not providers:
            raise AIError([], daily_limit=True)

    errors = []
    rate_limited = []

    for name in providers:
        for attempt in range(1, RETRIES + 2):
            started = False
            try:
                think_filter = ThinkFilter()
                for piece in functions[name](system_prompt, history, images or []):
                    visible = think_filter.feed(piece)
                    if visible:
                        started = True
                        info["provider"] = name
                        yield visible
                visible = think_filter.flush()
                if visible:
                    started = True
                    info["provider"] = name
                    yield visible
                if started:
                    return
                raise RuntimeError("Empty response.")
            except Exception as error:
                if started:
                    # Already showed part of the answer, cannot switch now.
                    info["partial"] = True
                    leftover = think_filter.flush()
                    if leftover:
                        yield leftover
                    return
                errors.append(f"{name} (attempt {attempt}): {error}")
                if is_rate_limit(error):
                    rate_limited.append(name)
                    break  # retrying the same service will not help

    raise AIError(
        errors,
        rate_limited=bool(rate_limited) and len(rate_limited) == len(providers),
    )


def ask_ai(system_prompt, history, images=None):
    """Same as stream_ai, but waits and returns (text, provider)."""
    info = {}
    text = "".join(stream_ai(system_prompt, history, images, info)).strip()
    return text, info.get("provider")


def extract_documents(analysis_text, language):
    """Second small AI call: pull a document list out for the checklist."""
    system = (
        "You extract data. Reply with ONLY a JSON array of short strings. "
        "No explanation and no markdown fences."
    )
    history = [
        {
            "role": "user",
            "content": (
                "From the explanation below, list the supporting documents "
                "the applicant must keep ready. Use short names written in "
                f"{language}. If none are mentioned, return [].\n\n"
                f"{analysis_text}"
            ),
        }
    ]
    try:
        text, _ = ask_ai(system, history)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        documents = json.loads(match.group(0))
        return [str(doc).strip() for doc in documents if str(doc).strip()][:15]
    except Exception:
        return []  # checklist is a bonus, never block the main result


# ==========================================
# 5. DISPLAY HELPERS
# ==========================================

def split_sections(markdown_text):
    """Split the answer at top-level '# ' headings."""
    parts = re.split(r"(?m)^# +(.+)$", markdown_text)
    preamble = parts[0].strip()
    sections = [
        (parts[i].strip(), parts[i + 1].strip())
        for i in range(1, len(parts) - 1, 2)
    ]
    return preamble, sections


def render_result(markdown_text):
    """Show the answer in tabs when it has several main sections."""
    preamble, sections = split_sections(markdown_text)

    if len(sections) < 2:
        st.markdown(markdown_text)
        return

    if preamble:
        st.markdown(preamble)

    tabs = st.tabs([title[:28] for title, _ in sections])
    for tab, (title, body) in zip(tabs, sections):
        with tab:
            st.markdown(f"### {title}")
            st.markdown(body)


def read_aloud(text, lang_code):
    """Browser text-to-speech (no extra library). Works best in Chrome."""
    clean = re.sub(r"[#*_`>|]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    safe_text = json.dumps(clean).replace("</", "<\\/")

    widget = """
    <div style="font-family: sans-serif;">
      <button id="play" style="padding:8px 14px;margin-right:8px;">
        🔊 Read aloud
      </button>
      <button id="stop" style="padding:8px 14px;">⏹ Stop</button>
    </div>
    <script>
      const text = __TEXT__;
      const lang = __LANG__;
      const chunks = text.match(/[^.!?।]+[.!?।]*/g) || [text];

      document.getElementById("play").onclick = () => {
        window.speechSynthesis.cancel();
        chunks.forEach((chunk) => {
          const u = new SpeechSynthesisUtterance(chunk.trim());
          u.lang = lang;
          u.rate = 0.95;
          window.speechSynthesis.speak(u);
        });
      };
      document.getElementById("stop").onclick = () => {
        window.speechSynthesis.cancel();
      };
    </script>
    """
    widget = widget.replace("__TEXT__", safe_text).replace(
        "__LANG__", json.dumps(lang_code)
    )
    components.html(widget, height=60)


def build_download_html(markdown_text):
    """Printable HTML page (open it, then Print -> Save as PDF)."""
    try:
        import markdown  # optional package

        body = markdown.markdown(markdown_text, extensions=["tables"])
    except Exception:
        body = (
            "<pre style='white-space:pre-wrap'>"
            f"{escape(markdown_text)}</pre>"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FormFriend Guide</title>
<style>
  body {{
    font-family: Arial, 'Noto Sans Tamil', 'Noto Sans Devanagari', sans-serif;
    max-width: 800px; margin: auto; padding: 24px; line-height: 1.6;
  }}
  h1, h2 {{ color: #1a4d8f; }}
  table {{ border-collapse: collapse; }}
  td, th {{ border: 1px solid #ccc; padding: 6px; }}
</style>
</head>
<body>
{body}
<hr>
<p><small>Generated by FormFriend. Please verify details on the
official website.</small></p>
</body>
</html>"""


def open_gallery_on_mobile():
    """
    Streamlit limits uploads with file extensions (.jpg, .png), which makes
    Android open the file manager. Setting accept="image/*" on the upload
    input makes phones open the photo gallery / photo picker instead.
    """
    components.html(
        """
        <script>
          try {
            const fix = () => {
              window.parent.document
                .querySelectorAll('input[type="file"]')
                .forEach((input) => {
                  if (input.accept !== "image/*") {
                    input.accept = "image/*";
                  }
                });
            };
            fix();
            new MutationObserver(fix).observe(
              window.parent.document.body,
              { childList: true, subtree: true }
            );
          } catch (e) {}
        </script>
        """,
        height=0,
    )


DAILY_LIMIT_MESSAGE = (
    "🌙 Today's free limit of FormFriend is over. Please come back "
    "tomorrow. Thank you for your patience! 💙"
)
BUSY_MESSAGE = (
    "🚦 FormFriend is very busy right now because many people are using it. "
    "Please wait about a minute and try again."
)


def show_errors(error, debug):
    """Friendly message for normal users, technical details only in debug."""
    if error.daily_limit:
        st.warning(DAILY_LIMIT_MESSAGE)
    elif error.rate_limited:
        st.warning(BUSY_MESSAGE)
    else:
        st.error(
            "😕 Can't read your document at the moment. "
            "Please try again in a little while."
        )
    if debug and error.errors:
        with st.expander("Technical error details"):
            for line in error.errors:
                st.code(line)


def show_stream(generator):
    """Show the answer word by word while it is being written."""
    placeholder = st.empty()
    text = ""
    last_draw = 0.0
    for piece in generator:
        text += piece
        now = time.monotonic()
        if now - last_draw > 0.06:  # do not redraw on every tiny piece
            placeholder.markdown(text + " ▌")
            last_draw = now
    placeholder.markdown(text)
    return text


def admin_requested():
    """Admin login is only shown when the page is opened with ?admin=1"""
    try:
        return "admin" in st.query_params
    except Exception:
        return False


# ==========================================
# 6. DESIGN (navy blue + pastel pink)
# Change these colours to restyle the whole app.
# ==========================================

NAVY = "#0B1F4B"        # header, headings, main buttons
ROYAL = "#1E3A8A"       # lighter royal blue
PINK = "#F8C8DC"        # pastel pink (fills, badges, soft accents)
PINK_DEEP = "#E58FB0"   # deeper pink for thin lines and hover borders
PAGE_BG = "#F4F7FC"     # calm light page background
CARD_BORDER = "#DCE4F3"

CSS = """
<style>
.stApp { background: __PAGE_BG__; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 820px; }
footer { visibility: hidden; }
h2, h3, h4 { color: __NAVY__; }
[data-testid="stSidebar"] { border-right: 1px solid __CARD_BORDER__; }

/* ---------- hero banner ---------- */
.ff-hero {
  background: linear-gradient(135deg, __NAVY__ 0%, __ROYAL__ 100%);
  border-radius: 18px;
  padding: 28px 26px 22px;
  margin-bottom: 18px;
  color: #fff;
  border-bottom: 5px solid __PINK__;
  box-shadow: 0 10px 26px rgba(11, 31, 75, 0.25);
}
.ff-badge {
  display: inline-block;
  background: __PINK__;
  color: __NAVY__;
  padding: 3px 12px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.ff-title {
  font-size: 2.3rem;
  font-weight: 800;
  line-height: 1.15;
  margin: 12px 0 6px;
  color: #fff;
}
.ff-tagline { font-size: 1.02rem; color: #D9E3F7; margin: 0 0 16px; }
.ff-steps { display: flex; flex-wrap: wrap; gap: 8px; }
.ff-step {
  background: rgba(255, 255, 255, 0.10);
  border: 1px solid rgba(255, 255, 255, 0.20);
  border-radius: 10px;
  padding: 6px 12px 6px 8px;
  font-size: 0.88rem;
  color: #fff;
}
.ff-step b {
  display: inline-block;
  background: __PINK__;
  color: __NAVY__;
  border-radius: 50%;
  width: 22px;
  height: 22px;
  line-height: 22px;
  text-align: center;
  margin-right: 7px;
  font-size: 0.8rem;
}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid __ROYAL__;
  color: __NAVY__;
  font-weight: 600;
  transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: __PINK__;
  border-color: __PINK_DEEP__;
  color: __NAVY__;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, __NAVY__, __ROYAL__);
  border: none;
  border-bottom: 3px solid __PINK__;
  padding: 0.6rem 1rem;
}
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p { color: #fff; }
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, __ROYAL__, __NAVY__);
  border-bottom: 3px solid __PINK_DEEP__;
}

/* ---------- cards and widgets ---------- */
[data-testid="stFileUploader"] section {
  background: #fff;
  border: 2px dashed __ROYAL__;
  border-radius: 14px;
}
[data-testid="stImage"] img {
  border-radius: 12px;
  box-shadow: 0 4px 14px rgba(11, 31, 75, 0.15);
}
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stChatMessage"] {
  background: #fff;
  border: 1px solid __CARD_BORDER__;
  border-radius: 14px;
}
.stTabs [data-baseweb="tab-list"] { gap: 2px; flex-wrap: wrap; }
.stTabs [aria-selected="true"] { color: __NAVY__; font-weight: 700; }
.stTabs [data-baseweb="tab-highlight"] { background-color: __PINK_DEEP__; }
.stTabs [data-baseweb="tab-panel"] {
  background: #fff;
  border: 1px solid __CARD_BORDER__;
  border-radius: 0 0 14px 14px;
  padding: 16px 18px;
}

/* ---------- phones ---------- */
@media (max-width: 640px) {
  .ff-hero { padding: 20px 16px 16px; }
  .ff-title { font-size: 1.75rem; }
  .ff-tagline { font-size: 0.95rem; }
}
</style>
"""


def inject_css():
    css = CSS
    for token, value in {
        "__NAVY__": NAVY,
        "__ROYAL__": ROYAL,
        "__PINK__": PINK,
        "__PINK_DEEP__": PINK_DEEP,
        "__PAGE_BG__": PAGE_BG,
        "__CARD_BORDER__": CARD_BORDER,
    }.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


def render_hero():
    st.markdown(
        '<div class="ff-hero">'
        '<span class="ff-badge">AI Application Guide</span>'
        '<div class="ff-title">FormFriend 🤝</div>'
        '<p class="ff-tagline">Upload any application form and get a clear, '
        "step-by-step explanation in English, Tamil or Hindi.</p>"
        '<div class="ff-steps">'
        '<span class="ff-step"><b>1</b>Upload your form</span>'
        '<span class="ff-step"><b>2</b>Get it explained</span>'
        '<span class="ff-step"><b>3</b>Ask questions</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


# ==========================================
# 7. PAGE SETUP + SESSION STATE
# ==========================================

st.set_page_config(
    page_title="FormFriend",
    page_icon="🤝",
    layout="centered",
)
inject_css()

ss = st.session_state

DEFAULT_STATE = {
    "result": None,           # the main analysis text
    "result_language": "English",
    "provider": None,         # which AI answered
    "partial": False,         # True if the answer broke off half way
    "history": [],            # whole conversation (for follow-ups)
    "analysis_images": [],    # images the analysis was based on
    "fingerprint": None,      # id of those images
    "docs": [],               # checklist items
    "use_sample": False,
    "uploader_version": 0,    # changing this clears the upload widgets
}
for key, value in DEFAULT_STATE.items():
    ss.setdefault(key, value)


def reset_all():
    version = ss.get("uploader_version", 0) + 1
    for key in list(ss.keys()):
        del ss[key]
    ss["uploader_version"] = version


# ==========================================
# 8. SIDEBAR
# ==========================================

is_admin = False
debug = False

with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("🌐 Language", list(LANGUAGES))
    st.button("🔄 Start over", on_click=reset_all)

    # ---- daily usage counter ----
    usage = usage_summary()
    st.divider()
    st.subheader("📊 Today's usage")
    if usage["capacity"]:
        st.progress(
            min(usage["used"] / usage["capacity"], 1.0),
            text=f"{usage['used']} of {usage['capacity']} free analyses used",
        )
        if usage["left"] > 0:
            st.caption(f"✅ {usage['left']} analyses left today")
        else:
            st.caption("🌙 Limit reached. Please come back tomorrow.")

    st.divider()
    st.subheader("How to use")
    st.markdown(
        "1. Upload a clear photo of your form\n"
        "2. Pick your language\n"
        "3. Click **Understand My Form**\n"
        "4. Read, listen, tick your documents\n"
        "5. Ask follow-up questions below the answer"
    )

    # ---- admin area (hidden unless the URL has ?admin=1) ----
    if ADMIN_PASSWORD and admin_requested():
        st.divider()
        with st.expander("🔐 Admin"):
            password = st.text_input("Admin password", type="password")
            if password:
                if hmac.compare_digest(
                    password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")
                ):
                    is_admin = True
                else:
                    st.error("Wrong password.")
            if is_admin:
                debug = st.checkbox(
                    "🛠️ Debug mode", help="Shows technical error details."
                )
                for name in configured_providers():
                    st.caption(
                        f"{name}: {usage['counts'][name]} / {DAILY_LIMIT} today"
                    )


# ==========================================
# 9. HEADER + UPLOAD SECTION
# ==========================================

render_hero()

st.info(
    "🔒 Privacy tip: only upload what is needed. Avoid photos that show "
    "Aadhaar numbers, bank details or passwords."
)

st.subheader("📄 Upload Your Application Form")
st.write(
    f"Choose up to {MAX_IMAGES} photos (one per page) from your phone "
    "or computer."
)

gallery_files = st.file_uploader(
    "🖼️ Choose photo(s) from Gallery / Photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"gallery_{ss.uploader_version}",
    help="Select existing photos of your application form.",
)

open_gallery_on_mobile()

camera_file = None
if st.checkbox("📷 Take a photo with the camera", key=f"cam_{ss.uploader_version}"):
    camera_file = st.camera_input(
        "Take a photo of your form",
        key=f"camera_{ss.uploader_version}",
    )

if st.button("🧪 Try with a sample form"):
    ss.use_sample = True

# Collect the images
raw_images = [f.getvalue() for f in (gallery_files or [])]
if camera_file is not None:
    raw_images.append(camera_file.getvalue())

using_sample = False
if not raw_images and ss.use_sample:
    raw_images = [make_sample_form()]
    using_sample = True

if len(raw_images) > MAX_IMAGES:
    st.warning(f"Only the first {MAX_IMAGES} images will be used.")
    raw_images = raw_images[:MAX_IMAGES]

images = []
for raw in raw_images:
    try:
        images.append(prepare_image(raw))
    except Exception:
        st.warning("One of the files could not be read as an image.")

current_fingerprint = (
    hashlib.md5(b"".join(images)).hexdigest() if images else None
)


# ==========================================
# 10. PREVIEW + ANALYSE (answer appears word by word)
# ==========================================

if images:
    if using_sample:
        st.caption("Showing a dummy sample form 🧪")

    columns = st.columns(min(len(images), 3))
    for index, img in enumerate(images):
        with columns[index % len(columns)]:
            show_image(img, f"Page {index + 1}")
        for warning in quality_warnings(img):
            st.warning(f"Page {index + 1}: {warning}")

    if st.button("✨ Understand My Form", type="primary"):
        request = (
            f"Please analyse the attached form ({len(images)} image(s)) "
            f"and explain it in {language_label(language)} only, "
            "following your response format."
        )
        first_message = [{"role": "user", "content": request}]
        info = {}

        try:
            st.markdown("##### ✍️ Writing your guide...")
            text = show_stream(
                stream_ai(
                    build_system_prompt(language),
                    first_message,
                    images,
                    info,
                    check_limit=True,  # respects the daily counter
                )
            ).strip()

            if not text:
                raise AIError(["Empty response."])

            provider = info.get("provider")
            get_usage_tracker().add(provider)  # count this analysis

            with st.spinner("Preparing your document checklist... 📋"):
                docs = extract_documents(text, language)

            ss.history = first_message + [
                {"role": "assistant", "content": text}
            ]
            ss.result = text
            ss.result_language = language
            ss.provider = provider
            ss.partial = bool(info.get("partial"))
            ss.analysis_images = images
            ss.fingerprint = current_fingerprint
            ss.docs = docs
            st.rerun()  # show the final tabs + updated counter

        except AIError as error:
            show_errors(error, debug)


# ==========================================
# 11. RESULT, TOOLS AND FOLLOW-UP CHAT
# ==========================================

if ss.result:
    st.divider()

    if current_fingerprint and current_fingerprint != ss.fingerprint:
        st.info(
            "You changed the images. Click **Understand My Form** "
            "to analyse the new ones."
        )

    if ss.result_language != language:
        st.info(
            f"The answer below is in {ss.result_language}. To get it in "
            f"{language}, click **Understand My Form** again. "
            f"(Follow-up questions will already use {language}.)"
        )

    st.success("Analysis completed! ✅")
    if ss.partial:
        st.warning(
            "The answer may be cut short. Click **Understand My Form** "
            "again if something looks missing."
        )
    if is_admin:
        st.caption(f"Answered by {ss.provider}")

    render_result(ss.result)

    # ---- Listen + download ----
    st.subheader("🔊 Listen or save")
    read_aloud(ss.result, LANGUAGES.get(ss.result_language, "en-IN"))
    st.caption(
        "Voice uses your browser. Tamil and Hindi voices depend on "
        "your device."
    )

    left, right = st.columns(2)
    with left:
        st.download_button(
            "⬇️ Download (.md)",
            data=ss.result,
            file_name="formfriend_guide.md",
            mime="text/markdown",
        )
    with right:
        st.download_button(
            "🖨️ Printable page (save as PDF)",
            data=build_download_html(ss.result),
            file_name="formfriend_guide.html",
            mime="text/html",
        )

    # ---- Document checklist ----
    if ss.docs:
        st.subheader("✅ My document checklist")
        ready = 0
        for index, doc in enumerate(ss.docs):
            if st.checkbox(doc, key=f"doc_{ss.fingerprint}_{index}"):
                ready += 1
        st.progress(ready / len(ss.docs), text=f"{ready} of {len(ss.docs)} ready")
        st.caption(
            "This list is made by AI from your form. Please confirm it "
            "against the official instructions."
        )

    # ---- Follow-up chat ----
    st.subheader("💬 Ask a follow-up question")
    include_image = st.checkbox(
        "📎 Include the form image with my question",
        help=(
            "Normally the image is NOT sent again, which saves your free "
            "daily limit. Tick this only if the answer needs the image."
        ),
    )

    for message in ss.history[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about this form...")
    if question:
        ss.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                info = {}
                answer = show_stream(
                    stream_ai(
                        build_system_prompt(language, image_attached=include_image),
                        ss.history,
                        ss.analysis_images if include_image else [],
                        info,
                    )
                ).strip()
                if not answer:
                    raise AIError(["Empty response."])
                ss.history.append({"role": "assistant", "content": answer})
                ss.provider = info.get("provider", ss.provider)
            except AIError as error:
                ss.history.pop()  # remove the question that got no answer
                show_errors(error, debug)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()
st.caption(
    "FormFriend gives general guidance. Always check the official "
    "website or office before you submit a form."
)
