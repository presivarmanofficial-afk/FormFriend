"""
FormFriend 🤝 - AI application-form guide

Upload (or photograph) a form and FormFriend explains it in simple
English, Tamil or Hindi. Groq is tried first, Gemini is the backup.

Run with:   streamlit run app.py
"""

import base64
import hashlib
import io
import json
import os
import re
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


def build_system_prompt(language):
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
    return f"{rule}\n\n{SYSTEM_PROMPT}\n\n## 15. SESSION LANGUAGE\n\n{rule}\n"


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
# 4. AI HELPERS (Groq first, Gemini backup)
# ==========================================

class AIError(Exception):
    """Raised when every API attempt failed."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def call_groq(system_prompt, history, images):
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

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def call_gemini(system_prompt, history, images):
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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


def ask_ai(system_prompt, history, images=None):
    """
    Try Groq, then Gemini. Each gets 1 + RETRIES attempts.
    Returns (text, provider_name). Raises AIError if everything fails.
    """
    providers = []
    if GROQ_API_KEY:
        providers.append(("Groq", call_groq))
    if GEMINI_API_KEY:
        providers.append(("Gemini", call_gemini))

    if not providers:
        raise AIError(["No API keys found. Add them to your .env file."])

    errors = []
    for name, function in providers:
        for attempt in range(1, RETRIES + 2):
            try:
                text = function(system_prompt, history, images or [])
                # some reasoning models add <think>...</think> blocks
                text = re.sub(
                    r"<think>.*?</think>", "", text or "", flags=re.DOTALL
                ).strip()
                if not text:
                    raise RuntimeError("Empty response.")
                return text, name
            except Exception as error:
                errors.append(f"{name} (attempt {attempt}): {error}")

    raise AIError(errors)


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


def show_errors(errors, debug):
    st.error(
        "😕 Can't read your document at the moment. "
        "Please try again in a little while."
    )
    if debug:
        with st.expander("Technical error details"):
            for line in errors:
                st.code(line)


# ==========================================
# 6. PAGE SETUP + SESSION STATE
# ==========================================

st.set_page_config(
    page_title="FormFriend",
    page_icon="🤝",
    layout="centered",
)

ss = st.session_state

DEFAULT_STATE = {
    "result": None,           # the main analysis text
    "result_language": "English",
    "provider": None,         # which AI answered
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
# 7. SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("🌐 Language", list(LANGUAGES))
    debug = st.checkbox("🛠️ Debug mode", help="Shows technical error details.")
    st.button("🔄 Start over", on_click=reset_all)

    st.divider()
    st.subheader("How to use")
    st.markdown(
        "1. Upload a clear photo of your form\n"
        "2. Pick your language\n"
        "3. Click **Understand My Form**\n"
        "4. Read, listen, tick your documents\n"
        "5. Ask follow-up questions below the answer"
    )


# ==========================================
# 8. HEADER + UPLOAD SECTION
# ==========================================

st.title("FormFriend 🤝")
st.subheader("Your AI Application Guide")
st.write("Upload a form and let AI explain it in simple language.")

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
# 9. PREVIEW + ANALYSE
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

        try:
            with st.spinner("Analysing your form with FormFriend AI... ⏳"):
                text, provider = ask_ai(
                    build_system_prompt(language), first_message, images
                )
                docs = extract_documents(text, language)

            ss.history = first_message + [
                {"role": "assistant", "content": text}
            ]
            ss.result = text
            ss.result_language = language
            ss.provider = provider
            ss.analysis_images = images
            ss.fingerprint = current_fingerprint
            ss.docs = docs

        except AIError as error:
            show_errors(error.errors, debug)


# ==========================================
# 10. RESULT, TOOLS AND FOLLOW-UP CHAT
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
    for message in ss.history[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about this form...")
    if question:
        ss.history.append({"role": "user", "content": question})
        try:
            with st.spinner("Thinking... 🤔"):
                answer, provider = ask_ai(
                    build_system_prompt(language),
                    ss.history,
                    ss.analysis_images,
                )
            ss.history.append({"role": "assistant", "content": answer})
            ss.provider = provider
        except AIError as error:
            ss.history.pop()  # remove the question that got no answer
            show_errors(error.errors, debug)
        else:
            st.rerun()


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()
st.caption(
    "FormFriend gives general guidance. Always check the official "
    "website or office before you submit a form."
)
