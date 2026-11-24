
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

# Load API keys
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page
st.title("FormFriend 🤝")
st.subheader("Your AI Application Guide")
st.write("Upload a form and let AI explain it in simple language.")

# Upload section
st.subheader("📄 Upload Your Application Form")

st.write("Choose a photo of your form from your phone or computer.")

st.info(
    "📱 Mobile users: Tap the image upload option and select "
    "your form photo from Gallery, Photos, or your phone's picker."
)

# Select photo from gallery/files
gallery_file = st.file_uploader(
    "🖼️ Choose a photo from Gallery / Photos",
    type=["jpg", "jpeg", "png"],
    help="Select an existing photo of your application form."
)

# Take a photo directly using camera
camera_file = st.camera_input(
    "📷 Take a photo of your form"
)

# Choose whichever image the user provided
uploaded_file = gallery_file if gallery_file is not None else camera_file

# Prompt
prompt = """
You are FormFriend, a friendly application guide.

Analyze the uploaded application form.

Explain in simple English:

1. What is this form used for?
2. Who should fill it?
3. Explain the fields in simple language.
4. List documents mentioned or apparently required.
5. Explain how to complete it step by step.
6. Identify confusing or unclear sections.

Do not invent official requirements.
Clearly state anything you cannot verify.
Do not request passwords or OTPs.
"""

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Form")

    if st.button("Understand My Form"):

        image_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type

        # Try Gemini first
        try:
            if not GEMINI_API_KEY:
                raise Exception("Gemini API key is missing.")

            st.info("ANALYSING YOUR DOCUMENT.....")

            gemini_client = genai.Client(
                api_key=GEMINI_API_KEY
            )

            response = gemini_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    )
                ]
            )

            st.success("Analysis completed!")
            st.markdown(response.text)

        # If Gemini fails, try Groq
        except Exception as gemini_error:

            st.warning(
                "This may Take some time......Please wait "
            )

            try:
                if not GROQ_API_KEY:
                    raise Exception("Groq API key is missing.")

                groq_client = Groq(
                    api_key=GROQ_API_KEY
                )

                image_base64 = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                groq_response = (
                    groq_client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": (
                                                f"data:{mime_type};"
                                                f"base64,{image_base64}"
                                            )
                                        }
                                    }
                                ]
                            }
                        ]
                    )
                )

                st.success("Analysis completed....")
                st.markdown(
                    groq_response.choices[0]
                    .message.content
                )

            except Exception as groq_error:

                st.error("Can't read your document at the movement Please try after some time.")

                st.write("Gemini error:")
                st.code(str(gemini_error))

                st.write("Groq error:")
                st.code(str(groq_error))
