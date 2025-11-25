import os
from typing import List, Optional

import google.generativeai as genai
from PIL import Image

# Model name (you can change if needed)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Configure API key (must be set in your environment or .env file)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. Please set it in your environment or .env file."
    )

genai.configure(api_key=API_KEY)


def call_gemini(
    prompt: str,
    image_paths: Optional[List[str]] = None,
) -> str:
    """
    Send prompt and optional images to Gemini LVLM using the official google-generativeai SDK.
    - prompt: user/system prompt text
    - image_paths: list of paths to image files (PNG/JPG, etc.)
    Returns the response text from Gemini.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Build parts list: first the text prompt
    parts: List = [prompt]

    # If images are provided, load them and append
    if image_paths:
        for path in image_paths:
            if not path:
                continue
            try:
                img = Image.open(path)
                parts.append(img)
            except Exception as e:
                # If some image fails to load, just skip it
                print(f"Warning: could not load image {path}: {e}")

    # Call Gemini with text (+ optional images)
    response = model.generate_content(parts)

    # Safely get the text output
    if hasattr(response, "text") and response.text is not None:
        return response.text
    return str(response)
