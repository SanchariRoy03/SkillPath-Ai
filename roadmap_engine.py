# roadmap_engine.py
import google.generativeai as genai
import time
import streamlit as st

# -----------------------------
# Global variable for Gemini API key
# -----------------------------
gemini_api_key = "AIzaSyBDAluH00q61idrFJX50Tmh1csJ363g0Vc"

def configure_gemini(key: str):
    """Set Gemini API key globally for roadmap generation."""
    global gemini_api_key
    gemini_api_key = key
    genai.configure(api_key=gemini_api_key)

# -----------------------------
# Custom Exception
# -----------------------------
class RoadmapGenerationError(Exception):
    """Raised when roadmap generation fails."""
    pass

# -----------------------------
# Prompt validation
# -----------------------------
def validate_prompt(prompt: str) -> bool:
    """Validate the prompt for roadmap generation."""
    if not prompt or not isinstance(prompt, str):
        return False
    if len(prompt) < 50:
        return False
    if len(prompt) > 4000:  # Gemini's context window limit
        return False
    return True

# -----------------------------
# Roadmap Generation
# -----------------------------
def generate_roadmap(prompt: str, max_retries: int = 3) -> str:
    """
    Generate a learning roadmap using Google's Gemini model.

    Args:
        prompt (str): The prompt for roadmap generation
        max_retries (int): Maximum number of retry attempts

    Returns:
        str: Generated roadmap text

    Raises:
        RoadmapGenerationError: If generation fails after retries
    """
    if not validate_prompt(prompt):
        raise RoadmapGenerationError(
            "Invalid prompt. Please provide a detailed prompt between 50 and 4000 characters."
        )

    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            if not response or not response.text:
                raise RoadmapGenerationError("Empty response from model")

            return response.text.strip()

        except Exception as e:
            last_error = str(e)
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2 ** retry_count)  # Exponential backoff
            continue

    raise RoadmapGenerationError(
        f"Failed to generate roadmap after {max_retries} attempts. Last error: {last_error}"
    )

# -----------------------------
# Present roadmap in Streamlit
# -----------------------------
def present_roadmap(roadmap_text: str):
    """Display the roadmap in Streamlit."""
    st.markdown("### 📅 6-Month Learning Roadmap")
    st.markdown(roadmap_text)
