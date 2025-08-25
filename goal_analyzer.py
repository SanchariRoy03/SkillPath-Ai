import os
import google.generativeai as genai

def analyze_goals(text, api_key=None):
    """
    Analyze career goals using Gemini AI model.
    Returns formatted markdown for Streamlit display.
    """
    if not text or not isinstance(text, str):
        return "⚠️ Please provide a valid career goal text."

    try:
        # Prefer API key from argument, fallback to environment
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            return "⚠️ Gemini API key not set. Please enter it in the sidebar."

        genai.configure(api_key=key)

        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Analyze the following career goal and identify:\n"
            "1. **Key Skills**\n"
            "2. **Relevant Roles**\n"
            "3. **Industry/Domain Keywords**\n"
            "Present the output as a neatly formatted markdown list for easy reading.\n\n"
            f"Career Goal:\n{text}"
        )

        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "⚠️ AI did not return a valid response."

    except Exception as e:
        return f"⚠️ Error analyzing goal with AI: {str(e)}"
