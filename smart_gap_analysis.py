import google.generativeai as genai
import streamlit as st

class SmartGapAnalysisError(Exception):
    pass

def get_smart_gap_analysis(resume_text: str, target_role: str, user_goal: str = "") -> str:
    """
    Performs a smart gap analysis using Gemini LLM.
    """
    if not st.session_state.get("gemini_key"):
        raise SmartGapAnalysisError("Gemini API key not configured in session state.")

    genai.configure(api_key=st.session_state.gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # or your preferred model

    prompt_parts = [
        f"Analyze the provided resume against the typical requirements and expectations for a professional in the role of '{target_role}'.",
        f"The user's career goal is: '{user_goal}'." if user_goal else "",
        f"\nResume Text:\n---\n{resume_text}\n---",
        "Please provide a concise, insightful gap analysis. Focus on:",
        "1. Key skills or technologies the resume demonstrates for this role.",
        f"2. Specific skills, experiences, or qualifications missing for an ideal '{target_role}' candidate.",
        "3. Suggest 1-2 actionable areas for improvement or focus.",
        "4. Present this as a narrative, not just a list.",
        f"Avoid generic advice. Be specific to the resume content and the role '{target_role}'."
    ]

    prompt = "\n".join(filter(None, prompt_parts))

    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            raise SmartGapAnalysisError("Received an empty response from the LLM.")
    except Exception as e:
        raise SmartGapAnalysisError(f"Failed to generate smart gap analysis: {e}")
