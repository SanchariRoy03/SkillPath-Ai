# smart_gap_analysis_error.py

class SmartGapAnalysisError(Exception):
    """
    Custom exception for Smart Gap Analysis errors.

    This exception is raised when:
    1. The Gemini API key is missing or not configured.
    2. The LLM fails to generate a valid gap analysis response.
    3. Any other error occurs during the smart gap analysis process.
    """
    pass
