import streamlit as st
import tempfile
import os
import re
import json
import fitz
import pandas as pd
import plotly.graph_objects as go
from roadmap_engine import generate_roadmap, present_roadmap, configure_gemini, RoadmapGenerationError
from goal_analyzer import analyze_goals
from PIL import Image
from PyPDF2 import PdfReader
from smart_gap_analysis import get_smart_gap_analysis, SmartGapAnalysisError
# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="SkillPath AI", page_icon="📈", layout="wide")

# -----------------------------
# Theme Setter
# -----------------------------
def set_theme(theme):
    if theme == "Dark":
        bg_gradient = "linear-gradient(135deg, #0f172a, #1e293b)"
        text_color = "#f1f5f9"
        label_color = "#f1f5f9"
        logo_color = "#f1f5f9"
        sidebar_bg = "#1e293b"
    else:
        bg_gradient = "linear-gradient(135deg, #ffffff, #f1f5f9)"
        text_color = "#000000"
        label_color = "#000000"
        logo_color = "#000000"
        sidebar_bg = "#f8f9fa"

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', sans-serif;
        color: {text_color} !important;
    }}
    .stApp {{
        background: {bg_gradient};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}

    /* Sidebar labels */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] .stCheckbox label p,
    section[data-testid="stSidebar"] .stSlider label {{
        color: {label_color} !important;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }}

    /* Main widget labels */
    div[data-baseweb="select"] label,
    div[data-baseweb="input"] label,
    div[data-baseweb="textarea"] label,
    section.main label,
    .stFileUploader label,
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stMultiSelect label,
    .stDateInput label,
    .stTimeInput label {{
        color: {label_color} !important;
        font-weight: 700;
    }}

    /* ✅ Fix skill / course / expander text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stExpander, .stExpander p, .stExpander span,
    [data-testid="stMarkdownContainer"] * {{
        color: {text_color} !important;
    }}

    /* Ensure DataFrame text adapts */
    [data-testid="stDataFrame"] .row_heading,
    [data-testid="stDataFrame"] .blank,
    [data-testid="stDataFrame"] .col_heading,
    [data-testid="stDataFrame"] .cell {{
        color: {text_color} !important;
    }}

    /* Tagline + headings */
    .tagline-text {{
        font-size: 1rem;
        font-style: italic;
        color: {logo_color} !important;
    }}
    .main-heading {{
        color: {logo_color} !important;
    }}
    .main-subheading {{
        color: {logo_color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
# -----------------------------
# Sidebar: Toggleable Settings
# -----------------------------
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

# Apply theme first, regardless of button click
theme_choice = st.session_state.get("theme_choice", "Dark")
set_theme(theme_choice)

# Settings button
if st.sidebar.button("⚙️ Settings"):
    st.session_state.show_settings = not st.session_state.show_settings

# Only show settings controls when toggled
if st.session_state.show_settings:
    # Theme selector
    theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0 if theme_choice=="Light" else 1)
    st.session_state.theme_choice = theme_choice
    set_theme(theme_choice)

    # Other settings
    bar_height = st.sidebar.slider("Bar Height per Skill (px)", min_value=30, max_value=80, value=st.session_state.get("bar_height", 50))
    st.session_state.bar_height = bar_height

    show_tables = st.sidebar.checkbox("Show Skills Table", value=st.session_state.get("show_tables", True))
    st.session_state.show_tables = show_tables

    # Gemini API Key
    # -----------------------------
# -----------------------------
# Gemini API Key (Hidden)
# -----------------------------
if "gemini_key" not in st.session_state:
    # Load from secrets.toml
    st.session_state.gemini_key = st.secrets.get("GEMINI_API_KEY", "")

# Configure Gemini if key exists
if st.session_state.gemini_key:
    try:
        configure_gemini(st.session_state.gemini_key)
    except RoadmapGenerationError as e:
        st.sidebar.error(f"Gemini config error: {e}")
else:
    st.sidebar.warning("⚠️ Gemini API key not found in secrets.toml, roadmap generation disabled.")




# -----------------------------
# Branding (above uploader)
# -----------------------------
# -----------------------------
# Branding (above uploader)
# -----------------------------
logo = Image.open("logo.png")

col1, col2 = st.columns([2, 18])  # ratio between logo and text
with col1:
    st.image(logo, width=150)  # ⬅️ increase size here (try 150–220)
with col2:
    st.markdown("""
        <h1 class='main-heading' style='margin-bottom:0; font-size:2.5rem;'>
            SkillPath AI
        </h1>
        <p class='main-subheading' style='margin-top:0; font-size:1.3rem; font-style:italic;'>
            Your AI-powered career navigator
        </p>
    """, unsafe_allow_html=True)

st.markdown("---")


# -----------------------------
# Load Skills Data
# -----------------------------
with open("skills_data.json", "r") as f:
    skills_data = json.load(f)

required_skills_data = skills_data["required_skills"]
expanded_terms = skills_data["expanded_skill_terms"]
course_recommendations = skills_data["course_recommendations"]

# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def extract_skills_from_text(text, expanded_terms):
    cleaned_text = text.lower()
    found_skills = set()
    for skill, variants in expanded_terms.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant.lower())}\b", cleaned_text):
                found_skills.add(skill.lower())
                break
    return found_skills

def extract_extra_skills(resume_text, exclude_list):
    """
    Extract extra skills strictly from the SKILLS section of the resume,
    ignoring Education, Achievements, Certifications, and other sections.
    """
    # Normalize text
    text = resume_text.lower()

    # Extract only the Skills section
    match = re.search(
        r"(skills|technical skills)\s*[:\-]?\s*(.+?)(education|experience|projects|certifications|summary|$)",
        text, re.S
    )

    if not match:
        return []

    skills_section = match.group(2)

    # Split by common separators like commas, newlines, semicolons
    candidates = re.split(r"[,;\n]", skills_section)
    candidates = [c.strip() for c in candidates if c.strip()]

    # Remove duplicates & anything already in predefined skills
    extra_skills = [c for c in candidates if c.lower() not in exclude_list]

    # Optional: capitalize first letters properly
    extra_skills = [c.title() for c in extra_skills]

    return sorted(extra_skills)

def extract_extra_skills(resume_text, exclude_list):
    """
    Extract extra skills strictly from the SKILLS section of the resume,
    ignoring Education, Achievements, Certifications, etc.
    """
    text = resume_text.lower()

    # Extract only the Skills section
    match = re.search(
        r"(skills|technical skills)\s*[:\-]?\s*(.+?)(education|experience|projects|certifications|summary|$)",
        text, re.S
    )

    if not match:
        return []

    skills_section = match.group(2)

    # Split by common separators
    candidates = re.split(r"[,;\n]", skills_section)
    candidates = [c.strip() for c in candidates if c.strip()]

    # Remove duplicates & anything already in predefined skills
    extra_skills = [c for c in candidates if c.lower() not in exclude_list]

    # Capitalize properly
    extra_skills = [c.title() for c in extra_skills]

    return sorted(extra_skills)

def analyze_resume(resume_text, job_title):
    # Extract found skills from Skills section
    found_skills = extract_skills_from_text(resume_text, expanded_terms)

    # Extract extra skills from Skills section only
    extra_skills = extract_extra_skills(resume_text, exclude_list=found_skills)

    if job_title not in required_skills_data:
        return pd.DataFrame(), [], {}, extra_skills

    # Job-required skills
    job_skills = set(skill.lower() for skill in required_skills_data[job_title])
    matched_skills = found_skills & job_skills
    missing_skills = job_skills - found_skills  # 🔹 missing skills

    skills_df = pd.DataFrame({
        "Skill": list(job_skills),
        "Match (%)": [100 if s in matched_skills else 0 for s in job_skills],
        "Color": ['#2563eb' if s in matched_skills else '#ef4444' for s in job_skills]
    }).sort_values(by="Match (%)", ascending=True).reset_index(drop=True)

    course_suggestions = {
        skill: [{"title": course_recommendations.get(skill.capitalize(), "No course found"), "url": "#"}]
        for skill in missing_skills
    }

    return skills_df, list(missing_skills), course_suggestions, extra_skills


# -----------------------------
# Main UI
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
job_roles = sorted(list(required_skills_data.keys()))
desired_role = st.selectbox("🎯 Target Job Role", options=["Select a role"] + job_roles)
career_objective = st.text_area("📝 Career Objective (Optional)", placeholder="Your career goals...")

if st.button("🔍 Analyze Resume", type="primary"):
    if not uploaded_file:
        st.warning("Please upload your resume first.")
    elif desired_role == "Select a role":
        st.warning("Please select your target job role.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        resume_text = extract_text_from_pdf(tmp_path)
        skills_df, missing_skills, course_suggestions, extra_skills = analyze_resume(resume_text, desired_role)
        st.session_state.extra_skills = extra_skills
        st.session_state.skills_df = skills_df
        st.session_state.missing_skills = missing_skills
        st.session_state.course_suggestions = course_suggestions
        st.session_state.desired_role = desired_role
        st.session_state.career_objective = career_objective
        st.session_state.resume_uploaded = True

        if career_objective.strip():
            st.session_state.career_analysis = analyze_goals(career_objective.strip())
        else:
            st.session_state.career_analysis = None

        os.remove(tmp_path)

if st.session_state.get("resume_uploaded", False):
    skills_df = st.session_state.skills_df
    missing_skills = st.session_state.missing_skills
    course_suggestions = st.session_state.course_suggestions

    if st.session_state.get("career_analysis"):
        st.markdown("<div class='stCard'><h3>📊 Career Goal Analysis</h3>", unsafe_allow_html=True)
        st.write(st.session_state.career_analysis)
        st.markdown("</div>", unsafe_allow_html=True)

    # Smart Gap Analysis Button
    if st.session_state.get("resume_uploaded", False):
        if st.button("🤖 Generate Smart Gap Analysis"):
            try:
                analysis = get_smart_gap_analysis(
                    resume_text=resume_text,
                    target_role=st.session_state.desired_role,
                    user_goal=st.session_state.career_objective
                )
                st.markdown("<div class='stCard'><h3>📝 Smart Gap Analysis</h3>", unsafe_allow_html=True)
                st.write(analysis)
                st.markdown("</div>", unsafe_allow_html=True)
            except SmartGapAnalysisError as e:
                st.error(f"Gap Analysis Error: {e}")
    if st.session_state.get("show_tables", True):
        # Left column → matched/extracted skills table
        st.markdown("<div class='stCard'><h3>✅ Extracted Skills</h3>", unsafe_allow_html=True)
        st.dataframe(skills_df[["Skill", "Match (%)"]], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Separate section for missing skills
        st.markdown("<div class='stCard'><h3>❌ Missing Skills</h3>", unsafe_allow_html=True)
        if st.session_state.missing_skills:
            st.write(", ".join(f"**{skill}**" for skill in st.session_state.missing_skills))
        else:
            st.success("No missing skills detected!")
        st.markdown("</div>", unsafe_allow_html=True)

        # Separate section for additional skills found in resume
        if st.session_state.extra_skills:
            st.markdown("<div class='stCard'><h3>📌 Additional Skills Found in Resume</h3>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            for i, skill in enumerate(st.session_state.extra_skills):
                if i % 2 == 0:
                    col1.write(f"- {skill}")
                else:
                    col2.write(f"- {skill}")

            st.markdown("</div>", unsafe_allow_html=True)

    fig = go.Figure(data=[
        go.Bar(
            x=skills_df['Match (%)'],
            y=skills_df['Skill'],
            orientation='h',
            marker=dict(color=skills_df['Color']),
            text=skills_df['Match (%)'],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Skill Match Overview",
        xaxis=dict(title="Match %", range=[0, 100]),
        yaxis=dict(title="Skills"),
        height=max(300, st.session_state.get("bar_height", 50) * len(skills_df))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='stCard'><h3>📚 Suggested Courses</h3>", unsafe_allow_html=True)
    for skill, courses in course_suggestions.items():
        with st.expander(f"📌 {skill.capitalize()}"):
            for course in courses:
                st.markdown(f"- [{course['title']}]({course['url']})")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("gemini_key", ""):
        if st.button("🤖 Generate AI-Powered Roadmap"):
            try:
                prompt = (
                    f"Create a learning roadmap for becoming a {st.session_state.desired_role}. "
                    f"Missing skills: {', '.join(st.session_state.missing_skills)}. "
                    f"Career goal: {st.session_state.career_objective}"
                )
                roadmap_text = generate_roadmap(prompt)
                st.markdown("<div class='stCard'><h3>🛣️ AI-Powered Roadmap</h3>", unsafe_allow_html=True)
                present_roadmap(roadmap_text)
                st.markdown("</div>", unsafe_allow_html=True)
            except RoadmapGenerationError as e:
                st.error(str(e))
    else:
        st.warning("Enter your Gemini API key in the sidebar to enable AI roadmap generation.")

