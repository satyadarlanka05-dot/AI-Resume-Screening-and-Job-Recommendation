import streamlit as st
import pandas as pd
import re
import os

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Resume Screening & Job Recommendation System")

st.write(
    """
    Upload a resume PDF to analyze its skills, compare it with a
    job description, identify missing skills, and recommend suitable
    job roles.
    """
)

st.info(
    "🎓 Academic Project Demo: This system provides resume-job "
    "alignment information and career guidance. It should not be "
    "used as an automated hiring decision system."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📌 About the Project")

st.sidebar.write(
    """
    This project uses:

    • Python  
    • Pandas  
    • Machine Learning concepts  
    • NLP  
    • Sentence Transformers  
    • Semantic Similarity  
    • Skill Matching  
    • Job Recommendation  
    • Streamlit
    """
)


# ============================================================
# SKILLS LIST
# ============================================================

skills_list = [
    "python",
    "java",
    "c",
    "sql",
    "dbms",
    "html",
    "css",
    "javascript",
    "excel",
    "tableau",
    "power bi",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai",
    "data science",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "tensorflow",
    "scikit-learn",
    "git",
    "github"
]


# ============================================================
# JOB ROLES
# ============================================================

job_roles = {
    "Data Scientist": [
        "python",
        "machine learning",
        "pandas",
        "numpy",
        "sql"
    ],

    "Data Analyst": [
        "python",
        "sql",
        "tableau",
        "power bi",
        "excel"
    ],

    "Python Developer": [
        "python",
        "sql",
        "git"
    ],

    "Business Intelligence Analyst": [
        "sql",
        "tableau",
        "power bi",
        "excel"
    ]
}


# ============================================================
# JOB DESCRIPTION
# ============================================================

default_job_description = """
We are looking for a Data Scientist with skills in
Python, SQL, Machine Learning, Pandas, NumPy,
Data Science, Tableau, Power BI and GitHub.
"""


# ============================================================
# LOAD NLP MODEL
# ============================================================

@st.cache_resource
def load_semantic_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_resume_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    return resume_text


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):

    text_lower = text.lower()

    found_skills = []

    for skill in skills_list:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text_lower):

            found_skills.append(skill)

    return found_skills


# ============================================================
# CALCULATE SKILL MATCH
# ============================================================

def calculate_skill_match(found_skills, required_skills):

    found_set = set(found_skills)

    required_set = set(required_skills)

    matching_skills = found_set.intersection(required_set)

    missing_skills = required_set.difference(found_set)

    if len(required_set) > 0:

        score = (
            len(matching_skills)
            / len(required_set)
        ) * 100

    else:

        score = 0

    return (
        matching_skills,
        missing_skills,
        score
    )


# ============================================================
# CALCULATE SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_similarity(
    model,
    resume_text,
    job_description
):

    resume_embedding = model.encode(
        [resume_text]
    )

    job_embedding = model.encode(
        [job_description]
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    score = similarity * 100

    return score


# ============================================================
# JOB RECOMMENDATION
# ============================================================

def recommend_jobs(found_skills):

    job_scores = {}

    found_set = set(found_skills)

    for job, required_skills in job_roles.items():

        required_set = set(required_skills)

        matched = found_set.intersection(
            required_set
        )

        if len(required_set) > 0:

            score = (
                len(matched)
                / len(required_set)
            ) * 100

        else:

            score = 0

        job_scores[job] = score

    return job_scores


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Upload Resume")

uploaded_file = st.file_uploader(
    "Upload your resume PDF",
    type=["pdf"]
)


# ============================================================
# JOB DESCRIPTION INPUT
# ============================================================

st.header("💼 Job Description")

job_description = st.text_area(
    "Enter the job description",
    value=default_job_description,
    height=180
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload a resume PDF first."
        )

    elif not job_description.strip():

        st.warning(
            "⚠️ Please enter a job description."
        )

    else:

        # ----------------------------------------------------
        # Extract Resume Text
        # ----------------------------------------------------

        with st.spinner(
            "📖 Reading resume..."
        ):

            resume_text = extract_resume_text(
                uploaded_file
            )


        if not resume_text.strip():

            st.error(
                "❌ Could not extract text from this PDF."
            )

            st.stop()


        # ----------------------------------------------------
        # Display Resume Text
        # ----------------------------------------------------

        with st.expander(
            "📄 View Extracted Resume Text"
        ):

            st.write(resume_text)


        # ----------------------------------------------------
        # Extract Skills
        # ----------------------------------------------------

        found_skills = extract_skills(
            resume_text
        )


        # ----------------------------------------------------
        # Required Skills from Job Description
        # ----------------------------------------------------

        required_skills = extract_skills(
            job_description
        )


        # ----------------------------------------------------
        # Skill Matching
        # ----------------------------------------------------

        (
            matching_skills,
            missing_skills,
            skill_match_score
        ) = calculate_skill_match(
            found_skills,
            required_skills
        )


        # ----------------------------------------------------
        # Load Semantic Model
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Calculating semantic similarity..."
        ):

            semantic_model = load_semantic_model()

            semantic_score = calculate_semantic_similarity(
                semantic_model,
                resume_text,
                job_description
            )


        # ----------------------------------------------------
        # Overall Resume-Job Alignment
        # ----------------------------------------------------

        alignment_score = (
            0.5 * skill_match_score
            + 0.5 * semantic_score
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            "✅ Resume analysis completed!"
        )


        # ----------------------------------------------------
        # SCORE CARDS
        # ----------------------------------------------------

        st.header("📊 Resume Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Skill Match",
                f"{skill_match_score:.2f}%"
            )

        with col2:

            st.metric(
                "Semantic Similarity",
                f"{semantic_score:.2f}%"
            )

        with col3:

            st.metric(
                "Overall Alignment",
                f"{alignment_score:.2f}%"
            )


        # ----------------------------------------------------
        # FOUND SKILLS
        # ----------------------------------------------------

        st.header("🛠️ Skills Found in Resume")

        if found_skills:

            for skill in found_skills:

                st.write(
                    f"✅ {skill}"
                )

        else:

            st.warning(
                "No skills from the predefined skill list were found."
            )


        # ----------------------------------------------------
        # REQUIRED SKILLS
        # ----------------------------------------------------

        st.header("🎯 Skills Required by Job")

        if required_skills:

            for skill in required_skills:

                st.write(
                    f"📌 {skill}"
                )

        else:

            st.warning(
                "No predefined skills were detected in the job description."
            )


        # ----------------------------------------------------
        # MATCHING SKILLS
        # ----------------------------------------------------

        st.header("✅ Matching Skills")

        if matching_skills:

            for skill in sorted(matching_skills):

                st.write(
                    f"✅ {skill}"
                )

        else:

            st.write(
                "No matching skills found."
            )


        # ----------------------------------------------------
        # MISSING SKILLS
        # ----------------------------------------------------

        st.header("📚 Skills to Improve")

        if missing_skills:

            for skill in sorted(missing_skills):

                st.write(
                    f"📖 {skill}"
                )

        else:

            st.success(
                "🎉 All detected required skills are present!"
            )


        # ====================================================
        # JOB RECOMMENDATION
        # ====================================================

        st.header("💼 Recommended Job Roles")

        job_scores = recommend_jobs(
            found_skills
        )


        sorted_jobs = sorted(
            job_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )


        for job, score in sorted_jobs:

            st.write(
                f"**{job}** — {score:.2f}% skill alignment"
            )

            st.progress(
                int(min(score, 100))
            )


        # ----------------------------------------------------
        # BEST ROLE
        # ----------------------------------------------------

        if sorted_jobs:

            best_job = sorted_jobs[0][0]

            best_job_score = sorted_jobs[0][1]

            st.subheader(
                "🌟 Best Skill-Based Role Match"
            )

            st.success(
                f"{best_job} — "
                f"{best_job_score:.2f}%"
            )


        # ====================================================
        # CAREER ADVICE
        # ====================================================

        st.header("💡 Career Advice")

        if missing_skills:

            st.write(
                """
                Based on the detected skill gaps, you can improve
                your profile by learning the missing skills and
                creating practical projects using those technologies.
                """
            )

            st.write(
                "**Recommended learning areas:**"
            )

            for skill in sorted(missing_skills):

                st.write(
                    f"📘 Learn {skill}"
                )

        else:

            st.write(
                """
                Your resume contains the detected skills required
                by this job description. Consider strengthening your
                profile with practical projects and interview preparation.
                """
            )


        # ====================================================
        # FINAL REPORT
        # ====================================================

        st.header("📋 Resume Screening Report")

        result = {

            "Resume File":
                uploaded_file.name,

            "Skill Match Score":
                round(skill_match_score, 2),

            "Semantic Similarity":
                round(semantic_score, 2),

            "Overall Alignment":
                round(alignment_score, 2),

            "Matching Skills":
                ", ".join(
                    sorted(matching_skills)
                ),

            "Missing Skills":
                ", ".join(
                    sorted(missing_skills)
                ),

            "Best Job Role":
                best_job if sorted_jobs else "N/A",

            "Best Job Skill Alignment":
                round(best_job_score, 2)
                if sorted_jobs else 0

        }


        result_df = pd.DataFrame(
            [result]
        )


        st.dataframe(
            result_df,
            use_container_width=True
        )


        # ====================================================
        # DOWNLOAD RESULT
        # ====================================================

        csv_data = result_df.to_csv(
            index=False
        )


        st.download_button(
            label="⬇️ Download Analysis Report",
            data=csv_data,
            file_name="resume_screening_result.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Resume Screening & Job Recommendation System | "
    "Academic Project"
)