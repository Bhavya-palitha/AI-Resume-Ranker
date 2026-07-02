import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background:#0F172A;
}

section[data-testid="stSidebar"]{
    background:#0F172A;
    border-right:2px solid #2563EB;
}
            
.block-container{
    padding-top:1.5rem;
    padding-left:3rem;
    padding-right:3rem;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(135deg,#3B82F6,#2563EB);
    color:white;
    border:none;
    border-radius:14px;
    font-size:18px;
    font-weight:700;
    height:56px;
    transition:0.3s;
}

.stButton>button:hover{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    transform:translateY(-2px);
    box-shadow:0 10px 25px rgba(37,99,235,.35);
}

/* Metrics */
div[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #334155;
    border-radius:18px;
    padding:18px;
}

/* Tables */
div[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

/* Cards */
.score-card{
    background:#1E293B;
    border:1px solid #334155;
    border-radius:18px;
    padding:20px;
    margin-bottom:20px;
    box-shadow:0 10px 30px rgba(0,0,0,.25);
}

/* Upload Box */
[data-testid="stFileUploader"]{
    background:#111827;
    border-radius:15px;
    border:2px dashed #334155;
    padding:15px;
}

h1,h2,h3{
color:white;
}

</style>
""",unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

# -------------------------------
# HEADER
# -------------------------------

st.markdown("""
<div style="
display:flex;
justify-content:space-between;
align-items:center;
padding:10px 0px 25px 0px;
">

<div>

<h1 style="
font-size:54px;
font-weight:900;
color:white;
margin:0;
">
🤖 AI Resume Ranker
</h1>

<p style="
font-size:22px;
color:#94A3B8;
margin-top:5px;
">
Enterprise Applicant Tracking System
</p>

</div>

<div style="
background:#111827;
padding:15px 25px;
border-radius:15px;
border:1px solid #334155;
">

<h3 style="color:#22C55E;margin:0;">
🟢 AI ONLINE
</h3>

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:linear-gradient(135deg,#1D4ED8,#3B82F6);
padding:28px;
border-radius:20px;
box-shadow:0px 15px 35px rgba(0,0,0,.35);
margin-top:20px;
margin-bottom:30px;
">

<h1 style="color:white;font-size:42px;font-weight:800;">
🚀 Intelligent Resume Screening Platform
            </h1>

<p style="color:#F8FAFC;font-size:22px;line-height:1.9;font-weight:500;">

Upload a <b>Job Description</b> and multiple <b>Resume PDFs</b>.

<br><br>

Our AI automatically:

</p>

<ul style="color:white;font-size:18px;line-height:2;">

<li>✅ Understands the Job Description</li>

<li>✅ Reads Every Resume</li>

<li>✅ Detects Skills & Experience</li>

<li>✅ Calculates Semantic Match Score</li>

<li>✅ Recommends the Best Candidate</li>

</ul>

</div>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------

# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:10px;">
        <h2 style="color:#FFFFFF;margin-bottom:0;">🏢 Recruiter Workspace</h2>
        <p style="color:#94A3B8;">Enterprise AI Hiring Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📄 Job Description")

    job_description = st.text_area(
        "",
        height=220,
        placeholder="""Example:

Looking for a Python Developer

Skills:
• Python
• SQL
• Machine Learning
• Streamlit
• Communication
"""
    )

    st.markdown("---")

    st.markdown("### 📂 Upload Resume PDFs")

    uploaded_files = st.file_uploader(
        "",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more candidate resumes"
    )

    st.markdown("---")

    rank_button = st.button(
        "🚀 Analyze Candidates",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("""
    <div style="
    background:#0B1220;
    padding:15px;
    border-radius:12px;
    border:1px solid #334155;
    ">

    <h4 style="color:white;">📌 Workflow</h4>

    <p style="color:#CBD5E1;">

    1️⃣ Paste Job Description<br><br>

    2️⃣ Upload Resume PDFs<br><br>

    3️⃣ Click Analyze<br><br>

    4️⃣ View AI Ranking

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.success("✅ Ready for Analysis")

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# PDF READER
# -------------------------------

def extract_text(pdf):

    text=""

    with pdfplumber.open(pdf) as p:

        for page in p.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text

# -------------------------------
# SCORING FUNCTION
# -------------------------------

def calculate_score(job, resume):

    embeddings = model.encode([job, resume])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    score = round(similarity * 100, 2)

    resume_lower = resume.lower()

    bonus = 0
    skills = []
    highlights = []

    keywords = [
        "python","java","c","c++","sql","mysql",
        "html","css","javascript","react",
        "machine learning","deep learning",
        "tensorflow","pytorch",
        "aws","docker","git","linux",
        "streamlit","flask","django"
    ]

    for skill in keywords:
        if skill in resume_lower:
            skills.append(skill.title())
            bonus += 1

    if "project" in resume_lower:
        bonus += 5
        highlights.append("Projects")

    if "internship" in resume_lower:
        bonus += 5
        highlights.append("Internship")

    if "certificate" in resume_lower or "certification" in resume_lower:
        bonus += 3
        highlights.append("Certification")

    if "leadership" in resume_lower:
        bonus += 2
        highlights.append("Leadership")

    final = min(score + bonus, 100)

    return final, skills, highlights
# -------------------------------
# MAIN LOGIC
# -------------------------------

if rank_button:

    if job_description=="":
        st.error("Please enter Job Description")
        st.stop()

    if uploaded_files is None or len(uploaded_files)==0:
        st.error("Please upload resumes")
        st.stop()

    results=[]

    progress=st.progress(0)

    for i,file in enumerate(uploaded_files):

        resume_text=extract_text(file)

        score, skills, highlights = calculate_score(
    job_description,
    resume_text
)

        results.append({
            "Candidate": file.name,
            "Score": score,
            "Skills": ", ".join(skills),
            "Highlights": ", ".join(highlights)
        })

        progress.progress((i+1)/len(uploaded_files))
            # -------------------------------
    # SORT RESULTS
    # -------------------------------

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Score",
        ascending=False
    ).reset_index(drop=True)

    df.index = df.index + 1

    st.success("✅ Ranking Completed Successfully!")
    st.markdown("## 📊 Recruiter Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("📄 Resumes", len(df))

    with c2:
        st.metric("🏆 Highest Score", f"{df['Score'].max()}%")

    with c3:
        st.metric("📈 Average Score", f"{round(df['Score'].mean(),1)}%")

    with c4:
        st.metric("⭐ Selected", df.iloc[0]["Candidate"])

   

    st.subheader("🏆 Candidate Leaderboard")

    for idx, row in df.iterrows():

        if idx == 1:
            medal = "🥇"
        elif idx == 2:
            medal = "🥈"
        elif idx == 3:
            medal = "🥉"
        else:
            medal = "⭐"

        st.markdown(f"""
        <div class="score-card">

        <h3>{medal} {row['Candidate']}</h3>

        <div class="metric">{row['Score']}%</div>

        <p>💻 <b>Skills:</b> {row['Skills']}</p>

        <p>⭐ <b>Highlights:</b> {row['Highlights']}</p>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Ranking Table")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🤖 AI Recommendation")

    best = df.iloc[0]

    st.markdown(f"""
    <div style="
    background:#064E3B;
    padding:25px;
    border-radius:18px;
    border-left:8px solid #22C55E;
    ">

    <h2 style="color:#F8FAFC;">
    🏆 Best Candidate
    </h2>

    <h1 style="color:#22C55E;">
    {best['Candidate']}
    </h1>

    <h3 style="color:white;">
    AI Match Score : {best['Score']}%
    </h3>

    <p style="color:#E2E8F0;">
    This candidate is highly recommended based on semantic similarity,
    skills detected and overall resume quality.
    </p>

    </div>
    """, unsafe_allow_html=True)
    st.metric(
        label="🏅 Best Match",
        value=best["Candidate"],
        delta=f"{best['Score']}%"
    )
    st.divider()

    st.subheader("📈 Candidate Scores")

    chart = df.set_index("Candidate")

    st.line_chart(chart["Score"])

    st.divider()

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Ranked Candidates CSV",
        data=csv,
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )

    st.divider()

    st.info(
        "💡 AI Ranking is based on semantic similarity using Sentence Transformers with bonus scoring for Projects, Internships, Certifications, and Leadership experience."
    )
else:
    st.markdown("""
    <h2 style="color:white;">🚀 Start Screening Candidates</h2>

    <div style="
    background:#1E293B;
    padding:25px;
    border-radius:18px;
    border:1px solid #334155;
    ">

    <p style="color:white;font-size:20px;">

    <b>Step 1</b> 📄 Paste the Job Description

    <br><br>

    <b>Step 2</b> 📂 Upload Resume PDFs

    <br><br>

    <b>Step 3</b> 🤖 Click <b>Rank Candidates</b>

    <br><br>

    <b>Step 4</b> 🏆 View AI Ranking Dashboard

    <br><br>

    <b>Step 5</b> 📥 Download the Final Report

    </p>

    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <center>

    <h3 style="color:white;">
    AI Resume Ranker Pro
    </h3>

    <p style="color:#94A3B8;">
    Enterprise Applicant Tracking System
    </p>

    </center>
    """, unsafe_allow_html=True)