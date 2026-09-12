# 🤖 AI Resume Screening and Job Recommendation System

## 📌 Project Overview

The **AI Resume Screening and Job Recommendation System** is a Python-based application that analyzes a candidate's resume and compares it with a job description.

The system uses **NLP and machine learning techniques** to identify matching skills, missing skills, calculate resume-job similarity, and recommend suitable job roles.

This project is developed as an **academic/demo project** to demonstrate AI, NLP, machine learning, and Streamlit concepts.

## 🎯 Objectives

- Analyze resume content automatically
- Extract relevant skills from resumes
- Compare resume skills with job requirements
- Identify matching and missing skills
- Calculate semantic similarity between resume and job description
- Recommend suitable career roles
- Provide career improvement suggestions
- Display results through an interactive Streamlit interface

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyPDF2
- Sentence Transformers
- PyTorch
- Torchvision
- Streamlit

## 🧠 Machine Learning Models

The project includes machine learning experiments using:

- Decision Tree
- Logistic Regression
- Random Forest

The models are evaluated using accuracy, confusion matrix, and classification report.

## 🔤 NLP Component

The system uses the **Sentence Transformers** model:

`all-MiniLM-L6-v2`

It converts resume and job-description text into numerical embeddings and calculates their semantic similarity.

## ⚙️ How the System Works

1. Upload a resume PDF.
2. Extract text from the resume.
3. Enter or provide a job description.
4. Extract relevant skills.
5. Compare resume skills with required skills.
6. Identify matching and missing skills.
7. Calculate semantic similarity.
8. Generate an overall alignment score.
9. Recommend suitable job roles.
10. Provide career improvement suggestions.

## 📊 Dataset

The project uses a resume screening dataset containing features such as:

- Years of Experience
- Skills Match Score
- Education Level
- Project Count
- Resume Length
- GitHub Activity
- Shortlisted

## 💼 Job Roles

The system can provide recommendations for roles such as:

- Data Scientist
- Data Analyst
- Python Developer
- Business Intelligence Analyst

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Resume-Screening-and-Job-Recommendation.git
