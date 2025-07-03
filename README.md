# 🏥 AI-Powered Health & Wellness System

This is a full-stack AI/ML health monitoring and recommendation system built using **Python**, **Streamlit**, and **Machine Learning**. It provides users with real-time health risk assessments, personalized wellness recommendations, and an interactive AI chatbot — all in one place.

---

## 🚀 Features

- 🔍 **Health Risk Assessment** using Random Forest Classifier
- 💡 **Personalized Diet and Exercise Recommendations**
- 📈 **Activity Tracking** with beautiful Plotly charts
- 🤖 **AI Chatbot Assistant** powered by LLM-style natural language logic
- 🗃️ **SQLite3 Database Integration** to persist user metrics and progress
- 📊 **Interactive Visuals** with Plotly for trends and reports

---

## 🧠 Tech Stack

- **Language:** Python 3.10
- **Framework:** Streamlit
- **ML Model:** Random Forest (via Scikit-learn)
- **Visualization:** Plotly
- **Database:** SQLite3
- **Deployment-Ready:** Can be hosted on Streamlit Cloud / Render / Hugging Face Spaces

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-health-wellness-system.git
   cd ai-health-wellness-system
2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
    venv\Scripts\activate      # On Windows
    # OR
    source venv/bin/activate   # On macOS/Linux
   
   pip install -r requirements.txt
   
    streamlit run health_app.py

App Modules

Module	                   Description

health_app.py	            Main Streamlit frontend + navigation
HealthRiskPredictor	      ML model for predicting health risk levels
RecommendationEngine	   Diet & exercise suggestions based on risk profile
HealthChatbot	            LLM-inspired chatbot for general health queries
SQLite3 Database	        Tracks user data and health metrics persistently

