import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sqlite3
import hashlib

# Configuration
st.set_page_config(
    page_title="Health & Wellness System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database Setup
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('health_app.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            height REAL,
            weight REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create health metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date DATE,
            heart_rate INTEGER,
            steps INTEGER,
            sleep_hours REAL,
            calories_burned INTEGER,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Create recommendations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            rec_type TEXT,
            recommendation TEXT,
            date DATE,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Health Risk Prediction Model
class HealthRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['age', 'bmi', 'heart_rate', 'steps', 'sleep_hours', 'systolic_bp']
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample health data for training"""
        np.random.seed(42)
        
        # Generate realistic health data
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'heart_rate': np.random.randint(60, 100, n_samples),
            'steps': np.random.randint(2000, 15000, n_samples),
            'sleep_hours': np.random.normal(7.5, 1.5, n_samples),
            'systolic_bp': np.random.randint(90, 160, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create risk labels based on realistic health criteria
        risk_labels = []
        for _, row in df.iterrows():
            risk_score = 0
            
            # Age factor
            if row['age'] > 60:
                risk_score += 2
            elif row['age'] > 45:
                risk_score += 1
                
            # BMI factor
            if row['bmi'] > 30:
                risk_score += 2
            elif row['bmi'] > 25:
                risk_score += 1
                
            # Heart rate factor
            if row['heart_rate'] > 90:
                risk_score += 1
                
            # Steps factor
            if row['steps'] < 5000:
                risk_score += 1
                
            # Sleep factor
            if row['sleep_hours'] < 6:
                risk_score += 1
                
            # Blood pressure factor
            if row['systolic_bp'] > 140:
                risk_score += 2
            elif row['systolic_bp'] > 130:
                risk_score += 1
                
            # Assign risk level
            if risk_score >= 5:
                risk_labels.append('High')
            elif risk_score >= 3:
                risk_labels.append('Medium')
            else:
                risk_labels.append('Low')
                
        df['risk_level'] = risk_labels
        return df
    
    def train_model(self):
        """Train the health risk prediction model"""
        # Generate sample data
        data = self.generate_sample_data()
        
        # Prepare features
        X = data[self.feature_names]
        y = data['risk_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        accuracy = self.model.score(X_test_scaled, y_test)
        return accuracy
    
    def predict_risk(self, user_data):
        """Predict health risk for a user"""
        if self.model is None:
            self.train_model()
            
        # Prepare user data
        user_features = [user_data[feature] for feature in self.feature_names]
        user_scaled = self.scaler.transform([user_features])
        
        # Make prediction
        risk_level = self.model.predict(user_scaled)[0]
        probabilities = self.model.predict_proba(user_scaled)[0]
        
        # Get feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return {
            'risk_level': risk_level,
            'probabilities': dict(zip(self.model.classes_, probabilities)),
            'feature_importance': feature_importance
        }

# Recommendation System
class RecommendationEngine:
    def __init__(self):
        self.diet_recommendations = {
            'Low': [
                "Maintain a balanced diet with plenty of fruits and vegetables",
                "Include lean proteins like chicken, fish, and legumes",
                "Stay hydrated with 8-10 glasses of water daily",
                "Choose whole grains over refined carbohydrates"
            ],
            'Medium': [
                "Reduce sodium intake to less than 2300mg per day",
                "Increase omega-3 fatty acids with fatty fish twice a week",
                "Limit processed foods and added sugars",
                "Include more fiber-rich foods like oats and berries"
            ],
            'High': [
                "Consider a heart-healthy Mediterranean diet",
                "Drastically reduce saturated fats and trans fats",
                "Increase potassium-rich foods like bananas and spinach",
                "Consult with a registered dietitian for personalized meal planning"
            ]
        }
        
        self.exercise_recommendations = {
            'Low': [
                "Aim for 150 minutes of moderate exercise per week",
                "Include strength training 2-3 times per week",
                "Try activities like brisk walking, swimming, or cycling",
                "Add flexibility exercises like yoga or stretching"
            ],
            'Medium': [
                "Increase cardio to 200-300 minutes per week",
                "Focus on low-impact exercises to protect joints",
                "Include balance and coordination exercises",
                "Monitor heart rate during exercise"
            ],
            'High': [
                "Start with gentle activities like walking or water aerobics",
                "Gradually increase intensity under medical supervision",
                "Include stress-reduction activities like meditation",
                "Consider working with a certified personal trainer"
            ]
        }
    
    def get_recommendations(self, risk_level, user_data):
        """Get personalized recommendations based on risk level"""
        diet_recs = random.sample(self.diet_recommendations[risk_level], 3)
        exercise_recs = random.sample(self.exercise_recommendations[risk_level], 3)
        
        return {
            'diet': diet_recs,
            'exercise': exercise_recs
        }

# Enhanced Chatbot System
class HealthChatbot:
    def __init__(self):
        self.health_knowledge = {
            # Nutrition and Diet
            'diet': {
                'keywords': ['diet', 'food', 'nutrition', 'eat', 'meal', 'calories', 'weight loss', 'healthy eating'],
                'responses': {
                    'general': "A balanced diet includes fruits, vegetables, lean proteins, whole grains, and healthy fats. Aim for variety and moderation.",
                    'weight_loss': "For healthy weight loss, create a moderate calorie deficit through portion control and regular exercise. Aim for 1-2 pounds per week.",
                    'nutrients': "Focus on getting essential nutrients: protein for muscle health, fiber for digestion, omega-3s for heart health, and vitamins from colorful fruits and vegetables.",
                    'hydration': "Drink 8-10 glasses of water daily. Proper hydration supports metabolism, joint health, and cognitive function."
                }
            },
            
            # Exercise and Fitness
            'exercise': {
                'keywords': ['exercise', 'workout', 'fitness', 'gym', 'cardio', 'strength', 'training', 'activity', 'sports'],
                'responses': {
                    'general': "Regular exercise should include 150 minutes of moderate cardio weekly, plus 2-3 strength training sessions.",
                    'cardio': "Cardio exercises like walking, running, cycling, or swimming strengthen your heart and improve endurance.",
                    'strength': "Strength training builds muscle, improves bone density, and boosts metabolism. Include all major muscle groups.",
                    'flexibility': "Incorporate stretching or yoga to improve flexibility, reduce injury risk, and promote relaxation."
                }
            },
            
            # Sleep and Recovery
            'sleep': {
                'keywords': ['sleep', 'rest', 'tired', 'insomnia', 'bedtime', 'fatigue', 'energy'],
                'responses': {
                    'general': "Adults need 7-9 hours of quality sleep nightly. Good sleep improves immunity, memory, and emotional well-being.",
                    'hygiene': "Create a sleep routine: consistent bedtime, cool dark room, no screens 1 hour before bed, and avoid caffeine late in the day.",
                    'quality': "Signs of good sleep: falling asleep within 20 minutes, staying asleep, and waking up refreshed.",
                    'problems': "For persistent sleep issues, consider sleep disorders like sleep apnea. Consult a healthcare provider if problems continue."
                }
            },
            
            # Mental Health and Stress
            'mental_health': {
                'keywords': ['stress', 'anxiety', 'depression', 'mental health', 'mood', 'worried', 'emotional', 'therapy'],
                'responses': {
                    'general': "Mental health is as important as physical health. Regular self-care, social connections, and stress management are crucial.",
                    'stress': "Manage stress through deep breathing, meditation, regular exercise, adequate sleep, and time management techniques.",
                    'anxiety': "For anxiety, try mindfulness practices, progressive muscle relaxation, and limiting caffeine. Seek professional help if symptoms persist.",
                    'support': "Don't hesitate to reach out for professional help. Therapy, counseling, and support groups can be very beneficial."
                }
            },
            
            # Heart Health
            'heart_health': {
                'keywords': ['heart', 'blood pressure', 'cholesterol', 'cardiovascular', 'chest pain', 'heart rate'],
                'responses': {
                    'general': "Heart health depends on regular exercise, healthy diet, not smoking, managing stress, and maintaining healthy weight.",
                    'blood_pressure': "Normal blood pressure is less than 120/80. Reduce sodium, increase potassium, exercise regularly, and manage stress.",
                    'cholesterol': "Maintain healthy cholesterol with fiber-rich foods, lean proteins, healthy fats, and regular physical activity.",
                    'warning': "Seek immediate medical attention for chest pain, shortness of breath, or severe heart palpitations."
                }
            },
            
            # Preventive Care
            'preventive': {
                'keywords': ['checkup', 'screening', 'prevention', 'vaccine', 'doctor', 'health exam'],
                'responses': {
                    'general': "Regular preventive care includes annual checkups, screenings, vaccinations, and monitoring key health metrics.",
                    'screenings': "Important screenings include blood pressure, cholesterol, diabetes, cancer screenings, and bone density tests.",
                    'vaccines': "Stay up-to-date with recommended vaccines including flu shots, COVID-19 boosters, and age-appropriate immunizations."
                }
            },
            
            # Common Health Issues
            'common_issues': {
                'keywords': ['headache', 'back pain', 'joint pain', 'cold', 'flu', 'allergies', 'digestion'],
                'responses': {
                    'headache': "For tension headaches, try stress reduction, regular sleep, hydration, and gentle neck stretches. See a doctor for severe or frequent headaches.",
                    'back_pain': "Prevent back pain with good posture, core strengthening, proper lifting techniques, and ergonomic workspaces.",
                    'joint_pain': "Joint health benefits from regular low-impact exercise, maintaining healthy weight, and anti-inflammatory foods.",
                    'digestion': "Support digestive health with fiber-rich foods, probiotics, staying hydrated, and managing stress."
                }
            }
        }
    
    def analyze_intent(self, user_input):
        """Analyze user input to determine health topic and specific intent"""
        user_input_lower = user_input.lower()
        
        # Check for specific health topics
        for topic, data in self.health_knowledge.items():
            if any(keyword in user_input_lower for keyword in data['keywords']):
                # Determine specific subtopic
                for subtopic, response in data['responses'].items():
                    if subtopic != 'general':
                        subtopic_keywords = subtopic.replace('_', ' ').split()
                        if any(keyword in user_input_lower for keyword in subtopic_keywords):
                            return topic, subtopic
                return topic, 'general'
        
        # Check for greetings
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return 'greeting', None
        
        # Check for general health questions
        if any(word in user_input_lower for word in ['health', 'healthy', 'wellness', 'advice', 'tips']):
            return 'general_health', None
        
        return 'unknown', None
    
    def get_response(self, user_input):
        """Generate a comprehensive response based on user input"""
        topic, subtopic = self.analyze_intent(user_input)
        
        if topic == 'greeting':
            greetings = [
                "Hello! I'm your AI health assistant. I can help you with questions about nutrition, exercise, sleep, mental health, and general wellness. What would you like to know?",
                "Hi there! I'm here to provide evidence-based health guidance. Feel free to ask about diet, fitness, sleep, stress management, or any health concerns.",
                "Welcome! I'm your personal health AI. I can assist with wellness questions, healthy lifestyle tips, and general health information. How can I help you today?"
            ]
            return random.choice(greetings)
        
        elif topic == 'general_health':
            return """Here are key pillars of good health:
            
**Nutrition**: Eat a balanced diet with plenty of fruits, vegetables, lean proteins, and whole grains
**Exercise**: Aim for 150 minutes of moderate activity weekly plus strength training
**Sleep**: Get 7-9 hours of quality sleep nightly
**Stress Management**: Practice relaxation techniques and maintain work-life balance
**Hydration**: Drink 8-10 glasses of water daily
**Preventive Care**: Regular checkups and screenings

What specific area would you like to know more about?"""
        
        elif topic in self.health_knowledge:
            base_response = self.health_knowledge[topic]['responses'].get(subtopic, 
                           self.health_knowledge[topic]['responses']['general'])
            
            # Add personalized tips based on topic
            additional_tips = self.get_additional_tips(topic, subtopic)
            if additional_tips:
                return f"{base_response}\n\n**Additional Tips:**\n{additional_tips}"
            return base_response
        
        else:
            return """I'm here to help with health and wellness questions! I can provide information about:

**Nutrition & Diet** - Healthy eating, weight management, nutrients
**Exercise & Fitness** - Workout routines, cardio, strength training
**Sleep & Recovery** - Sleep hygiene, rest, energy management
**Mental Health** - Stress management, anxiety, emotional wellness
**Heart Health** - Blood pressure, cholesterol, cardiovascular fitness
**Preventive Care** - Checkups, screenings, health monitoring
**Common Issues** - Headaches, back pain, joint health, digestion

Please ask me about any of these topics, and I'll provide evidence-based guidance!"""
    
    def get_additional_tips(self, topic, subtopic):
        """Provide additional personalized tips based on topic"""
        tips = {
            'diet': "Try meal prepping on weekends to maintain healthy eating habits throughout the week.",
            'exercise': "Start with just 10-15 minutes daily if you're new to exercise. Consistency matters more than intensity.",
            'sleep': "Consider keeping a sleep diary to identify patterns and factors affecting your sleep quality.",
            'mental_health': "Practice the 5-4-3-2-1 grounding technique: 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
            'heart_health': "Take the stairs instead of elevators when possible - small changes add up for heart health!",
            'preventive': "Keep a health journal to track symptoms, medications, and questions for your doctor visits."
        }
        return tips.get(topic, "")
    
    def get_emergency_response(self, user_input):
        """Detect emergency situations and provide appropriate response"""
        emergency_keywords = ['chest pain', 'heart attack', 'stroke', 'severe pain', 'emergency', 'urgent', 'can\'t breathe']
        
        if any(keyword in user_input.lower() for keyword in emergency_keywords):
            return """⚠️ **EMERGENCY NOTICE** ⚠️
            
If you're experiencing a medical emergency, please:
• Call 911 (US) or your local emergency number immediately
• Go to the nearest emergency room
• Contact emergency services right away

For chest pain, difficulty breathing, severe injuries, or other urgent symptoms, seek immediate medical attention. This AI assistant cannot replace emergency medical care."""
        
        return None

# Initialize components
@st.cache_resource
def load_models():
    """Load and cache ML models"""
    predictor = HealthRiskPredictor()
    recommender = RecommendationEngine()
    chatbot = HealthChatbot()
    return predictor, recommender, chatbot

# Streamlit App
def main():
    # Initialize database
    init_database()
    
    # Load models
    predictor, recommender, chatbot = load_models()
    
    # Sidebar navigation
    st.sidebar.title("Health & Wellness AI")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Home", "Health Assessment", "Recommendations", "Activity Tracking", "AI Assistant"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Health Assessment":
        show_health_assessment(predictor)
    elif page == "Recommendations":
        show_recommendations(recommender)
    elif page == "Activity Tracking":
        show_activity_tracking()
    elif page == "AI Assistant":
        show_chatbot(chatbot)

def show_home_page():
    """Display home page"""
    st.title("Health & Wellness System Using AI/ML")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to Your Personal Health Assistant!
    
    This AI-powered system helps you:
    - **Assess your health risk** based on vital signs and lifestyle factors
    - **Get personalized recommendations** for diet and exercise
    - **Track your daily activities** and progress over time
    - **Chat with an AI assistant** for instant health guidance
    
    ### Key Features:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Risk Assessment**
        - Advanced ML algorithms
        - Personalized risk scoring
        - Feature importance analysis
        """)
    
    with col2:
        st.markdown("""
        **Smart Recommendations**
        - Tailored diet plans
        - Exercise routines
        - Lifestyle modifications
        """)
    
    with col3:
        st.markdown("""
        **Progress Tracking**
        - Daily metrics monitoring
        - Trend analysis
        - Goal setting & tracking
        """)
    
    st.markdown("---")
    st.info("Get started by navigating to the Health Assessment page to evaluate your current health status!")

def show_health_assessment(predictor):
    """Display health assessment page"""
    st.title("Health Risk Assessment")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Your Health Information")
        
        # User inputs
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
        col_a, col_b = st.columns(2)
        with col_a:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        
        with col_b:
            heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=70)
            steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
        
        col_c, col_d = st.columns(2)
        with col_c:
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.5, step=0.5)
        
        with col_d:
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
        
        if st.button("Assess Health Risk", type="primary"):
            # Calculate BMI
            bmi = weight / (height / 100) ** 2
            
            # Prepare user data
            user_data = {
                'age': age,
                'bmi': bmi,
                'heart_rate': heart_rate,
                'steps': steps,
                'sleep_hours': sleep_hours,
                'systolic_bp': systolic_bp
            }
            
            # Get prediction
            result = predictor.predict_risk(user_data)
            
            # Store in session state
            st.session_state.assessment_result = result
            st.session_state.user_data = user_data
    
    with col2:
        st.markdown("### Health Metrics Guide")
        st.markdown("""
        **Normal Ranges:**
        - BMI: 18.5-24.9
        - Heart Rate: 60-100 bpm
        - Sleep: 7-9 hours
        - Blood Pressure: <120/80
        - Steps: 8,000+ daily
        """)
    
    # Display results
    if hasattr(st.session_state, 'assessment_result'):
        st.markdown("---")
        st.markdown("### Assessment Results")
        
        result = st.session_state.assessment_result
        user_data = st.session_state.user_data
        
        # Risk level display
        risk_level = result['risk_level']
        risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Level", risk_level)
            st.markdown(f"<div style='background-color: {risk_colors[risk_level]}; padding: 10px; border-radius: 5px; color: white; text-align: center;'><strong>{risk_level} Risk</strong></div>", unsafe_allow_html=True)
        
        with col2:
            st.metric("BMI", f"{user_data['bmi']:.1f}")
            bmi_status = "Normal" if 18.5 <= user_data['bmi'] <= 24.9 else "Attention Needed"
            st.markdown(f"Status: {bmi_status}")
        
        with col3:
            confidence = max(result['probabilities'].values())
            st.metric("Confidence", f"{confidence:.1%}")
            st.markdown("Model certainty")
        
        # Probability distribution
        st.markdown("#### Risk Probability Distribution")
        prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Risk Level', 'Probability'])
        fig = px.bar(prob_df, x='Risk Level', y='Probability', color='Risk Level',
                     color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("#### What Influences Your Risk Score")
        importance_df = pd.DataFrame(list(result['feature_importance'].items()), columns=['Factor', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Factor', orientation='h',
                     title="Feature Importance in Risk Assessment")
        st.plotly_chart(fig, use_container_width=True)

def show_recommendations(recommender):
    """Display recommendations page"""
    st.title("Personalized Recommendations")
    st.markdown("---")
    
    if hasattr(st.session_state, 'assessment_result'):
        risk_level = st.session_state.assessment_result['risk_level']
        user_data = st.session_state.user_data
        
        recommendations = recommender.get_recommendations(risk_level, user_data)
        
        st.markdown(f"### Recommendations for {risk_level} Risk Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Diet Recommendations")
            for i, rec in enumerate(recommendations['diet'], 1):
                st.markdown(f"{i}. {rec}")
        
        with col2:
            st.markdown("#### Exercise Recommendations")
            for i, rec in enumerate(recommendations['exercise'], 1):
                st.markdown(f"{i}. {rec}")
        
        # Weekly meal plan visualization
        st.markdown("---")
        st.markdown("#### Sample Weekly Plan")
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        meals = ['Breakfast', 'Lunch', 'Dinner']
        
        # Generate sample meal plan
        meal_plan = {}
        sample_meals = {
            'Breakfast': ['Oatmeal with berries', 'Greek yogurt with nuts', 'Whole grain toast with avocado', 'Smoothie bowl'],
            'Lunch': ['Grilled chicken salad', 'Quinoa bowl', 'Vegetable soup', 'Turkey sandwich'],
            'Dinner': ['Baked salmon', 'Lean beef stir-fry', 'Vegetarian pasta', 'Grilled vegetables']
        }
        
        for day in days:
            meal_plan[day] = {meal: random.choice(sample_meals[meal]) for meal in meals}
        
        # Display meal plan
        meal_df = pd.DataFrame(meal_plan).T
        st.dataframe(meal_df, use_container_width=True)
        
        # Exercise schedule
        st.markdown("#### Weekly Exercise Schedule")
        
        exercise_plan = {
            'Monday': 'Cardio - 30 min brisk walk',
            'Tuesday': 'Strength training - Upper body',
            'Wednesday': 'Flexibility - Yoga session',
            'Thursday': 'Cardio - Swimming or cycling',
            'Friday': 'Strength training - Lower body',
            'Saturday': 'Active recovery - Light stretching',
            'Sunday': 'Rest day or gentle walk'
        }
        
        for day, exercise in exercise_plan.items():
            st.markdown(f"**{day}:** {exercise}")
            
    else:
        st.warning("Please complete the Health Assessment first to get personalized recommendations!")
        if st.button("Go to Health Assessment"):
            st.switch_page("Health Assessment")

def show_activity_tracking():
    """Display activity tracking page"""
    st.title("Activity Tracking & Progress")
    st.markdown("---")
    
    # Generate sample activity data
    @st.cache_data
    def generate_activity_data():
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'steps': np.random.randint(3000, 15000, len(dates)),
            'calories': np.random.randint(1800, 3000, len(dates)),
            'sleep_hours': np.random.normal(7.5, 1, len(dates)),
            'heart_rate': np.random.randint(60, 100, len(dates)),
            'weight': 70 + np.random.normal(0, 2, len(dates))
        }
        
        return pd.DataFrame(data)
    
    activity_data = generate_activity_data()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Filter data
    filtered_data = activity_data[
        (activity_data['date'] >= pd.Timestamp(start_date)) & 
        (activity_data['date'] <= pd.Timestamp(end_date))
    ]
    
    # Metrics overview
    st.markdown("### Activity Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_steps = filtered_data['steps'].mean()
        st.metric("Avg Daily Steps", f"{avg_steps:,.0f}")
    
    with col2:
        avg_calories = filtered_data['calories'].mean()
        st.metric("Avg Daily Calories", f"{avg_calories:,.0f}")
    
    with col3:
        avg_sleep = filtered_data['sleep_hours'].mean()
        st.metric("Avg Sleep Hours", f"{avg_sleep:.1f}")
    
    with col4:
        avg_hr = filtered_data['heart_rate'].mean()
        st.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")
    
    # Activity charts
    st.markdown("### Activity Trends")
    
    # Steps chart
    fig_steps = px.line(filtered_data, x='date', y='steps', 
                       title='Daily Steps Over Time',
                       labels={'steps': 'Steps', 'date': 'Date'})
    fig_steps.add_hline(y=8000, line_dash="dash", line_color="red", 
                       annotation_text="Recommended: 8,000 steps")
    st.plotly_chart(fig_steps, use_container_width=True)
    
    # Multi-metric chart
    fig_multi = go.Figure()
    
    # Normalize data for comparison
    normalized_data = filtered_data.copy()
    normalized_data['steps_norm'] = normalized_data['steps'] / normalized_data['steps'].max()
    normalized_data['calories_norm'] = normalized_data['calories'] / normalized_data['calories'].max()
    normalized_data['sleep_norm'] = normalized_data['sleep_hours'] / normalized_data['sleep_hours'].max()
    
    fig_multi.add_trace(go.Scatter(x=normalized_data['date'], y=normalized_data['steps_norm'],
                                  mode='lines', name='Steps (normalized)'))
    fig_multi.add_trace(go.Scatter(x=normalized_data['date'], y=normalized_data['calories_norm'],
                                  mode='lines', name='Calories (normalized)'))
    fig_multi.add_trace(go.Scatter(x=normalized_data['date'], y=normalized_data['sleep_norm'],
                                  mode='lines', name='Sleep (normalized)'))
    
    fig_multi.update_layout(title='Multi-Metric Comparison (Normalized)',
                           xaxis_title='Date',
                           yaxis_title='Normalized Value')
    st.plotly_chart(fig_multi, use_container_width=True)
    
    # Progress goals
    st.markdown("### Progress Goals")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Weekly Goals")
        weekly_steps = filtered_data['steps'].sum()
        weekly_goal = 56000  # 8000 steps * 7 days
        progress = min(weekly_steps / weekly_goal, 1.0)
        
        st.progress(progress)
        st.markdown(f"Steps: {weekly_steps:,} / {weekly_goal:,} ({progress:.1%})")
        
        sleep_goal = 7.5 * 7  # 7.5 hours * 7 days
        weekly_sleep = filtered_data['sleep_hours'].sum()
        sleep_progress = min(weekly_sleep / sleep_goal, 1.0)
        
        st.progress(sleep_progress)
        st.markdown(f"Sleep: {weekly_sleep:.1f} / {sleep_goal:.1f} hours ({sleep_progress:.1%})")
    
    with col2:
        st.markdown("#### Achievement Badges")
        
        # Calculate achievements
        achievements = []
        if avg_steps >= 10000:
            achievements.append("Step Master")
        if avg_sleep >= 7:
            achievements.append("Sleep Champion")
        if filtered_data['steps'].min() >= 5000:
            achievements.append("Consistency King")
        
        if achievements:
            for achievement in achievements:
                st.success(achievement)
        else:
            st.info("Keep going! Achievements unlocked at milestones.")

def show_chatbot(chatbot):
    """Display chatbot interface"""
    st.title("AI Health Assistant")
    st.markdown("---")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your health and wellness..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check for emergency situations first
        emergency_response = chatbot.get_emergency_response(prompt)
        if emergency_response:
            response = emergency_response
        else:
            # Get normal bot response
            response = chatbot.get_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
    
    # Enhanced quick action buttons with more health topics
    st.markdown("### Quick Health Topics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Nutrition Tips"):
            quick_response = chatbot.get_response("nutrition and healthy eating tips")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
        
        if st.button("Heart Health"):
            quick_response = chatbot.get_response("heart health and cardiovascular fitness")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
    
    with col2:
        if st.button("Exercise Guide"):
            quick_response = chatbot.get_response("exercise and fitness workout")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
        
        if st.button("Mental Health"):
            quick_response = chatbot.get_response("mental health and stress management")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
    
    with col3:
        if st.button("Sleep Better"):
            quick_response = chatbot.get_response("sleep tips and rest")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
        
        if st.button("Common Issues"):
            quick_response = chatbot.get_response("headache back pain joint pain")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
    
    with col4:
        if st.button("Preventive Care"):
            quick_response = chatbot.get_response("preventive care and health screenings")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
        
        if st.button("Hydration"):
            quick_response = chatbot.get_response("hydration and water intake")
            st.session_state.chat_history.append({"role": "assistant", "content": quick_response})
            st.rerun()
    
    # Sample questions for users
    st.markdown("### Sample Questions You Can Ask:")
    
    sample_questions = [
        "How can I lose weight safely?",
        "What exercises are best for beginners?",
        "How do I improve my sleep quality?",
        "What foods are good for heart health?",
        "How can I manage stress better?",
        "What are signs of high blood pressure?",
        "How much water should I drink daily?",
        "What stretches help with back pain?",
        "How do I start a healthy morning routine?",
        "What vitamins should I take?",
        "How can I boost my immune system?",
        "What are healthy snack options?"
    ]
    
    # Display sample questions in columns
    col1, col2 = st.columns(2)
    
    with col1:
        for i, question in enumerate(sample_questions[:6]):
            if st.button(f"{question}", key=f"sample_q_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": question})
                response = chatbot.get_response(question)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    with col2:
        for i, question in enumerate(sample_questions[6:], 6):
            if st.button(f"{question}", key=f"sample_q_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": question})
                response = chatbot.get_response(question)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()

# Additional utility functions for future enhancement
def save_user_data(user_data):
    """Save user data to database"""
    conn = sqlite3.connect('health_app.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, age, gender, height, weight)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_data['username'], user_data['age'], user_data['gender'], 
              user_data['height'], user_data['weight']))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def save_health_metrics(user_id, metrics):
    """Save health metrics to database"""
    conn = sqlite3.connect('health_app.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO health_metrics 
        (user_id, date, heart_rate, steps, sleep_hours, calories_burned, systolic_bp, diastolic_bp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, datetime.now().date(), metrics['heart_rate'], metrics['steps'],
          metrics['sleep_hours'], metrics.get('calories_burned', 0), 
          metrics['systolic_bp'], metrics.get('diastolic_bp', 80)))
    
    conn.commit()
    conn.close()

def get_user_history(user_id):
    """Retrieve user's health history"""
    conn = sqlite3.connect('health_app.db')
    query = '''
        SELECT * FROM health_metrics 
        WHERE user_id = ? 
        ORDER BY date DESC
    '''
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df

