import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv

import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report

# --- Quiz Performance Analysis Functions ---  
def fetch_data():
    current_quiz_url = os.getenv("QUIZ_URL")
    hist_data_url = os.getenv("HISTORICAL_DATA_URL")
    current_quiz_data = requests.get(current_quiz_url, verify=False).json()
    hist_data = requests.get(hist_data_url, verify=False).json()
    hist_df = pd.DataFrame(hist_data)
    return current_quiz_data, hist_df

def prep_data(hist_df):
    hist_df['topic'] = hist_df['quiz'].apply(lambda x: x.get('topic', 'Unknown'))
    hist_df['accuracy'] = hist_df.get('correct_answers', 0) / hist_df.get('total_questions', 1)
    hist_df['duration_minutes'] = (
        pd.to_datetime(hist_df['ended_at'], errors='coerce') - 
        pd.to_datetime(hist_df['started_at'], errors='coerce')
    ).dt.total_seconds() / 60
    topic_error_counts = hist_df.groupby('topic')['incorrect_answers'].sum().reset_index()
    topic_error_counts.columns = ['topic', 'error_count']
    hist_df = hist_df.merge(topic_error_counts, on='topic', how='left')
    hist_df['weak_topic'] = hist_df.apply(lambda x: x['topic'] if x['error_count'] > 0 else 'None', axis=1)
    features = hist_df[['score', 'accuracy', 'duration_minutes']]
    labels = hist_df['weak_topic']
    return features, labels

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_train)
    predictions = clf_model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    return clf_model, report

def recommend_topics(model, current_data):
    current_data['accuracy'] = current_data.get('correct_answers', 0) / current_data.get('total_questions', 1)
    current_data['duration_minutes'] = (
        pd.to_datetime(current_data.get('ended_at', '1970-01-01')) - 
        pd.to_datetime(current_data.get('started_at', '1970-01-01'))
    ).total_seconds() / 60
    features = pd.DataFrame([{
        'score': current_data.get('score', 0),
        'accuracy': current_data['accuracy'],
        'duration_minutes': current_data['duration_minutes']
    }])
    predictions = model.predict(features)
    unique_topics, counts = np.unique(predictions, return_counts=True)
    topic_error_map = dict(zip(unique_topics, counts))
    sorted_topics = sorted(topic_error_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_topics[:3]

def generate_structured_insights(current_quiz_data, historical_df, model):
    historical_df['accuracy'] = historical_df['correct_answers'] / historical_df['total_questions'] * 100
    historical_df['submitted_at'] = pd.to_datetime(historical_df['submitted_at'])
    topic_predictions = recommend_topics(model, current_quiz_data)
    insights = {}
    for topic, _ in topic_predictions:
        topic_data = historical_df[historical_df['quiz'].apply(lambda x: x.get('topic') == topic)].sort_values('submitted_at')
        if topic_data.empty:
            continue
        trend = "improving" if topic_data['accuracy'].diff().mean() > 0 else "declining"
        historical_performance = topic_data[['submitted_at', 'score', 'accuracy', 'total_questions', 'correct_answers', 'incorrect_answers']].to_dict(orient='records')
        historical_scores = topic_data[['submitted_at', 'score']]
        recommendations = [
            "Review the core concepts using notes or flashcards.",
            "Focus on practice questions for common mistakes.",
            "Use visual aids like flowcharts to understand better.",
            "Analyze trends and target specific subtopics."
        ]
        insights[topic] = {
            "performance_trend": trend,
            "reason_for_weakness": f"The topic '{topic}' is weak due to frequent mistakes or low scores.",
            "historical_performance": historical_performance,
            "historical_scores": historical_scores,
            "recommendations": recommendations
        }
    return insights


def train_rank_model():
    np.random.seed(42)
    X = pd.DataFrame({
         'score': np.random.randint(0, 100, size=100),
         'accuracy': np.random.uniform(50, 100, size=100),
         'duration_minutes': np.random.uniform(30, 120, size=100)
    })
  
    y = 10000 - (X['score'] * X['accuracy']) + np.random.randint(-100, 100, size=100)
    rank_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rank_model.fit(X, y)
    return rank_model

def predict_student_rank(rank_model, current_quiz_data):
    current_data = {
        'score': current_quiz_data.get('score', 0),
        'accuracy': current_quiz_data.get('correct_answers', 0) / current_quiz_data.get('total_questions', 1),
        'duration_minutes': (
            pd.to_datetime(current_quiz_data.get('ended_at', '1970-01-01')) -
            pd.to_datetime(current_quiz_data.get('started_at', '1970-01-01'))
        ).total_seconds() / 60
    }
    features = pd.DataFrame([current_data])
    predicted_rank = rank_model.predict(features)[0]
    return predicted_rank


colleges = [
    {"name": "B.J.Government Medical College, Pune", "estd": 1964, "cutoff": 250},
    {"name": "Dr. Vaishampayan Memorial Medical College, Solapur", "estd": 1963, "cutoff": 200},
    {"name": "Government Medical College, Baramati", "estd": 2019, "cutoff": 100},
    {"name": "Grant Medical College, Mumbai", "estd": 1845, "cutoff": 250},
    {"name": "Government Medical College, Jalgaon", "estd": 2018, "cutoff": 150},
    {"name": "Government Medical College, Sangli,Miraj", "estd": 1962, "cutoff": 200},
    {"name": "H.B.T Medical College &Dr.R.N.Cooper Muncipal General Hospital,Juhu, Mumbai", "estd": 2015, "cutoff": 200},
    {"name": "Lokmanya Tilak Muncipal Medical College, Sion, Mumbai", "estd": 1964, "cutoff": 200},
    {"name": "Rajashree Chatrapati Sahu Maharaj Govt. Medical College, Kolhapur", "estd": 2001, "cutoff": 150},
    {"name": "Rajiv Gandhi Medical College & Chatrapati Shivaji Maharaj Hospital, Thane", "estd": 1992, "cutoff": 100},
    {"name": "Shri bhausaheb Hire Government Medical College, Dhule", "estd": 1988, "cutoff": 150},
    {"name": "Seth GS Medical College, Mumbai", "estd": 1925, "cutoff": 250},
    {"name": "Topiwala National Medical College, Mumbai", "estd": 1964, "cutoff": 150},
    {"name": "Government Medical College, Akola", "estd": 2002, "cutoff": 200},
    {"name": "Government Medical College, Chandrapur", "estd": 2015, "cutoff": 150},
    {"name": "Government Medical College, Gondia", "estd": 2016, "cutoff": 150},
    {"name": "Government Medical College, Nagpur", "estd": 1947, "cutoff": 250},
    {"name": "Indira Gandhi Medical College & Hospital, Nagpur", "estd": 1968, "cutoff": 200},
    {"name": "Shri Vasant Rao Naik Memorial Medical College, Yavatmal", "estd": 1989, "cutoff": 200},
    {"name": "Dr.Shankarrao Chavan Government Medical College", "estd": 1988, "cutoff": 150},
    {"name": "Government Medical College, Aurangabad", "estd": 1956, "cutoff": 200},
    {"name": "Government Medical College, Latur", "estd": 2002, "cutoff": 150},
    {"name": "Swami Ramananda Teertha Rural Gov Medical College, Ambajogi", "estd": 1974, "cutoff": 150},
    {"name": "GMC Sindhudurg", "estd": 2021, "cutoff": 100},
    {"name": "GMC Satara", "estd": 2021, "cutoff": 100},
    {"name": "GMC Parbhani", "estd": 2023, "cutoff": 100},
    {"name": "GMC Osmanabad", "estd": 2022, "cutoff": 100},
    {"name": "GMC Nandurbar", "estd": 2020, "cutoff": 100},
    {"name": "GMC Ratnagiri", "estd": 2023, "cutoff": 100},
    {"name": "GMC Alibag", "estd": 2021, "cutoff": 100},
    {"name": "AIIMS Nagpur", "estd": 2018, "cutoff": 125},
    {"name": "AFMC Pune", "estd": None, "cutoff": None}
]

def predict_college_by_rank(predicted_rank, colleges):
   
    if predicted_rank < 3000:
         filtered = [c for c in colleges if c.get("cutoff") == 250]
         if not filtered: filtered = colleges
    elif predicted_rank < 6000:
         filtered = [c for c in colleges if c.get("cutoff") == 200]
         if not filtered: filtered = colleges
    elif predicted_rank < 9000:
         filtered = [c for c in colleges if c.get("cutoff") == 150]
         if not filtered: filtered = colleges
    else:
         filtered = [c for c in colleges if c.get("cutoff") == 100]
         if not filtered: filtered = colleges
    filtered_sorted = sorted(filtered, key=lambda x: x["name"])
    return filtered_sorted[0]["name"]

# --- Streamlit App ---
def main():
    st.title("NEET Testline Analysis")
    
    # Quiz performance analysis
    st.subheader("Fetching and Preparing Data")
    current_quiz_data, historical_df = fetch_data()
    features, labels = prep_data(historical_df)
    
    st.subheader("Training Performance Model")
    perf_model, report = train_model(features, labels)
    st.json(report)
    
    st.subheader("Overall Historical Quiz Scores")
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(historical_df['submitted_at']), historical_df['score'], marker='o', label='Quiz Scores')
    plt.title('Historical Quiz Performance')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("AI-Generated Insights for Weak Topics")
    weak_topics_insights = generate_structured_insights(current_quiz_data, historical_df, perf_model)
    for topic, insights in weak_topics_insights.items():
        st.markdown(f"### {topic}")
        st.write(f"Performance Trend: {insights['performance_trend']}")
        st.write(f"Reason: {insights['reason_for_weakness']}")
        st.write("Previous Performance:")
        st.table(insights['historical_performance'])
        st.write("Recommended Actions:")
        for rec in insights['recommendations']:
            st.write(f"- {rec}")
        
        st.subheader(f"Performance Trend for {topic}")
        topic_scores = insights['historical_scores']
        plt.figure(figsize=(8, 4))
        plt.plot(pd.to_datetime(topic_scores['submitted_at']), topic_scores['score'], marker='o', label=f'{topic} Scores')
        plt.title(f'Performance Trend for {topic}')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()
    
    st.subheader("Student NEET Rank Prediction")
    rank_model = train_rank_model()
    predicted_rank = predict_student_rank(rank_model, current_quiz_data)
    st.write(f"Predicted NEET Rank: {int(predicted_rank)}")
    
    st.subheader("College Prediction Based on Predicted Rank")
    college = predict_college_by_rank(predicted_rank, colleges)
    st.write(f"Most likely college admission: {college}")

    print("Matplotlib is working!")


if __name__ == '__main__':
    main()
