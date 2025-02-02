# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import streamlit as st
# import requests
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import classification_report

# def fetch_data():
#     current_quiz_url = "https://www.jsonkeeper.com/b/LLQT"
#     hist_data_url = "https://api.jsonserve.com/XgAgFJ"
#     try:
#         current_quiz_response = requests.get(current_quiz_url, verify=False)
#         hist_data_response = requests.get(hist_data_url, verify=False)
#         current_quiz_response.raise_for_status()
#         hist_data_response.raise_for_status()
#         current_quiz_data = current_quiz_response.json()
#         hist_data = hist_data_response.json()
#         if not isinstance(hist_data, list):
#             raise ValueError("Unexpected data format in historical data")
#         hist_df = pd.DataFrame(hist_data)
#         return current_quiz_data, hist_df
#     except (requests.RequestException, ValueError) as e:
#         st.error(f"Error fetching data: {e}")
#         return None, None

# def prep_data(hist_df):
#     if hist_df is None or hist_df.empty:
#         return None, None
    
#     hist_df['topic'] = hist_df.get('quiz', {}).apply(lambda x: x.get('topic', 'Unknown'))
#     hist_df['accuracy'] = hist_df.get('correct_answers', 0) / hist_df.get('total_questions', 1)
#     hist_df['duration_minutes'] = (
#         pd.to_datetime(hist_df['ended_at'], errors='coerce') - 
#         pd.to_datetime(hist_df['started_at'], errors='coerce')
#     ).dt.total_seconds() / 60
#     hist_df.dropna(subset=['duration_minutes'], inplace=True)
    
#     topic_error_counts = hist_df.groupby('topic')['incorrect_answers'].sum().reset_index()
#     topic_error_counts.columns = ['topic', 'error_count']
#     hist_df = hist_df.merge(topic_error_counts, on='topic', how='left')
#     hist_df['weak_topic'] = hist_df.apply(lambda x: x['topic'] if x['error_count'] > 0 else 'None', axis=1)
    
#     features = hist_df[['score', 'accuracy', 'duration_minutes']]
#     labels = hist_df['weak_topic']
#     return features, labels

# def train_model(features, labels):
#     if features is None or labels is None:
#         return None, None
    
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#     clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf_model.fit(X_train, y_train)
#     predictions = clf_model.predict(X_test)
#     report = classification_report(y_test, predictions, output_dict=True)
#     return clf_model, report

# def train_rank_model():
#     np.random.seed(42)
#     X = pd.DataFrame({
#         'score': np.random.randint(0, 100, size=100),
#         'accuracy': np.random.uniform(50, 100, size=100),
#         'duration_minutes': np.random.uniform(30, 120, size=100)
#     })
#     y = 10000 - (X['score'] * X['accuracy']) + np.random.randint(-100, 100, size=100)
#     rank_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rank_model.fit(X, y)
#     return rank_model

# def predict_student_rank(rank_model, current_quiz_data, user_id, hist_df):
#     if rank_model is None or hist_df is None or current_quiz_data is None:
#         return None
    
#     user_data = hist_df[hist_df['user_id'] == user_id]
#     if user_data.empty:
#         return None
    
#     current_data = {
#         'score': current_quiz_data.get('score', 0),
#         'accuracy': current_quiz_data.get('correct_answers', 0) / current_quiz_data.get('total_questions', 1),
#         'duration_minutes': (
#             pd.to_datetime(current_quiz_data.get('ended_at', '1970-01-01')) -
#             pd.to_datetime(current_quiz_data.get('started_at', '1970-01-01'))
#         ).total_seconds() / 60
#     }
#     features = pd.DataFrame([current_data])
#     predicted_rank = rank_model.predict(features)[0]
#     return predicted_rank

# def predict_college_by_rank(predicted_rank):
#     colleges = [
#         {"name": "B.J.Government Medical College, Pune", "cutoff": 250},
#         {"name": "Grant Medical College, Mumbai", "cutoff": 250},
#         {"name": "Seth GS Medical College, Mumbai", "cutoff": 250},
#         {"name": "Government Medical College, Nagpur", "cutoff": 250},
#         {"name": "Indira Gandhi Medical College & Hospital, Nagpur", "cutoff": 200},
#         {"name": "Government Medical College, Aurangabad", "cutoff": 200},
#         {"name": "Government Medical College, Latur", "cutoff": 150},
#         {"name": "GMC Sindhudurg", "cutoff": 100},
#         {"name": "AIIMS Nagpur", "cutoff": 125},
#         {"name": "AFMC Pune", "cutoff": None}
#     ]
    
#     if predicted_rank < 3000:
#         return next((c["name"] for c in colleges if c["cutoff"] == 250), "No Match")
#     elif predicted_rank < 6000:
#         return next((c["name"] for c in colleges if c["cutoff"] == 200), "No Match")
#     elif predicted_rank < 9000:
#         return next((c["name"] for c in colleges if c["cutoff"] == 150), "No Match")
#     else:
#         return next((c["name"] for c in colleges if c["cutoff"] == 100), "No Match")

# def main():
#     st.title("NEET Testline Performance Analysis Report")
    
#     user_id = st.text_input("Enter User ID:")
    
#     if user_id:
#         st.subheader("Fetching and Preparing Data")
#         current_quiz_data, historical_df = fetch_data()
#         if historical_df is None:
#             return
        
#         features, labels = prep_data(historical_df)
        
#         st.subheader("Training Performance Model")
#         perf_model, report = train_model(features, labels)
#         if report:
#             st.json(report)
        
#         st.subheader("Student NEET Rank Prediction")
#         rank_model = train_rank_model()
#         predicted_rank = predict_student_rank(rank_model, current_quiz_data, user_id, historical_df)
#         if predicted_rank:
#             st.write(f"Predicted NEET Rank for User {user_id}: {int(predicted_rank)}")
            
#             st.subheader("College Prediction Based on Predicted Rank")
#             college = predict_college_by_rank(predicted_rank)
#             st.write(f"Most likely college admission: {college}")
#         else:
#             st.write("No data available for this User ID.")
#     else:
#         st.write("Please enter a valid User ID to proceed.")

# if __name__ == '__main__':
#     main()

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

def main():
    st.title("Student Rank Analysis")
    
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
    
    # New Histogram Plot
    
    
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

if __name__ == '__main__':
    main()
