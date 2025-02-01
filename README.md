# Rankpredictor
# NEET Testline Analysis Tool

## 📌 Project Overview
The **NEET Testline Analysis Tool** is a machine learning-based web application that helps students analyze their **quiz performance**, predict their **NEET rank**, and suggest **medical colleges** based on expected rank.

### **Key Features**
- **Quiz Performance Analysis** 📝
  - Identifies weak topics using machine learning.
  - Generates AI-based insights & improvement recommendations.
- **NEET Rank Prediction** 🎯
  - Uses Random Forest Regression to estimate the student’s rank.
- **College Recommendation System** 🏥
  - Suggests government medical colleges based on the predicted rank.
- **Data Visualization** 📊
  - Displays historical quiz performance trends with interactive graphs.

---

## 🚀 Setup Instructions

### **1️⃣ Install Dependencies**
Ensure you have **Python 3.8+** installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### **2️⃣ Set Up Environment Variables**
Create a `.env` file and add your quiz data URLs:
```
QUIZ_URL=<your_current_quiz_data_url>
HISTORICAL_DATA_URL=<your_historical_quiz_data_url>
```

### **3️⃣ Run the Application**
```bash
streamlit run main.py
```

### **4️⃣ View in Browser**
Once the app is running, open **http://localhost:8501** in your browser.

---

## 🎯 Approach & Methodology

### **1️⃣ Data Collection** 📥
- Fetches the latest quiz data and historical performance from external APIs.
- Stores them in a **Pandas DataFrame** for analysis.

### **2️⃣ Data Preprocessing** 🛠️
- Extracts key metrics like **accuracy, duration, and weak topics**.
- Handles missing or incorrect data using Pandas operations.

### **3️⃣ Machine Learning Models** 🤖
- **RandomForestClassifier** → Predicts weak topics based on historical quiz performance.
- **RandomForestRegressor** → Predicts the NEET rank based on quiz scores, accuracy, and time spent.

### **4️⃣ Visualization & Insights** 📊
- Uses **Matplotlib** to generate interactive graphs.
- Provides AI-powered suggestions for topic improvement.

### **5️⃣ College Prediction** 🏥
- Compares the predicted rank with medical college cutoffs.
- Suggests the most likely college based on historical cutoffs.

---

## 📸 Screenshots

### **1️⃣ Overall Performance Trend**
![Quiz Performance Graph](screenshots/quiz_performance.png)

### **2️⃣ Weak Topics Insights**
![Weak Topics Insights](screenshots/weak_topics.png)

### **3️⃣ Rank Prediction & College Suggestion**
![Rank Prediction](screenshots/rank_prediction.png)

---

## 🏆 Future Enhancements
- 🔹 Improve model accuracy with more historical data.
- 🔹 Add a **personalized dashboard** for tracking student progress.
- 🔹 Integrate **adaptive learning suggestions** based on weak areas.

---

## 📜 License
This project is **open-source** and free to use for educational purposes.

---

## 🤝 Contributing
Pull requests are welcome! If you find any bugs or have feature requests, feel free to open an issue.

---



