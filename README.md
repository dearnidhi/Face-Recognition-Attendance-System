# Face Recognition Attendance System with Streamlit 📸🧠
This project is a **face recognition–based attendance system** built using **OpenCV**, **KNN (scikit-learn)**, and **Streamlit**. It captures facial data, trains a simple classifier, marks attendance in CSV files, and displays live attendance records through a Streamlit dashboard.
The system is designed for **learning and demonstration purposes**, focusing on computer vision fundamentals and real-time data handling.

---

## 🚀 Features

- Face data collection using webcam  
- Face recognition using K-Nearest Neighbors (KNN)  
- Real-time attendance marking  
- CSV-based daily attendance logs  
- Voice confirmation for attendance (Windows)  
- Streamlit dashboard with auto-refresh  
- Live attendance table visualization  

---

## 🧠 How It Works

### 1. Face Collection
- Captures face images from webcam
- Stores flattened face data and labels using pickle

### 2. Model Training
- Uses a KNN classifier to recognize faces
- Trained dynamically from stored face data

### 3. Attendance Marking
- Detects faces in real time
- Press **`o`** to mark attendance
- Saves name and timestamp to a CSV file

### 4. Dashboard Visualization
- Streamlit app auto-refreshes every 2 seconds
- Displays daily attendance with highlights

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt

2. Collect Face Data
python collect_faces.py
This step generates faces.pkl and names.pkl locally.

3. Start Attendance System
python test.py
Press o to mark attendance
Press q to quit

4. Run Streamlit Dashboard
streamlit run dashboard.py

🛠️ Tech Stack
Python
OpenCV
Scikit-learn
Streamlit
Pandas
NumPy

⚠️ Notes
This project is intended for educational use only
Uses a simple ML model, not deep learning
Accuracy depends on lighting and camera quality
Trained data files (faces.pkl, names.pkl) are not included in the repository
CSV storage is not suitable for large-scale or production systems

