AI Fabric Quality Grading System (MLOps Simulation)

Project Overview

This project is a minimal, end-to-end simulation of an MLOps pipeline for industrial quality control. It consists of a Machine Learning model deployed as a REST API (Flask) and a simple HTML dashboard used as the front-end interface.

The system is designed to take real-time production parameters (like RPM, temperature, and humidity) and predict the fabric's quality grade (A, B, or C), immediately providing a Quality Management System (QMS) Recommended Action Plan.

Core Components

app.py: A Python Flask API that trains a mock Random Forest Classifier on simulated factory data and serves the prediction model.

defect_dashboard.html: A clean, responsive HTML/Tailwind CSS front-end that captures user input and communicates with the Python API via a POST request.

🛠️ Technology Stack

Backend/API: Python, Flask, scikit-learn, joblib

Front-end/UI: HTML5, JavaScript (Fetch API), Tailwind CSS

Model: Random Forest Classifier (Trained on synthetic data)

🚀 Setup and Run Instructions

Follow these steps to get the API and the dashboard running locally.

Prerequisites

You must have Python 3.x installed.

Step 1: Install Dependencies

Open your terminal in the project directory and install the necessary Python packages:

pip install flask scikit-learn pandas joblib flask-cors


Step 2: Run the Machine Learning API (app.py)

The first time you run the API, it will train the model and save it as fabric_classifier.joblib. It will then start the web server.

Keep this terminal window open and running.

python app.py


You should see confirmation logs like:

... Model trained and saved successfully as fabric_classifier.joblib
... Starting Flask API server on [http://127.0.0.1:5000](http://127.0.0.1:5000)


Step 3: Launch the Dashboard (defect_dashboard.html)

With the Flask server running, open the front-end:

Locate the defect_dashboard.html file in your project folder.

Double-click the file to open it in your web browser.

🧪 Testing the System

The dashboard is designed to demonstrate two key quality scenarios.

1. Test Case: Grade A (Low Risk)

This simulation shows parameters leading to high-quality fabric.

Parameter

Value

Machine Setting (RPM)

700

Humidity (%)

65

Expected Result: Predicted Grade A, Severity LOW.

2. Test Case: Grade C (Critical Risk)

This simulation shows parameters that are out of the ideal range and often lead to defects (high RPM or high humidity).

Parameter

Value

Machine Setting (RPM)

1150

Humidity (%)

92

Expected Result: Predicted Grade C, Severity CRITICAL. The recommended action will be IMMEDIATE STOPPAGE and flagging the machine for maintenance.