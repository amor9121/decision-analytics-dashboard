## 🌐 Live Demo

- GitHub Repository: https://github.com/amor9121/decision-analytics-dashboard/
- Deployed Application: https://decision-analytics-dashboard.streamlit.app/

## 🚀 How to Run the Application

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```

2. Navigate to the project directory:
   ```bash
   cd decision-analytics-dashboard
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

The application will open automatically in your web browser.

---

# 📊 Decision Analytics Dashboard

An interactive decision analytics dashboard developed as part of the **Python Programming for Business Intelligence & Analytics** coursework.  
The application integrates optimisation, statistical analysis, and machine learning into a single Streamlit-based interface, with an emphasis on **reproducibility, auditability, and decision support**.

---

## 🔍 Project Overview

This project is designed to support data-driven decision-making through a structured analytics workflow.  
Users can explore data, run analytical models, and review results interactively, while ensuring that all data modifications remain traceable and reversible.

The dashboard demonstrates how multiple analytical techniques can be integrated into a unified decision-support system rather than implemented as isolated scripts.

---

## 🧩 Key Features

- Optimisation models for scheduling and resource allocation  
- Descriptive analytics for workload and performance analysis  
- Statistical testing to identify factors associated with key outcomes  
- Machine learning models (e.g. logistic regression with feature selection)  
- Audit logging of user-triggered data modifications  
- Dataset reset function to ensure reproducibility  

---

## 🗂 Project Structure

decision-analytics-dashboard/
│
├── app.py            # Main Streamlit application  
├── main.py           # Core execution logic  
├── core/             # Core shared logic  
├── tasks/            # Task-specific analytics  
├── utils/            # Shared utility functions  
├── data/             # Datasets  
├── doc/              # Documentation  
├── requirements.txt  
├── runtime.txt  
└── README.md  

