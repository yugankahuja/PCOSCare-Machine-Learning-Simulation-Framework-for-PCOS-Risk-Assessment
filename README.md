# PCOSCare: Machine Learning & Simulation Framework for PCOS Risk Assessment

## 🎯 Project Overview
PCOSCare is an integrated web application that combines machine learning risk prediction with agent-based simulation to:
- Provide percentage-based PCOS risk assessment (Random Forest with 98.2% accuracy)
- Simulate long-term health outcomes under different treatment plans
- Visualize progression of key PCOS indicators (weight, insulin resistance, etc.)

## ✨ Key Features
| Feature | Description |
|---------|-------------|
| 🔍 Risk Assessment | Percentage-based prediction (not binary) using 14 clinical/lifestyle factors |
| 🧪 Treatment Simulation | Compare outcomes for medication, diet, exercise, and no treatment |
| 📈 Interactive Visuals | Dynamic plots of health metrics over time |
| 🔄 Feedback System | Tracks user inputs and outcomes in Firebase |
| 📱 Responsive UI | Streamlit-based web interface |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Firebase project with Firestore enabled

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/PCOSCare.git
cd PCOSCare

# Install dependencies
pip install streamlit firebase-admin mesa scikit-learn pandas matplotlib numpy

# Download your Firebase service account JSON and rename to:
mv path/to/your-firebase-key.json firebase-config.json

# Run the application
streamlit run final.py
```

## 🧩 Core Components
```python
# System Architecture Highlights
1. RandomForestClassifier()  # PCOS risk prediction
2. PatientAgent()           # MESA agent for individual health simulation  
3. PCOSModel()              # Manages multi-agent treatment scenarios
4. Streamlit UI             # User input forms and visualization dashboard
```

## 📊 Sample Outputs
### Risk Prediction
```
Predicted PCOS Risk: 72.5%
```

### Simulation Results
| Treatment | Weight (kg) | Insulin Resistance | Hirsutism | Menstrual Regularity |
|-----------|-------------|--------------------|-----------|----------------------|
| Medication | 68.2 | 1.15 | Mild | Regular |
| Exercise | 71.5 | 1.32 | Moderate | Regular |
| Diet | 73.1 | 1.41 | Moderate | Irregular |
| None | 76.8 | 1.89 | Severe | Irregular |

## 📝 Dataset Information
- We use the Diet, Exercise and PCOS Insights dataset from Kaggle with these key features:
- Age, Weight, Height, BMI
- Menstrual regularity
- Hormonal imbalance indicators
- Lifestyle factors (exercise, diet)
- PCOS diagnosis (target variable)

## 🧩 Dependencies
- Python 3.8+
- Streamlit
- Firebase Admin SDK
- scikit-learn
- Mesa
- pandas
- matplotlib
- numpy

## 📂 File Structure
```
PCOSCare/
├── final.py                # Main application (Streamlit + ML + Simulation)
├── firebase-config.json    # Firebase credentials (gitignored)
├── Cleaned-Data-new.csv    # Training dataset (gitignored)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── assets/                 # For screenshots/diagrams
```

## 🌐 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)

*(Replace with your actual deployment URL)*

## 🤝 How to Contribute
1. Report bugs via [Issues](https://github.com/yourusername/PCOSCare/issues)
2. Fork and submit Pull Requests
3. Suggest new features or improvements

## 📜 License
MIT License - See [LICENSE](LICENSE) for details.

## 📧 Contact
For questions or collaborations:  
📩 your.email@example.com  
🔗 [Project Website](https://your-project-site.com)

---

> **Note**: Remember to:
> - Add your actual Firebase config file (keep it secret!)
> - Include the dataset if not proprietary
> - Update all placeholder URLs


## 👥 Contributors
I thank the following contributors for their work on PCOSCare:

| Name                  | GitHub Profile |
|-----------------------|----------------|
| Yugank Ahuja          | [@yugankahuja](https://github.com/yugankahuja) |
| Spatika Girirajan     | [@spatikag](https://github.com/spatikag) |
| Namratha Muraleedhaan | [@namratham](https://github.com/namratham) |
| Sana Sehgal           | [@sanasehgal](https://github.com/sanasehgal) |
| Saima Khatoon         | [@saimak](https://github.com/saimak) |
| Yuchen Liu            | [@yuchenl](https://github.com/yuchenl) |

