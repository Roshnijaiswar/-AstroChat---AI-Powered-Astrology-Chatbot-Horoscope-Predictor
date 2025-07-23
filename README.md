# -AstroChat---AI-Powered-Astrology-Chatbot-Horoscope-Predictor
**AstroChat** generates astrological insights using user inputs like date, time, and place of birth.
It predicts planetary positions, zodiac signs, doshas with remedies, horoscope forecasts, and personality traits.
With AI and chatbot integration, it modernizes traditional astrology, making it accurate, interactive, and accessible.

## 🚀 Features
✅ AI-based Horoscope Prediction (Multi-output Classification)  
✅ NLP-Powered Astrology Chatbot (Cohere API)  
✅ Birth Chart Visualization using Matplotlib  
✅ Dosha Detection and Remedies  
✅ Synthetic Data Generation for Model Training  
✅ PDF Report Generation (FPDF)  
✅ Streamlit UI with Sidebar Navigation  
✅ Live Planetary Calculation using Ephem and Geopy  
## 🗃️ Project Structure

AstroChat-AI/
ASTRO_CHAT.py # Main app logic
UI.py # User interface setup (Streamlit)
training.py # Model training using RandomForestClassifier
synthetic.py # Synthetic data generation script
chart.py # Chart rendering and layout
check.py # Birth chart drawing logic
astro_model.joblib # Trained model file
data.csv # Kaggle-sourced dataset
requirement.txt # Python dependencies
.env # (Not uploaded) API keys if needed
README.md # This file

## 📦 Tech Stack & Libraries
- **Frontend**: Streamlit  
- **Backend**: Python  
- **ML Libraries**: Scikit-learn, Joblib, Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **APIs**:  
  - 🪐 `ephem` for planetary positions  
  - 🌍 `geopy` for latitude/longitude  
  - 💬 `cohere` for chatbot NLP  
- **Others**: Pytz, Pillow, Pytesseract, FPDF, OS, Streamlit Components
