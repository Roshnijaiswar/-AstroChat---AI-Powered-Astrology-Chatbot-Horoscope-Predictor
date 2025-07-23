import streamlit as st
from datetime import datetime, timedelta, date
import ephem
import pytz
from interface.chatbot import get_chatbot_response  # Correct import from 'interface' folder
import matplotlib.pyplot as plt
import numpy as np
from pytz import timezone
from geopy import Nominatim
import time
import pytesseract
from PIL import Image
import cohere
from fpdf import FPDF
import os
import streamlit.components.v1 as components
import joblib
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# ----------------------------------- Function to get latitude & longitude ----------------------------------------------
from geopy.geocoders import Nominatim
def get_lat_lon(place):
    geolocator = Nominatim(user_agent="astro_app")
    location = geolocator.geocode(place)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# ------------------------------------- Function to calculate planetary positions ----------------------------------------------------
def get_planet_positions(birth_date_time, lat, lon):
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)
    observer.date = ephem.Date(birth_date_time)
    planets = {
        "Sun": ephem.Sun(observer),
        "Moon": ephem.Moon(observer),
        "Mars": ephem.Mars(observer),
        "Mercury": ephem.Mercury(observer),
        "Jupiter": ephem.Jupiter(observer),
        "Venus": ephem.Venus(observer),
        "Saturn": ephem.Saturn(observer),
        "Rahu": ephem.Moon(observer),  
        "Ketu": ephem.Moon(observer)   
    }
    planet_positions = {name: planet.ra * (180 / ephem.pi) for name, planet in planets.items()}
    return planet_positions

# ----------------------------------- Function to determine house positions -----------------------------------------------
def get_house_positions(birth_date_time, lat, lon):
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)
    observer.date = ephem.Date(birth_date_time)
    sidereal_time = observer.sidereal_time()  
    ascendant = float(ephem.degrees(sidereal_time)) * (180 / ephem.pi)
    house_positions = {i: (ascendant + (i - 1) * 30) % 360 for i in range(1, 13)}
    return house_positions

# ------------------------------ Function to map planetary positions to house numbers -----------------------------------------------
def map_positions_to_houses(planet_positions, house_positions):
    mapped_positions = {}
    for planet, degrees in planet_positions.items():
        house_number = min(house_positions, key=lambda h: abs(house_positions[h] - degrees))
        mapped_positions[planet] = house_number
    return mapped_positions

# --------------------------------------------House positions for plotting--------------------------------------------------
house_positions_plot = {
    1:  (5, 6),   
    2:  (2.5, 6),  
    3:  (1, 5),   
    4:  (2.5, 3), 
    5:  (1, 1.5),   
    6:  (2.5, 0.8),  
    7:  (5, 1.5), 
    8:  (7.5, 0.5),   
    9:  (9, 1.5), 
    10: (7.5, 3),  
    11: (9.2, 4.8), 
    12: (7.5, 6)  
}

# -------------------------------------- Function to calculate the Sun Sign (Vedic Zodiac Sign) ------------------------------------------
def get_sun_sign(birth_date):
    sun_signs = [
        ("Aries", (3, 21), (4, 19)),
        ("Taurus", (4, 20), (5, 20)),
        ("Gemini", (5, 21), (6, 20)),
        ("Cancer", (6, 21), (7, 22)),
        ("Leo", (7, 23), (8, 22)),
        ("Virgo", (8, 23), (9, 22)),
        ("Libra", (9, 23), (10, 22)),
        ("Scorpio", (10, 23), (11, 21)),
        ("Sagittarius", (11, 22), (12, 21)),
        ("Capricorn", (12, 22), (1, 19)),
        ("Aquarius", (1, 20), (2, 18)),
        ("Pisces", (2, 19), (3, 20)),
    ]
    month, day = birth_date.month, birth_date.day
    for sign, start_date, end_date in sun_signs:
        if (month == start_date[0] and day >= start_date[1]) or (month == end_date[0] and day <= end_date[1]):
            return sign
    return None

# ------------------------------------ Function to calculate Nakshatra, Paadam, and Rasi ----------------------------------------------
def get_nakshatra_and_rasi(birth_date_time):
    observer = ephem.Observer()
    observer.date = ephem.Date(birth_date_time)
    moon = ephem.Moon(observer)
    moon_longitude = moon.ra * 180 / 3.14159
    nakshatra_num = int(moon_longitude / 13.3333)
    nakshatras = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Aridra", 
        "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
        "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", 
        "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", 
        "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
    ]
    nakshatra = nakshatras[nakshatra_num % len(nakshatras)]
    nakshatra_position = moon_longitude % 13.3333
    paadam = int(nakshatra_position / 3.3333) + 1
    rasis = [
        "Makar", "Kumbh", "Meen", "Mesh", "Vrishabh", "Mithun", "Kark", 
        "Singh", "Kanya", "Tula", "Vrishchik", "Dhanu"
    ]
    rasi = rasis[int(moon_longitude / 30) % len(rasis)]
    return nakshatra, paadam, rasi

# -----------------------------------  Function to calculate the Moon phase ---------------------------------------------------------
def get_moon_phase(user_date):
    observer = ephem.Observer()
    observer.date = ephem.Date(user_date)
    sun = ephem.Sun(observer)
    moon = ephem.Moon(observer)
    
    # Convert to ecliptic coordinates
    sun_ecl = ephem.Ecliptic(sun)
    moon_ecl = ephem.Ecliptic(moon)
    
    # Calculate phase angle in radians and convert to degrees
    phase_angle = (moon_ecl.lon - sun_ecl.lon) % (2 * ephem.pi)
    phase_angle_deg = phase_angle * 180 / ephem.pi

    # Classify the moon phase based on the phase angle (boundaries are approximate)
    if phase_angle_deg < 15 or phase_angle_deg > 345:
        return "New Moon"
    elif 15 <= phase_angle_deg < 75:
        return "Waxing Crescent"
    elif 75 <= phase_angle_deg < 105:
        return "First Quarter"
    elif 105 <= phase_angle_deg < 165:
        return "Waxing Gibbous"
    elif 165 <= phase_angle_deg < 195:
        return "Full Moon"
    elif 195 <= phase_angle_deg < 255:
        return "Waning Gibbous"
    elif 255 <= phase_angle_deg < 285:
        return "Last Quarter"
    elif 285 <= phase_angle_deg <= 345:
        return "Waning Crescent"
    else:
        return "Unknown Phase"

# -------------------------------------------Function to calculate Doshas and Remedies-------------------------------------------------
def get_doshas_and_remedies(birth_date_time):
    observer = ephem.Observer()
    observer.date = ephem.Date(birth_date_time)
    moon = ephem.Moon(observer)
    moon_longitude = moon.ra * 180 / 3.14159
    doshas = []
    remedies = []
    if 0 <= moon_longitude < 30 or 180 <= moon_longitude < 210:
        doshas.append("Mangal Dosha")
        remedies.append("Chant 'Mangal Stotra' and offer red flowers to Lord Hanuman.")
    if 120 < moon_longitude < 150:
        doshas.append("Kaal Sarp Dosha")
        remedies.append("Perform Kaal Sarp Dosh Nivaran Pooja at a temple.")
    if 210 < moon_longitude < 240:
        doshas.append("Nadi Dosha")
        remedies.append("Perform Nadi Dosha Puja and consult an astrologer before marriage.")
    if 240 < moon_longitude < 270:
        doshas.append("Pitra Dosha")
        remedies.append("Offer water to ancestors (Tarpan) and donate food to Brahmins.")
    if 270 < moon_longitude < 300:
        doshas.append("Chandal Dosha")
        remedies.append("Recite 'Guru Chandal Dosh Nivaran Mantra' and worship Jupiter.")
    if 300 < moon_longitude < 330:
        doshas.append("Shani Dosha")
        remedies.append("Donate black sesame seeds and feed crows on Saturdays.")
    if 30 < moon_longitude < 60:
        doshas.append("Guru Chandal Dosha")
        remedies.append("Chant 'Om Brihaspataye Namah' and donate yellow clothes.")
    if 60 < moon_longitude < 90:
        doshas.append("Kemdrum Dosha")
        remedies.append("Recite 'Chandra Graha Shanti Mantra' and wear silver ornaments.")
    if 90 < moon_longitude < 120:
        doshas.append("Grahan Dosha")
        remedies.append("Perform Chandra or Surya Grahan Shanti Puja and donate food.")
    if 150 < moon_longitude < 180:
        doshas.append("Gandmool Dosha")
        remedies.append("Perform Nakshatra Shanti Puja on the 27th day after birth.")
    if 180 < moon_longitude < 210:
        doshas.append("Vish Yoga")
        remedies.append("Chant 'Maha Mrityunjaya Mantra' and worship Lord Shiva.")
    if 330 < moon_longitude < 360:
        doshas.append("Angarak Dosha")
        remedies.append("Chant 'Hanuman Chalisa' and avoid conflicts on Tuesdays.")
    if 0 <= moon_longitude <= 30 or 150 <= moon_longitude <= 180:
        mangalik_status = "Mangalik"
        remedies.append("Mangalik Dosha Remedy: Worship Lord Hanuman and perform Navagraha Shanti Puja.")
    else:
        mangalik_status = "Non-Mangalik"
    return doshas, remedies, mangalik_status

#------------------------------------------------------------------------------ Model Training and Data Handling --------------
import joblib
import os
from datetime import datetime
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

model_filename = "astro_model.joblib"

if os.path.exists(model_filename):
    print("✅ Model already exists. Skipping training...")
else:
    print("🔄 Model not found. Training a new model...")
    df = pd.read_csv("data.csv")
    if "Place" in df.columns:
        df.rename(columns={"Place": "POB"}, inplace=True)
    drop_columns = ["Name", "Sun_Sign", "Moon_Phase", "Nakshatra", "Paadam", 
                    "Rasi", "Doshas", "Remedies", "Mangalik_Status", "Personality", 
                    "Interest"]
    df = df.drop(columns=drop_columns, errors="ignore")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df = df[df["DOB"].notnull()]
    df["Birth_Year"] = df["DOB"].dt.year
    df = df.drop(columns=["DOB"])
    def convert_time_to_minutes(time_str):
        try:
            dt = datetime.strptime(time_str, "%I:%M %p")
            return dt.hour * 60 + dt.minute
        except:
            return np.nan
    df["Time"] = df["Time"].apply(convert_time_to_minutes)
    df = df.dropna(subset=["Time"])
    df["POB"] = df["POB"].fillna("Unknown")
    le_pob = LabelEncoder()
    df["POB"] = le_pob.fit_transform(df["POB"])
    label_encoders = {"POB": le_pob}
    X = df[["Birth_Year", "POB", "Time"]]
    y = df[["General_Horoscope", "Health_Horoscope", "Work_Horoscope", "Relationship_Horoscope"]]
    for col in y.columns:
        le = LabelEncoder()
        y[col] = le.fit_transform(y[col])
        label_encoders[col] = le
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for i, col in enumerate(y.columns):
        print(f"Classification Report for {col}:\n")
        print(classification_report(y_test[col], y_pred[:, i]))
    joblib.dump({"model": model, "encoders": label_encoders}, model_filename)
    print("✅ Model training completed and saved successfully!")

# ----------------- Sidebar Page Selection -----------------
page = st.sidebar.radio("Select Page", ["Home", "Dashboard"])

# Initialize session state for chatbot follow-ups if not present
if "followup_answers" not in st.session_state:
    st.session_state.followup_answers = {}

# Enhanced CSS Styling
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #1b1b2f, #2a2a4d);
            padding: 20px;
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #FFD700;
            font-weight: bold;
            text-align: center;
        }
        .disclaimer {
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: #FFD700;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.7);
            margin-bottom: 20px;
        }
        .astro-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.7);
            margin: 20px 0;
            text-align: center;
        }
        .astro-box h3 {
            color: #FFD700;
            font-size: 22px;
            font-weight: bold;
        }
        .astro-box p {
            font-size: 18px;
            color: #ffffff;
        }
        .astro-highlight {
            font-weight: bold;
            color: #00ffcc;
        }
        div[data-baseweb="input"] {
            border-radius: 10px;
            border: 1px solid #FFD700 !important;
            box-shadow: 0 0 5px rgba(0,0,0,0.7);
        }
        .stButton>button {
            background: linear-gradient(90deg, #00c9ff, #92fe9d);
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 15px;
            transition: 0.3s;
            box-shadow: 0 0 10px rgba(0,0,0,0.4);
            min-width: 200px !important;
            text-overflow: ellipsis;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #92fe9d, #00c9ff);
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0,0,0,0.6);
        }
        .astro-img {
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.7);
            width: 250px;
            height: 250px;
        }
        .caption {
            font-size: 16px;
            color: #cccccc;
            margin-top: 5px;
            text-align: center;
        }
        h1 { text-align: center; }
        .chatbot-button {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border-radius: 10px;
            background: linear-gradient(to right, #800080, #0000FF);
            color: white;
            border: none;
            transition: 0.3s;
            margin-bottom: 5px;
            box-shadow: 0 0 10px black;
        }
        .chatbot-button:hover {
            background: linear-gradient(to right, #0000FF, #800080);
            color: white;
            box-shadow: 0 0 15px black;
        }
        button[data-key*="followup"] {
            background: linear-gradient(90deg, #ff9966, #ff5e62) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 0 6px rgba(0,0,0,0.4) !important;
            border-radius: 8px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            text-overflow: ellipsis;
            overflow: hidden;
            white-space: nowrap;
        }
        button[data-key*="followup"]:hover {
            background: linear-gradient(90deg, #ff5e62, #ff9966) !important;
            box-shadow: 0 0 10px rgba(0,0,0,0.6) !important;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Page Rendering Based on Sidebar Selection -----------------
if page == "Home":
    st.title("Welcome to Astrology World 🪐")
    st.markdown('<p class="disclaimer">Disclaimer: This is not completely accurate data, but some parts may be true based on Vedic astrology.</p>', unsafe_allow_html=True)
    st.write("Please enter your details below to get your astrology insights and birth chart:")
    name = st.text_input("Enter your Name:", key="name")
    dob = st.date_input("Enter your Date of Birth:", min_value=datetime(1900,1,1), max_value=datetime(2100,12,31), key="dob")
    time_input = st.text_input("Enter your Birth Time (HH:MM AM/PM)", placeholder="e.g., 07:18 AM", key="time")
    pob = st.text_input("Enter Place of Birth", key="pob")

    col1, col2 = st.columns(2)
    with col1:
        submit_btn = st.button("Submit")
    with col2:
        birthchart_btn = st.button("Generate Birth Chart")

    if submit_btn:
        if not name or not dob or not time_input or not pob:
            st.error("Please fill in all fields before submitting.")
        else:
            try:
                birth_time_obj = datetime.strptime(time_input, "%I:%M %p")
                birth_date_time = datetime.combine(dob, datetime.min.time()) + timedelta(hours=birth_time_obj.hour, minutes=birth_time_obj.minute)
                sun_sign = get_sun_sign(dob)
                moon_phase = get_moon_phase(dob)
                nakshatra, paadam, rasi = get_nakshatra_and_rasi(birth_date_time)
                doshas, remedies, mangalik_status = get_doshas_and_remedies(birth_date_time)
                st.markdown(f"""
                    <div class='astro-box'>
                        <h3>Your Sun Sign is: <span class='astro-highlight'>{sun_sign}</span></h3>
                        <h3>Your Moon Phase is: <span class='astro-highlight'>{moon_phase}</span></h3>
                        <p>Your Nakshatra is: <span class='astro-highlight'>{nakshatra}</span></p>
                        <p>Your Paadam (quarter of Nakshatra) is: <span class='astro-highlight'>{paadam}</span></p>
                        <p>Your Rasi (Moon Sign) is: <span class='astro-highlight'>{rasi}</span></p>
                        <p>Your Doshas: <span class='astro-highlight'>{', '.join(doshas) if doshas else 'None'}</span></p>
                        <p>Remedies: <span class='astro-highlight'>{', '.join(remedies) if remedies else 'None'}</span></p>
                        <p>Mangalik Status: <span class='astro-highlight'>{mangalik_status}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                col1_img, col2_img = st.columns(2)
                with col1_img:
                    vedic_zodiac_images = {
                        "Gemini": "assets/images/gemini.png",
                        "Capricorn": "assets/images/capricorn.png",
                        "Aquarius": "assets/images/aquarius.png",
                        "Taurus": "assets/images/taurus.png",
                        "Cancer": "assets/images/cancer.png",
                        "Leo": "assets/images/leo.png",
                        "Virgo": "assets/images/virgo.png",
                        "Libra": "assets/images/libra.png",
                        "Scorpio": "assets/images/scorpio.png",
                        "Sagittarius": "assets/images/sagittarius.png",
                        "Aries": "assets/images/aries.png",
                        "Pisces": "assets/images/pisces.png"
                    }
                    vedic_zodiac_image_path = vedic_zodiac_images.get(sun_sign, "assets/images/default_zodiac.png")
                    st.image(vedic_zodiac_image_path, caption=f"{sun_sign} Zodiac", use_container_width=True)
                with col2_img:
                    moon_phase_images = {
                        "Waning Crescent": "assets/images/waning_crescent.webp",
                        "New Moon": "assets/images/new_moon.jpg",
                        "Waxing Crescent": "assets/images/waxing_crescent.jpg",
                        "First Quarter": "assets/images/first_quarter.jpg",
                        "Waxing Gibbous": "assets/images/waxing_gibbous.jpg",
                        "Full Moon": "assets/images/full_moon.webp",
                        "Waning Gibbous": "assets/images/waning_gibbous.webp",
                        "Last Quarter": "assets/images/last_quarter.webp",
                    }
                    moon_image_path = moon_phase_images.get(moon_phase, "assets/images/full_moon.jpg")
                    st.image(moon_image_path, caption=f"Moon Phase: {moon_phase}", use_container_width=True)

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Personalized Astrological Report", ln=True, align="C")
                    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
                    pdf.cell(200, 10, txt=f"DOB: {dob}", ln=True)
                    pdf.cell(200, 10, txt=f"Birth Time: {time_input}", ln=True)
                    pdf.cell(200, 10, txt=f"Place of Birth: {pob}", ln=True)
                    pdf.cell(200, 10, txt=f"Sun Sign: {sun_sign}", ln=True)
                    pdf.cell(200, 10, txt=f"Moon Phase: {moon_phase}", ln=True)
                    pdf.cell(200, 10, txt=f"Nakshatra: {nakshatra}", ln=True)
                    pdf.cell(200, 10, txt=f"Doshas: {', '.join(doshas) if doshas else 'None'}", ln=True)
                    pdf.cell(200, 10, txt=f"Mangalik Status: {mangalik_status}", ln=True)

                    current_y = pdf.get_y() + 10
                    pdf.image(vedic_zodiac_image_path, x=10, y=current_y, w=50)
                    pdf.image(moon_image_path, x=70, y=current_y, w=50)
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    st.session_state["pdf_output"] = pdf_output

                    if "pdf_output" in st.session_state:
                        st.download_button("Download Report as PDF",
                        data=st.session_state["pdf_output"],
                        file_name="AstroReport.pdf",
                        mime="application/pdf")
            except Exception as e:
                st.error(f"Error generating report: {e}")

    if birthchart_btn:
        try:
            if not name.strip() or not time_input.strip() or not pob.strip():
                st.warning("⚠️ Please fill in all fields.")
            else:
                lat, lon = get_lat_lon(pob)
                if lat is None or lon is None:
                    st.error("❌ Invalid city name! Please enter a valid location.")
                else:
                    try:
                        birth_time_obj = datetime.strptime(time_input, "%I:%M %p")
                    except ValueError:
                        st.error("❌ Invalid time format! Use HH:MM AM/PM.")
                        raise ValueError("Invalid time format.")
                    local_tz = pytz.timezone("Asia/Kolkata")
                    birth_local = local_tz.localize(datetime.combine(dob, birth_time_obj.time()))
                    birth_date_time = birth_local.astimezone(pytz.utc)
                    planet_positions = get_planet_positions(birth_date_time, lat, lon)
                    house_positions_calc = get_house_positions(birth_date_time, lat, lon)
                    mapped_planets = map_positions_to_houses(planet_positions, house_positions_calc)
                    house_planet_map = {h: [] for h in range(1, 13)}
                    for planet, house in mapped_planets.items():
                        house_planet_map[house].append(planet)
                    fig, ax = plt.subplots(figsize=(22, 14))
                    ax.set_xlim(-1, 11)
                    ax.set_ylim(-2, 8)
                    ax.plot([0, 10, 10, 0, 0], [0, 0, 7, 7, 0], color='blue', lw=2)
                    lines = [
                        [(5, 7), (0, 3.5)], [(5, 7), (10, 3.5)],
                        [(0, 3.5), (5, 0)], [(10, 3.5), (5, 0)]
                    ]
                    lines += [
                        [(0, 7), (10, 0)], [(0, 0), (10, 7)]
                    ]
                    for line in lines:
                        ax.plot(*zip(*line), color='orange', lw=2)
                    for house, (x, y) in house_positions_plot.items():
                        ax.text(x, y, str(house), fontsize=18, ha="center", va="center", fontweight="bold")
                    planet_offsets = {house: 0 for house in house_positions_plot}
                    for planet, house in mapped_planets.items():
                        x, y = house_positions_plot[house]
                        y_offset = planet_offsets[house] * 0.3
                        ax.text(x, y - 0.3 - y_offset, planet, fontsize=15, ha="center", va="center", color="blue")
                        planet_offsets[house] += 1
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Birth Chart for {name}", ha='center', fontsize=25, fontweight='bold')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    
    st.markdown("---")
    st.subheader("Astrology Chatbot 🤖")
    if "chatbot_answers" not in st.session_state:
        st.session_state.chatbot_answers = {}
    main_questions = [
        "What is the current Moon Phase?",
        "Explain the significance of the Moon's position in astrology",
        "What is Mangal Dosha?",
        "What is Nadi Dosha?",
        "Explain how all the functions in this code work together"
    ]
    st.markdown("---")
    followups_map = {
        "What is the current Moon Phase?": [
            "How does the Moon phase affect daily energy?",
            "What are the cultural interpretations of the Moon phase?"
        ],
        "Explain the significance of the Moon's position in astrology": [
            "How does the Moon's position affect personality?",
            "What aspects of life are influenced by the Moon's position?"
        ],
        "What is Mangal Dosha?": [
            "What impact does Mangal Dosha have on relationships?",
            "How can one mitigate the effects of Mangal Dosha?"
        ],
        "What is Nadi Dosha?": [
            "What does Nadi Dosha indicate in a horoscope?",
            "Can Nadi Dosha be remedied, and how?"
        ],
        "Explain how all the functions in this code work together": [
            "How are planetary positions and house calculations integrated?",
            "How does the chatbot use the code to answer queries?"
        ]
    }
    st.subheader("Main Questions")
    for question in main_questions:
        clicked = st.button(question, key=f"main_{question}")
        if clicked:
            answer = get_chatbot_response(question)
            st.session_state.chatbot_answers[question] = answer
        if question in st.session_state.chatbot_answers:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {st.session_state.chatbot_answers[question]}")
            if question in followups_map:
                st.write("**Follow-up Questions:**")
                followup_questions = followups_map[question]
                cols = st.columns(len(followup_questions))
                for i, fquestion in enumerate(followup_questions):
                    with cols[i]:
                        fclicked = st.button(fquestion, key=f"followup_{fquestion}")
                        if fclicked:
                            fanswer = get_chatbot_response(fquestion)
                            st.session_state.chatbot_answers[f"f_{fquestion}"] = fanswer
                for fquestion in followup_questions:
                    if f"f_{fquestion}" in st.session_state.chatbot_answers:
                        st.write(f"**Q:** {fquestion}")
                        st.write(f"**A:** {st.session_state.chatbot_answers[f'f_{fquestion}']}")
    st.markdown("---")
    st.subheader("Free-Text Chatbot")
    with st.form(key="free_text_chatbot_form"):
        user_q = st.text_input("Ask your astrology question:")
        sub_q = st.form_submit_button("Submit Question")
        if sub_q and user_q:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "text": user_q})
            response = get_chatbot_response(user_q)
            st.session_state.messages.append({"role": "bot", "text": response})
    if "messages" in st.session_state:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                align = "right"
                color = "#FFB6C1"
            else:
                align = "left"
                color = "#ADD8E6"
            st.markdown(
                f"""
                <div style="text-align: {align};">
                    <div style="
                        display: inline-block;
                        background-color: {color};
                        padding: 10px;
                        border-radius: 10px;
                        margin: 5px;
                        max-width: 70%;
                        font-size: 16px;
                        color: black;">
                        {msg["text"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "Dashboard":
    st.markdown("### 📊 Explore, Predict & Discover Insights in Astrology Data Using AI & ML!")
    uploaded_file = st.file_uploader("📂 Upload an Astrology Data CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File Uploaded Successfully!")
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"⚠️ Your dataset contains {missing_values} missing values. Consider cleaning it.")
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())
        st.subheader("⚙️ Smart Feature Selection")
        if "Sun_Sign" in df.columns and "Moon_Phase" in df.columns:
            label_encoder = LabelEncoder()
            df["Mangalik_Status"] = label_encoder.fit_transform(df["Mangalik_Status"])
            df_encoded = pd.get_dummies(df[["Sun_Sign", "Moon_Phase"]])
            selected_features = []
            for col in df_encoded.columns:
                chi2_stat, p, _, _ = chi2_contingency(pd.crosstab(df_encoded[col], df["Mangalik_Status"]))
                if p < 0.05:
                    selected_features.append(col)
            if not selected_features:
                selected_features = df_encoded.columns.tolist()
            df_final = df_encoded[selected_features]
            df_final["Mangalik_Status"] = df["Mangalik_Status"]
            st.write(f"🔍 Selected Features: **{', '.join(selected_features)}**")
            st.subheader("📊 Dynamic Data Insights")
            viz_option = st.radio("Select Visualization:", ["Sun Sign Distribution", "Moon Phase Trend", "Correlation Heatmap"])
            if viz_option == "Sun Sign Distribution":
                fig = px.histogram(df, x="Sun_Sign", title="🌞 Sun Sign Distribution", color="Sun_Sign")
            elif viz_option == "Moon Phase Trend":
                fig = px.line(df, x=df.index, y="Moon_Phase", title="🌙 Moon Phase Trend", markers=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df_encoded.corr(), cmap="coolwarm", annot=True)
                st.pyplot(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("🧠 AI-Optimized Model Training")
            X = df_final.drop(columns=["Mangalik_Status"])
            y = df_final["Mangalik_Status"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_n_estimators = random.choice([50, 100, 150, 200])
            model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"🎯 **Best Model Accuracy:** {accuracy * 100:.2f}% (Auto-Tuned: {best_n_estimators} trees)")
            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
            st.pyplot(fig)
            st.subheader("🌌 Astrology Galaxy Clustering")
            k_clusters = st.slider("🔢 Select Number of Clusters", 2, 5, 3)
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            df_final["cluster"] = kmeans.fit_predict(X)
            fig = px.scatter_3d(df_final, x=df_final.columns[0], y=df_final.columns[1], z=df_final.columns[2],
                                color=df_final["cluster"].astype(str), title="🌌 Astrology Cluster Universe")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Data Science & AI Astrology Analysis Complete! 🚀")
