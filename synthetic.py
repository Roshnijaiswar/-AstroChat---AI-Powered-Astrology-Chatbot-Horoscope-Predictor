import subprocess
import sys
import importlib.util

def install_library(lib_name):
    """Install the given library using pip."""
    print(f"Installing missing library: {lib_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])

def check_and_install_libraries():
    """Scan for required libraries and install missing ones."""
    required_libraries = {
        "pandas", "faker", "numpy", "matplotlib", "seaborn", "cohere"
    }

    for lib in required_libraries:
        if importlib.util.find_spec(lib) is None:
            install_library(lib)

# ✅ Run before importing libraries
check_and_install_libraries()

# ✅ Now import all libraries safely
import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
import cohere  # Now it won't throw ModuleNotFoundError















import streamlit as st
from datetime import datetime, timedelta
import ephem
import pytz
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
from faker import Faker

fake = Faker('IN')  # Set locale to generate Indian names
import random


# Additional imports for Data Science functionality
import random
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
        "Rahu": ephem.degrees(ephem.Moon(observer).ra - 180),  
        "Ketu": ephem.degrees(ephem.Moon(observer).ra + 180)   
    }
    planet_positions = {name: ephem.degrees(ephem.Ecliptic(planet).lon) for name, planet in planets.items()}
    return planet_positions

# ----------------------------------- Function to determine house positions -----------------------------------------------
def get_house_positions(birth_date_time, lat, lon):
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)
    observer.date = ephem.Date(birth_date_time)
    sidereal_time = observer.sidereal_time()  
    ascendant = ephem.Ecliptic(ephem.Sun(observer)).lon  # Approximate Lagna
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
    nakshatra_num = int(moon_longitude / 13.3333333333)
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






traits = {
    "Aries": {"interests": ["Adventure", "Sports", "Leadership"], "personality": ["Bold", "Energetic", "Confident"]},
    "Taurus": {"interests": ["Music", "Cooking", "Gardening"], "personality": ["Patient", "Reliable", "Practical"]},
    "Gemini": {"interests": ["Reading", "Writing", "Travel"], "personality": ["Witty", "Curious", "Adaptable"]},
    "Cancer": {"interests": ["Family", "Art", "Helping Others"], "personality": ["Sensitive", "Loyal", "Caring"]},
    "Leo": {"interests": ["Theater", "Leadership", "Luxury"], "personality": ["Confident", "Creative", "Charismatic"]},
    "Virgo": {"interests": ["Puzzles", "Health", "Organizing"], "personality": ["Analytical", "Kind", "Diligent"]},
    "Libra": {"interests": ["Fashion", "Socializing", "Music"], "personality": ["Charming", "Fair-minded", "Diplomatic"]},
    "Scorpio": {"interests": ["Mystery", "Investigation", "Psychology"], "personality": ["Passionate", "Intense", "Strategic"]},
    "Sagittarius": {"interests": ["Travel", "Philosophy", "Adventure"], "personality": ["Optimistic", "Independent", "Curious"]},
    "Capricorn": {"interests": ["Business", "Discipline", "Climbing"], "personality": ["Responsible", "Hardworking", "Practical"]},
    "Aquarius": {"interests": ["Technology", "Innovation", "Social Causes"], "personality": ["Visionary", "Intellectual", "Rebellious"]},
    "Pisces": {"interests": ["Art", "Spirituality", "Music"], "personality": ["Dreamy", "Compassionate", "Intuitive"]}
}




general_horoscope = {
        "Aries": ["Bold moves will lead to success.", "Trust your instincts and take action.", "New opportunities are on the horizon.", "Take charge of your destiny today.", "Energy and enthusiasm will drive your success."],
        "Taurus": ["Patience will bring rewards.", "Financial stability is in your favor.", "Ground yourself in nature for clarity.", "Stay determined and success will follow.", "A steady approach will lead to gains."],
        "Gemini": ["Your communication skills will shine.", "A short trip may bring new opportunities.", "Keep your mind open to new ideas.", "Adaptability is your strength today.", "A new perspective will open doors."],
        "Cancer": ["Emotional balance is key today.", "Family time will bring joy.", "Listen to your intuition carefully.", "Trust your emotions to guide you.", "A caring approach will bring peace."],
        "Leo": ["Leadership opportunities are emerging.", "Your energy will inspire others.", "Take center stage in your career.", "Confidence will open new doors.", "A bold step forward will pay off."],
        "Virgo": ["Focus on organization and efficiency.", "A new learning opportunity awaits.", "Pay attention to details.", "Practicality will lead to success.", "Precision is your key to achievement."],
        "Libra": ["Balance your personal and professional life.", "Your charm will open new doors.", "Make thoughtful decisions in relationships.", "Fairness and diplomacy will serve you well.", "A harmonious approach will bring results."],
        "Scorpio": ["Deep introspection will bring clarity.", "A powerful transformation is ahead.", "Trust the process of change.", "Your intensity will drive major breakthroughs.", "A secret may be revealed today."],
        "Sagittarius": ["Adventure is calling you.", "Stay optimistic and embrace change.", "A new learning journey is ahead.", "Your thirst for knowledge will be rewarded.", "A spontaneous decision may bring joy."],
        "Capricorn": ["Hard work will pay off soon.", "Financial success is near.", "Discipline is your key to success.", "Patience and persistence will yield results.", "A long-term goal is within reach."],
        "Aquarius": ["Innovative ideas will lead to success.", "Collaboration will bring growth.", "A surprise encounter may change your perspective.", "Embrace your uniqueness today.", "An unconventional approach may work best."],
        "Pisces": ["Creativity will flow effortlessly.", "Pay attention to your dreams.", "Express your emotions freely.", "A deep emotional connection may form.", "Your intuition will guide you forward."],
}
health_horoscope =  {
        "young": ["Hit the gym and stay active!", "Maintain a healthy diet for long-term fitness.", "Keep exercising regularly for mental clarity."],
        "adult": ["Take regular body checkups to stay healthy.", "Maintain a balanced diet and manage stress.", "Preventive healthcare will save you in the long run."]
}
work_horoscope = {
        "young": ["Invest wisely and plan your career path.", "Work hard now for a bright future.", "Opportunities are waiting—be proactive!"],
        "adult": ["Stay updated with market trends.", "Avoid financial scams and invest carefully.", "Maintain professional relationships wisely."]
}
relationship_horoscope = {
        "single": ["Love may be around the corner.", "Enjoy your independence and explore new connections.", "Focus on self-love before seeking romance."],
        "married": ["Communication will strengthen your bond.", "Plan a romantic getaway with your partner.", "Balance work and love life effectively."],
        "divorced": ["Healing takes time, but new love awaits.", "Self-care is your priority now.", "Open your heart to new beginnings."],
        "widowed": ["Cherish your memories, but stay open to joy.", "Surround yourself with supportive people.", "Focus on personal growth and happiness."]
}





# ---------------------------------------------- Synthetic Data Generation for Data Science Page ----------------------------------
def generate_synthetic_data(num_samples=1000):
    cities = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Mumbai": (19.0760, 72.8777),
    "Tokyo": (35.6895, 139.6917),
    "Sydney": (-33.8688, 151.2093),

    # Indian State Capitals
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),  # Maharashtra
    "Kolkata": (22.5726, 88.3639),  # West Bengal
    "Chennai": (13.0827, 80.2707),  # Tamil Nadu
    "Bangalore": (12.9716, 77.5946),  # Karnataka
    "Hyderabad": (17.3850, 78.4867),  # Telangana
    "Ahmedabad": (23.0225, 72.5714),  # Gujarat
    "Jaipur": (26.9124, 75.7873),  # Rajasthan
    "Lucknow": (26.8467, 80.9462),  # Uttar Pradesh
    "Bhopal": (23.2599, 77.4126),  # Madhya Pradesh
    "Patna": (25.5941, 85.1376),  # Bihar
    "Thiruvananthapuram": (8.5241, 76.9366),  # Kerala
    "Bhubaneswar": (20.2961, 85.8245),  # Odisha
    "Raipur": (21.2514, 81.6296),  # Chhattisgarh
    "Ranchi": (23.3441, 85.3096),  # Jharkhand
    "Dispur": (26.1445, 91.7362),  # Assam
    "Shillong": (25.5788, 91.8933),  # Meghalaya
    "Aizawl": (23.7271, 92.7173),  # Mizoram
    "Imphal": (24.8170, 93.9368),  # Manipur
    "Itanagar": (27.1025, 93.6920),  # Arunachal Pradesh
    "Gangtok": (27.3314, 88.6138),  # Sikkim
    "Kohima": (25.6747, 94.1100),  # Nagaland
    "Agartala": (23.8315, 91.2868),  # Tripura
    "Panaji": (15.4909, 73.8278),  # Goa
    "Shimla": (31.1048, 77.1734),  # Himachal Pradesh
    "Dehradun": (30.3165, 78.0322),  # Uttarakhand
    "Chandigarh": (30.7333, 76.7794),  # Punjab & Haryana (Union Territory)
    "Puducherry": (11.9416, 79.8083),  # Puducherry (Union Territory)
    "Port Blair": (11.6234, 92.7265),  # Andaman and Nicobar Islands (UT)
    "Daman": (20.3974, 72.8328),  # Dadra and Nagar Haveli and Daman and Diu (UT)
    }

    

    data = []
    for i in range(num_samples):
        name = fake.name()
        start_date = datetime(1980, 1, 1)
        end_date = datetime(2010, 12, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        random_date = start_date + timedelta(days=random_days)
        random_hour = random.randint(0, 23)
        random_minute = random.randint(0, 59)
        birth_time = datetime(2000, 1, 1, random_hour, random_minute)
        time_str = birth_time.strftime("%I:%M %p")
        pob = random.choice(list(cities.keys()))
        lat, lon = cities[pob]
        birth_date_time = datetime.combine(random_date, birth_time.time())
        sun_sign = get_sun_sign(random_date)
        sun_sign = get_sun_sign(random_date)
        
        # Get traits based on sun sign
        if sun_sign in traits:
            interests = traits[sun_sign]["interests"]
            personalities = traits[sun_sign]["personality"]
            # Select random interest and personality from the lists
            interest = random.choice(interests)
            personality = random.choice(personalities)
        else:
            interest = "Unknown"
            personality = "Unknown"
        # Calculate age and relationship status
        age_category = "young" if (datetime.now().year - random_date.year) < 30 else "adult"
        relationship_status = random.choice(["single", "married", "divorced", "widowed"])

        moon_phase = get_moon_phase(random_date)
        nakshatra, paadam, rasi = get_nakshatra_and_rasi(birth_date_time)
        doshas, remedies, mangalik_status = get_doshas_and_remedies(birth_date_time)
        record = {
            "Name": name,
            "DOB": random_date,
            "Time": time_str,
            "Place": pob,
            "Sun_Sign": sun_sign,
            "Moon_Phase": moon_phase,
            "Nakshatra": nakshatra,
            "Paadam": paadam,
            "Rasi": rasi,
            # ... previous fields ...
            "Doshas": ", ".join(doshas) if doshas else "None",
            "Remedies": ", ".join(remedies) if remedies else "None",

            "Mangalik_Status": mangalik_status,
            "Personality": personality,
            "Interest": interest,
            "General_Horoscope": random.choice(general_horoscope[sun_sign]) if sun_sign in general_horoscope else "Not available",
            "Health_Horoscope": random.choice(health_horoscope[age_category]),
            "Work_Horoscope": random.choice(work_horoscope[age_category]),
            "Relationship_Horoscope": random.choice(relationship_horoscope[relationship_status])

        }
        data.append(record)
        df = pd.DataFrame(data)
        df.to_csv("data.csv", index=False)  # Saves as CSV without index
        print("CSV file saved successfully!")


generate_synthetic_data(1000)  # Generate 5000 records