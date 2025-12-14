import requests
import pandas as pd
import folium
from sklearn.ensemble import IsolationForest

API_KEY = "01c08cae76c7ffeb584451a8c674331c"

print("SCRIPT STARTED")

# ----------------------------
# Indian cities (coordinates)
# ----------------------------
cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567)
}


# ----------------------------
# Fetch AQI data
# ----------------------------
def get_aqi(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution?"
        f"lat={lat}&lon={lon}&appid={API_KEY}"
    )

    r = requests.get(url, timeout=5)
    data = r.json()["list"][0]["components"]

    pm25 = data["pm2_5"]
    pm10 = data["pm10"]
    no2 = data["no2"]

    return pm25, pm10, no2

# ----------------------------
# Collect data
# ----------------------------
rows = []

for city, (lat, lon) in cities.items():
    pm25, pm10, no2 = get_aqi(lat, lon)
    print(city, pm25, pm10, no2)
    rows.append([city, lat, lon, pm25, pm10, no2])

df = pd.DataFrame(
    rows,
    columns=["City", "Lat", "Lon", "PM2.5", "PM10", "NO2"]
)

# ----------------------------
# Data science logic
# ----------------------------
df["Risk_Score"] = (
    0.5 * df["PM2.5"] +
    0.3 * df["PM10"] +
    0.2 * df["NO2"]
)

model = IsolationForest(contamination=0.2, random_state=42)
df["Anomaly"] = model.fit_predict(df[["Risk_Score"]])
df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Abnormal"})

def classify(score):
    if score < 50:
        return "Good"
    elif score < 100:
        return "Moderate"
    else:
        return "Hazardous"

df["Health_Risk"] = df["Risk_Score"].apply(classify)

# ----------------------------
# Create map
# ----------------------------
india_map = folium.Map(location=[22.5, 80.9], zoom_start=5)

colors = {
    "Good": "green",
    "Moderate": "orange",
    "Hazardous": "red"
}

for _, row in df.iterrows():
    popup = f"""
    <b>{row['City']}</b><br>
    PM2.5: {row['PM2.5']}<br>
    PM10: {row['PM10']}<br>
    NO₂: {row['NO2']}<br>
    Risk: {row['Health_Risk']}<br>
    Status: {row['Anomaly']}
    """

    folium.Marker(
        [row["Lat"], row["Lon"]],
        popup=popup,
        icon=folium.Icon(color=colors[row["Health_Risk"]])
    ).add_to(india_map)

india_map.save("india_aqi_openweather.html")
print("Map generated: india_aqi_openweather.html")
print("SCRIPT FINISHED")

