import requests
import pandas as pd
import folium
from sklearn.ensemble import IsolationForest

# ----------------------------
# City coordinates (India)
# ----------------------------
cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867)
}

# ----------------------------
# Fetch AQI data
# ----------------------------
def get_air_quality(city):
    url = f"https://api.openaq.org/v2/latest?city={city}&limit=1"
    response = requests.get(url).json()

    pm25 = pm10 = no2 = None

    try:
        measurements = response["results"][0]["measurements"]
        for m in measurements:
            if m["parameter"] == "pm25":
                pm25 = m["value"]
            elif m["parameter"] == "pm10":
                pm10 = m["value"]
            elif m["parameter"] == "no2":
                no2 = m["value"]
    except:
        pass

    return pm25, pm10, no2

# ----------------------------
# Collect data
# ----------------------------
data = []

for city in cities:
    pm25, pm10, no2 = get_air_quality(city)
    data.append([city, pm25, pm10, no2])

df = pd.DataFrame(data, columns=["City", "PM2.5", "PM10", "NO2"])
df.fillna(df.mean(numeric_only=True), inplace=True)

# ----------------------------
# Health Risk Score (DS logic)
# ----------------------------
df["Risk_Score"] = (
    0.5 * df["PM2.5"] +
    0.3 * df["PM10"] +
    0.2 * df["NO2"]
)

# ----------------------------
# Anomaly Detection
# ----------------------------
model = IsolationForest(contamination=0.2, random_state=42)
df["Anomaly"] = model.fit_predict(df[["Risk_Score"]])
df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Abnormal"})

# ----------------------------
# Risk Classification
# ----------------------------
def classify_risk(score):
    if score < 50:
        return "Good"
    elif score < 100:
        return "Moderate"
    else:
        return "Hazardous"

df["Health_Risk"] = df["Risk_Score"].apply(classify_risk)

# ----------------------------
# Create Map
# ----------------------------
india_map = folium.Map(location=[22.5, 80.9], zoom_start=5)

color_map = {
    "Good": "green",
    "Moderate": "orange",
    "Hazardous": "red"
}

for _, row in df.iterrows():
    lat, lon = cities[row["City"]]

    popup = f"""
    <b>{row['City']}</b><br>
    PM2.5: {row['PM2.5']}<br>
    PM10: {row['PM10']}<br>
    NO2: {row['NO2']}<br>
    <b>Risk:</b> {row['Health_Risk']}<br>
    <b>Status:</b> {row['Anomaly']}
    """

    folium.Marker(
        location=[lat, lon],
        popup=popup,
        icon=folium.Icon(
            color=color_map[row["Health_Risk"]],
            icon="info-sign"
        )
    ).add_to(india_map)

india_map.save("india_air_quality.html")

print("Map generated: open india_air_quality.html")

