from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data/aqi_data.csv")

# Load model
model = joblib.load("models/aqi_model.pkl")

# Prepare pollutant mapping
data["pollutant_id"] = data["pollutant_id"].str.strip().str.upper()
unique_pollutants = sorted(data["pollutant_id"].unique())
pollutant_map = {p: i for i, p in enumerate(unique_pollutants)}

# =============================
# ROUTES
# =============================
@app.route("/")
def index():
    states = sorted(data["state"].unique())
    return render_template("index.html", states=states)

# Get cities for a given state
@app.route("/get_cities", methods=["POST"])
def get_cities():
    state = request.form.get("state")
    cities = sorted(data[data["state"] == state]["city"].unique().tolist())
    return jsonify({"success": True, "cities": cities})

# Fetch pollutant info for state+city
@app.route("/fetch", methods=["POST"])
def fetch():
    state = request.form.get("state")
    city = request.form.get("city")

    df = data[(data["state"] == state) & (data["city"] == city)]

    if df.empty:
        return jsonify({"success": False, "message": "No data found!"})

    info = {
        "pollutant_min": round(df["pollutant_min"].mean(), 2),
        "pollutant_max": round(df["pollutant_max"].mean(), 2),
        "latitude": round(df["latitude"].mean(), 6),
        "longitude": round(df["longitude"].mean(), 6),
        "pollutants": df["pollutant_id"].unique().tolist()
    }

    return jsonify({"success": True, "data": info})

# Predict pollutant_avg
@app.route("/predict", methods=["POST"])
def predict():
    state = request.form["state"]
    city = request.form["city"]

    pollutant_min = float(request.form["pollutant_min"])
    pollutant_max = float(request.form["pollutant_max"])
    latitude = float(request.form["latitude"])
    longitude = float(request.form["longitude"])

    pollutants = data[(data["state"] == state) & (data["city"] == city)]["pollutant_id"].unique().tolist()
    pollutant_nums = [pollutant_map[p] for p in pollutants]
    pollutant_id_value = np.mean(pollutant_nums)

    # Model input (5 features)
    X = [[pollutant_min, pollutant_max, latitude, longitude, pollutant_id_value]]
    predicted_avg = model.predict(X)[0]

    # AQI Category (based on pollutant_avg)
    if predicted_avg < 50:
        category = "Good"
        css_class = "good"
    elif predicted_avg < 100:
        category = "Moderate"
        css_class = "moderate"
    else:
        category = "Poor"
        css_class = "poor"

    return render_template(
        "result.html",
        aqi_value=round(predicted_avg, 2),
        category=category,
        css_class=css_class,
        pollutants=pollutants,
        pollutant_min=pollutant_min,
        pollutant_max=pollutant_max,
        latitude=latitude,
        longitude=longitude,
        state=state,
        city=city
    )

if __name__ == "__main__":
    app.run(debug=True)
