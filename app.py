from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# ================= SAFE MATPLOTLIB =================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= ML =================
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================= NLP CHATBOT =================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= UTILS =================
from io import BytesIO
import base64

app = Flask(__name__)

# ======================================================
# 1️⃣ LOAD DATASET
# ======================================================
df = pd.read_csv("my_data.csv")

FEATURES = [
    "NoofRooms", "Occupancy", "HeavyAppliances",
    "HeatingCoolingSystems", "AvgDailyUsageHours",
    "Season", "TariffPerUnit", "PeakUsage"
]

X = df[FEATURES]
y = df["ElectricityBill"]

np.random.seed(42)
y = y + np.random.normal(0, 120, size=len(y))

# ======================================================
# 2️⃣ TRAIN / TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================================================
# 3️⃣ PIPELINE (PRODUCTION SAFE)
# ======================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.2,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# ======================================================
# 4️⃣ METRICS
# ======================================================
y_pred = pipeline.predict(X_test)

R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

accuracy_percent = round(R2 * 100, 2)

# ======================================================
# 5️⃣ STATE-WISE TARIFF LOGIC (INDIA)
# ======================================================
STATE_TARIFF = {
    "UP": (6.5, "Government subsidy & rural mix"),
    "Delhi": (7.0, "Urban slab-based pricing"),
    "Maharashtra": (9.0, "High industrial & peak demand"),
    "Karnataka": (8.5, "IT-sector driven consumption"),
    "Tamil Nadu": (7.5, "Balanced hydro-thermal mix")
}

# ======================================================
# 6️⃣ USAGE CATEGORY
# ======================================================
def usage_category(bill):
    if bill < 800:
        return "🟢 Low Usage"
    elif bill < 1400:
        return "🟡 Moderate Usage"
    return "🔴 High Usage"

# ======================================================
# 7️⃣ ENERGY OPTIMIZER
# ======================================================
def energy_optimizer(rooms, occupancy, appliances, hvac, bill):
    tips, savings = [], 0

    if hvac >= 4:
        tips.append("❄️ Reduce AC/Heater usage by 1 hour/day → Save ~₹180")
        savings += 180

    if appliances >= 4:
        tips.append("🔌 Switch to 5-star rated appliances → Save ~₹120")
        savings += 120

    if occupancy >= 5:
        tips.append("⏱️ Shift heavy usage to non-peak hours → Save ~₹90")
        savings += 90

    if rooms >= 5:
        tips.append("💡 Use LED lighting in all rooms → Save ~₹70")
        savings += 70

    optimized_bill = max(bill - savings, bill * 0.6)
    return tips, round(optimized_bill, 2), savings

# ======================================================
# 8️⃣ ENERGY SCORE
# ======================================================
def energy_score(bill, optimized):
    return min(100, max(30, int(((bill - optimized) / bill) * 120 + 50)))

# ======================================================
# 9️⃣ CONFIDENCE SCORE
# ======================================================
def confidence_score(r2):
    return round(70 + r2 * 30, 2)

# ======================================================
# 🔟 FEATURE IMPACT (EXPLAINABILITY)
# ======================================================
def feature_impact(input_data):
    importance = pipeline.named_steps["model"].feature_importances_
    impact = {}
    for f, imp, val in zip(FEATURES, importance, input_data):
        impact[f] = round(imp * val * 100, 2)
    return dict(sorted(impact.items(), key=lambda x: x[1], reverse=True))

# ======================================================
# 1️⃣1️⃣ ANOMALY DETECTION
# ======================================================
def detect_anomaly(bill, rooms):
    avg = df[df["NoofRooms"] == rooms]["ElectricityBill"].mean()
    if not np.isnan(avg) and bill > avg * 1.25:
        return "⚠️ Usage significantly higher than similar households."
    return "✅ Usage within normal range."

# ======================================================
# 1️⃣2️⃣ WHAT-IF SIMULATION (SAFE)
# ======================================================
def what_if(base):
    base = list(base)
    result = {}

    temp = base.copy()
    temp[3] = max(1, temp[3] - 1)
    result["Reduce HVAC usage"] = round(pipeline.predict([temp])[0], 2)

    temp = base.copy()
    temp[2] = max(1, temp[2] - 1)
    result["Efficient appliances"] = round(pipeline.predict([temp])[0], 2)

    temp = base.copy()
    temp[4] = max(1, temp[4] - 1)
    result["Reduce daily usage"] = round(pipeline.predict([temp])[0], 2)

    return result

# ======================================================
# 1️⃣3️⃣ GRAPH
# ======================================================
def generate_plot(actual, predicted):
    plt.figure(figsize=(6,4))
    plt.scatter(actual, predicted, alpha=0.7)
    plt.plot([actual.min(), actual.max()],
             [actual.min(), actual.max()], "k--")
    plt.xlabel("Actual Bill (₹)")
    plt.ylabel("Predicted Bill (₹)")
    plt.title("Actual vs Predicted Electricity Bill")

    img = BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plt.close()
    return base64.b64encode(img.read()).decode()

plot_img = generate_plot(y_test, y_pred)

# ======================================================
# 1️⃣4️⃣ SMART CHATBOT
# ======================================================
KB = {
    "model": "We use XGBoost to capture non-linear energy consumption patterns.",
    "accuracy": "R² score shows how much real usage variance the model explains.",
    "save": "Optimization simulates reduced appliance and HVAC usage.",
    "tariff": "Tariffs differ by state due to policy and infrastructure.",
    "anomaly": "Anomaly means your usage is higher than similar homes.",
    "why high": "High bill is mainly due to HVAC, appliances, and tariff impact."
}

vectorizer = TfidfVectorizer()
keys = list(KB.keys())
vectors = vectorizer.fit_transform(keys)

def chatbot_response(text):
    text = text.lower()

    for state, (_, reason) in STATE_TARIFF.items():
        if state.lower() in text:
            return f"📍 {state}: {reason}"

    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, vectors)
    idx = sim.argmax()

    if sim[0][idx] > 0.15:
        return KB[keys[idx]]

    return "🤖 Ask about bill, savings, tariff, or model logic."

# ======================================================
# 1️⃣5️⃣ ROUTES
# ======================================================
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        rooms = float(request.form["size"])
        people = float(request.form["people"])
        appliances = float(request.form["insulation"])
        hvac = float(request.form["heating_cooling_systems"])
        state = request.form.get("state", "UP")

        tariff, _ = STATE_TARIFF.get(state, (8.0,"Standard"))

        input_data = [
            rooms, people, appliances, hvac,
            7, 1, tariff, 1
        ]

        bill = pipeline.predict([input_data])[0]
        units = round(bill / tariff, 2)

        tips, optimized, savings = energy_optimizer(
            rooms, people, appliances, hvac, bill
        )

        return render_template(
            "index.html",
            size=rooms,
            people=people,
            insulation=appliances,
            heating_cooling_systems=hvac,
            predicted_energy=round(bill,2),
            predicted_energy_divided=units,
            accuracy=accuracy_percent,
            mae=round(MAE,2),
            mse=round(MSE,2),
            usage_category=usage_category(bill),
            confidence=confidence_score(R2),
            suggestions=tips,
            optimized_bill=optimized,
            savings=savings,
            energy_score=energy_score(bill, optimized),
            anomaly=detect_anomaly(bill, rooms),
            what_if=what_if(input_data),
            impact=feature_impact(input_data),
            plot=plot_img
        )

    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    return chatbot_response(request.form["message"])

# ======================================================
# 🚀 SAFE RUN
# ======================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
