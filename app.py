"""
app.py — SleepSenseAI Flask server
============================================================
Serves predictions from the pipeline trained by model.py
(sleep_model.pkl). This file never trains anything — it is
model-agnostic and will keep working if the underlying
estimator in model.py changes.

Beyond the raw class probabilities, this endpoint also returns:
  - a composite Sleep Risk Score (0-100)
  - a per-vital wellness score against clinically reasonable
    reference bands (for the UI's vitals radar)
  - a comparison of the user's vitals against the average
    profile of the predicted class (from the training data)
  - a short list of plain-language contributing factors

None of this is a medical diagnosis — see the disclaimer
returned with every response and shown in the UI.

Before running this, train the model once with:
    python model.py

Then start the server with:
    python app.py
    open http://127.0.0.1:5000
"""

import math
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

from model import engineer_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "sleep_model.pkl")

app = Flask(__name__)

with open(MODEL_FILE, "rb") as f:
    ARTIFACT = pickle.load(f)

PIPELINE = ARTIFACT["pipeline"]
TARGET_MAPPING_INV = ARTIFACT["target_mapping_inv"]
FEATURE_ORDER = ARTIFACT["feature_order"]
CLASS_PROFILES = ARTIFACT["class_profiles"]

REQUIRED_FIELDS = [
    "gender", "age", "occupation", "sleepDuration", "sleepQuality",
    "physicalActivity", "stressLevel", "bmiCategory", "dailySteps",
    "systolicBP", "diastolicBP", "heartRate",
]

NUMERIC_RULES = {
    "age": (18.0, 80.0),
    "sleepDuration": (4.0, 10.0),
    "sleepQuality": (1.0, 10.0),
    "stressLevel": (1.0, 10.0),
    "physicalActivity": (0.0, 120.0),
    "dailySteps": (1000.0, 20000.0),
    "systolicBP": (90.0, 200.0),
    "diastolicBP": (55.0, 130.0),
    "heartRate": (40.0, 140.0),
}

FIELD_TO_COLUMN = {
    "gender": "Gender",
    "age": "Age",
    "occupation": "Occupation",
    "sleepDuration": "Sleep Duration",
    "sleepQuality": "Quality of Sleep",
    "physicalActivity": "Physical Activity Level",
    "stressLevel": "Stress Level",
    "bmiCategory": "BMI Category",
    "dailySteps": "Daily Steps",
    "systolicBP": "Systolic BP",
    "diastolicBP": "Diastolic BP",
    "heartRate": "Heart Rate",
}

# These values are read from the fitted encoder when possible, so backend
# validation follows the actual trained model rather than a duplicated list.
def _trained_categories(column):
    try:
        preprocessor = PIPELINE.named_steps["preprocessor"]
        cat_pipeline = dict(preprocessor.transformers_)["cat"]
        encoder = cat_pipeline.named_steps["encoder"]
        cat_columns = list(dict(preprocessor.transformers_)["cat"][2])
        idx = cat_columns.index(column)
        return {str(v) for v in encoder.categories_[idx]}
    except (KeyError, ValueError, AttributeError, IndexError):
        return set()


VALID_CATEGORIES = {
    "gender": _trained_categories("Gender"),
    "occupation": _trained_categories("Occupation"),
    "bmiCategory": _trained_categories("BMI Category"),
}

# Clinically-reasonable reference bands used only for the UI's wellness
# scoring (NOT the ML model).
VITAL_REFERENCE = {
    "sleepDuration":    ("Sleep Duration", "hrs",  7.0,  9.0,  3.0, 11.0, True),
    "sleepQuality":     ("Sleep Quality", "/10",  7.0, 10.0,  1.0, 10.0, True),
    "stressLevel":      ("Stress Level", "/10",  1.0,  4.0,  1.0, 10.0, False),
    "physicalActivity": ("Physical Activity", "min", 30.0, 90.0, 0.0, 150.0, True),
    "dailySteps":       ("Daily Steps", "steps", 7000.0, 12000.0, 500.0, 20000.0, True),
    "heartRate":        ("Resting Heart Rate", "bpm", 60.0, 80.0, 40.0, 140.0, False),
    "systolicBP":       ("Systolic BP", "mmHg", 90.0, 120.0, 80.0, 200.0, False),
    "diastolicBP":      ("Diastolic BP", "mmHg", 60.0, 80.0, 50.0, 130.0, False),
}

FEATURE_KEY_TO_COLUMN = {
    "age": "Age", "sleepDuration": "Sleep Duration", "sleepQuality": "Quality of Sleep",
    "physicalActivity": "Physical Activity Level", "stressLevel": "Stress Level",
    "heartRate": "Heart Rate", "dailySteps": "Daily Steps",
    "systolicBP": "Systolic BP", "diastolicBP": "Diastolic BP",
}


def _parse_number(field, value):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric.")
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    low, high = NUMERIC_RULES[field]
    if not low <= number <= high:
        raise ValueError(f"{field} must be between {low:g} and {high:g}.")
    return number


def validate_and_prepare_input(form_data):
    """Validate the API contract, then apply the shared model feature engineering."""
    if not isinstance(form_data, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [field for field in REQUIRED_FIELDS if field not in form_data]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")

    # Reject unexpected fields in the ML payload. Full name and other UI-only
    # metadata must never enter the model endpoint.
    unexpected = sorted(set(form_data) - set(REQUIRED_FIELDS))
    if unexpected:
        raise ValueError(f"Unexpected field(s): {', '.join(unexpected)}")

    categorical = {}
    for field in ("gender", "occupation", "bmiCategory"):
        value = form_data[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        value = value.strip()
        allowed = VALID_CATEGORIES.get(field, set())
        if allowed and value not in allowed:
            raise ValueError(f"Invalid {field}: {value}")
        categorical[field] = value

    numeric = {field: _parse_number(field, form_data[field]) for field in NUMERIC_RULES}

    raw_row = {
        "Gender": categorical["gender"],
        "Age": numeric["age"],
        "Occupation": categorical["occupation"],
        "Sleep Duration": numeric["sleepDuration"],
        "Quality of Sleep": numeric["sleepQuality"],
        "Physical Activity Level": numeric["physicalActivity"],
        "Stress Level": numeric["stressLevel"],
        "BMI Category": categorical["bmiCategory"],
        "Daily Steps": numeric["dailySteps"],
        "Systolic BP": numeric["systolicBP"],
        "Diastolic BP": numeric["diastolicBP"],
        "Heart Rate": numeric["heartRate"],
    }

    input_df = pd.DataFrame([raw_row])
    input_df = engineer_features(input_df)

    # The artifact's feature order is the final contract consumed by the fitted pipeline.
    missing_features = [c for c in FEATURE_ORDER if c not in input_df.columns]
    if missing_features:
        raise ValueError(f"Model feature(s) missing after engineering: {', '.join(missing_features)}")
    return input_df[FEATURE_ORDER], {**categorical, **numeric}


def score_vital(key, value):
    """Maps a raw vital reading to a 0-100 wellness score using a
    clinically-reasonable optimal band, plus a status label."""
    label, unit, opt_low, opt_high, hard_low, hard_high, higher_is_better = VITAL_REFERENCE[key]
    value = float(value)

    if opt_low <= value <= opt_high:
        score = 100.0
    elif value < opt_low:
        span = max(opt_low - hard_low, 1e-6)
        score = max(0.0, 100.0 * (1 - (opt_low - value) / span))
    else:
        span = max(hard_high - opt_high, 1e-6)
        score = max(0.0, 100.0 * (1 - (value - opt_high) / span))

    if score >= 75:
        status = "optimal"
    elif score >= 45:
        status = "borderline"
    else:
        status = "attention"

    return {
        "key": key,
        "label": label,
        "unit": unit,
        "value": round(value, 1),
        "score": round(score, 1),
        "status": status,
        "optimalRange": [opt_low, opt_high],
    }


def build_vitals(form_data):
    return [score_vital(key, form_data[key]) for key in VITAL_REFERENCE]


def build_profile_comparison(form_data, predicted_label):
    """Compares the user's raw vitals against the training-data
    average for the predicted class and for a healthy baseline."""
    healthy_profile = CLASS_PROFILES.get("No Sleep Disorder", {})
    predicted_profile = CLASS_PROFILES.get(predicted_label, {})

    comparison = []
    for key, column in FEATURE_KEY_TO_COLUMN.items():
        if column not in healthy_profile:
            continue
        comparison.append({
            "key": key,
            "label": column,
            "yourValue": round(float(form_data[key]), 1),
            "healthyAverage": round(healthy_profile.get(column, 0), 1),
            "predictedClassAverage": round(predicted_profile.get(column, 0), 1),
        })
    return comparison


def build_factors(form_data, predicted_label):
    factors = []
    if float(form_data["stressLevel"]) >= 7:
        factors.append(f"Elevated stress level ({form_data['stressLevel']}/10)")
    if float(form_data["sleepQuality"]) <= 5:
        factors.append(f"Below-average self-rated sleep quality ({form_data['sleepQuality']}/10)")
    if float(form_data["sleepDuration"]) < 6.5:
        factors.append(f"Shorter sleep duration ({float(form_data['sleepDuration']):.1f} hrs)")
    if form_data["bmiCategory"] != "Normal":
        factors.append(f"{form_data['bmiCategory']} BMI category")
    if float(form_data["systolicBP"]) > 130 or float(form_data["diastolicBP"]) > 85:
        factors.append("Blood pressure above typical resting range")
    if float(form_data["heartRate"]) > 80:
        factors.append(f"Resting heart rate above typical range ({form_data['heartRate']} bpm)")
    if float(form_data["physicalActivity"]) < 30:
        factors.append(f"Lower daily physical activity ({form_data['physicalActivity']} min)")
    if float(form_data["age"]) > 45:
        factors.append("Age above 45, a mild Sleep Apnea risk factor")
    if not factors:
        factors.append("No strong risk indicators found — your metrics look well balanced.")
    return factors


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.get_json(silent=True)
    try:
        input_df, clean_data = validate_and_prepare_input(form_data)
    except ValueError as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    probabilities = PIPELINE.predict_proba(input_df)[0]
    class_indices = PIPELINE.named_steps["model"].classes_
    class_names = [TARGET_MAPPING_INV[i] for i in class_indices]

    probability_by_class = {
        name: round(float(prob) * 100, 1) for name, prob in zip(class_names, probabilities)
    }
    predicted_label = max(probability_by_class, key=probability_by_class.get)

    healthy_probability = probability_by_class.get("No Sleep Disorder", 0.0)
    risk_score = round(100 - healthy_probability, 1)

    response = {
        "probabilities": probability_by_class,
        "predictedClass": predicted_label,
        "riskScore": risk_score,
        "vitals": build_vitals(form_data),
        "profileComparison": build_profile_comparison(form_data, predicted_label),
        "factors": build_factors(form_data, predicted_label),
        "disclaimer": (
            "This result is a screening estimate based on the information you "
            "provided, not a medical diagnosis. Please consult a healthcare "
            "professional for any concerns about your sleep health."
        ),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=False)
