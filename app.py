"""
app.py — SleepSense AI Flask server
============================================================

Serves predictions from the pipeline trained by model.py
(sleep_model.pkl).

This file never trains the model.

The application:
    - Loads sleep_model.pkl
    - Validates incoming prediction data
    - Applies the same feature engineering used during training
    - Sends the engineered data through the saved pipeline
    - Returns prediction probabilities, risk score, vitals,
      profile comparison, risk factors, and disclaimer

None of this is a medical diagnosis.
"""

import math
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

from model import engineer_features


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "sleep_model.pkl",
)


# ------------------------------------------------------------------
# Flask application
# ------------------------------------------------------------------

app = Flask(__name__)


# ------------------------------------------------------------------
# Load trained model artifact
# ------------------------------------------------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_FILE}. "
        "Make sure sleep_model.pkl exists in the project root."
    )

with open(MODEL_FILE, "rb") as f:
    ARTIFACT = pickle.load(f)


# ------------------------------------------------------------------
# Validate model artifact
# ------------------------------------------------------------------

REQUIRED_ARTIFACT_KEYS = [
    "pipeline",
    "target_mapping_inv",
    "feature_order",
    "class_profiles",
]

missing_artifact_keys = [
    key
    for key in REQUIRED_ARTIFACT_KEYS
    if key not in ARTIFACT
]

if missing_artifact_keys:
    raise ValueError(
        "sleep_model.pkl is missing required artifact key(s): "
        + ", ".join(missing_artifact_keys)
    )


PIPELINE = ARTIFACT["pipeline"]

TARGET_MAPPING_INV = ARTIFACT["target_mapping_inv"]

FEATURE_ORDER = ARTIFACT["feature_order"]

CLASS_PROFILES = ARTIFACT["class_profiles"]


# ------------------------------------------------------------------
# Expected model feature contract
# ------------------------------------------------------------------

# The current notebook/model uses:
#
#   Raw features
#   +
#   Pulse Pressure
#   Sleep Quality Index
#
# IMPORTANT:
# Stress-Sleep Interaction is intentionally NOT included.

EXPECTED_ENGINEERED_FEATURES = [
    "Pulse Pressure",
    "Sleep Quality Index",
]

missing_engineered_features = [
    feature
    for feature in EXPECTED_ENGINEERED_FEATURES
    if feature not in FEATURE_ORDER
]

unexpected_stress_interaction = (
    "Stress-Sleep Interaction" in FEATURE_ORDER
)

if missing_engineered_features:
    raise ValueError(
        "The loaded model artifact is missing expected engineered "
        "feature(s): "
        + ", ".join(missing_engineered_features)
    )

if unexpected_stress_interaction:
    raise ValueError(
        "The loaded sleep_model.pkl contains the obsolete "
        "'Stress-Sleep Interaction' feature. "
        "Please rebuild sleep_model.pkl using the latest model.py "
        "without that feature."
    )


# ------------------------------------------------------------------
# Validation schema
# ------------------------------------------------------------------

# This schema is used by the frontend through /validation-rules.
#
# It is intentionally kept separate from feature engineering.
# Stress Level is still an input because it is a raw model feature.
# It is NOT multiplied with Sleep Duration.

VALIDATION_SCHEMA = {
    "gender": {
        "type": "select",
        "label": "Gender",
        "options": [
            "Male",
            "Female",
        ],
    },

    "age": {
        "type": "number",
        "label": "Age",
        "unit": "years",
        "min": 18,
        "max": 80,
        "step": 1,
    },

    "occupation": {
        "type": "select",
        "label": "Occupation",
        "options": [
            "Accountant",
            "Doctor",
            "Engineer (Non-Software)",
            "Entrepreneur",
            "Lawyer",
            "Manager",
            "Nurse",
            "Sales Representative",
            "Scientist",
            "Software Engineer",
            "Student",
            "Teacher",
        ],
    },

    "sleepDuration": {
        "type": "number",
        "label": "Sleep Duration",
        "unit": "hours/night",
        "min": 4,
        "max": 10,
        "step": 0.1,
    },

    "sleepQuality": {
        "type": "number",
        "label": "Quality of Sleep",
        "unit": "/10",
        "min": 1,
        "max": 10,
        "step": 1,
    },

    "physicalActivity": {
        "type": "number",
        "label": "Physical Activity",
        "unit": "minutes/day",
        "min": 0,
        "max": 120,
        "step": 5,
    },

    "stressLevel": {
        "type": "number",
        "label": "Stress Level",
        "unit": "/10",
        "min": 1,
        "max": 10,
        "step": 1,
    },

    "bmiCategory": {
        "type": "select",
        "label": "BMI Category",
        "options": [
            "Underweight",
            "Normal",
            "Overweight",
            "Obese",
        ],
    },

    "dailySteps": {
        "type": "number",
        "label": "Daily Steps",
        "unit": "steps/day",
        "min": 1000,
        "max": 20000,
        "step": 100,
    },

    "systolicBP": {
        "type": "number",
        "label": "Systolic BP",
        "unit": "mmHg",
        "min": 90,
        "max": 180,
        "step": 1,
    },

    "diastolicBP": {
        "type": "number",
        "label": "Diastolic BP",
        "unit": "mmHg",
        "min": 60,
        "max": 120,
        "step": 1,
    },

    "heartRate": {
        "type": "number",
        "label": "Resting Heart Rate",
        "unit": "bpm",
        "min": 50,
        "max": 120,
        "step": 1,
    },
}


# ------------------------------------------------------------------
# API fields
# ------------------------------------------------------------------

REQUIRED_FIELDS = [
    "gender",
    "age",
    "occupation",
    "sleepDuration",
    "sleepQuality",
    "physicalActivity",
    "stressLevel",
    "bmiCategory",
    "dailySteps",
    "systolicBP",
    "diastolicBP",
    "heartRate",
]


# ------------------------------------------------------------------
# Numeric validation rules
# ------------------------------------------------------------------

NUMERIC_RULES = {
    field: (
        float(rule["min"]),
        float(rule["max"]),
    )
    for field, rule in VALIDATION_SCHEMA.items()
    if rule.get("type") == "number"
}


# ------------------------------------------------------------------
# Frontend field -> model column mapping
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Extract categories from fitted OneHotEncoder
# ------------------------------------------------------------------

def _trained_categories(column):
    """
    Return categories learned by the fitted OneHotEncoder.

    This keeps backend validation aligned with the actual
    categories used during model training.
    """

    try:
        preprocessor = PIPELINE.named_steps["preprocessor"]

        transformers = dict(
            preprocessor.transformers_
        )

        cat_pipeline = transformers["cat"]

        encoder = cat_pipeline.named_steps["encoder"]

        cat_columns = list(
            transformers["cat"][2]
        )

        index = cat_columns.index(column)

        return {
            str(value)
            for value in encoder.categories_[index]
        }

    except (
        KeyError,
        ValueError,
        AttributeError,
        IndexError,
    ):
        return set()


# ------------------------------------------------------------------
# Valid categorical values
# ------------------------------------------------------------------

VALID_CATEGORIES = {}

for field, rule in VALIDATION_SCHEMA.items():

    if rule.get("type") != "select":
        continue

    column = FIELD_TO_COLUMN.get(field)

    trained_values = (
        _trained_categories(column)
        if column
        else set()
    )

    schema_values = {
        str(value)
        for value in rule.get("options", [])
    }

    # Prefer categories actually learned by the model.
    # Fall back to the validation schema.
    VALID_CATEGORIES[field] = (
        trained_values
        or schema_values
    )


# ------------------------------------------------------------------
# Wellness reference ranges
# ------------------------------------------------------------------

# These ranges are used only for UI wellness scoring.
# They are NOT ML model features.

VITAL_REFERENCE = {

    "sleepDuration": (
        "Sleep Duration",
        "hrs",
        7.0,
        9.0,
        3.0,
        11.0,
        True,
    ),

    "sleepQuality": (
        "Sleep Quality",
        "/10",
        7.0,
        10.0,
        1.0,
        10.0,
        True,
    ),

    "stressLevel": (
        "Stress Level",
        "/10",
        1.0,
        4.0,
        1.0,
        10.0,
        False,
    ),

    "physicalActivity": (
        "Physical Activity",
        "min",
        30.0,
        90.0,
        0.0,
        150.0,
        True,
    ),

    "dailySteps": (
        "Daily Steps",
        "steps",
        7000.0,
        12000.0,
        500.0,
        20000.0,
        True,
    ),

    "heartRate": (
        "Resting Heart Rate",
        "bpm",
        60.0,
        80.0,
        50.0,
        120.0,
        False,
    ),

    "systolicBP": (
        "Systolic BP",
        "mmHg",
        90.0,
        120.0,
        90.0,
        180.0,
        False,
    ),

    "diastolicBP": (
        "Diastolic BP",
        "mmHg",
        60.0,
        80.0,
        60.0,
        120.0,
        False,
    ),
}


# ------------------------------------------------------------------
# Model feature -> frontend field mapping
# ------------------------------------------------------------------

FEATURE_KEY_TO_COLUMN = {
    "age": "Age",
    "sleepDuration": "Sleep Duration",
    "sleepQuality": "Quality of Sleep",
    "physicalActivity": "Physical Activity Level",
    "stressLevel": "Stress Level",
    "heartRate": "Heart Rate",
    "dailySteps": "Daily Steps",
    "systolicBP": "Systolic BP",
    "diastolicBP": "Diastolic BP",
}


# ------------------------------------------------------------------
# Validation endpoint
# ------------------------------------------------------------------

@app.get("/validation-rules")
def validation_rules():
    """
    Return frontend validation rules.
    """

    return jsonify(VALIDATION_SCHEMA)


# ------------------------------------------------------------------
# Numeric parser and validator
# ------------------------------------------------------------------

def _parse_number(field, value):
    """
    Convert input value to float and validate its range.
    """

    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be numeric."
        )

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            f"{field} must be numeric."
        )

    if not math.isfinite(number):
        raise ValueError(
            f"{field} must be finite."
        )

    if field not in NUMERIC_RULES:
        raise ValueError(
            f"No numeric validation rule exists for {field}."
        )

    low, high = NUMERIC_RULES[field]

    if not low <= number <= high:
        raise ValueError(
            f"{field} must be between "
            f"{low:g} and {high:g}."
        )

    return number


# ------------------------------------------------------------------
# Input validation and feature preparation
# ------------------------------------------------------------------

def validate_and_prepare_input(form_data):
    """
    Validate API input and apply the exact feature engineering
    used during model training.

    IMPORTANT:
    This calls engineer_features() from model.py.

    The current model.py must therefore contain only:
        - Pulse Pressure
        - Sleep Quality Index

    No Stress-Sleep Interaction is created here.
    """

    if not isinstance(form_data, dict):
        raise ValueError(
            "Request body must be a JSON object."
        )

    # --------------------------------------------------------------
    # Required fields
    # --------------------------------------------------------------

    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in form_data
    ]

    if missing:
        raise ValueError(
            "Missing field(s): "
            + ", ".join(missing)
        )

    # --------------------------------------------------------------
    # Reject unexpected fields
    # --------------------------------------------------------------

    unexpected = sorted(
        set(form_data)
        - set(REQUIRED_FIELDS)
    )

    if unexpected:
        raise ValueError(
            "Unexpected field(s): "
            + ", ".join(unexpected)
        )

    # --------------------------------------------------------------
    # Categorical fields
    # --------------------------------------------------------------

    categorical = {}

    for field in (
        "gender",
        "occupation",
        "bmiCategory",
    ):

        value = form_data[field]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"{field} must be a non-empty string."
            )

        value = value.strip()

        allowed = VALID_CATEGORIES.get(
            field,
            set(),
        )

        if allowed and value not in allowed:
            raise ValueError(
                f"Invalid {field}: {value}"
            )

        categorical[field] = value

    # --------------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------------

    numeric = {
        field: _parse_number(
            field,
            form_data[field],
        )
        for field in NUMERIC_RULES
    }

    # --------------------------------------------------------------
    # Build raw model row
    # --------------------------------------------------------------

    raw_row = {
        "Gender": categorical["gender"],
        "Age": numeric["age"],
        "Occupation": categorical["occupation"],
        "Sleep Duration": numeric["sleepDuration"],
        "Quality of Sleep": numeric["sleepQuality"],
        "Physical Activity Level": numeric[
            "physicalActivity"
        ],
        "Stress Level": numeric[
            "stressLevel"
        ],
        "BMI Category": categorical[
            "bmiCategory"
        ],
        "Daily Steps": numeric["dailySteps"],
        "Systolic BP": numeric[
            "systolicBP"
        ],
        "Diastolic BP": numeric[
            "diastolicBP"
        ],
        "Heart Rate": numeric[
            "heartRate"
        ],
    }

    input_df = pd.DataFrame(
        [raw_row]
    )

    # --------------------------------------------------------------
    # Apply SAME feature engineering as model.py
    # --------------------------------------------------------------

    input_df = engineer_features(
        input_df
    )

    # --------------------------------------------------------------
    # Verify final feature contract
    # --------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_ORDER
        if column not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Model feature(s) missing after engineering: "
            + ", ".join(missing_features)
        )

    # Ensure the app does not silently send an obsolete
    # Stress-Sleep Interaction feature.
    if "Stress-Sleep Interaction" in input_df.columns:
        input_df = input_df.drop(
            columns=["Stress-Sleep Interaction"]
        )

    # Return exact feature order expected by saved pipeline.
    prepared_df = input_df[
        FEATURE_ORDER
    ]

    clean_data = {
        **categorical,
        **numeric,
    }

    return prepared_df, clean_data


# ------------------------------------------------------------------
# Wellness score
# ------------------------------------------------------------------

def score_vital(key, value):
    """
    Convert raw vital reading to a 0-100 wellness score.

    This score is for UI presentation only.
    It is NOT part of the ML prediction.
    """

    (
        label,
        unit,
        opt_low,
        opt_high,
        hard_low,
        hard_high,
        higher_is_better,
    ) = VITAL_REFERENCE[key]

    value = float(value)

    if opt_low <= value <= opt_high:

        score = 100.0

    elif value < opt_low:

        span = max(
            opt_low - hard_low,
            1e-6,
        )

        score = max(
            0.0,
            100.0
            * (
                1
                - (
                    (opt_low - value)
                    / span
                )
            ),
        )

    else:

        span = max(
            hard_high - opt_high,
            1e-6,
        )

        score = max(
            0.0,
            100.0
            * (
                1
                - (
                    (value - opt_high)
                    / span
                )
            ),
        )

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
        "optimalRange": [
            opt_low,
            opt_high,
        ],
    }


# ------------------------------------------------------------------
# Build vitals response
# ------------------------------------------------------------------

def build_vitals(form_data):
    return [
        score_vital(
            key,
            form_data[key],
        )
        for key in VITAL_REFERENCE
    ]


# ------------------------------------------------------------------
# Profile comparison
# ------------------------------------------------------------------

def build_profile_comparison(
    form_data,
    predicted_label,
):
    """
    Compare user measurements against:

        1. No Sleep Disorder average
        2. Predicted-class average
    """

    healthy_profile = CLASS_PROFILES.get(
        "No Sleep Disorder",
        {},
    )

    predicted_profile = CLASS_PROFILES.get(
        predicted_label,
        {},
    )

    comparison = []

    for key, column in FEATURE_KEY_TO_COLUMN.items():

        if column not in healthy_profile:
            continue

        comparison.append(
            {
                "key": key,
                "label": column,
                "yourValue": round(
                    float(form_data[key]),
                    1,
                ),
                "healthyAverage": round(
                    float(
                        healthy_profile.get(
                            column,
                            0,
                        )
                    ),
                    1,
                ),
                "predictedClassAverage": round(
                    float(
                        predicted_profile.get(
                            column,
                            0,
                        )
                    ),
                    1,
                ),
            }
        )

    return comparison


# ------------------------------------------------------------------
# Risk factors
# ------------------------------------------------------------------

def build_factors(
    form_data,
    predicted_label,
):
    """
    Build human-readable factors for the UI.

    These are explanatory UI rules.
    They are NOT XGBoost feature importance values.
    """

    factors = []

    if float(
        form_data["stressLevel"]
    ) >= 7:

        factors.append(
            "Elevated stress level "
            f"({form_data['stressLevel']}/10)"
        )

    if float(
        form_data["sleepQuality"]
    ) <= 5:

        factors.append(
            "Below-average self-rated "
            "sleep quality "
            f"({form_data['sleepQuality']}/10)"
        )

    if float(
        form_data["sleepDuration"]
    ) < 6.5:

        factors.append(
            "Shorter sleep duration "
            f"{float(form_data['sleepDuration']):.1f} hrs"
        )

    if (
        form_data["bmiCategory"]
        != "Normal"
    ):

        factors.append(
            f"{form_data['bmiCategory']} "
            "BMI category"
        )

    if (
        float(form_data["systolicBP"]) > 130
        or float(form_data["diastolicBP"]) > 85
    ):

        factors.append(
            "Blood pressure above "
            "typical resting range"
        )

    if float(
        form_data["heartRate"]
    ) > 80:

        factors.append(
            "Resting heart rate above "
            "typical range "
            f"({form_data['heartRate']} bpm)"
        )

    if float(
        form_data["physicalActivity"]
    ) < 30:

        factors.append(
            "Lower daily physical activity "
            f"({form_data['physicalActivity']} min)"
        )

    if float(
        form_data["age"]
    ) > 45:

        factors.append(
            "Age above 45, a mild "
            "Sleep Apnea risk factor"
        )

    if not factors:

        factors.append(
            "No strong risk indicators found — "
            "your metrics look well balanced."
        )

    return factors


# ------------------------------------------------------------------
# Home page
# ------------------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok"
        }
    )


# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"],
)
def predict():

    form_data = request.get_json(
        silent=True
    )

    try:

        input_df, clean_data = (
            validate_and_prepare_input(
                form_data
            )
        )

    except ValueError as exc:

        return jsonify(
            {
                "error": (
                    f"Invalid input: {exc}"
                )
            }
        ), 400

    try:

        # ----------------------------------------------------------
        # Model prediction
        # ----------------------------------------------------------

        probabilities = (
            PIPELINE
            .predict_proba(input_df)[0]
        )

        # Get class indices from fitted model.
        class_indices = (
            PIPELINE
            .named_steps["model"]
            .classes_
        )

        class_names = [
            TARGET_MAPPING_INV[int(index)]
            for index in class_indices
        ]

        probability_by_class = {
            name: round(
                float(probability) * 100,
                1,
            )
            for name, probability in zip(
                class_names,
                probabilities,
            )
        }

        predicted_label = max(
            probability_by_class,
            key=probability_by_class.get,
        )

        # ----------------------------------------------------------
        # Risk score
        # ----------------------------------------------------------

        healthy_probability = (
            probability_by_class.get(
                "No Sleep Disorder",
                0.0,
            )
        )

        risk_score = round(
            100.0
            - healthy_probability,
            1,
        )

        # ----------------------------------------------------------
        # API response
        # ----------------------------------------------------------

        response = {
            "probabilities": (
                probability_by_class
            ),

            "predictedClass": (
                predicted_label
            ),

            "riskScore": risk_score,

            "vitals": build_vitals(
                clean_data
            ),

            "profileComparison": (
                build_profile_comparison(
                    clean_data,
                    predicted_label,
                )
            ),

            "factors": build_factors(
                clean_data,
                predicted_label,
            ),

            "disclaimer": (
                "This result is a screening estimate "
                "based on the information you provided, "
                "not a medical diagnosis. Please consult "
                "a healthcare professional for any concerns "
                "about your sleep health."
            ),
        }

        return jsonify(response)

    except Exception:

        # Do not expose internal exception details
        # through the production API.
        app.logger.exception(
            "Prediction failed."
        )

        return jsonify(
            {
                "error": (
                    "Prediction failed. "
                    "Please check the model artifact "
                    "and input configuration."
                )
            }
        ), 500


# ------------------------------------------------------------------
# Local development / Render entry point
# ------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000,
            )
        ),
        debug=False,
    )