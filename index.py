import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import numpy as np
import zipfile
from datetime import datetime
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PACKAGE = BASE_DIR / "diabetes_model_package.zip"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)

# Global variables for model and scaler
model = None
scaler = None

def load_components():
    """Load the model and scaler from the package"""
    global model, scaler
    
    try:
        # Unzip the model package
        with zipfile.ZipFile(MODEL_PACKAGE, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        
        # Load components
        model = load_model(MODEL_DIR / "diabetes_model.keras")
        scaler = joblib.load(MODEL_DIR / "scaler.save")
        
        print("✓ Medical model loaded successfully")
        return True
        
    except Exception as e:
        print(f"× Error loading components: {str(e)}", file=sys.stderr)
        model = None
        scaler = None
        return False

# Load components when starting
load_components()

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "service": "Clinical Diabetes Prediction API",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Clinical prediction endpoint for healthcare providers"""
    if model is None or scaler is None:
        return jsonify({
            "error": "Diagnostic model not available",
            "status": "error",
            "severity": "critical"
        }), 503

    try:
        data = request.get_json()
        
        # Validate input with medical ranges
        required = ['age', 'bloodSugar', 'systolicBP', 'diastolicBP', 'patientId']
        if not all(field in data for field in required):
            return jsonify({
                "error": "Missing required clinical data",
                "missing_fields": [f for f in required if f not in data],
                "status": "error",
                "severity": "high"
            }), 400

        try:
            # Convert and validate with clinical thresholds
            age = float(data['age'])
            blood_sugar = float(data['bloodSugar'])
            systolic = float(data['systolicBP'])
            diastolic = float(data['diastolicBP'])
            
            if not (0 < age <= 120):
                raise ValueError("Age must be 0-120 years")
            if not (20 <= blood_sugar <= 1000):
                raise ValueError("Blood sugar must be 20-1000 mg/dL")
            if not (50 <= systolic <= 250):
                raise ValueError("Systolic BP must be 50-250 mmHg")
            if not (30 <= diastolic <= 150):
                raise ValueError("Diastolic BP must be 30-150 mmHg")
                
        except ValueError as ve:
            return jsonify({
                "error": "Invalid clinical values",
                "details": str(ve),
                "status": "error",
                "severity": "medium"
            }), 400

        # Prepare features and predict
        features = np.array([[age, blood_sugar, systolic, diastolic]])
        scaled_features = scaler.transform(features)
        probability = float(model.predict(scaled_features)[0][0])
        is_diabetic = probability > 0.5
        
        # Generate clinical report
        risk_level = get_risk_level(probability)
        report = generate_clinical_report(
            patient_id=data['patientId'],
            age=age,
            blood_sugar=blood_sugar,
            systolic=systolic,
            diastolic=diastolic,
            probability=probability,
            risk_level=risk_level
        )
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({
            "error": "Clinical prediction failed",
            "details": str(e),
            "status": "error",
            "severity": "critical"
        }), 500

def get_risk_level(probability):
    """Medical risk stratification"""
    if probability < 0.3:
        return "Low Risk"
    elif 0.3 <= probability < 0.6:
        return "Moderate Risk"
    elif 0.6 <= probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def generate_clinical_report(patient_id, age, blood_sugar, systolic, diastolic, probability, risk_level):
    """Generate comprehensive clinical report"""
    is_diabetic = probability > 0.5
    confidence = probability * 100 if is_diabetic else (1 - probability) * 100
    
    # Medical interpretation
    if risk_level == "Low Risk":
        interpretation = "Minimal diabetes risk markers detected"
    elif risk_level == "Moderate Risk":
        interpretation = "Early metabolic dysregulation indicators present"
    elif risk_level == "High Risk":
        interpretation = "Strong evidence of prediabetes/diabetes"
    else:
        interpretation = "Urgent diabetes probability with complications risk"
    
    # Doctor-focused recommendations
    recommendations = {
        "diagnostic": get_diagnostic_recommendations(risk_level, blood_sugar),
        "therapeutic": get_therapeutic_recommendations(risk_level, age),
        "monitoring": get_monitoring_plan(risk_level),
        "referrals": get_specialist_referrals(risk_level)
    }
    
    return {
        "patientId": patient_id,
        "assessment": {
            "status": "Diabetic" if is_diabetic else "Non-Diabetic",
            "probability": round(probability, 4),
            "confidence": round(confidence, 2),
            "riskStratification": risk_level,
            "interpretation": interpretation,
            "biomarkers": {
                "age": age,
                "bloodSugar": blood_sugar,
                "bloodPressure": f"{systolic}/{diastolic} mmHg",
                "bmi": "Not provided"  # Can be added to input
            }
        },
        "clinicalActions": recommendations,
        "timestamp": datetime.now().isoformat(),
        "modelVersion": "1.0-clinical"
    }

def get_diagnostic_recommendations(risk_level, blood_sugar):
    """Evidence-based diagnostic recommendations"""
    recommendations = []
    
    if risk_level in ["High Risk", "Very High Risk"]:
        recommendations.extend([
            "Order HbA1c test immediately",
            "Fasting plasma glucose test",
            "Oral glucose tolerance test",
            "Urinalysis for ketones if blood sugar > 240 mg/dL"
        ])
    elif risk_level == "Moderate Risk":
        recommendations.extend([
            "Repeat fasting blood glucose",
            "Consider HbA1c screening",
            "Assess for metabolic syndrome"
        ])
    
    if blood_sugar > 200:
        recommendations.append("Point-of-care glucose confirmation")
    
    return recommendations

def get_therapeutic_recommendations(risk_level, age):
    """Personalized treatment recommendations"""
    recommendations = [
        "Lifestyle modification counseling"
    ]
    
    if risk_level in ["High Risk", "Very High Risk"]:
        recommendations.extend([
            "Consider metformin therapy",
            "Initiate statin if LDL > 100 mg/dL",
            "ACE inhibitor/ARB if hypertensive"
        ])
        
        if age > 50:
            recommendations.append("Aspirin therapy consideration")
    
    if risk_level == "Very High Risk":
        recommendations.extend([
            "Immediate diabetes education referral",
            "Consider GLP-1 RA or SGLT2 inhibitor",
            "Comprehensive foot exam"
        ])
    
    return recommendations

def get_monitoring_plan(risk_level):
    """Monitoring frequency recommendations"""
    if risk_level == "Very High Risk":
        return {
            "glucose": "4x daily monitoring",
            "a1c": "Quarterly",
            "bp": "Weekly",
            "weight": "Monthly",
            "retinal": "Annual exam",
            "renal": "Urine microalbumin now + annual"
        }
    elif risk_level == "High Risk":
        return {
            "glucose": "Daily fasting + occasional postprandial",
            "a1c": "Semi-annual",
            "bp": "Bi-weekly",
            "weight": "Monthly"
        }
    else:
        return {
            "glucose": "Annual screening",
            "a1c": "Consider baseline",
            "bp": "Routine monitoring",
            "weight": "Annual assessment"
        }

def get_specialist_referrals(risk_level):
    """Interdisciplinary care recommendations"""
    referrals = ["Registered dietitian"]
    
    if risk_level in ["High Risk", "Very High Risk"]:
        referrals.extend([
            "Endocrinology consult",
            "Diabetes educator",
            "Ophthalmology screening"
        ])
    
    if risk_level == "Very High Risk":
        referrals.extend([
            "Podiatry evaluation",
            "Cardiology risk assessment"
        ])
    
    return referrals

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    