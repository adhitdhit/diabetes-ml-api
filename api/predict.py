from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- KONFIGURASI DATABASE ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority")
client = MongoClient(MONGO_URI)
db = client["Database"]
collection = db["Database_3"]

# --- LOAD MODEL ---
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'diabetes_model.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
    print("✅ Model loaded & DB connected!")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Simpan data ke DB (status: processing)
        doc = {
            "patientName": data.get('patientName', 'Unknown'),
            "patientGender": data.get('patientGender', 'Unknown'),
            "Pregnancies": data.get('Pregnancies', 0),
            "Glucose": data.get('Glucose'),
            "BloodPressure": data.get('BloodPressure', 0),
            "SkinThickness": data.get('SkinThickness', 0),
            "Insulin": data.get('Insulin', 0),
            "BMI": data.get('BMI', 0),
            "DiabetesPedigreeFunction": data.get('DiabetesPedigreeFunction', 0),
            "Age": data.get('Age'),
            "status": "processing",
            "createdAt": datetime.now()
        }
        
        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        # 2. Prediksi
        features = np.array([[
            data.get('Pregnancies', 0),
            data.get('Glucose'),
            data.get('BloodPressure', 0),
            data.get('SkinThickness', 0),
            data.get('Insulin', 0),
            data.get('BMI', 0),
            data.get('DiabetesPedigreeFunction', 0),
            data.get('Age')
        ]])
        
        features_scaled = scaler.transform(features)
        prediction_val = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        risk_score = round(probability * 100)
        
        if probability >= 0.75: risk_level = "Sangat Tinggi"
        elif probability >= 0.50: risk_level = "Tinggi"
        elif probability >= 0.25: risk_level = "Sedang"
        else: risk_level = "Rendah"
        
        recommendations = [
            "Segera konsultasi ke dokter untuk pemeriksaan lebih lanjut.",
            "Lakukan tes HbA1c untuk konfirmasi diagnosis diabetes.",
            "Mulai pengaturan pola makan ketat (kurangi gula & karbohidrat).",
            "Monitor glukosa darah secara rutin."
        ]

        # 3. Update DB dengan hasil
        collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {
                "Prediction_Result": prediction_val,
                "Risk_Score": risk_score,
                "Risk_Level": risk_level,
                "Recommendations": recommendations,
                "Probability": probability,
                "status": "completed",
                "processedAt": datetime.now()
            }}
        )
        
        # 4. Return dengan savedId ✅
        return jsonify({
            "success": True,
            "savedId": doc_id,  # ← INI YANG KURANG!
            "prediction": prediction_val,
            "probability": probability,
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "status": "completed"
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/prediction/<id>', methods=['GET'])
def get_prediction(id):
    try:
        doc = collection.find_one({"_id": ObjectId(id)})
        if doc:
            doc['_id'] = str(doc['_id'])
            return jsonify({"success": True, "data": doc})
        return jsonify({"success": False, "error": "Not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Diabetes ML API with MongoDB is running! 🎉"})

if __name__ == '__main__':
    app.run(debug=True)