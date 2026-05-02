from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.json_util import dumps
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app)
load_dotenv()

# ============================================
# 1. MONGODB CONNECTION
# ============================================
# Pastikan URI lu bener. Kalau pakai password khusus, ganti di sini.
MONGO_URI = "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority&authSource=admin"

db = None
collection = None

try:
    # Timeout 5 detik aja, biar kalau gagal gak nunggu lama
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Ping buat test koneksi
    client.admin.command('ping')
    db = client["Database"]
    collection = db["Database_3"]
    print("✅ MongoDB Connected!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")
    db = None
    collection = None

# ============================================
# 2. LOAD MODEL PIPELINE (SESUAI COLAB)
# ============================================
pipeline = None
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        pipeline = package['pipeline']  # ✅ INI PIPELINE LU
    print("✅ Pipeline Loaded!")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")

# ============================================
# 3. ROUTES
# ============================================

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "mongo": "connected" if db else "failed",
        "model": "loaded" if pipeline else "failed"
    })

# ✅ CORS PREFLIGHT
@app.route('/predict', methods=['OPTIONS'])
@app.route('/history', methods=['OPTIONS'])
def options_handler():
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response, 200

# === PREDICT ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
            
        data = request.json
        now_wib = datetime.utcnow() + timedelta(hours=7)

        # 1. Simpan data mentah dulu (Status: Processing)
        doc = {
            "patientName": data.get('patientName', 'Anonim'),
            "patientGender": data.get('patientGender', '-'),
            "Pregnancies": data.get('Pregnancies'),
            "Glucose": data.get('Glucose'),
            "BloodPressure": data.get('BloodPressure'),
            "SkinThickness": data.get('SkinThickness'),
            "Insulin": data.get('Insulin'),
            "BMI": data.get('BMI'),
            "DiabetesPedigreeFunction": data.get('DiabetesPedigreeFunction'),
            "Age": data.get('Age'),
            "Prediction_Result": None,
            "Risk_Score": None,
            "Risk_Level": None,
            "Recommendations": None,
            "Probability": None,
            "status": "processing",
            "createdAt": now_wib,
            "processedAt": None
        }
        
        # Cek koneksi DB sebelum insert
        if collection is None:
            raise Exception("Database not connected")

        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        # 2. Persiapan Fitur (Null -> NaN)
        raw_features = [
            data.get('Pregnancies'), data.get('Glucose'), data.get('BloodPressure'),
            data.get('SkinThickness'), data.get('Insulin'), data.get('BMI'),
            data.get('DiabetesPedigreeFunction'), data.get('Age')
        ]
        
        features = np.array([[np.nan if val is None else val for val in raw_features]], dtype=float)
        
        # 3. Prediksi PAKAI PIPELINE
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        # Logic Risk Level (Sama kayak dulu)
        if probability < 0.25:
            risk_level = "✅ RENDAH - Masih aman"
            recommendations = ["Pertahankan pola hidup sehat.", "Check-up rutin."]
        elif probability < 0.50:
            risk_level = "⚠️ SEDANG - Pre-diabetes"
            recommendations = ["Kurangi gula.", "Olahraga rutin."]
        elif probability < 0.75:
            risk_level = "🔴 TINGGI - Indikasi diabetes"
            recommendations = ["Segera konsultasi dokter.", "Tes HbA1c."]
        else:
            risk_level = "🚨 SANGAT TINGGI"
            recommendations = ["Wajib ke dokter.", "Terapi medis."]

        # 4. Update DB dengan hasil
        collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {
                "Prediction_Result": prediction_val,
                "Risk_Score": risk_score,
                "Risk_Level": risk_level,
                "Recommendations": recommendations,
                "Probability": probability,
                "status": "completed",
                "processedAt": datetime.utcnow() + timedelta(hours=7)
            }}
        )
        
        return jsonify({
            "success": True,
            "savedId": doc_id,
            "prediction": prediction_val,
            "probability": probability,
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "status": "completed"
        })
        
    except Exception as e:
        # ❌ INI PENTING: Kasi tau error detailnya ke frontend
        return jsonify({"success": False, "error": str(e)}), 500

# === HISTORY (YANG SERING ERROR) ===
@app.route('/history', methods=['GET'])
def get_history():
    try:
        if collection is None:
            return jsonify({"success": False, "error": "Database not connected"}), 500
            
        # Ambil data
        cursor = collection.find().sort("createdAt", -1).limit(50)
        
        history_data = []
        for doc in cursor:
            # ✅ KONVERSI MANUAL BIAR GAK CRASH JSON
            clean_doc = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    clean_doc[key] = str(value)
                elif isinstance(value, datetime):
                    clean_doc[key] = value.isoformat()
                else:
                    clean_doc[key] = value
            history_data.append(clean_doc)
        
        return jsonify({
            "success": True,
            "count": len(history_data),
            "data": history_data
        })
        
    except Exception as e:
        # ❌ KASI TAU ERROR DETAILNYA
        return jsonify({
            "success": False, 
            "error": "HISTORY ERROR: " + str(e),
            "data": []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)