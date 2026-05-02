from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

# ============================================
# 1. MONGODB CONNECTION
# ============================================
MONGO_URI = "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority&authSource=admin"

db = None
collection = None

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client["Database"]
    collection = db["Database_3"]
    print("✅ MongoDB Connected!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

# ============================================
# 2. LOAD MODEL PIPELINE
# ============================================
pipeline = None
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        pipeline = package['pipeline']
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
@app.route('/prediction/<id>', methods=['OPTIONS'])
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
        
        # Simpan dalam format ISO WIB agar konsisten
        wib_tz = timezone(timedelta(hours=7))
        now_wib = datetime.now(wib_tz)

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
        
        if collection is None:
            raise Exception("Database not connected")

        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        raw_features = [
            data.get('Pregnancies'), data.get('Glucose'), data.get('BloodPressure'),
            data.get('SkinThickness'), data.get('Insulin'), data.get('BMI'),
            data.get('DiabetesPedigreeFunction'), data.get('Age')
        ]
        
        features = np.array([[np.nan if val is None else val for val in raw_features]], dtype=float)
        
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        # Rekomendasi Lengkap
        if probability < 0.25:
            risk_level = "✅ RENDAH - Masih aman"
            recommendations = [
                "Pertahankan pola makan sehat & bergizi seimbang.",
                "Lakukan aktivitas fisik rutin minimal 150 menit/minggu.",
                "Lakukan pemeriksaan kesehatan tahunan untuk deteksi dini.",
                "Hindari konsumsi gula & lemak jenuh berlebihan."
            ]
        elif probability < 0.50:
            risk_level = "⚠️ SEDANG - Pre-diabetes"
            recommendations = [
                "Kurangi konsumsi karbohidrat sederhana & gula tambahan.",
                "Tingkatkan aktivitas fisik (jalan cepat/sepeda 30 menit/hari).",
                "Monitor kadar gula darah secara berkala.",
                "Konsultasikan dengan ahli gizi untuk pengaturan diet."
            ]
        elif probability < 0.75:
            risk_level = "🔴 TINGGI - Indikasi diabetes"
            recommendations = [
                "Segera konsultasi ke dokter untuk evaluasi klinis.",
                "Lakukan tes HbA1c & profil lipid lengkap.",
                "Terapkan diet rendah glikemik & tinggi serat.",
                "Hindari gaya hidup tidak sehat, perbanyak gerak aktif."
            ]
        else:
            risk_level = "🚨 SANGAT TINGGI - Segera konsultasi"
            recommendations = [
                "Wajib konsultasi ke dokter spesialis penyakit dalam.",
                "Lakukan pemeriksaan laboratorium lengkap segera.",
                "Mulai terapi medis sesuai anjuran dokter.",
                "Pantau gula darah harian & catat pola makan."
            ]

        collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {
                "Prediction_Result": prediction_val,
                "Risk_Score": risk_score,
                "Risk_Level": risk_level,
                "Recommendations": recommendations,
                "Probability": probability,
                "status": "completed",
                "processedAt": datetime.now(wib_tz)
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
        return jsonify({"success": False, "error": str(e)}), 500

# === GET PREDICTION BY ID ===
@app.route('/prediction/<id>', methods=['GET'])
def get_prediction_by_id(id):
    try:
        if collection is None:
            return jsonify({"success": False, "error": "Database not connected"}), 500
        
        doc = collection.find_one({"_id": ObjectId(id)})
        if not doc:
            return jsonify({"success": False, "error": "Prediction not found"}), 404
        
        clean_doc = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                clean_doc[key] = str(value)
            elif isinstance(value, datetime):
                # ✅ FIX: Pakai ISO Format biar Frontend React bisa baca
                if value.tzinfo is None:
                    wib_time = value.replace(tzinfo=timezone.utc) + timedelta(hours=7)
                else:
                    wib_time = value.astimezone(timezone(timedelta(hours=7)))
                clean_doc[key] = wib_time.isoformat()
            else:
                clean_doc[key] = value
        
        return jsonify({"success": True, "data": clean_doc})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# === HISTORY ===
@app.route('/history', methods=['GET'])
def get_history():
    try:
        if collection is None:
            return jsonify({"success": False, "error": "Database not connected"}), 500
            
        cursor = collection.find().sort("createdAt", -1).limit(50)
        history_data = []
        
        for doc in cursor:
            clean_doc = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    clean_doc[key] = str(value)
                elif isinstance(value, datetime):
                    # ✅ FIX: ISO Format agar new Date() di JS valid
                    if value.tzinfo is None:
                        wib_time = value.replace(tzinfo=timezone.utc) + timedelta(hours=7)
                    else:
                        wib_time = value.astimezone(timezone(timedelta(hours=7)))
                    clean_doc[key] = wib_time.isoformat()
                else:
                    clean_doc[key] = value
            history_data.append(clean_doc)
        
        return jsonify({
            "success": True,
            "count": len(history_data),
            "data": history_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)