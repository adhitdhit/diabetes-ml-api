from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

# ============================================
# 1. MONGODB CONNECTION
# ============================================
MONGO_URI = "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority&authSource=admin"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    client.admin.command('ping')  # Test connection
    db = client["Database"]
    collection = db["Database_3"]
    print("✅ Connected to MongoDB successfully")
except Exception as e:
    print(f"❌ MongoDB Error: {e}")
    db = None
    collection = None

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
    print("✅ Pipeline loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")

# ============================================
# 3. ROUTES
# ============================================

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "mongo_connected": db is not None,
        "model_loaded": pipeline is not None
    })

# ✅ CORS Preflight
@app.route('/predict', methods=['OPTIONS'])
@app.route('/history', methods=['OPTIONS'])
def options_handler():
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response, 200

# === PREDICT ENDPOINT (SAVE + PREDICT + UPDATE) ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        data = request.json
        
        # 1. Simpan data ke DB (status: processing)
        now_wib = datetime.utcnow() + timedelta(hours=7)

        doc = {
            "patientName": data.get('patientName'),
            "patientGender": data.get('patientGender'),
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
        
        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        # 2. Persiapan fitur (null → np.nan)
        raw_features = [
            data.get('Pregnancies'),
            data.get('Glucose'),
            data.get('BloodPressure'),
            data.get('SkinThickness'),
            data.get('Insulin'),
            data.get('BMI'),
            data.get('DiabetesPedigreeFunction'),
            data.get('Age')
        ]
        
        features = np.array([[
            np.nan if val is None else val 
            for val in raw_features
        ]], dtype=float)
        
        # 3. Predict pakai PIPELINE
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        # 4. Risk Level & Rekomendasi
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
                "Hindari gaya hidup sedentari, perbanyak gerak aktif."
            ]
        else:
            risk_level = "🚨 SANGAT TINGGI - Segera konsultasi"
            recommendations = [
                "Wajib konsultasi ke dokter spesialis penyakit dalam.",
                "Lakukan pemeriksaan laboratorium lengkap segera.",
                "Mulai terapi medis sesuai anjuran dokter.",
                "Pantau gula darah harian & catat pola makan."
            ]

        # 5. Update DB dengan hasil
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
        
        # 6. Return ke Frontend
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
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# === HISTORY ENDPOINT ===
@app.route('/history', methods=['GET'])
def get_history():
    try:
        if not collection:
            return jsonify({
                "success": False, 
                "error": "Database not connected",
                "data": []
            }), 500
        
        # Ambil 50 data terbaru
        cursor = collection.find().sort("createdAt", -1).limit(50)
        
        history_data = []
        for doc in cursor:
            # Convert ObjectId ke string
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            # Convert datetime ke ISO string
            if 'createdAt' in doc and hasattr(doc['createdAt'], 'isoformat'):
                doc['createdAt'] = doc['createdAt'].isoformat()
            if 'processedAt' in doc and hasattr(doc['processedAt'], 'isoformat'):
                doc['processedAt'] = doc['processedAt'].isoformat()
            history_data.append(doc)
        
        return jsonify({
            "success": True,
            "count": len(history_data),
            "data": history_data
        })
        
    except Exception as e:
        print(f"❌ History Error: {str(e)}")
        return jsonify({
            "success": False, 
            "error": str(e),
            "data": []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)