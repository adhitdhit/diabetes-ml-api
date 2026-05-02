from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

# ============================================
# 1. KONFIGURASI DATABASE
# ============================================
# Gunakan format mongodb+srv untuk kestabilan di Vercel/Serverless
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority")

try:
    client = MongoClient(MONGO_URI)
    db = client["Database"]
    prediction_collection = db["Database_3"]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    db = None
    prediction_collection = None

# ============================================
# 2. LOAD MODEL PIPELINE
# ============================================
try:
    # Path yang benar untuk Vercel: dari api/predict.py ke root/diabetes_model.pkl
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        pipeline = package['pipeline']
    print("✅ Pipeline loaded successfully! (Auto handle NaN + Scaling + Prediction)")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    pipeline = None

# ============================================
# 3. ROUTES
# ============================================
@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes ML API is running! 🎉",
        "status": "ok",
        "endpoints": {
            "predict": "POST /predict",
            "history": "GET /history",
            "health": "GET /"
        }
    })

# ✅ CORS PREFLIGHT HANDLER (Handle /predict & /history)
@app.route('/predict', methods=['OPTIONS'])
@app.route('/history', methods=['OPTIONS'])
def cors_preflight():
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response, 200

# === ROUTE PREDICT (POST) + AUTO SAVE TO DB ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None:
            raise ValueError("Pipeline tidak dimuat. Cek log deployment.")
            
        data = request.json
        print(f"📥 Received prediction request: {data}")
        
        # 1. ✅ PERSIAPAN FITUR (Null → np.nan)
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
        
        # 2. ✅ PREDIKSI LANGSUNG MENGGUNAKAN PIPELINE
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        print(f"✅ Prediction done: Class={prediction_val}, Prob={probability:.4f}, Score={risk_score}%")
        
        # 3. Risk Level & Rekomendasi 
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

        result_payload = {
            "success": True,
            "prediction": prediction_val,
            "probability": probability,
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "status": "completed"
        }

        # ✅ SIMPAN HASIL KE MONGODB
        if db and prediction_collection:
            try:
                db_record = {
                    "patientName": data.get('patientName', 'Tanpa Nama'),
                    "patientGender": data.get('patientGender', 'Tidak Diketahui'),
                    "Pregnancies": data.get('Pregnancies'),
                    "Glucose": data.get('Glucose'),
                    "BloodPressure": data.get('BloodPressure'),
                    "SkinThickness": data.get('SkinThickness'),
                    "Insulin": data.get('Insulin'),
                    "BMI": data.get('BMI'),
                    "DiabetesPedigreeFunction": data.get('DiabetesPedigreeFunction'),
                    "Age": data.get('Age'),
                    "Prediction_Result": prediction_val,
                    "Risk_Score": risk_score,
                    "Risk_Level": risk_level,
                    "Recommendations": recommendations,
                    "Probability": probability,
                    "createdAt": datetime.utcnow()
                }
                prediction_collection.insert_one(db_record)
                print("✅ Saved prediction to Database_3")
            except Exception as db_err:
                print(f"⚠️ DB Save Error: {db_err}")

        return jsonify(result_payload)
        
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# === ROUTE HISTORY (GET) ===
@app.route('/history', methods=['GET'])
def get_history():
    try:
        if not db or not prediction_collection:
            return jsonify({"success": False, "error": "Database not connected"}), 500
            
        # Ambil 50 data terakhir, urutkan berdasarkan waktu terbaru
        cursor = prediction_collection.find().sort("createdAt", -1).limit(50)
        
        history_data = []
        for doc in cursor:
            # Convert ObjectId & datetime agar bisa di-JSON serialize
            doc['_id'] = str(doc['_id'])
            if 'createdAt' in doc and isinstance(doc['createdAt'], datetime):
                doc['createdAt'] = doc['createdAt'].isoformat()
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
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)