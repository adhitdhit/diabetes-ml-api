from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import pickle  
import numpy as np 
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

# --- KONFIGURASI DATABASE ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority")
client = MongoClient(MONGO_URI)
db = client["Database"]
collection = db["Database_3"]

# --- LOAD PIPELINE (IMPUTER + SCALER + MODEL) ---
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'diabetes_model.pkl')
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        # ✅ PERUBAHAN 1: Load pipeline lengkap, bukan model/scaler terpisah
        pipeline = package['pipeline']
    print("✅ Pipeline loaded successfully! (Auto handle NaN + Scaling + Prediction)")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    pipeline = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Simpan data ke DB (status: processing) - DATA MENTAH
        now_wib = datetime.utcnow() + timedelta(hours=7)

        doc = {
            "patientName": data.get('patientName'),
            "patientGender": data.get('patientGender'),
            "Pregnancies": data.get('Pregnancies'),      # ✅ null tetap null
            "Glucose": data.get('Glucose'),              # ✅ null tetap null
            "BloodPressure": data.get('BloodPressure'),  # ✅ null tetap null
            "SkinThickness": data.get('SkinThickness'),  # ✅ null tetap null
            "Insulin": data.get('Insulin'),              # ✅ null tetap null
            "BMI": data.get('BMI'),                      # ✅ null tetap null
            "DiabetesPedigreeFunction": data.get('DiabetesPedigreeFunction'), # ✅ null tetap null
            "Age": data.get('Age'),                      # ✅ null tetap null
            "Prediction_Result": None,
            "Risk_Score": None,
            "Risk_Level": None,
            "Recommendations": None,
            "status": "processing",
            "createdAt": now_wib,
            "processedAt": None
        }
        
        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        # 2. ✅ PERSIAPAN FITUR (Null → np.nan)
        # Urutan HARUS sama persis dengan kolom saat training di Colab
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
        
        # Konversi None → np.nan agar Pipeline SimpleImputer bisa bekerja
        features = np.array([[
            np.nan if val is None else val 
            for val in raw_features
        ]], dtype=float)
        
        print(f"🔍 Input features (with NaN): {features}")
        
        # 3. ✅ PREDIKSI LANGSUNG MENGGUNAKAN PIPELINE
        if pipeline is None:
            raise ValueError("Pipeline belum dimuat. Cek file diabetes_model.pkl")
            
        # Pipeline secara otomatis melakukan:
        # 1. SimpleImputer: Ganti np.nan dengan mean (sesuai data training)
        # 2. StandardScaler: Scaling fitur
        # 3. RandomForest: Prediksi kelas & probabilitas
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        print(f"✅ Prediction done: Class={prediction_val}, Prob={probability:.4f}, Score={risk_score}%")
        
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
    
# ... (route lain seperti /prediction/<id>, /history, dll tetap sama) ...

if __name__ == '__main__':
    app.run(debug=True)