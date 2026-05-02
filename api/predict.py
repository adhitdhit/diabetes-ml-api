from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

# --- LOAD PIPELINE (IMPUTER + SCALER + MODEL) ---
# ✅ TIDAK ADA KONEKSI DATABASE DI SINI LAGI!
try:
    # Path yang benar untuk Vercel: dari api/index.py ke root/diabetes_model.pkl
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        pipeline = package['pipeline']
    print("✅ Pipeline loaded successfully! (Auto handle NaN + Scaling + Prediction)")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    pipeline = None

@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes ML API is running! 🎉",
        "status": "ok",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ VALIDASI: Pipeline harus loaded
        if pipeline is None:
            raise ValueError("Pipeline tidak dimuat. Cek log deployment.")
            
        data = request.json
        print(f"📥 Received prediction request: {data}")
        
        # 1. ✅ PERSIAPAN FITUR (Null → np.nan)
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
        
        # 2. ✅ PREDIKSI LANGSUNG MENGGUNAKAN PIPELINE
        # Pipeline secara otomatis melakukan:
        # 1. SimpleImputer: Ganti np.nan dengan mean (sesuai data training)
        # 2. StandardScaler: Scaling fitur
        # 3. RandomForest: Prediksi kelas & probabilitas
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

        # 4. ✅ RETURN HASIL SAJA (TANPA SIMPAN KE DATABASE!)
        # Database handling sekarang 100% di server.js (Express backend)
        return jsonify({
            "success": True,
            "prediction": prediction_val,
            "probability": probability,
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "status": "completed"
            # ❌ HAPUS: "savedId" — tidak ada lagi karena Flask tidak simpan ke DB
        })
        
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)