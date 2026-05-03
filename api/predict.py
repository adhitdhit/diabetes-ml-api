from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

# MongoDB Connection
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
    print(f"❌ MongoDB Error: {e}")

# Load Model
pipeline = None
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        pipeline = package['pipeline']
    print("✅ Pipeline Loaded!")
except Exception as e:
    print(f"❌ Model Error: {e}")

# Routes
@app.route('/')
def home():
    return jsonify({"status": "ok"})

# ✅ CORS PREFLIGHT - Fix Content-Type header
@app.route('/predict', methods=['OPTIONS'])
@app.route('/history', methods=['OPTIONS'])
@app.route('/prediction/<id>', methods=['OPTIONS'])
def options_handler():
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    return response, 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        data = request.json
        
        # ✅ UTC TIME (Standar Internasional)
        utc_time = datetime.utcnow()
        
        # Simpan data dengan timestamp UTC
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
            "createdAt": utc_time,  # ✅ UTC
            "processedAt": None
        }
        
        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        # Prediksi
        raw_features = [
            data.get('Pregnancies'), data.get('Glucose'), data.get('BloodPressure'),
            data.get('SkinThickness'), data.get('Insulin'), data.get('BMI'),
            data.get('DiabetesPedigreeFunction'), data.get('Age')
        ]
        
        features = np.array([[np.nan if val is None else val for val in raw_features]], dtype=float)
        
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        # Rekomendasi LENGKAP
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

        # ✅ UTC TIME untuk processedAt
        utc_processed = datetime.utcnow()
        
        collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {
                "Prediction_Result": prediction_val,
                "Risk_Score": risk_score,
                "Risk_Level": risk_level,
                "Recommendations": recommendations,
                "Probability": probability,
                "status": "completed",
                "processedAt": utc_processed  # ✅ UTC
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
        return jsonify({"error": str(e)}), 500

@app.route('/prediction/<id>', methods=['GET'])
def get_prediction_by_id(id):
    try:
        if collection is None:
            return jsonify({"error": "Database not connected"}), 500
            
        doc = collection.find_one({"_id": ObjectId(id)})
        if not doc:
            return jsonify({"error": "Not found"}), 404
        
        # BUILD CLEAN DICT
        clean_doc = {
            '_id': str(doc['_id']),
            'patientName': doc.get('patientName', 'Unknown'),
            'patientGender': doc.get('patientGender', 'Unknown'),
            'Pregnancies': doc.get('Pregnancies'),
            'Glucose': doc.get('Glucose'),
            'BloodPressure': doc.get('BloodPressure'),
            'SkinThickness': doc.get('SkinThickness'),
            'Insulin': doc.get('Insulin'),
            'BMI': doc.get('BMI'),
            'DiabetesPedigreeFunction': doc.get('DiabetesPedigreeFunction'),
            'Age': doc.get('Age'),
            'Prediction_Result': doc.get('Prediction_Result'),
            'Risk_Score': doc.get('Risk_Score'),
            'Risk_Level': doc.get('Risk_Level'),
            'Recommendations': doc.get('Recommendations', []),
            'Probability': doc.get('Probability'),
            'status': doc.get('status', 'unknown'),
            'createdAt': doc['createdAt'].isoformat() if 'createdAt' in doc and hasattr(doc['createdAt'], 'isoformat') else None,
            'processedAt': doc['processedAt'].isoformat() if 'processedAt' in doc and hasattr(doc['processedAt'], 'isoformat') else None
        }
        
        return jsonify({"success": True, "data": clean_doc})
    except Exception as e:
        print(f"❌ Error get_prediction_by_id: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        if collection is None:
            return jsonify({"error": "Database not connected"}), 500
            
        cursor = collection.find().sort("createdAt", -1).limit(50)
        
        history_data = []
        for doc in cursor:
            try:
                # BUILD CLEAN DICT
                clean_doc = {
                    '_id': str(doc['_id']),
                    'patientName': doc.get('patientName', 'Unknown'),
                    'patientGender': doc.get('patientGender', 'Unknown'),
                    'Pregnancies': doc.get('Pregnancies'),
                    'Glucose': doc.get('Glucose'),
                    'BloodPressure': doc.get('BloodPressure'),
                    'SkinThickness': doc.get('SkinThickness'),
                    'Insulin': doc.get('Insulin'),
                    'BMI': doc.get('BMI'),
                    'DiabetesPedigreeFunction': doc.get('DiabetesPedigreeFunction'),
                    'Age': doc.get('Age'),
                    'Prediction_Result': doc.get('Prediction_Result'),
                    'Risk_Score': doc.get('Risk_Score'),
                    'Risk_Level': doc.get('Risk_Level'),
                    'Recommendations': doc.get('Recommendations', []),
                    'Probability': doc.get('Probability'),
                    'status': doc.get('status', 'unknown'),
                    'createdAt': doc['createdAt'].isoformat() if 'createdAt' in doc and hasattr(doc['createdAt'], 'isoformat') else None,
                    'processedAt': doc['processedAt'].isoformat() if 'processedAt' in doc and hasattr(doc['processedAt'], 'isoformat') else None
                }
                history_data.append(clean_doc)
            except Exception as doc_error:
                print(f"⚠️ Error processing doc: {doc_error}")
                continue
        
        return jsonify({
            "success": True,
            "count": len(history_data),
            "data": history_data
        })
    except Exception as e:
        print(f"❌ Error get_history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)