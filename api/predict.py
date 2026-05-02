from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle  
import numpy as np 
from datetime import datetime
from dotenv import load_dotenv

# Import pymongo dengan try-except (biar gak crash kalau gagal import)
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("⚠️ pymongo not installed - MongoDB features disabled")

app = Flask(__name__)
CORS(app)
load_dotenv()

# ============================================
# 1. KONFIGURASI DATABASE
# ============================================
db = None
prediction_collection = None

if MONGO_AVAILABLE:
    try:
        # Gunakan format mongodb+srv untuk Vercel
        MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://dbUser:admin@cluster0.toqswqk.mongodb.net/Database?retryWrites=true&w=majority&authSource=admin")
        
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        
        db = client["Database"]
        prediction_collection = db["Database_3"]
        print("✅ Connected to MongoDB successfully")
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {str(e)}")
        db = None
        prediction_collection = None
else:
    print("⚠️ MongoDB disabled - pymongo not available")

# ============================================
# 2. LOAD MODEL PIPELINE
# ============================================
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'diabetes_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
        pipeline = package['pipeline']
    print("✅ Pipeline loaded successfully!")
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
        "mongo_connected": db is not None,
        "model_loaded": pipeline is not None,
        "endpoints": {
            "predict": "POST /predict",
            "history": "GET /history"
        }
    })

# ✅ CORS PREFLIGHT
@app.route('/predict', methods=['OPTIONS'])
@app.route('/history', methods=['OPTIONS'])
def cors_preflight():
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response, 200

# === ROUTE PREDICT ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
            
        data = request.json
        
        # Persiapan fitur
        raw_features = [
            data.get('Pregnancies'), data.get('Glucose'), data.get('BloodPressure'),
            data.get('SkinThickness'), data.get('Insulin'), data.get('BMI'),
            data.get('DiabetesPedigreeFunction'), data.get('Age')
        ]
        
        features = np.array([[np.nan if v is None else v for v in raw_features]], dtype=float)
        
        # Prediksi
        prediction_val = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        risk_score = round(probability * 100)
        
        # Risk level
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

        result = {
            "success": True,
            "prediction": prediction_val,
            "probability": probability,
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "status": "completed"
        }

        # Simpan ke DB (optional - tidak blocking)
        if prediction_collection:
            try:
                db_record = {
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
                    "Prediction_Result": prediction_val,
                    "Risk_Score": risk_score,
                    "Risk_Level": risk_level,
                    "Recommendations": recommendations,
                    "Probability": probability,
                    "createdAt": datetime.utcnow()
                }
                prediction_collection.insert_one(db_record)
            except Exception as db_err:
                print(f"⚠️ DB save failed: {db_err}")
                # Tidak return error, prediction tetap sukses

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# === ROUTE HISTORY ===
@app.route('/history', methods=['GET'])
def get_history():
    try:
        if not prediction_collection:
            return jsonify({
                "success": False, 
                "error": "Database not connected",
                "data": []
            }), 500
        
        # Ambil data
        cursor = prediction_collection.find().sort("createdAt", -1).limit(50)
        
        history_data = []
        for doc in cursor:
            try:
                # Convert ObjectId ke string
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                # Convert datetime ke ISO string
                if 'createdAt' in doc and hasattr(doc['createdAt'], 'isoformat'):
                    doc['createdAt'] = doc['createdAt'].isoformat()
                history_data.append(doc)
            except Exception as doc_err:
                print(f"⚠️ Error processing document: {doc_err}")
                continue
        
        return jsonify({
            "success": True,
            "count": len(history_data),
            "data": history_data
        })
        
    except Exception as e:
        print(f"❌ History route error: {str(e)}")
        return jsonify({
            "success": False, 
            "error": f"Failed to fetch history: {str(e)}",
            "data": []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)