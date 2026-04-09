from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
try:
    # Untuk Vercel, path relatif dari root
    model_path = os.path.join(os.path.dirname(__file__), '..', 'diabetes_model.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return jsonify({
        'message': 'Diabetes ML API is running! 🎉',
        'endpoints': {
            'POST /predict': 'Send patient data to get prediction'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        data = request.json
        
        # Extract features
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
        
        # Scaling
        features_scaled = scaler.transform(features)
        
        # Prediction
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        risk_score = round(probability * 100)
        
        # Risk level
        if probability >= 0.75:
            risk_level = "Sangat Tinggi"
        elif probability >= 0.50:
            risk_level = "Tinggi"
        elif probability >= 0.25:
            risk_level = "Sedang"
        else:
            risk_level = "Rendah"
        
        # Recommendations
        recommendations = get_recommendations(probability, prediction)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability, 4),
            'riskScore': risk_score,
            'riskLevel': risk_level,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_recommendations(probability, prediction):
    if prediction == 1 or probability >= 0.75:
        return [
            "🚨 SANGAT TINGGI - Segera konsultasi",
            "⚠️ Segera konsultasi ke dokter untuk pemeriksaan lebih lanjut.",
            "Lakukan tes HbA1c untuk konfirmasi diagnosis diabetes.",
            "Mulai pengaturan pola makan ketat (kurangi gula & karbohidrat)."
        ]
    elif probability >= 0.50:
        return [
            "🔴 TINGGI - Indikasi diabetes",
            "Kurangi konsumsi gula & karbohidrat sederhana.",
            "Tingkatkan aktivitas fisik minimal 30 menit/hari.",
            "Monitor glukosa darah secara berkala."
        ]
    elif probability >= 0.25:
        return [
            "⚠️ SEDANG - Pre-diabetes",
            "Pertahankan pola makan sehat dengan porsi seimbang.",
            "Lakukan aktivitas fisik ringan-sedang secara teratur.",
            "Periksa kesehatan tahunan untuk deteksi dini."
        ]
    else:
        return [
            "✅ RENDAH - Masih aman",
            "Lanjutkan mempertahankan kebiasaan gaya hidup sehat.",
            "Pemeriksaan kesehatan tahunan direkomendasikan.",
            "Tetap aktif secara fisik dan jaga pola makan bergizi."
        ]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)