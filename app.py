from flask import Flask, request, jsonify, render_template
import joblib
import json
import os
from datetime import datetime, timedelta

# ==========================================
# APP SETUP
# ==========================================
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ==========================================
# STORAGE (for dashboard)
# ==========================================
DATA_FILE = 'reports_data.json'

def load_reports():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_report(text, is_safe, confidence):
    reports = load_reports()
    reports.append({
        'text': text,
        'safe': is_safe,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })
    # Keep only last 500 reports
    reports = reports[-500:]
    with open(DATA_FILE, 'w') as f:
        json.dump(reports, f, indent=2)

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the text from the form
    report_text = request.form.get('report', '')
    
    # 2. Handle empty input
    if not report_text.strip():
        return render_template(
            'index.html',
            prediction="⚠️ Please enter a report.",
            report_text="",
            confidence=None
        )
    
    # 3. Vectorize and predict
    vectorized_text = vectorizer.transform([report_text])
    prediction = model.predict(vectorized_text)[0]
    
    # 4. Confidence score
    try:
        probability = model.predict_proba(vectorized_text)[0]
        confidence = round(max(probability) * 100, 2)
    except:
        confidence = None
    
    # 5. Convert prediction to readable text
    if str(prediction).lower() in ['0', 'safe', '✅ safe']:
        prediction = "✅ SAFE — No immediate concern."
    else:
        prediction = "🚨 DANGER — Immediate action required!"
    
    # 6. Save to JSON file for dashboard
    is_safe = "safe" in str(prediction).lower()
    save_report(report_text, is_safe, confidence if confidence else 0)
    
    # 7. Return template
    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        report_text=report_text
    )


@app.route('/dashboard')
def dashboard():
    reports = load_reports()
    total_reports = len(reports)
    safe_count = sum(1 for r in reports if r.get('safe'))
    danger_count = total_reports - safe_count
    
    confidences = [r['confidence'] for r in reports if r.get('confidence')]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
    
    # Last 7 days bar chart
    week_labels = []
    week_counts = []
    today = datetime.now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        week_labels.append(day.strftime('%a'))
        count = sum(
            1 for r in reports
            if datetime.fromisoformat(r['timestamp']).date() == day
        )
        week_counts.append(count)
    
    # Last 10 reports confidence trend
    recent = reports[-10:]
    conf_labels = [f"#{i+1}" for i in range(len(recent))]
    conf_values = [r.get('confidence', 0) for r in recent]
    
    return render_template(
        'dashboard.html',
        total_reports=total_reports,
        safe_count=safe_count,
        danger_count=danger_count,
        avg_confidence=avg_confidence,
        week_labels=week_labels,
        week_counts=week_counts,
        conf_labels=conf_labels,
        conf_values=conf_values
    )


# ==========================================
# RUN
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)