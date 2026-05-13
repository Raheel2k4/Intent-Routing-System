import os
import threading
import webbrowser
import time
from flask import Flask, request, jsonify, render_template
from waitress import serve
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_PATH = os.path.join("model", "intent_classifier.pkl")
CONFIDENCE_THRESHOLD = 0.35
MAX_QUESTION_LENGTH = 500
HOST = "0.0.0.0"
PORT = 5000

# --- Initialise Flask ---
app = Flask(__name__)

# --- Load models (transformer and classifier) ---
print("Loading sentence transformer model...")
# This loads from the HuggingFace cache (downloaded during train.py), so offline works.
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence transformer loaded.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Classifier model not found at {MODEL_PATH}. "
        "Run train.py first to generate it."
    )
classifier = joblib.load(MODEL_PATH)
print(f"Classifier loaded. Classes: {classifier.classes_}")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Validate input
    data = request.get_json(force=True)
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" field in JSON'}), 400

    question = str(data['question']).strip()
    if not question:
        return jsonify({'error': 'Question must not be empty'}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({'error': f'Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters'}), 400

    # 2. Encode question
    embedding = embedder.encode([question])

    # 3. Predict probabilities
    try:
        proba = classifier.predict_proba(embedding)[0]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    max_prob = np.max(proba)
    predicted_idx = np.argmax(proba)

    # 4. Confidence threshold → unclassified if below threshold
    if max_prob < CONFIDENCE_THRESHOLD:
        label = "unclassified"
        confidence = 0.0
    else:
        label = classifier.classes_[predicted_idx]
        confidence = float(max_prob)

    # 5. Return full probability distribution
    all_probs = {cls: float(prob) for cls, prob in zip(classifier.classes_, proba)}
    return jsonify({
        "question": question,
        "label": label,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs
    })

# --- Start the server ---
if __name__ == '__main__':
    def open_browser():
        """Wait a moment for the server to start, then open the browser."""
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    # Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting Waitress production server on http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)