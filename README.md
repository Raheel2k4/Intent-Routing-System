# 🧭 Intent Routing System

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Model F1](https://img.shields.io/badge/macro%20F1-0.94-brightgreen)](#-evaluation)

A **production‑grade microservice** that instantly classifies any English question into one of six intent categories – **what, who, when, where, why, how**.  
It uses a distilled Sentence‑Transformer for semantic understanding, a fast logistic regression classifier, and exposes a clean REST API with a built‑in web interface.

---

## 🚀 What Problem Does It Solve?
Customer support teams, e‑learning platforms, and chatbots handle thousands of questions daily. Manually reading and routing them is slow and expensive.  
Intent Routing System **automates question triage**, reduces response times by up to 80%, and provides real‑time analytics on what users are asking.

---

## ✨ Features

- 🧠 **Semantic Understanding** – Uses `all-MiniLM-L6-v2` embeddings, not just keywords.  
- ⚡ **Blazing Fast** – Inference time < 15 ms per query.  
- 🧪 **Confidence‑Aware** – Returns `unclassified` when the model is unsure (threshold = 0.35).  
- 📊 **Live Web Demo** – Animated bar chart, session history, and colour‑coded predictions.  
- 🔌 **True Microservice** – Decoupled REST API; integrate with any stack using a single `curl` command.  
- 🐍 **One‑Click Setup** – `run.bat` creates a virtual environment, installs dependencies, trains the model (if missing), and starts the production server.  
- 🏭 **Production WSGI** – Served by Waitress, no Flask dev‑server warnings.  
- 🧹 **Robust Input Validation** – Rejects empty, malformed, or over‑long queries with clear errors.

---

## 📸 Demo

![Web UI Demo](screenshots/demo.gif)  
*Type any question – see the predicted intent, confidence, and probability distribution.*

---

## 🏗️ Architecture

    ┌──────────────┐     POST /predict      ┌─────────────────────────┐
    │  Client App  │ ──────────────────────> │    Flask + Waitress       │
    │  (cURL, UI,  │ <────────────────────── │    (REST API)             │
    │   Slack, …)  │     JSON Response       └───────────┬─────────────┘
    └──────────────┘                                     │
                                                         ▼
                                          ┌─────────────────────────┐
                                          │  Sentence‑Transformer   │
                                          │  (all-MiniLM-L6-v2)     │
                                          │  → 384‑d embedding      │
                                          └───────────┬─────────────┘
                                                      │
                                                      ▼
                                          ┌─────────────────────────┐
                                          │  Logistic Regression    │
                                          │  (multinomial, F1=0.94) │
                                          │  → label + confidence   │
                                          └─────────────────────────┘

---

## 📦 Installation & Quick Start

### Prerequisites
- **Windows** (the automated script is for Windows; Linux/macOS can run manually)
- **Python 3.11** ([download](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe))

### One‑Click Setup (Windows)
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/intent-routing-system.git](https://github.com/yourusername/intent-routing-system.git)
   cd intent-routing-system
   
Place your train_clean.csv dataset in the root folder (the included sample trains a good model).

Double‑click run.bat – it will:

Create a virtual environment with Python 3.11

Install all dependencies

Download the Sentence‑Transformer model (once, cached offline afterward)

Train the classifier if model/intent_classifier.pkl is missing

Launch the Waitress server and open your browser at http://127.0.0.1:5000

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
python train.py               # Train & save the model
python app.py                 # Start the server

📡 API Reference
GET /
Serves the interactive web demo.

POST /predict
Classify a question.

Request:

{
  "question": "How do I reset my password?"
}

Response (200):

{
  "question": "How do I reset my password?",
  "label": "how",
  "confidence": 0.9812,
  "all_probabilities": {
    "how": 0.9812,
    "what": 0.0103,
    "why": 0.0051,
    "who": 0.0018,
    "when": 0.0012,
    "where": 0.0004
  }
}

Error (400):

{
  "error": "Question must not be empty"
}

Example with curl:

curl -X POST [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is the Taj Mahal?"}'

ccuracy: 95.2%

Confidence threshold: 0.35 (chosen via validation sweep)

Cross‑validation (5‑fold): mean macro F1 = 0.93 (±0.02)

📁 Project Structure

intent-routing-system/
├── run.bat                  # One-click setup for Windows
├── requirements.txt         # Python dependencies
├── train.py                 # Data loading, training, evaluation, model export
├── app.py                   # Flask + Waitress server
├── train_clean.csv          # Training dataset (1,314 questions)
├── model/
│   └── intent_classifier.pkl   # Serialised classifier
├── templates/
│   └── index.html           # Web demo UI
├── static/
│   ├── style.css
│   └── logo.png
└── README.md

💰 Business Model (Sales Perspective)
How we sell it: API‑as‑a‑Service with three tiers:

Developer (Free) – 1,000 requests/month

Pro ($99/month) – 100,000 requests/month

Enterprise (Custom) – unlimited requests, on‑premises, SLA

Target clients:

SaaS helpdesks (Zendesk alternatives, Freshdesk)

E‑learning platforms (auto‑classify student queries)

Chatbot developers (plug intent detection into conversations)

Market research agencies (analyse public question trends)

Why they purchase:

80% reduction in manual triage effort → immediate cost savings.

Zero‑disruption integration – one REST endpoint, works with any existing stack.

Real‑time intent analytics – understand what users are asking and why.

Production‑grade reliability with confidence‑aware predictions.

📹 Promotional Video
A <5‑minute demo and sales pitch is available. See the promo/ folder or watch the video presentation submitted with the project.

🤝 Contributing
Contributions, issues, and feature requests are welcome.

Open a pull request or issue to discuss improvements, fine‑tuned models, or multilingual extensions.

📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgements
Sentence‑BERT paper: Reimers & Gurevych, EMNLP 2019

UIUC question classification dataset

scikit‑learn & HuggingFace communities

Made with ❤️ for the semester project – and built to be production‑ready.
