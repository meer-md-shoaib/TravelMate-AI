# 🌿 TravelMate AI

An intelligent travel companion for **Kerala and Karnataka** tourism, combining verified destination data with RAG vector search and natural conversational AI.

---

## ✨ Features

- **⚡ Instant Itineraries:** Curated 1-to-5 day schedules across 720+ verified spots in < 0.05s.
- **📚 RAG Semantic Search:** Uses ChromaDB and `all-MiniLM-L6-v2` to retrieve relevant destinations (waterfalls, temples, treks, viewpoints) and generate grounded answers.
- **💬 Conversational AI:** Powered by `Qwen/Qwen2.5-0.5B-Instruct` for natural chat, greetings, packing tips, and local travel advice.
- **🌓 Responsive Modern UI:** Interactive Gradio interface with Homepage hero banner, feature cards, and dark/light mode toggle.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
python app.py
```
Or:
```bash
python "GenStars first.py"
```

The application will launch on `http://127.0.0.1:7860`.

---

## 🛠️ Tech Stack
- **Language Model:** Qwen 2.5 (0.5B Instruct) via Hugging Face Transformers
- **Embeddings & Vector DB:** Sentence-Transformers (`all-MiniLM-L6-v2`) & ChromaDB
- **Dataset:** 720+ places across Kerala & Karnataka (`kagglehub`)
- **UI Framework:** Gradio with customized responsive CSS and Dark/Light mode
