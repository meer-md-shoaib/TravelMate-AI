# -*- coding: utf-8 -*-
"""
TravelMate AI: Smart Tourism Concierge for Kerala & Karnataka
-------------------------------------------------------------
- Full-featured Homepage + Chatbot below layout
- Native Dark / Light mode toggle with responsive glassmorphism aesthetic
- High-Speed Dataset Engine (<0.05s) + RAG Vector Search (ChromaDB + all-MiniLM-L6-v2)
- Conversational AI Engine: Qwen 2.5 (0.5B Instruct) for natural chat, greetings & travel advice
"""

import os
import sys
import re
from threading import Thread

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import numpy as np
import pandas as pd
import kagglehub
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import chromadb

# ==============================================================================
# 1. CORE AI CONFIGURATION & MODEL PARAMETERS
# ==============================================================================
TASK = "text-generation"
AI_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

# ------------------------------------------------------------------------------
# GENERATION DECODING PARAMETERS (Modify directly in code to experiment)
# ------------------------------------------------------------------------------
TEMPERATURE = 0.7         # Lower = deterministic, Higher = creative
TOP_K = 50                # Sample from top-K probability tokens
TOP_P = 0.9               # Nucleus sampling threshold
MAX_NEW_TOKENS = 180      # Maximum generated token length
DO_SAMPLE = True          # Stochastic sampling

SYSTEM_PROMPT = (
    "You are 'TravelMate AI', a warm, friendly, and knowledgeable travel companion for Kerala and Karnataka. "
    "Respond naturally, politely, and concisely to user greetings (Hi, Hello), general travel queries, "
    "packing, local culture, and season advice. Keep replies conversational, encouraging, and within 2-3 brief paragraphs."
)

# ==============================================================================
# 2. DATASET INGESTION & PREPARATION
# ==============================================================================
print("Loading Kerala & Karnataka Tourism Dataset...")
try:
    path = kagglehub.dataset_download("adilpasharazvi1/kerala-and-karnataka-tourism-places-dataset")
    print("Dataset directory:", path)
    csv_file = os.path.join(path, "Kerala_and_karnataka_Districts_tourism.csv")
    df = pd.read_csv(csv_file)
except Exception:
    cached_path = r"C:\Users\meerm\.cache\kagglehub\datasets\adilpasharazvi1\kerala-and-karnataka-tourism-places-dataset\versions\1\Kerala_and_karnataka_Districts_tourism.csv"
    df = pd.read_csv(cached_path)

df["District_Clean"] = df["District"].astype(str).str.strip().str.title()
districts_list = ["All Districts"] + sorted(df["District_Clean"].unique().tolist())
print(f"Dataset ready: {len(df)} authentic places across {len(districts_list)-1} districts.")

# ==============================================================================
# 3. TOKENIZATION & EMBEDDING SETUP
# ==============================================================================
print(f"Loading embedding model ({EMBEDDING_MODEL_ID})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

print(f"Loading conversational model ({AI_MODEL_ID})...")
device = "cuda" if torch.cuda.is_available() else "cpu"
ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_ID)
ai_model = AutoModelForCausalLM.from_pretrained(
    AI_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
if not torch.cuda.is_available():
    ai_model = ai_model.to(device)
ai_model.eval()
print(f"AI Models loaded successfully on {device.upper()}!")

# ==============================================================================
# 4. RAG PIPELINE (RECURSIVE CHUNKING & VECTOR DATABASE)
# ==============================================================================
print("Building RAG Vector Knowledge Base...")

def recursive_chunk_text(text, chunk_size=280):
    """Recursive chunking adhering to natural text boundaries"""
    if len(text) <= chunk_size:
        return [text]
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    
    def split_rec(segment, idx=0):
        if len(segment) <= chunk_size or idx >= len(separators):
            chunks.append(segment.strip())
            return
        sep = separators[idx]
        parts = segment.split(sep)
        curr = ""
        for p in parts:
            cand = curr + (sep if curr else "") + p
            if len(cand) <= chunk_size:
                curr = cand
            else:
                if curr:
                    chunks.append(curr.strip())
                if len(p) > chunk_size:
                    split_rec(p, idx + 1)
                    curr = ""
                else:
                    curr = p
        if curr:
            chunks.append(curr.strip())

    split_rec(text)
    return [c for c in chunks if c]

all_chunks = []
chunk_metadata = []

for idx, row in df.iterrows():
    name = str(row.get("Name of the Place", "")).strip()
    dist = str(row.get("District_Clean", "")).strip()
    famous = str(row.get("Famous_For", "")).strip()
    acts = str(row.get("Activities", "")).strip()
    
    doc = (
        f"Place: {name} | District: {dist}\n"
        f"Famous For: {famous}\n"
        f"Activities: {acts}"
    )
    for sub in recursive_chunk_text(doc):
        all_chunks.append(sub)
        chunk_metadata.append({"district": dist, "name": name})

chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("tourism_rag")
except Exception:
    pass

rag_collection = chroma_client.create_collection(
    name="tourism_rag",
    metadata={"hnsw:space": "cosine"}
)

batch_size = 256
all_embeddings = []
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i + batch_size]
    embs = embedding_model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    all_embeddings.extend(embs.tolist())

rag_collection.add(
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    documents=all_chunks,
    embeddings=all_embeddings,
    metadatas=chunk_metadata
)
print(f"RAG Knowledge Base indexed: {len(all_chunks)} chunks ready.")

def retrieve_rag_context(query, district=None, top_k=4):
    q_emb = embedding_model.encode([query], normalize_embeddings=True).tolist()
    where_filter = {"district": district} if (district and district != "All Districts") else None
    try:
        res = rag_collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where=where_filter
        )
        return res["documents"][0] if res and "documents" in res else []
    except Exception:
        return []

# ==============================================================================
# 5. FAST DATASET ITINERARY GENERATOR (<0.05s)
# ==============================================================================
def generate_fast_itinerary(district_name, days=3, travel_style="🌿 Nature & Eco-Trails"):
    if district_name and district_name != "All Districts":
        places = df[df["District_Clean"].str.lower() == district_name.strip().lower()].copy()
        if len(places) == 0:
            places = df[df["District_Clean"].str.contains(district_name.strip(), case=False, na=False)].copy()
    else:
        places = df[df["District_Clean"].isin(["Munnar", "Coorg", "Wayanad", "Hampi", "Alappuzha"])].copy()

    if len(places) == 0:
        places = df.copy()

    style_keywords = {
        "🌿 Nature & Eco-Trails": ["nature", "fall", "peak", "garden", "lake", "forest", "hill", "sanctuary", "river", "tea"],
        "🏛️ Heritage & Temples": ["temple", "palace", "fort", "heritage", "monument", "historic", "architecture", "church", "ruins"],
        "🧗 Adventure & Treks": ["trek", "climb", "safari", "raft", "cave", "camp", "paragliding", "adventure"],
        "☕ Relaxed & Cafes": ["beach", "lake", "sunset", "park", "boat", "dam", "scenic", "relax", "promenade", "coffee"],
    }
    keywords = style_keywords.get(travel_style, [])

    def score_place(row):
        text = f"{row['Name of the Place']} {row['Famous_For']} {row['Activities']}".lower()
        return sum(1 for kw in keywords if kw in text)

    places["relevance"] = places.apply(score_place, axis=1)
    places = places.sort_values(by="relevance", ascending=False).reset_index(drop=True)

    target_district = district_name if district_name != "All Districts" else "Kerala & Karnataka"
    out = [
        f"### 🌿 **{days}-Day {travel_style.split(' ')[1]} Itinerary for {target_district}**\n",
        f"*⚡ Instant Verified Itinerary (Curated from {len(places)} local destinations)*\n\n---\n"
    ]

    idx = 0
    for day in range(1, int(days) + 1):
        out.append(f"### 📅 **Day {day} Schedule**\n")
        slots = ["🌅 Morning Trail", "☀️ Afternoon Exploration", "🌇 Sunset & Unwind"]
        for slot in slots:
            if idx < len(places):
                r = places.iloc[idx]
                out.append(
                    f"#### {slot}: **{r['Name of the Place']}** *({r['District_Clean']})*\n"
                    f"- 🌟 **Highlights:** {r['Famous_For']}\n"
                    f"- 🎯 **Activities:** {r['Activities']}\n"
                )
                idx += 1
        out.append("💡 *Local Tip: Start early in the morning to beat the rush and enjoy fresh local breakfast.*\n\n---\n")

    return "\n".join(out)

# ==============================================================================
# 6. ROUTER & STREAMING AI ENGINE
# ==============================================================================
def is_travel_or_rag_query(msg, district):
    msg_lower = msg.lower().strip()
    words = re.findall(r"\b\w+\b", msg_lower)

    # Pure greetings
    greetings = ["hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "howdy", "wassup"]
    if len(words) <= 3 and any(w in greetings for w in words):
        return False

    # Identity and friendly chit-chat
    chitchat = [
        "who are you", "what are you", "what can you do", "help me", "how are you",
        "tell me a joke", "thank you", "thanks", "bye", "goodbye", "who made you"
    ]
    if any(c in msg_lower for c in chitchat):
        return False

    for dist in df["District_Clean"].unique():
        if dist.lower() in msg_lower:
            return True

    travel_keywords = [
        "plan", "itinerary", "trip", "days", "day", "visit", "tour", "guide",
        "waterfall", "temple", "beach", "trek", "hiking", "palace", "fort",
        "scenic", "spots", "places", "food", "resort", "lake", "cave",
        "wildlife", "safari", "botanical", "monument", "kerala", "karnataka",
        "hills", "valley", "viewpoint", "boating", "tea", "coffee"
    ]
    if any(k in msg_lower for k in travel_keywords):
        return True

    if district and district != "All Districts":
        return True

    return False

def stream_llm_response(messages):
    prompt_text = ai_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ai_tokenizer(prompt_text, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(ai_tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
    )
    thread = Thread(target=ai_model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for token in streamer:
        accumulated += token
        yield accumulated

# ==============================================================================
# 7. MODERN RESPONSIVE UI WITH HOMEPAGE + CHATBOT BELOW & DUAL THEMES
# ==============================================================================
custom_css = """
/* Responsive typography & full page layout */
:root {
    --brand-primary: #1b7a43;
    --brand-accent: #e59819;
    --card-radius: 16px;
    --transition-fast: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Light Mode Styling (Default) */
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 16px 20px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* ---------------- HERO HOMEPAGE BANNER ---------------- */
.hero-wrapper {
    position: relative;
    border-radius: var(--card-radius);
    padding: 42px 36px;
    margin-bottom: 24px;
    overflow: hidden;
    background: linear-gradient(135deg, #e4f3e7 0%, #d8ecdd 45%, #f6efe2 100%);
    border: 1px solid rgba(27, 122, 67, 0.15);
    box-shadow: 0 10px 30px rgba(27, 122, 67, 0.06);
}

.dark .hero-wrapper {
    background: linear-gradient(135deg, #13271c 0%, #18221b 45%, #271f16 100%) !important;
    border: 1px solid rgba(74, 222, 128, 0.2) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(27, 122, 67, 0.2);
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1b7a43;
    margin-bottom: 12px;
}
.dark .hero-pill {
    background: rgba(22, 101, 52, 0.25) !important;
    color: #86efac !important;
    border-color: rgba(74, 222, 128, 0.3) !important;
}

.hero-title {
    font-size: 2.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(135deg, #124d29 0%, #1e824c 50%, #b45309 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.dark .hero-title {
    background: linear-gradient(135deg, #4ade80 0%, #86efac 50%, #fcd34d 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.hero-subtext {
    font-size: 1.1rem;
    line-height: 1.55;
    color: #35533f;
    max-width: 720px;
}
.dark .hero-subtext {
    color: #cbd5cb !important;
}

/* ---------------- FEATURE CARDS (HOMEPAGE) ---------------- */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}

.feature-box {
    background: #ffffff;
    border: 1px solid #dce8de;
    border-radius: 14px;
    padding: 18px 20px;
    transition: all var(--transition-fast);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
}
.dark .feature-box {
    background: #17241c !important;
    border-color: #26382b !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}
.feature-box:hover {
    transform: translateY(-3px);
    border-color: #1b7a43;
    box-shadow: 0 8px 20px rgba(27, 122, 67, 0.1);
}
.dark .feature-box:hover {
    border-color: #4ade80 !important;
}

.feature-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}
.feature-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: #1a3824;
    margin-bottom: 4px;
}
.dark .feature-title {
    color: #e5ece5 !important;
}
.feature-desc {
    font-size: 0.88rem;
    color: #556c5c;
    line-height: 1.4;
}
.dark .feature-desc {
    color: #9cb1a0 !important;
}

/* ---------------- SECTION DIVIDER ---------------- */
.section-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-radius: 12px;
    background: #eef6f0;
    border-left: 4px solid #1b7a43;
    margin: 10px 0 18px 0;
}
.dark .section-banner {
    background: #15251b !important;
    border-left-color: #4ade80 !important;
}
.section-banner h3 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #173d24;
}
.dark .section-banner h3 {
    color: #e2ede4 !important;
}

/* ---------------- CHATBOT CARD & SIDEBAR ---------------- */
.chat-wrapper-card {
    background: #ffffff;
    border: 1px solid #dce8de;
    border-radius: var(--card-radius);
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.03);
}
.dark .chat-wrapper-card {
    background: #142018 !important;
    border-color: #243528 !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
}

#send-button {
    background: linear-gradient(135deg, #1b7a43 0%, #155e34 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px rgba(27, 122, 67, 0.25) !important;
    transition: all var(--transition-fast) !important;
}
#send-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(27, 122, 67, 0.35) !important;
}

.chip-btn {
    background: #f1f7f2 !important;
    border: 1px solid #c9decb !important;
    color: #1c4d2d !important;
    border-radius: 10px !important;
    padding: 9px 12px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
}
.dark .chip-btn {
    background: #1c2b20 !important;
    border-color: #2d4233 !important;
    color: #cfe5d3 !important;
}
.chip-btn:hover {
    background: #dbeef0 !important;
    border-color: #1b7a43 !important;
    transform: translateY(-2px);
}
.dark .chip-btn:hover {
    background: #253d2d !important;
    border-color: #4ade80 !important;
}

/* Mode Switch Button Styling */
#theme-toggle-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 6px 14px !important;
}

@media (max-width: 768px) {
    .hero-title { font-size: 2rem !important; }
    .features-grid { grid-template-columns: 1fr; }
}
"""

with gr.Blocks(title="TravelMate AI — Kerala & Karnataka") as demo:

    # ==========================================================================
    # SECTION A: TOP HOMEPAGE HERO & THEME TOGGLE
    # ==========================================================================
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
                """
                <div class="hero-wrapper">
                    <div class="hero-pill">🌿 Kerala & Karnataka Tourism Concierge</div>
                    <div class="hero-title">Discover Your Next Journey</div>
                    <div class="hero-subtext">
                        Explore 720+ authentic destinations across mist-covered hills, historic palaces, pristine beaches, and serene backwaters. 
                        Instant curated day-plans and smart conversational assistance in one place.
                    </div>
                </div>
                """
            )
        with gr.Column(scale=2, min_width=120):
            theme_btn = gr.Button("🌓 Toggle Dark / Light", elem_id="theme-toggle-btn")

    # ==========================================================================
    # SECTION B: HOMEPAGE HIGHLIGHT CARDS
    # ==========================================================================
    gr.Markdown(
        """
        <div class="features-grid">
            <div class="feature-box">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Instant Itineraries</div>
                <div class="feature-desc">Curated 1 to 5-day schedules from 720+ verified spots in under 0.05 seconds.</div>
            </div>
            <div class="feature-box">
                <div class="feature-icon">📚</div>
                <div class="feature-title">RAG Vector Search</div>
                <div class="feature-desc">Dense semantic vector search pinpoints waterfalls, temples, treks & viewpoints.</div>
            </div>
            <div class="feature-box">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Conversational AI</div>
                <div class="feature-desc">Friendly Qwen 2.5 neural chat for greetings, packing tips, seasons, and advice.</div>
            </div>
            <div class="feature-box">
                <div class="feature-icon">☕</div>
                <div class="feature-title">Custom Travel Styles</div>
                <div class="feature-desc">Tailor trips to eco-trails, ancient architecture, thrilling treks, or relaxed cafes.</div>
            </div>
        </div>
        """
    )

    # ==========================================================================
    # SECTION C: CHATBOT & INTERACTIVE CONCIERGE (BELOW HOMEPAGE)
    # ==========================================================================
    gr.Markdown(
        """
        <div class="section-banner">
            <h3>💬 Travel Concierge & Chatbot</h3>
            <span style="font-size: 0.9rem; color: #406b4b; font-weight: 500;">Ask questions below or customize your trip</span>
        </div>
        """
    )

    with gr.Row(elem_classes=["chat-wrapper-card"]):
        # LEFT: Trip Preferences & 1-Click Fast Itineraries
        with gr.Column(scale=4):
            gr.Markdown("#### 🧭 **Trip Preferences**")

            district_dropdown = gr.Dropdown(
                label="Destination District",
                choices=districts_list,
                value="All Districts",
            )

            days_slider = gr.Slider(
                label="Duration (Days)",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
            )

            style_radio = gr.Radio(
                label="Travel Style",
                choices=[
                    "🌿 Nature & Eco-Trails",
                    "🏛️ Heritage & Temples",
                    "🧗 Adventure & Treks",
                    "☕ Relaxed & Cafes",
                ],
                value="🌿 Nature & Eco-Trails",
            )

            gr.Markdown("---")
            gr.Markdown("#### 🍃 **1-Click Quick Itineraries**")
            b_munnar = gr.Button("🌴 3 Days in Munnar (Tea & Peaks)", elem_classes=["chip-btn"])
            b_coorg = gr.Button("☕ 2 Days in Coorg (Coffee & Waterfalls)", elem_classes=["chip-btn"])
            b_hampi = gr.Button("🏛️ 2 Days in Hampi (Ancient Ruins)", elem_classes=["chip-btn"])
            b_alappuzha = gr.Button("🛶 2 Days in Alappuzha (Houseboats)", elem_classes=["chip-btn"])

        # RIGHT: Chatbot Console
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                height=490,
                layout="bubble",
                placeholder=(
                    "### 🌿 Welcome to TravelMate AI!\n\n"
                    "- 💬 Say **'Hi'**, **'Hello'**, or ask **'What can you do?'** for natural conversation.\n"
                    "- 📍 Ask for recommendations: *'Best waterfalls in Wayanad'*, *'Trekking in Coorg'*, or *'Historic temples in Hampi'*.\n"
                    "- ⚡ Click quick buttons on the left for instant full-day itineraries!\n"
                ),
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Say 'Hi', ask a question, or type 'Plan a trip to Coorg'...",
                    show_label=False,
                    scale=9,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send 🌿", scale=2, elem_id="send-button")
                clear_btn = gr.Button("🗑️", scale=1)

    # --------------------------------------------------------------------------
    # CHAT PIPELINE IMPLEMENTATION
    # --------------------------------------------------------------------------
    def chat_pipeline(message, history, district, days, travel_style):
        if not message.strip():
            yield "", history
            return

        is_travel = is_travel_or_rag_query(message, district)

        if not is_travel:
            # Casual conversation branch
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for turn in history[-4:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": message})

            new_hist = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            yield "", new_hist

            for partial in stream_llm_response(messages):
                new_hist[-1]["content"] = partial
                yield "", new_hist
        else:
            # Travel & Itinerary Branch
            if any(w in message.lower() for w in ["plan", "itinerary", "days", "trip"]):
                target_dist = district if district != "All Districts" else "Munnar"
                for d in df["District_Clean"].unique():
                    if d.lower() in message.lower():
                        target_dist = d
                        break
                itinerary = generate_fast_itinerary(target_dist, days, travel_style)
                new_hist = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": itinerary}
                ]
                yield "", new_hist
                return

            # Semantic RAG Search
            target_filter = district if district != "All Districts" else None
            retrieved = retrieve_rag_context(message, target_filter, top_k=4)

            if retrieved:
                context_str = "\n---\n".join(retrieved)
                augmented_prompt = (
                    f"Use the verified tourism information below to answer the user inquiry accurately and pleasantly.\n\n"
                    f"Information:\n{context_str}\n\n"
                    f"Question: {message}\n\n"
                    f"Answer with specific highlights and tips:"
                )
                messages = [
                    {"role": "system", "content": "You are TravelMate AI. Answer travel questions helpfully based on provided context."},
                    {"role": "user", "content": augmented_prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ]

            new_hist = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            yield "", new_hist

            for partial in stream_llm_response(messages):
                new_hist[-1]["content"] = partial
                yield "", new_hist

    # Wiring chat events
    chat_inputs = [user_input, chatbot, district_dropdown, days_slider, style_radio]
    user_input.submit(chat_pipeline, inputs=chat_inputs, outputs=[user_input, chatbot])
    send_btn.click(chat_pipeline, inputs=chat_inputs, outputs=[user_input, chatbot])
    clear_btn.click(lambda: [], None, chatbot)

    # 1-Click Fast Itineraries
    def quick_plan(dest, days_count, style_name):
        plan = generate_fast_itinerary(dest, days_count, style_name)
        new_hist = [
            {"role": "user", "content": f"Plan a {days_count}-day trip to {dest}"},
            {"role": "assistant", "content": plan}
        ]
        return dest, days_count, style_name, new_hist

    b_munnar.click(
        lambda *_: quick_plan("Munnar", 3, "🌿 Nature & Eco-Trails"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )
    b_coorg.click(
        lambda *_: quick_plan("Coorg", 2, "☕ Relaxed & Cafes"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )
    b_hampi.click(
        lambda *_: quick_plan("Hampi", 2, "🏛️ Heritage & Temples"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )
    b_alappuzha.click(
        lambda *_: quick_plan("Alappuzha", 2, "☕ Relaxed & Cafes"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )

    # Client-side Dark / Light Mode Switcher
    theme_btn.click(
        None,
        js="""() => {
            document.body.classList.toggle('dark');
            const gradioApp = document.querySelector('gradio-app');
            if (gradioApp) gradioApp.classList.toggle('dark');
        }"""
    )

# ==============================================================================
# 8. LAUNCH APPLICATION
# ==============================================================================
if __name__ == "__main__":
    print("Launching TravelMate AI with Modern Homepage & Responsive Themes...")
    demo.queue().launch(
        css=custom_css,
        share=True,
        inbrowser=False,
        server_port=7860,
    )
