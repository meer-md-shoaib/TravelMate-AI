# -*- coding: utf-8 -*-
"""
TravelMate AI: Hybrid Tourism Concierge
- Fast Dataset Engine: Instant (<0.1s) responses for 720+ Kerala & Karnataka destinations
- Conversational AI Engine: Qwen 2.5 (0.5B) for natural chit-chat, greetings (Hi, Hello), & general travel advice
- Earthy Green & Warm Brown Aesthetic
"""

import os
import sys
import re

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import kagglehub
import pandas as pd
import gradio as gr
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# -------------------------------------------------------------
# 1. LOAD DATASET (720 PLACES)
# -------------------------------------------------------------
print("Loading Kerala & Karnataka Tourism Dataset...")
try:
    dataset_path = kagglehub.dataset_download("adilpasharazvi1/kerala-and-karnataka-tourism-places-dataset")
    csv_file = os.path.join(dataset_path, "Kerala_and_karnataka_Districts_tourism.csv")
    df = pd.read_csv(csv_file)
except Exception:
    cached_path = r"C:\Users\meerm\.cache\kagglehub\datasets\adilpasharazvi1\kerala-and-karnataka-tourism-places-dataset\versions\1\Kerala_and_karnataka_Districts_tourism.csv"
    df = pd.read_csv(cached_path)

df["District_Clean"] = df["District"].astype(str).str.strip().str.title()
districts_list = ["All Districts"] + sorted(df["District_Clean"].unique().tolist())
print(f"Dataset ready: {len(df)} places across {len(districts_list)-1} districts.")

# -------------------------------------------------------------
# 2. LOAD CONVERSATIONAL AI MODEL (Qwen 2.5 0.5B)
# -------------------------------------------------------------
AI_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading conversational model {AI_MODEL_ID} on {device.upper()}...")

ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_ID)
ai_model = AutoModelForCausalLM.from_pretrained(
    AI_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
if not torch.cuda.is_available():
    ai_model = ai_model.to(device)
ai_model.eval()
print("Conversational AI Engine Ready!")

AI_SYSTEM_PROMPT = (
    "You are 'TravelMate AI', a warm, enthusiastic, and knowledgeable travel companion. "
    "Respond politely and concisely to user greetings (Hi, Hello), general questions, travel banter, "
    "and packing or safety queries. Keep responses natural, encouraging, and within 2-3 short paragraphs."
)

# -------------------------------------------------------------
# 3. HYBRID ROUTER: FAST DATASET vs CONVERSATIONAL AI
# -------------------------------------------------------------
def is_dataset_query(msg, district):
    msg_lower = msg.lower()

    # Greetings and pure chitchat triggers
    greetings = ["hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "howdy", "wassup"]
    words = re.findall(r"\b\w+\b", msg_lower)
    
    # If it's a short message that's purely a greeting
    if len(words) <= 3 and any(w in greetings for w in words):
        return False

    # Chitchat / identity checks
    if any(p in msg_lower for p in ["who are you", "what are you", "how are you", "tell me a joke", "thank you", "thanks", "bye", "who made you"]):
        return False

    # Check for direct district mentions
    for dist in df["District_Clean"].unique():
        if dist.lower() in msg_lower:
            return True

    # Check for travel/itinerary keywords
    travel_keywords = [
        "plan", "itinerary", "trip", "days", "day", "visit", "tour", "guide",
        "waterfall", "temple", "beach", "trek", "hiking", "palace", "fort",
        "scenic", "spots", "places", "food", "resort", "lake", "cave",
        "wildlife", "safari", "botanical", "monument", "kerala", "karnataka"
    ]
    if any(k in msg_lower for k in travel_keywords):
        return True

    # If user has a specific district selected other than 'All Districts'
    if district and district != "All Districts":
        return True

    return False

def generate_dataset_itinerary(district_name, days=3, travel_style="🌿 Nature & Eco-Trails"):
    if district_name and district_name != "All Districts":
        places = df[df["District_Clean"].str.lower() == district_name.strip().lower()].copy()
        if len(places) == 0:
            places = df[df["District_Clean"].str.contains(district_name.strip(), case=False, na=False)].copy()
    else:
        places = df[df["District_Clean"].isin(["Munnar", "Coorg", "Wayanad", "Hampi", "Alappuzha"])].copy()

    if len(places) == 0:
        places = df.copy()

    style_keywords = {
        "🌿 Nature & Eco-Trails": ["nature", "fall", "peak", "garden", "lake", "forest", "hill", "sanctuary", "river"],
        "🏛️ Heritage & Temples": ["temple", "palace", "fort", "heritage", "monument", "historic", "architecture", "church", "ruins"],
        "🧗 Adventure & Treks": ["trek", "climb", "safari", "raft", "cave", "camp", "paragliding", "adventure"],
        "☕ Relaxed & Cafes": ["beach", "lake", "sunset", "park", "boat", "dam", "scenic", "relax", "promenade"],
    }
    keywords = style_keywords.get(travel_style, [])
    
    def score_place(row):
        text = f"{row['Name of the Place']} {row['Famous_For']} {row['Activities']}".lower()
        return sum(1 for kw in keywords if kw in text)

    places["relevance"] = places.apply(score_place, axis=1)
    places = places.sort_values(by="relevance", ascending=False).reset_index(drop=True)

    target_district = district_name if district_name != "All Districts" else "Kerala & Karnataka"
    output = []
    output.append(f"### 🌿 **{days}-Day {travel_style.split(' ')[1]} Itinerary for {target_district}**\n")
    output.append(f"*⚡ Instant Curated Plan (Source: {len(places)} verified destinations from dataset)*\n\n---\n")

    places_per_day = 3
    idx = 0
    for day in range(1, int(days) + 1):
        output.append(f"### 📅 **Day {day} Exploration**\n")
        
        # Morning spot
        if idx < len(places):
            r = places.iloc[idx]
            output.append(f"#### 🌅 **Morning Trail: {r['Name of the Place']}** *({r['District_Clean']})*\n- 🌟 **Famous For:** {r['Famous_For']}\n- 🎯 **Activities:** {r['Activities']}\n")
            idx += 1
        
        # Afternoon spot
        if idx < len(places):
            r = places.iloc[idx]
            output.append(f"#### ☀️ **Afternoon Discovery: {r['Name of the Place']}** *({r['District_Clean']})*\n- 🌟 **Famous For:** {r['Famous_For']}\n- 🎯 **Activities:** {r['Activities']}\n")
            idx += 1
            
        # Evening spot
        if idx < len(places):
            r = places.iloc[idx]
            output.append(f"#### 🌇 **Sunset & Unwind: {r['Name of the Place']}** *({r['District_Clean']})*\n- 🌟 **Famous For:** {r['Famous_For']}\n- 🎯 **Activities:** {r['Activities']}\n")
            idx += 1

        output.append("💡 *Local Tip: Start mornings around 7:30 AM to beat midday warmth and enjoy fresh local delicacies.*\n\n---\n")

    return "\n".join(output)

def search_dataset_spots(query, district):
    msg_lower = query.lower()
    stop_words = {"the", "a", "an", "in", "for", "best", "what", "where", "are", "places", "spots", "to", "and", "of", "some", "good", "show", "me"}
    key_terms = [t for t in re.findall(r"\w+", msg_lower) if t not in stop_words and len(t) > 2]

    matches = df.copy()
    if district and district != "All Districts":
        matches = matches[matches["District_Clean"].str.lower() == district.lower()]

    if key_terms:
        def matches_keywords(row):
            full_text = f"{row['Name of the Place']} {row['Famous_For']} {row['Activities']} {row['District_Clean']}".lower()
            return sum(1 for term in key_terms if term in full_text)
        
        matches["match_score"] = matches.apply(matches_keywords, axis=1)
        matches = matches[matches["match_score"] > 0].sort_values(by="match_score", ascending=False)

    if len(matches) > 0:
        top_matches = matches.head(6)
        resp_lines = [
            f"### 📍 **Top Recommended Spots matching '{query}'**\n",
            f"*⚡ Fast Dataset Match: Found {len(matches)} authentic spots:*\n"
        ]
        for _, r in top_matches.iterrows():
            resp_lines.append(
                f"- 🌟 **{r['Name of the Place']}** *({r['District_Clean']})*\n"
                f"  - **Famous For:** {r['Famous_For']}\n"
                f"  - **Activities:** {r['Activities']}\n"
            )
        return "\n".join(resp_lines)
    return None

def stream_ai_conversation(message, history):
    messages = [{"role": "system", "content": AI_SYSTEM_PROMPT}]
    # Add recent history
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    prompt_text = ai_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ai_tokenizer(prompt_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(ai_tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=150, temperature=0.7, do_sample=True, top_p=0.9)
    thread = Thread(target=ai_model.generate, kwargs=kwargs)
    thread.start()

    accumulated = ""
    for token in streamer:
        accumulated += token
        yield accumulated

# -------------------------------------------------------------
# 4. CLEAN UI WITH EARTHY GREEN & WARM BROWN PALETTE
# -------------------------------------------------------------
custom_css = """
.gradio-container {
    background: radial-gradient(circle at 12% 12%, #142218 0%, #0d1610 50%, #161009 100%) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #e5ece5 !important;
}

#header-box {
    background: linear-gradient(135deg, rgba(24, 40, 28, 0.85) 0%, rgba(36, 25, 17, 0.8) 100%);
    border: 1px solid rgba(134, 185, 137, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.header-title {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #4ade80, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.sidebar-box {
    background: rgba(20, 31, 23, 0.6) !important;
    border: 1px solid rgba(163, 130, 92, 0.25) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

#send-btn {
    background: linear-gradient(135deg, #15803d 0%, #166534 40%, #92400e 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid rgba(134, 239, 172, 0.3) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(22, 101, 52, 0.4) !important;
}

.chip-btn {
    background: rgba(43, 33, 23, 0.55) !important;
    border: 1px solid rgba(134, 185, 137, 0.25) !important;
    color: #e2e8e2 !important;
    border-radius: 10px !important;
    text-align: left !important;
    padding: 8px 12px !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}
.chip-btn:hover {
    background: rgba(34, 197, 94, 0.22) !important;
    border-color: rgba(74, 222, 128, 0.5) !important;
    transform: translateY(-2px);
}
"""

with gr.Blocks(title="TravelMate AI - Hybrid Travel Guide") as demo:

    # Header
    with gr.Row(elem_id="header-box"):
        with gr.Column(scale=8):
            gr.Markdown(
                """
                <div class="header-title">🌿 TravelMate AI</div>
                <div style="color: #cbd5cb; font-size: 1rem;">
                    Hybrid Travel Concierge — <strong>Instant Fast Answers</strong> for 720+ Kerala & Karnataka destinations + <strong>AI Neural Chat</strong> for greetings and questions.
                </div>
                """
            )
        with gr.Column(scale=4, min_width=220):
            gr.Markdown("⚡ **Fast Dataset Engine**: 720 Places\n\n🤖 **AI Model**: Qwen 2.5 (0.5B)")

    with gr.Row():
        # LEFT SIDEBAR
        with gr.Column(scale=4, elem_classes=["sidebar-box"]):
            gr.Markdown("### 🧭 **Trip Preferences**")

            district_dropdown = gr.Dropdown(
                label="Destination District",
                choices=districts_list,
                value="Munnar",
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
            gr.Markdown("#### 🍃 **1-Click Instant Itineraries (Fast)**")
            b_munnar = gr.Button("🌴 3 Days in Munnar (Tea & Peaks)", elem_classes=["chip-btn"])
            b_coorg = gr.Button("☕ 2 Days in Coorg (Coffee & Waterfalls)", elem_classes=["chip-btn"])
            b_hampi = gr.Button("🏛️ 2 Days in Hampi (Ancient Ruins)", elem_classes=["chip-btn"])
            b_alappuzha = gr.Button("🛶 2 Days in Alappuzha (Backwaters)", elem_classes=["chip-btn"])

        # RIGHT MAIN CHAT
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                height=530,
                layout="bubble",
                placeholder=(
                    "### 🌿 Welcome to TravelMate AI!\n\n"
                    "- 💬 Say **'Hi'** or **'Hello'** to chat with the AI model.\n"
                    "- ⚡ Ask for any destination (*'Best waterfalls in Wayanad'*, *'Plan 3 days in Coorg'*, *'Temples in Hampi'*) for **instant dataset itineraries**!\n"
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
                submit_btn = gr.Button("Send 🌿", scale=2, elem_id="send-btn")
                clear_btn = gr.Button("🗑️", scale=1)

    # -------------------------------------------------------------
    # 5. HYBRID DISPATCHER & EVENT HANDLERS
    # -------------------------------------------------------------
    def chat_pipeline(message, history, district, days, travel_style):
        if not message.strip():
            yield "", history
            return

        # Check if query is dataset-driven or AI conversational
        if is_dataset_query(message, district):
            # Fast Dataset Path (<0.05s)
            matched_spots = search_dataset_spots(message, district)
            if matched_spots:
                resp = matched_spots
            else:
                target_dist = district if district != "All Districts" else "Munnar"
                for d in df["District_Clean"].unique():
                    if d.lower() in message.lower():
                        target_dist = d
                        break
                resp = generate_dataset_itinerary(target_dist, days, travel_style)

            new_hist = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": resp}
            ]
            yield "", new_hist
        else:
            # AI Conversational Path (Qwen 2.5) with streaming
            new_hist = history + [{"role": "user", "content": message}]
            yield "", new_hist

            new_hist = new_hist + [{"role": "assistant", "content": ""}]
            for partial in stream_ai_conversation(message, history):
                new_hist[-1]["content"] = partial
                yield "", new_hist

    # Submit wiring
    inputs_list = [user_input, chatbot, district_dropdown, days_slider, style_radio]
    user_input.submit(chat_pipeline, inputs=inputs_list, outputs=[user_input, chatbot])
    submit_btn.click(chat_pipeline, inputs=inputs_list, outputs=[user_input, chatbot])
    clear_btn.click(lambda: [], None, chatbot)

    # 1-Click Fast Itinerary Handlers (Robust *args)
    def trigger_fast_itinerary(target_dist, num_days, style_val):
        query = f"Plan a {num_days}-day itinerary for {target_dist}"
        resp = generate_dataset_itinerary(target_dist, num_days, style_val)
        updated_hist = [{"role": "user", "content": query}, {"role": "assistant", "content": resp}]
        return target_dist, num_days, style_val, updated_hist

    b_munnar.click(
        lambda *_: trigger_fast_itinerary("Munnar", 3, "🌿 Nature & Eco-Trails"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )
    b_coorg.click(
        lambda *_: trigger_fast_itinerary("Coorg", 2, "☕ Relaxed & Cafes"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )
    b_hampi.click(
        lambda *_: trigger_fast_itinerary("Hampi", 2, "🏛️ Heritage & Temples"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )
    b_alappuzha.click(
        lambda *_: trigger_fast_itinerary("Alappuzha", 2, "☕ Relaxed & Cafes"),
        outputs=[district_dropdown, days_slider, style_radio, chatbot]
    )

# -------------------------------------------------------------
# 6. LAUNCH DEMO
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Launching Hybrid TravelMate on Port 7860...")
    earth_theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="amber",
        neutral_hue="stone",
    )
    demo.queue().launch(
        theme=earth_theme,
        css=custom_css,
        share=True,
        inbrowser=False,
        server_port=7860,
    )