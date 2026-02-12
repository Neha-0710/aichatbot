from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, random
from openai import OpenAI
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_ENABLED = False
client = None

if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_ENABLED = True
        print("OpenAI API Key Loaded")
    except Exception as e:
        print("OpenAI Init Failed:", e)
else:
    print("OpenAI API Key NOT found — fallback active")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
intent_path = os.path.join(BASE_DIR, "intent.json")
memory_path = os.path.join(BASE_DIR, "memory.json")
knowledge_path = os.path.join(BASE_DIR, "knowledge.json")

intents = {"intents": []}

try:
    with open(intent_path, "r", encoding="utf-8") as f:
        intents = json.load(f)
        print("intent.json loaded")
except Exception as e:
    print("intent.json missing or broken:", e)
    intents = {"intents": []}

if not os.path.exists(memory_path):
    with open(memory_path, "w") as f:
        json.dump({"long_term_memory": []}, f)

try:
    with open(memory_path, "r") as f:
        long_term_memory = json.load(f)
except:
    long_term_memory = {"long_term_memory": []}

def save_memory():
    with open(memory_path, "w") as f:
        json.dump(long_term_memory, f, indent=2)

def store_memory(user_msg, ai_msg):
    if len(user_msg) > 6:
        long_term_memory["long_term_memory"].append({
            "user": user_msg,
            "ai": ai_msg
        })

        long_term_memory["long_term_memory"] = long_term_memory["long_term_memory"][-200:]
        save_memory()

knowledge_path = os.path.join(BASE_DIR, "knowledge.json")
if not os.path.exists(knowledge_path):
    with open(knowledge_path, "w") as f:
        json.dump({"facts": []}, f)

with open(knowledge_path, "r") as f:
    knowledge = json.load(f)

def save_knowledge():
    with open(knowledge_path, "w") as f:
        json.dump(knowledge, f, indent=2)

def learn_fact(user_msg):
    if "remember" in user_msg.lower():
        knowledge["facts"].append(user_msg)
        save_knowledge()

def check_intent(user_message):
    msg = user_message.lower()

    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            if pattern.lower() in msg:
                return random.choice(intent.get("responses", []))

    return None

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

def get_ai_response(history):
    if not OPENAI_ENABLED:
        return "AI brain offline — but I'm still here "

    try:
        memory_context = "\n".join([
            f"User: {m['user']} | AI: {m['ai']}"
            for m in long_term_memory["long_term_memory"][-5:]
        ])

        system_prompt = f"""
You are a ChatGPT-like assistant.
Be friendly, helpful, and human.

Past memory:
{memory_context}
"""

        messages = [{"role": "system", "content": system_prompt}] + history

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        print("OpenAI Error:", e)

        fallback = [
            "I'm here to chat",
            "Tell me more!",
            "I'm listening",
            "Let's keep talking",
            "Sorry, my AI brain is a bit tired right now!",
            "Hmm, something's not working, but I'm still here for you!",
            "Oops, AI hiccup! But I'm still here to chat!",
            "Sorry, I'm having a little trouble thinking right now, but I'm all ears!"
        ]
        return random.choice(fallback)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)

        user_message = data.get("message", "")
        history = data.get("history", [])

        if not user_message.strip():
            return jsonify({"response": "Please type something!", "history": history})

        history.append({"role": "user", "content": user_message})

        intent_reply = check_intent(user_message)
        mood = get_sentiment(user_message)

        ai_response = intent_reply if intent_reply else get_ai_response(history)

        if mood == "negative":
            ai_response = "I'm here for you. " + ai_response
        elif mood == "positive":
            ai_response = "Love that energy! " + ai_response

        history.append({"role": "assistant", "content": ai_response})

        store_memory(user_message, ai_response)

        return jsonify({"response": ai_response, "history": history})

    except Exception as e:
        print("Backend Crash:", e)
        return jsonify({"response": "Backend error — check logs"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
