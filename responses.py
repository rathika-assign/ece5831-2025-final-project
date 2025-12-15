# src/responses.py
import random

DEFAULT_TEMPLATES = {
    "happy": [
        "That's wonderful to hear! 🎉 Can you tell me more?",
        "I'm glad you're feeling good — want to share why?",
        "Yay! That sounds great. How can I help keep the positivity going?"
    ],
    "sad": [
        "I'm sorry you're feeling down. Do you want to talk about what's bothering you?",
        "That sounds tough — I'm here to listen.",
        "I hear you — would you like some suggestions to feel better, or just someone to listen?"
    ],
    "angry": [
        "I can tell you're upset. Want to tell me what happened?",
        "That sounds frustrating. Do you want help figuring out next steps?",
        "I hear your anger — it's valid. Would you like to vent or look for solutions?"
    ],
    "neutral": [
        "Thanks for letting me know. Anything else on your mind?",
        "Alright. If you'd like, I can help summarize or take action.",
        "Noted. How else can I help?"
    ]
}

def choose_response(emotion, user_text=None, context=None):
    emotion = emotion.lower()
    templates = DEFAULT_TEMPLATES.get(emotion, DEFAULT_TEMPLATES["neutral"])
    # Basic context tweak: if user text contains a question, prefer helpful response
    if user_text and user_text.strip().endswith("?"):
        question_templates = [
            "Good question — here's how I see it: {}",
            "I can help with that — can you give me more detail?"
        ]
        # return one of the helpful templates if available
        return random.choice(templates)
    return random.choice(templates)
