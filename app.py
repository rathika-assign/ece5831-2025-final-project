# # # src/app.py
# # import gradio as gr
# # import pickle
# # import numpy as np
# # import tensorflow as tf
# # from responses import choose_response
# # from data_prep import basic_clean, make_sequences

# # MODEL_PATH = "models/best_model.h5"
# # TOKENIZER_PATH = "models/tokenizer.pkl"
# # INV_CLASS_MAP_PATH = "models/inv_class_map.pkl"
# # MAX_LEN = 64

# # def load_artifacts():
# #     tokenizer = None
# #     inv_map = None
# #     model = None
# #     try:
# #         with open(TOKENIZER_PATH, "rb") as f:
# #             tokenizer = pickle.load(f)
# #         with open(INV_CLASS_MAP_PATH, "rb") as f:
# #             inv_map = pickle.load(f)
# #         model = tf.keras.models.load_model(MODEL_PATH)
# #     except Exception as e:
# #         print("Failed to load artifacts:", e)
# #     return tokenizer, inv_map, model

# # tokenizer, inv_map, tf_model = load_artifacts()

# # def classify_and_respond(user_message, history=""):
# #     clean = basic_clean(user_message)
# #     if tokenizer is None or tf_model is None:
# #         # fallback naive rule-based if model missing
# #         # simple keyword rules
# #         low = clean
# #         if any(x in low for x in ["love", "happy", "great", "yay", "wonderful"]):
# #             emo = "happy"
# #         elif any(x in low for x in ["sad", "upset", "depressed", "down"]):
# #             emo = "sad"
# #         elif any(x in low for x in ["angry","furious","hate","unacceptable","mad"]):
# #             emo = "angry"
# #         else:
# #             emo = "neutral"
# #         response = choose_response(emo, user_text=user_message)
# #         return response
# #     seq = make_sequences(tokenizer, [clean], max_len=MAX_LEN)
# #     probs = tf_model.predict(seq)
# #     pred_idx = int(np.argmax(probs, axis=1)[0])
# #     emotion = inv_map.get(pred_idx, "neutral")
# #     response = choose_response(emotion, user_text=user_message)
# #     return response

# # with gr.Blocks() as demo:
# #     gr.Markdown("# EmotiSense — Talk. Detect. Empathize.")
# #     chatbot = gr.Chatbot()
# #     user_input = gr.Textbox(placeholder="Type your message here...", lines=2)
# #     def user_submit(msg, chat_history):
# #         bot_resp = classify_and_respond(msg)
# #         chat_history = chat_history or []
# #         chat_history.append(("You", msg))
# #         chat_history.append(("EmotiSense", bot_resp))
# #         return "", chat_history
# #     user_input.submit(user_submit, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
# #     demo.launch(server_name="0.0.0.0", share=False)



# import gradio as gr 
# import pickle
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from data_prep import basic_clean, make_sequences
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Paths
# MODEL_PATH = "models/best_model.h5"
# TOKENIZER_PATH = "models/tokenizer.pkl"
# INV_CLASS_MAP_PATH = "models/inv_class_map.pkl"
# MAX_LEN = 64

# # Emoji responses
# RESPONSES = {
#     "happy": [
#         "😄 That's wonderful! Can you tell me more?",
#         "😄 I'm glad you're feeling good! Want to share why?",
#         "😄 Yay! That sounds great. How can I help keep the positivity going?"
#     ],
#     "sad": [
#         "😢 I'm sorry you're feeling down. Do you want to talk about it?",
#         "😢 That sounds tough — I'm here to listen.",
#         "😢 I hear you — would you like some suggestions to feel better, or just someone to listen?"
#     ],
#     "angry": [
#         "😡 I can tell you're upset. Want to tell me what happened?",
#         "😡 That sounds frustrating. Do you want help figuring out next steps?",
#         "😡 I hear your anger — it's valid. Would you like to vent or look for solutions?"
#     ],
#     "neutral": [
#         "😐 Thanks for letting me know. Anything else on your mind?",
#         "😐 Alright. If you'd like, I can help summarize or take action.",
#         "😐 Noted. How else can I help?"
#     ]
# }
# tf_model = tf.keras.models.load_model(MODEL_PATH)
# # Load artifacts
# with open(TOKENIZER_PATH, "rb") as f:
#     tokenizer = pickle.load(f)
# with open(INV_CLASS_MAP_PATH, "rb") as f:
#     inv_map = pickle.load(f)

# def detect_emotion(text):
#     seq = tokenizer.texts_to_sequences([text])
#     padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding="post")
#     pred = tf_model.predict(padded)
#     label_index = pred.argmax(axis=1)[0]
#     return inv_map[label_index]
# # def detect_emotion(text):
# #     seq = tokenizer.texts_to_sequences([text])
# #     padded = tf.keras.preprocessing.sequence.pad_sequences(
# #         seq, maxlen=MAX_LEN, padding="post"
# #     )

# #     preds = tf_model.predict(padded)
# #     confidence = float(np.max(preds))
# #     label_index = int(np.argmax(preds))
# #     emotion = inv_map[label_index]

# #     return emotion, confidence

# # --- Memory to store all user messages ---
# memory = []

# def store_in_memory(user_message):
#     """Store all user messages for recall"""
#     clean_msg = basic_clean(user_message)
#     memory.append({"text": user_message, "clean": clean_msg})
# def refer_to_memory(user_message, response):
#     """
#     Semantic recall: returns the most relevant previous user input
#     excluding the current message itself.
#     """
#     if len(memory) < 2:  # need at least one previous message
#         return response

#     msg_lower = basic_clean(user_message)
#     question_words = ["what", "who", "where", "which", "when", "how", "do you remember", "recall", "previous"]

#     if any(qw in msg_lower for qw in question_words):
#         # Compare only against previous messages, not current one
#         prev_messages = [entry["clean"] for entry in memory[:-1]]
#         tfidf = TfidfVectorizer().fit_transform(prev_messages + [msg_lower])
#         sim_matrix = cosine_similarity(tfidf[-1], tfidf[:-1])
#         best_idx = sim_matrix.argmax()
#         if sim_matrix[0, best_idx] > 0.1:
#             return f"🤖 I remember you said: '{memory[best_idx]['text']}'"

#     return response


# def classify_and_respond(user_message, history=None):
#     history = history or []

#     # Store user input
#     store_in_memory(user_message)

#     # Use last 3 messages for context
#     context_text = " ".join([msg[0] for msg in history[-3:]] + [user_message])
#     clean = basic_clean(context_text)

#     # LSTM prediction
#     seq = make_sequences(tokenizer, [clean], max_len=MAX_LEN)
#     probs = tf_model.predict(seq)[0]
#     pred_idx = int(np.argmax(probs))
#     emotion = inv_map[pred_idx]

#     # Low-confidence fallback
#     low = basic_clean(user_message)
#     if probs[pred_idx] < 0.3 or np.count_nonzero(seq) == 0:
        
#         if any(x in low for x in ["happy", "joy", "joyful", "cheerful", "excited", "thrilled", "delighted",
#     "love", "loved", "loving", "amazing", "awesome", "great", "fantastic",
#     "wonderful", "excellent", "best", "fun", "enjoy", "enjoyed",
#     "smile", "smiling", "laugh", "laughing", "lol", "haha",
#     "proud", "relieved", "grateful", "thankful", "blessed",
#     "positive", "hopeful", "motivated", "confident", "satisfied",
#     "peaceful", "content", "yay", "woohoo", "nice", "cool"]):
#             emotion = "happy"
            
#         elif any(x in low for x in ["sad", "sadness", "unhappy", "down", "low", "depressed",
#     "depression", "lonely", "alone", "heartbroken","bad","embarrassed", "loneliness",
#     "guilty", "regretful", "disappointed", "downcast", "blue", 
#     "cry", "crying", "tears", "hurt", "painful","disappointing","disgusting","disgusted",
#     "upset", "miserable", "hopeless", "helpless",
#     "tired", "exhausted", "drained",
#     "anxious", "anxiety", "worried", "fear", "scared",
#     "stressed", "stressful", "overwhelmed",
#     "disappointed", "regret", "lost", "empty"]):
#             emotion = "sad"
            
#         elif any(x in low for x in ["angry", "anger", "mad", "furious", "rage", "raging",
#     "hate", "hated", "annoyed", "annoying", "irritated","offended","insulted",
#     "agitated", "aggravated", "provoked", "indignant", 
#     "frustrated", "frustrating", "upset", "outraged",
#     "unfair", "unacceptable", "disappointed", "fed up",
#     "pissed", "resentful", "bitter", "hostile",
#     "irritation", "aggressive", "argument", "conflict",
#     "complaint", "worst", "terrible", "awful"]):
#             emotion = "angry"
            
#         else:
#             emotion = "neutral"

#     # Pick a random emoji response
#     response = np.random.choice(RESPONSES[emotion])

#     # Check memory if user asks a question
#     response = refer_to_memory(user_message, response)

#     # Probabilities for visualization
#     prob_df = pd.DataFrame({
#         "Emotion": [inv_map[i].capitalize() for i in range(len(probs))],
#         "Probability": [float(probs[i]) for i in range(len(probs))]
#     })

#     # Update chat history
#     history.append([user_message, response])
#     return history, prob_df

# # --- Gradio UI ---
# with gr.Blocks() as demo:
#     gr.Markdown("# Emotion Detection in Text Using Pattern Recognition and Neural Networks ")
#     gr.Markdown("The bot remembers all your previous messages and answers questions based on them.")

#     chatbot = gr.Chatbot()
#     user_input = gr.Textbox(placeholder="Type your message here...", lines=2)
#     prob_plot = gr.BarPlot(x="Emotion", y="Probability", title="Emotion Probabilities", interactive=False)

#     def submit(msg, chat_history):
#         history, df = classify_and_respond(msg, chat_history)
#         return "", history, df

#     user_input.submit(submit, inputs=[user_input, chatbot], outputs=[user_input, chatbot, prob_plot])
#     demo.launch()

import gradio as gr 
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from data_prep import basic_clean, make_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
MODEL_PATH = "models/best_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
INV_CLASS_MAP_PATH = "models/inv_class_map.pkl"
MAX_LEN = 64

# Emoji responses
RESPONSES = {
    "happy": [
        "😄 That's wonderful! Can you tell me more?",
        "😄 I'm glad you're feeling good! Want to share why?",
        "😄 Yay! That sounds great. How can I help keep the positivity going?"
    ],
    "sad": [
        "😢 I'm sorry you're feeling down. Do you want to talk about it?",
        "😢 That sounds tough — I'm here to listen.",
        "😢 I hear you — would you like some suggestions to feel better, or just someone to listen?"
    ],
    "angry": [
        "😡 I can tell you're upset. Want to tell me what happened?",
        "😡 That sounds frustrating. Do you want help figuring out next steps?",
        "😡 I hear your anger — it's valid. Would you like to vent or look for solutions?"
    ],
    "neutral": [
        "😐 Thanks for letting me know. Anything else on your mind?",
        "😐 Alright. If you'd like, I can help summarize or take action.",
        "😐 Noted. How else can I help?"
    ]
}

# Load model and artifacts
tf_model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(INV_CLASS_MAP_PATH, "rb") as f:
    inv_map = pickle.load(f)

# Keywords for rule-based override
POSITIVE_KEYWORDS = [
    "excited", "happy", "thrilled", "delighted", "glad", "amazing",
    "awesome", "great", "fantastic", "wonderful", "excellent", "best",
    "fun", "enjoy", "smile", "yay", "woohoo", "nice", "cool"
]
NEGATIVE_KEYWORDS = [
    "angry", "frustrated", "mad", "furious", "rage", "annoyed",
    "upset", "pissed", "resentful", "bitter", "hostile", "worst",
    "terrible", "awful", "disappointed", "helpless", "hopeless"
]

# --- Memory for previous messages ---
memory = []

def store_in_memory(user_message):
    clean_msg = basic_clean(user_message)
    memory.append({"text": user_message, "clean": clean_msg})

def refer_to_memory(user_message, response):
    if len(memory) < 2:
        return response

    msg_lower = basic_clean(user_message)
    question_words = ["what", "who", "where", "which", "when", "how", "do you remember", "recall", "previous"]

    if any(qw in msg_lower for qw in question_words):
        prev_messages = [entry["clean"] for entry in memory[:-1]]
        tfidf = TfidfVectorizer().fit_transform(prev_messages + [msg_lower])
        sim_matrix = cosine_similarity(tfidf[-1], tfidf[:-1])
        best_idx = sim_matrix.argmax()
        if sim_matrix[0, best_idx] > 0.1:
            return f"🤖 I remember you said: '{memory[best_idx]['text']}'"
    return response

# --- Main classifier ---
def classify_and_respond(user_message, history=None):
    history = history or []

    store_in_memory(user_message)
    context_text = " ".join([msg[0] for msg in history[-3:]] + [user_message])
    clean = basic_clean(context_text)

    # LSTM prediction
    seq = make_sequences(tokenizer, [clean], max_len=MAX_LEN)
    probs = tf_model.predict(seq)[0]
    pred_idx = int(np.argmax(probs))
    emotion = inv_map[pred_idx]

    low = basic_clean(user_message)
    confidence = probs[pred_idx]

    # --- RULE-BASED OVERRIDE ---
    # High confidence positive/negative corrections
    if any(word in low for word in POSITIVE_KEYWORDS):
        if emotion in ["neutral", "happy"]:
            emotion = "happy"
    elif any(word in low for word in NEGATIVE_KEYWORDS):
        emotion = "angry"
    # Low confidence fallback
    elif confidence < 0.3 or np.count_nonzero(seq) == 0:
        emotion = "neutral"

    # Pick a random emoji response
    response = np.random.choice(RESPONSES[emotion])
    response = refer_to_memory(user_message, response)

    # Probabilities for visualization
    prob_df = pd.DataFrame({
        "Emotion": [inv_map[i].capitalize() for i in range(len(probs))],
        "Probability": [float(probs[i]) for i in range(len(probs))]
    })

    history.append([user_message, response])
    return history, prob_df

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Emotion Detection in Text Using Pattern Recognition and Neural Networks")
    gr.Markdown("The bot remembers all your previous messages and answers questions based on them.")

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your message here...", lines=2)
    prob_plot = gr.BarPlot(x="Emotion", y="Probability", title="Emotion Probabilities", interactive=False)

    def submit(msg, chat_history):
        history, df = classify_and_respond(msg, chat_history)
        return "", history, df

    user_input.submit(submit, inputs=[user_input, chatbot], outputs=[user_input, chatbot, prob_plot])
    demo.launch()
