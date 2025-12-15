# Emotion Detection in Text Using Pattern Recognition and Neural Networks

This project builds an **emotion-aware text chatbot** that detects emotion from user input and generates an empathetic reply in real time.  
The system classifies text into **four emotions**: **happy**, **sad**, **angry**, and **neutral**, using a **Bidirectional LSTM (BiLSTM)** model and compares performance against a **TF‑IDF + Logistic Regression** baseline.



## Key Features

- **Emotion classification (4 classes):** happy / sad / angry / neutral  
- **Neural model:** Bidirectional LSTM (BiLSTM) for sequence/context modeling  
- **Baseline model:** TF‑IDF + Logistic Regression for comparison  
- **Real-time chatbot UI:** **Gradio** interface for interactive testing  
- **Emoji + text responses:** emotion-conditioned response templates  
- **Conversational memory + recall:** stores session messages and answers recall queries using **TF‑IDF + cosine similarity**  
- **Confidence visualization:** probability bar chart shown in the UI  
- **Robust fallback behavior:** low-confidence or unclear inputs fall back to neutral / safe response  



## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib
- gradio
\

## Dataset

We used a public emotion-labeled text dataset (Kaggle) split into train/val/test.

- Train: 16,000 samples  
- Validation: 2,000 samples  
- Test: 2,000 samples  
- Total: 20,000 samples  

**Dataset link:**  
- Kaggle: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

## Setup

### 1) Create & activate a virtual environment
**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run

### 1) Put the dataset in the right location
Unzip Kaggle files so you have:
```
data/emotions-dataset-for-nlp/train.txt
data/emotions-dataset-for-nlp/val.txt
data/emotions-dataset-for-nlp/test.txt
```

### 2) Train the models (baseline + BiLSTM)
```bash
python train.py --folder data/emotions-dataset-for-nlp --epochs 10 --batch_size 64
```

After training, the following artifacts are typically created under `models/`:
- `best_model.h5` (trained BiLSTM)
- `tokenizer.pkl`, `class_map.pkl`, `inv_class_map.pkl`
- `training_acc.png`, `training_loss.png`, `confusion.png`

### 3) Run the chatbot UI (Gradio)
```bash
python app.py
```
Open the printed local URL in your browser (example: `http://127.0.0.1:7860`).

## Demo Behavior

- The chatbot shows the **detected emotion** and responds with **emotion-aware text + emoji**.
- If you ask recall-type questions (e.g., “Do you remember what I said?”), it retrieves a relevant previous message using **TF‑IDF similarity**.
- A **probability chart** displays confidence over all four emotions.


## Results Summary

- The BiLSTM model achieves **~90%+ test accuracy** in our experiments and improves over the TF‑IDF baseline.
- Confusion is most likely between **neutral** and weak emotional statements due to semantic overlap.

See the report for detailed metrics, curves, and the confusion matrix.


## Links / Resources

- **Pre-recorded presentation video:** <PASTE_YOUTUBE_OR_DRIVE_LINK_HERE>
- [Presentation slides](https://drive.google.com/drive/u/0/folders/1CkRIxO5U1YubWMCWU5C7bL1BCBMb6Ne1)
- [Report](https://drive.google.com/drive/u/0/folders/1aVrZKOWf4pZRA-OYw2YvfPENej-7eWUG)
- [Dataset](https://drive.google.com/drive/u/0/folders/1fpiklJQs3It5nCCo1hv6p6scADBCbNbF)
- **Demo video:** <PASTE_DEMO_VIDEO_LINK_HERE>

## Team

- Rathikadevi Mani  
- Anantha Gokul Sivakumar  
- Rakshita Telrandhe  



## References

1. C.I.V. and S.K.J., “Text-Based Emotion Recognition Using Deep Learning,” ICAIT 2024.  
2. J. Deng and F. Ren, “A Survey of Textual Emotion Recognition and Its Challenges,” IEEE TAC, 2023.  
3. Y. Kim, “Convolutional Neural Networks for Sentence Classification,” EMNLP 2014.
