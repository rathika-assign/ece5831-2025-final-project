import os
import re
import csv
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42

# Our 4-class target space
DEFAULT_CLASSES = ["happy", "sad", "angry", "neutral"]

# Kaggle emotions (6) → our 4-class mapping
KAGGLE_MAP_6TO4 = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "neutral",
    "love": "neutral",
    "surprise": "neutral",
}

def basic_clean(text: str) -> str:
    """Lightweight normalizer for social/text data.
    - lowercases
    - strips urls
    - removes non-alphanum (keeps spaces)
    - collapses whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------- Kaggle loaders -------------

def _read_split_tsv(path: str) -> pd.DataFrame:
    import csv
    import pandas as pd

    # Many copies of this dataset use ';' (semicolon), some use tabs.
    # Read with a regex separator that handles both.
    df = pd.read_csv(
        path,
        sep=r"\t|;",
        engine="python",
        header=None,              # files have no header
        names=["text", "label"],  # enforce column names
        quoting=csv.QUOTE_NONE,   # prevent quote parsing issues
        on_bad_lines="skip"       # skip any malformed lines
    )

    # basic cleanup
    df["text"] = df["text"].astype(str).apply(basic_clean)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df

def load_kaggle_emotions(folder: str = "data/emotions-dataset-for-nlp"):
    """Loads praveengovi/emotions-dataset-for-nlp split files and maps 6→4 labels.

    Expects: folder/train.txt, val.txt, test.txt with TAB format: `text\tlabel`
    """
    train_fp = os.path.join(folder, "train.txt")
    val_fp   = os.path.join(folder, "val.txt")
    test_fp  = os.path.join(folder, "test.txt")
    if not (os.path.exists(train_fp) and os.path.exists(val_fp) and os.path.exists(test_fp)):
        raise FileNotFoundError("Kaggle split files not found. Expected train.txt, val.txt, test.txt")

    train = _read_split_tsv(train_fp)
    val   = _read_split_tsv(val_fp)
    test  = _read_split_tsv(test_fp)

    # Map 6 → 4 and filter
    for df in (train, val, test):
        df["label"] = df["label"].map(KAGGLE_MAP_6TO4)
        df.dropna(subset=["label"], inplace=True)
        df[:] = df[df["label"].isin(DEFAULT_CLASSES)]

    return train, val, test

# ------------- Tokenizer & encoders -------------

def prepare_tokenizer(texts, num_words=15000, oov_token="<OOV>"):
    tok = Tokenizer(num_words=num_words, oov_token=oov_token)
    tok.fit_on_texts(texts)
    return tok

def make_sequences(tokenizer, texts, max_len=64):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

def encode_labels(labels, class_map=None):
    if class_map is None:
        classes = sorted(list(set(labels)))
        class_map = {c: i for i, c in enumerate(classes)}
    y = np.array([class_map[l] for l in labels])
    return y, class_map

# ------------- Public API -------------

def load_and_split_from_kaggle(folder="data/emotions-dataset-for-nlp", max_len=64, num_words=15000):
    """End-to-end: load Kaggle splits, tokenize, pad, and encode.

    Returns dict with: tokenizer, X_train/X_val/X_test, y_train/y_val/y_test,
    class_map, inv_class_map, and raw texts/labels for the baseline.
    """
    train, val, test = load_kaggle_emotions(folder)

    tokenizer = prepare_tokenizer(train["text"].tolist(), num_words=num_words)
    X_train = make_sequences(tokenizer, train["text"], max_len=max_len)
    X_val   = make_sequences(tokenizer, val["text"],   max_len=max_len)
    X_test  = make_sequences(tokenizer, test["text"],  max_len=max_len)

    y_train, class_map = encode_labels(train["label"].tolist())
    y_val, _  = encode_labels(val["label"].tolist(), class_map)
    y_test, _ = encode_labels(test["label"].tolist(), class_map)

    inv_class_map = {v: k for k, v in class_map.items()}

    return dict(
        tokenizer=tokenizer,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        class_map=class_map, inv_class_map=inv_class_map,
        train_texts=train["text"].tolist(), val_texts=val["text"].tolist(), test_texts=test["text"].tolist(),
        train_labels=train["label"].tolist(), val_labels=val["label"].tolist(), test_labels=test["label"].tolist(),
    )

if __name__ == "__main__":
    # quick smoke test (requires the Kaggle files present)
    try:
        d = load_and_split_from_kaggle()
        print("Loaded:", len(d['y_train']), len(d['y_val']), len(d['y_test']))
        print("Classes:", d['inv_class_map'])
    except Exception as e:
        print("Data prep error:", e)
