# src/train.py

import os
import argparse
import pickle
from sklearn.metrics import classification_report

import data_prep, baseline, model, utils


def main(folder: str,
         max_len: int = 64,
         num_words: int = 15000,
         batch_size: int = 64,
         epochs: int = 10):
    """
    Train baseline (TF-IDF + LR) and LSTM on Kaggle Emotions → 4-class mapping.
    Saves artifacts to ./models
    """
    # 1) Load & preprocess (uses the Kaggle train/val/test splits)
    d = data_prep.load_and_split_from_kaggle(
        folder=folder, max_len=max_len, num_words=num_words
    )

    inv_map = d["inv_class_map"]                # idx -> label
    label_names = [inv_map[i] for i in range(len(inv_map))]

    # 2) Baseline: TF-IDF + Logistic Regression (quick sanity/reference)
    print("\n==== Training baseline (TF-IDF + LogisticRegression) ====")
    vect, clf = baseline.train_baseline(
        d["train_texts"], d["train_labels"], d["val_texts"], d["val_labels"]
    )

    # 3) Build & compile LSTM
    vocab_size = num_words
    num_classes = len(inv_map)
    lstm = model.build_lstm_model(
        vocab_size=vocab_size,
        max_len=max_len,
        num_classes=num_classes,
        embed_dim=128,
        lstm_units=128,
        bidirectional=True,
        dropout=0.3,
    )
    lstm.build(input_shape=(None, max_len))
    model.compile_model(lstm)
    print(lstm.summary())

    # 4) Train LSTM (ModelCheckpoint inside saves models/best_model.h5)
    hist, _ = model.train(
        lstm,
        d["X_train"], d["y_train"],
        d["X_val"],   d["y_val"],
        out_dir="models",
        epochs=epochs,
        batch_size=batch_size,
    )

    # 5) Curves & evaluation
    utils.plot_history(hist, out_path="models/training.png")

    preds, _ = model.predict(lstm, d["X_test"])
    print("\nLSTM on TEST:\n")
    print(classification_report(d["y_test"], preds, target_names=label_names))
    utils.show_confusion(d["y_test"], preds, label_names, out_path="models/confusion.png")

    # 6) Save tokenizer and label maps
    os.makedirs("models", exist_ok=True)
    with open("models/class_map.pkl", "wb") as f:
        # class_map: label -> idx; inv_map is idx -> label
        pickle.dump({v: k for k, v in d["inv_class_map"].items()}, f)
    with open("models/inv_class_map.pkl", "wb") as f:
        pickle.dump(d["inv_class_map"], f)
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(d["tokenizer"], f)

    print("\nDone. Artifacts saved in ./models\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data/emotions-dataset-for-nlp",
                        help="Path containing train.txt / val.txt / test.txt")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--num_words", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    main(folder=args.folder,
         max_len=args.max_len,
         num_words=args.num_words,
         batch_size=args.batch_size,
         epochs=args.epochs)
