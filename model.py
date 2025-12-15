# src/model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def build_lstm_model(vocab_size, embed_dim=128, max_len=64, lstm_units=128, num_classes=4, bidirectional=True, dropout=0.3):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len))
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))
    else:
        model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout*0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model

def compile_model(model, lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train(model, X_train, y_train, X_val, y_val, out_dir="models", epochs=20, batch_size=64):
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "best_model.h5")
    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True),
        ModelCheckpoint(weights_path, save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return history, weights_path

def predict(model, X):
    probs = model.predict(X)
    preds = np.argmax(probs, axis=1)
    return preds, probs
