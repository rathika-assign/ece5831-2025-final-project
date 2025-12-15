# src/utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_history(history, out_path=None):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    if out_path:
        plt.savefig(out_path.replace(".png", "_acc.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    if out_path:
        plt.savefig(out_path.replace(".png", "_loss.png"))
    plt.close()

def show_confusion(y_true, y_pred, labels, out_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if out_path:
        plt.savefig(out_path)
    plt.close()
    return cm
