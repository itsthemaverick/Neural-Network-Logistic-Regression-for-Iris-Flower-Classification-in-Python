import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_nn(model, X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, cm


def evaluate_logr(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, cm
