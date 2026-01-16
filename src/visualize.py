import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

def plot_loss(losses):
    plt.plot(losses)
    plt.title("NN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_confusion(cm, title):
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.show()


def plot_pca(X, y, title):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
