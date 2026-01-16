import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.train_nn import train_nn
from src.train_logr import train_logr
from src.evaluate import evaluate_nn, evaluate_logr
from src.visualize import plot_loss, plot_confusion, plot_pca

# Load CSV
df = pd.read_csv("data/iris.csv")

# Features and target
X = df.drop("target", axis=1).values
y = df["target"]

# Encode string labels -> integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train models
nn_model, losses = train_nn(X_train, y_train)
logr_model = train_logr(X_train, y_train)

# Evaluate models
nn_acc, nn_cm = evaluate_nn(nn_model, X_test, y_test)
logr_acc, logr_cm = evaluate_logr(logr_model, X_test, y_test)

print(f"Neural Network Accuracy: {nn_acc:.4f}")
print(f"Logistic Regression Accuracy: {logr_acc:.4f}")

# Visualizations
plot_loss(losses)
plot_confusion(nn_cm, "Neural Network Confusion Matrix")
plot_confusion(logr_cm, "Logistic Regression Confusion Matrix")
plot_pca(X, y, "PCA Projection of Iris Dataset")
