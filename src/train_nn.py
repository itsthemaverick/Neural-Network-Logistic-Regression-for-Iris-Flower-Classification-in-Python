import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import IrisNN

def train_nn(X_train, y_train, epochs=200):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = IrisNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []

    for _ in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))

    return model, losses
