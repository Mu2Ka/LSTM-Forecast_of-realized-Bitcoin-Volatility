"""Подготовка последовательностей и обучение моделей PyTorch."""

import numpy as np
import torch
from torch import nn

def create_sequences(X, y, seq_length=60):
    X_sequences = []
    y_sequences = []

    for i in range(seq_length - 1, len(X)):
        X_sequences.append(
            X[i - seq_length + 1:i + 1]
        )
        y_sequences.append(y[i])

    return np.array(X_sequences),   np.array(y_sequences)


from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader

epochs = 100
patience = 10


def create_dataset(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
    )
    return dataloader


def pytorch_model_validation(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs,
    patience,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    valid_losses = []

    train_loader = create_dataset(X_train, y_train)
    valid_loader = create_dataset(X_valid, y_valid)

    best_valid_loss = np.inf
    best_model_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).reshape(-1)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        valid_loss_sum = 0
        with torch.inference_mode():
            for X_batch, y_batch in valid_loader:
                predictions = model(X_batch).reshape(-1)
                loss = criterion(predictions, y_batch)
                valid_loss_sum += loss.item() * len(X_batch)

        valid_loss = valid_loss_sum / len(valid_loader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train={train_loss:.6f} | "
            f"valid={valid_loss:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"остановка на эпохе {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    model.eval()
    return model, train_losses, valid_losses


def pytorch_model_fit(model, X_train, y_train, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = create_dataset(X_train, y_train)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).reshape(-1)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(train_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Final epoch {epoch + 1:03d} | "
                f"train={train_loss:.6f}"
            )

    model.eval()
    return model, train_losses


def predict_model(model, X_test, y_test):
    test_loader = create_dataset(X_test, y_test)
    criterion = nn.MSELoss()

    model.eval()
    test_loss_sum = 0
    predictions_list = []
    targets_list = []

    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch).reshape(-1)
            loss = criterion(predictions, y_batch)
            test_loss_sum += loss.item() * len(X_batch)

            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(y_batch.cpu().numpy())

    test_loss = test_loss_sum / len(test_loader.dataset)
    test_predictions = np.concatenate(predictions_list)
    test_targets = np.concatenate(targets_list)

    return test_loss, test_predictions, test_targets
