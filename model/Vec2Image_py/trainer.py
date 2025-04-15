import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Vec2ImageNet

def get_net_trainer(X_train, y_train, X_val, y_val):
    opt_vars = {
        'filter_size': 5,
        'initial_num_filters': 32,        
        'initial_learn_rate': 1e-3,       
        'momentum': 0.9,
        'l2_regularization': 1e-3
    }

    input_channels = 1
    num_classes = y_train.shape[1]

    model = Vec2ImageNet(
        input_channels=input_channels,
        num_classes=num_classes,
        filter_size=opt_vars['filter_size'],
        initial_filters=opt_vars['initial_num_filters']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt_vars['initial_learn_rate'],
        momentum=opt_vars['momentum'],
        weight_decay=opt_vars['l2_regularization']
    )

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.argmax(axis=1), dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.argmax(axis=1), dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=30)

    return model, train_loader, val_loader, criterion, optimizer, device


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=30):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:02d}: Loss={total_loss:.4f} | Train Acc={train_acc:.2%} | Val Acc={val_acc:.2%}")