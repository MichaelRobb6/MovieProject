import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

device = "cuda" if torch.cuda.is_available() else "cpu"

# Device setup
#%%
# New NN Model
class MovieModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(32, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0)
        self.activation = torch.sigmoid
        
    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.dropout(x)
        x = self.activation(self.layer_2(x))
        x = self.output(x)
        return x

    def set_dropout_rate(self, dr):
        self.dr = dr

def train_model(model, train_loader, criterion, optimizer):
    
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Reset optimizer to 0
        optimizer.zero_grad()
        
        # Predict and compute model loss
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Count correct model predictions
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # Perform back propogation and step
        loss.backward()
        optimizer.step()
        
        
    
    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples * 100 
    
    return average_loss, accuracy

def test_model(model, test_loader, criterion):
    model.eval()    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Predict and calculate model loss
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Count correct model predictions
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
    # Calucluate average batch loss and batch accuracy        
    average_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples * 100  # Accuracy in percentage

    return average_loss, accuracy


def train_test(output_size, train_loader, test_loader):

    train_losses = []
    test_losses = []
    train_percs = []
    test_percs = []
    
    weight_decay = 0.000001
    
    nn_model = MovieModel(output_size).to(device)  
    nn_model.set_dropout_rate(0.2)  
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=1)

    
    # Perform training loop
    epochs = 50
    for epoch in range(epochs):
        train_loss, train_perc = train_model(nn_model, train_loader, criterion, optimizer)
        test_loss, test_perc = test_model(nn_model, test_loader, criterion)
        
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        train_percs.append(train_perc)
        test_percs.append(test_perc)
        
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f'Train | Loss: {train_loss} | Accuracy: {train_perc}')
        print(f'Test  | Loss: {test_loss} | Accuracy: {test_perc}')
        print("-"*20)


    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.legend()
    plt.show()
    
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_percs, label="Train Percentage")
    plt.plot(range(epochs), test_percs, label="Test Percentage")
    plt.legend()
    plt.show()