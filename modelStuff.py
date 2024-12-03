import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

device = "cuda" if torch.cuda.is_available() else "cpu"

# Device setup
#%%
# New NN Model
class MovieModel(nn.Module):
    def __init__(self, input_size, output_size, method):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_4 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)
        self.method = method
        self.relu = nn.ReLU()
        self.pact = nn.ReLU()

    
    def forward(self, x):
        if self.method == 'p' or self.method == 'b':
            x = self.pact(self.layer_1(x))
            x = self.dropout(x)
            x = self.pact(self.layer_2(x))
            x = self.dropout(x)
            x = self.pact(self.layer_3(x))
            x = self.dropout(x)
            x = self.pact(self.layer_4(x))
            x = self.dropout(x)
            x = self.output(x)
        elif self.method =='r':
            x = self.relu(self.layer_1(x))
            x = self.dropout(x)
            x = self.relu(self.layer_2(x))
            x = self.output(x)
        return x

    def set_dropout_rate(self, dr):
        self.dr = dr

def train_model(model, train_loader, criterion, optimizer, method):
    
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        if method == 'r':
            labels = labels.unsqueeze(-1) 
            
        # Reset optimizer to 0
        optimizer.zero_grad()
        
        # Predict and compute model loss
        outputs = model(features).squeeze(-1)
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

def test_model(model, test_loader, criterion, method):
    model.eval()    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            if method == 'r':
                labels = labels.unsqueeze(-1) 
    
            # Predict and calculate model loss
            outputs = model(features).squeeze(-1)
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


def train_test(input_size, output_size, train_loader, test_loader, method, epochs, weight_decay, lr, gamma):

    train_losses = []
    test_losses = []
    train_percs = []
    test_percs = []

    best_test_loss = float('inf')  # Initialize best loss to a large value
    best_test_perc = 0  # Initialize best accuracy to the lowest value
    best_epoch_loss = -1
    best_epoch_acc = -1
    
    nn_model = MovieModel(input_size, output_size, method).to(device)  

    if method == 'r':
        #criterion = nn.MSELoss()
        #weight_decay = 0.001
        criterion = nn.HuberLoss(delta=0.5)
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=20, gamma=gamma)

    elif method == 'p':
        #weight_decay = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=20, gamma=gamma)

    elif method == 'b':
        #weight_decay = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=20, gamma=gamma)
  

    # Perform training loop
    epochs = epochs
    for epoch in range(epochs):
        train_loss, train_perc = train_model(nn_model, train_loader, criterion, optimizer, method)
        test_loss, test_perc = test_model(nn_model, test_loader, criterion, method)
        
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        train_percs.append(train_perc)
        test_percs.append(test_perc)
        
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f'Train | Loss: {train_loss} | Accuracy: {train_perc}')
        print(f'Test  | Loss: {test_loss} | Accuracy: {test_perc}')
        print("-"*20)
        
        # Update best test loss and corresponding epoch
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch_loss = epoch + 1
        
        # Update best test accuracy and corresponding epoch
        if test_perc > best_test_perc:  # Track highest accuracy
            best_test_perc = test_perc
            best_epoch_acc = epoch + 1 
    
    # Print the summary of best metrics
    print("="*20)
    print(f"Lowest Test Loss: {best_test_loss:.4f} achieved at Epoch {best_epoch_loss}")
    print(f"Highest Test Accuracy: {best_test_perc:.2f}% achieved at Epoch {best_epoch_acc}")
    
    param_caption = (f"Parameters\n"
                 f"Epochs: {epochs}, Weight Decay: {weight_decay}, "
                 f"Learning Rate: {lr}, Gamma: {gamma}, "
                 f"IS/PCA:{input_size}")
    
    plt.title(f"{method.capitalize()} Model Loss\n{param_caption}")

    if method == 'r':
        plt.ylabel("Huber Loss")
        #plt.title("Regression Model Huber Loss") 

    if method == 'p':
        plt.ylabel("Cross Entropy Loss")
        #plt.title("Profit/Loss Binary Classification CELoss")
        
    if method =='b':
        plt.ylabel("Cross Entropy Loss")
        #plt.title("Bins Multiclass Classification CELoss")
        
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.legend()
    plt.show()
    
    plt.title(f"{method.capitalize()} Model Accuracy\n{param_caption}")

    if method == 'p':
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.plot(range(epochs), train_percs, label="Train Percentage")
        plt.plot(range(epochs), test_percs, label="Test Percentage")
        plt.legend()
        
        plt.show()
        
    if method == 'b':
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.plot(range(epochs), train_percs, label="Train Percentage")
        plt.plot(range(epochs), test_percs, label="Test Percentage")
        plt.legend()
        
        plt.show()
    
    return best_test_loss, best_test_perc