import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define the custom dataset
class VEPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define the neural network architecture
class VEPNet(nn.Module):
    def __init__(self, input_size):
        super(VEPNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

        # Compute the size of the flattened feature map after conv and pool layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, input_size))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def convs(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

def load_data_from_json(files):
    true_periods = []
    false_periods = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            true_periods.extend(data['true_time_periods'])
            false_periods.extend(data['false_time_periods'])
    return true_periods, false_periods

def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

def augment_data(data):
    augmented_data = []
    for seq in data:
        noise = np.random.normal(0, 0.1, len(seq))
        augmented_seq = seq + noise
        augmented_data.append(augmented_seq)
    return np.array(augmented_data)

# Load the response dictionary from JSON files
json_files = [f'event_dictionairies/{file}' for file in os.listdir('event_dictionairies') if file.endswith('.json')]
true_data, false_data = load_data_from_json(json_files)

# Find the maximum length of the sequences
max_len = max(max(len(seq) for seq in true_data), max(len(seq) for seq in false_data))

# Pad the sequences
true_data = pad_sequences(true_data, max_len)
false_data = pad_sequences(false_data, max_len)

true_data = np.array(true_data)
false_data = np.array(false_data)

true_labels = np.ones(len(true_data))
false_labels = np.zeros(len(false_data))

# Data augmentation
augmented_true_data = augment_data(true_data)
augmented_false_data = augment_data(false_data)

all_data = np.concatenate((true_data, false_data, augmented_true_data, augmented_false_data), axis=0)
all_labels = np.concatenate((true_labels, false_labels, true_labels, false_labels), axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

# Convert data to Dataset and DataLoader
train_dataset = VEPDataset(X_train, y_train)
test_dataset = VEPDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = VEPNet(input_size=max_len)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)  # Add channel dimension
        labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
false_positives = []
false_negatives = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1)  # Add channel dimension
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect false positives and false negatives
        for i in range(len(predicted)):
            if predicted[i] == 1 and labels[i] == 0:
                false_positives.append(inputs[i].squeeze().tolist())
            elif predicted[i] == 0 and labels[i] == 1:
                false_negatives.append(inputs[i].squeeze().tolist())

print(f"Accuracy: {100 * correct / total}%")

# Save the false positives and false negatives
errors = {
    "false_positives": false_positives,
    "false_negatives": false_negatives
}

with open("errors.json", "w") as f:
    json.dump(errors, f)

# Save the model
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

existing_models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
model_idx = len(existing_models)
model_path = os.path.join(model_dir, f'model_{model_idx}.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
