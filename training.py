import sys
import os
import pickle
import torch
from torch.optim import Adam
from monai.losses import DiceLoss
from dataset import BraTSDataset
from model import get_model

# Ensure correct path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = get_model(device)

# Load train_loader
with open("train_loader.pkl", "rb") as f:
    train_loader = pickle.load(f)

print("train_loader loaded successfully!")

# Define optimizer & loss function
optimizer = Adam(model.parameters(), lr=1e-4)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)

# Enable mixed precision training
# Force CPU usage (no GPU required)
device = torch.device("cpu")
print(f"Using device: {device}")

num_epochs = 5
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # No AMP needed
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "brain_segmentation_model.pth")
print("Model saved successfully!")
