import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_dataloaders
from model import MalwareCNN

if __name__ == '__main__':
    # ✅ Detect CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    # Hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ✅ Load dataset with device parameter
    train_loader, test_loader = get_dataloaders("data/images", batch_size=batch_size, device=device)

    # ✅ Initialize model & move to GPU
    num_classes = len(os.listdir("data/images"))
    model = MalwareCNN(num_classes, device=device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # ✅ Move batch to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"malware_cnn_epoch{epoch+1}.pth")
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, checkpoint_path)
            print(f"💾 Checkpoint saved at {checkpoint_path}")

    # Save Final Model
    torch.save(model.state_dict(), "malware_cnn_final.pth")