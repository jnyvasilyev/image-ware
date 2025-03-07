import os
import torch
import torch.nn as nn
from tqdm import tqdm  # ✅ Progress bar for testing
from data_loader import get_dataloaders
from model import MalwareCNN

if __name__ == '__main__':
    # ✅ Detect CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    # ✅ Load test dataset
    _, test_loader = get_dataloaders("data/images", batch_size=64)

    # ✅ Load model
    num_classes = len(os.listdir("data/images"))
    class_names = sorted(os.listdir("data/images"))  # ✅ Get class names (folder names)
    model = MalwareCNN(num_classes).to(device)

    # ✅ Load checkpoint correctly
    checkpoint_path = "checkpoints\malware_cnn_epoch50.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode

    # ✅ Initialize per-class accuracy tracking
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}

    # ✅ Run inference with tqdm progress bar
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get highest probability class
            
            # ✅ Update overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # ✅ Update per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_name = class_names[label]  # Map label to class name
                class_correct[class_name] += (predicted[i] == label).item()
                class_total[class_name] += 1

    # ✅ Print overall accuracy
    accuracy = 100 * correct / total
    print(f"✅ Overall Test Accuracy: {accuracy:.2f}%")

    # ✅ Print per-class accuracy
    print("\n🔹 Per-Class Accuracy:")
    for class_name in class_names:
        if class_total[class_name] > 0:
            class_acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"   {class_name}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
        else:
            print(f"   {class_name}: No samples in test set")

