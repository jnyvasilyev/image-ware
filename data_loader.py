import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, random_split

class MalwareNPYDataset(Dataset):
    def __init__(self, data_folder, target_size=(128, 128)):
        self.samples = []
        self.family_labels = {family: idx for idx, family in enumerate(sorted(os.listdir(data_folder)))}
        
        for family in os.listdir(data_folder):
            family_folder = os.path.join(data_folder, family)
            for file in os.listdir(family_folder):
                self.samples.append((os.path.join(family_folder, file), self.family_labels[family]))

        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        image = np.load(npy_path)
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_folder, batch_size=64, split_ratio=0.75, num_workers=8, device="cpu"):
    dataset = MalwareNPYDataset(data_folder)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader