import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths to dataset files
path = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
test_csv = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'

# Load the training and test datasets
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

# Print dataset shapes
print(f'Training dataset shape: {train.shape}')
print(f'Test dataset shape: {test.shape}')

# Preprocess column names for consistency
train = train[['file_name', 'label']]
train.columns = ['id', 'label']

# Display columns for reference
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# Split the training data into training and validation sets (95% train, 5% validation)
train_df, val_df = train_test_split(
    train, 
    test_size=0.05, 
    random_state=42,  
    stratify=train['label'] 
)

# Print shapes of the splits
print(f'Train shape: {train_df.shape}')
print(f'Validation shape: {val_df.shape}')

# Check class distribution in both sets
print("\nTrain class distribution:")
print(train_df['label'].value_counts(normalize=True))

print("\nValidation class distribution:")
print(val_df['label'].value_counts(normalize=True))

# Training augmentations
train_transforms = transforms.Compose([
    transforms.Resize(232),  # Resize to match ConvNeXt preprocessing
    transforms.RandomCrop(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # New
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation and Test transforms
val_test_transforms = transforms.Compose([
    transforms.Resize(232),  # Resize to 232 as per ConvNeXt documentation
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class for training and validation
class AIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.dataframe.iloc[idx, 1]
        return image, label

# Dataset class for inference (validation and test)
class TestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)  # Return image and filename

class GANDetectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b3(pretrained=True)
        
        # Modify first layer
        original_layer = self.base.features[0][0]
        self.base.features[0][0] = nn.Conv2d(3, original_layer.out_channels, 
                                            kernel_size=3, stride=2, padding=1, bias=False)
        
        # Frequency analysis branch
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        num_ftrs = self.base.classifier[1].in_features
        self.base.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

def forward(self, x):
        features = self.base(x)
        freq_features = self.freq_branch(x)
        freq_features = freq_features.view(freq_features.size(0), -1)
        combined = torch.cat([features, freq_features], dim=1)
        return self.classifier(combined)

# Create datasets
train_dataset = AIImageDataset(train_df, root_dir=path, transform=train_transforms)

# For validation, create a list of file paths and store labels separately
val_file_list = [os.path.join(path, fname) for fname in val_df['id']]
val_labels = val_df['label'].values  # Store labels separately for later use
val_dataset = TestAIImageDataset(file_list=val_file_list, transform=val_test_transforms)

# For testing, create a list of file paths
test_file_list = [os.path.join(path, fname) for fname in test['id']]
test_dataset = TestAIImageDataset(file_list=test_file_list, transform=val_test_transforms)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pretrained ConvNeXt Base model
model = models.convnext_base(weights="DEFAULT")

# Freeze the backbone
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the last two stages 
for param in model.features[-2:].parameters(): 
    param.requires_grad = True

# Replace the classifier head with a custom one
model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 12)          
)

# Move the model to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function, optimizer, and learning rate scheduler
optimizer = torch.optim.AdamW([
    {'params': model.features[-2:].parameters(), 'lr': 1e-5},  # Lower LR for backbone
    {'params': model.classifier.parameters(), 'lr': 1e-4}      # Higher LR for classifier
])


# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4,weight_decay=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)


# Training Loop
epochs = 15

train_losses, train_accuracies, val_losses, val_accuracies, val_f1s = [], [], [], [], []

for epoch in range(epochs):
    # -- Training --
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        preds = output.argmax(dim=1)
        acc = (preds == label).float().mean().item()
        epoch_accuracy += acc
    
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    # -- Validation --
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_pred_classes = []  # To store predictions
    val_labels_list = []   # To store true labels
    
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
            data = data.to(device)
            output = model(data)
            
            # Get true labels from val_df
            batch_labels = val_labels[i * val_loader.batch_size : (i + 1) * val_loader.batch_size]
            batch_labels = torch.tensor(batch_labels, device=device)
            
            # Compute loss
            loss = criterion(output, batch_labels)
            val_loss += loss.item()
            
            # Compute predictions and accuracy
            preds = output.argmax(dim=1)
            acc = (preds == batch_labels).float().mean().item()
            val_acc += acc
            
            # Store predictions and true labels
            val_pred_classes.extend(preds.cpu().numpy())
            val_labels_list.extend(batch_labels.cpu().numpy())
    
    # Compute average validation metrics
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_f1 = f1_score(val_labels_list, val_pred_classes, average='binary')  # Binary classification
    
    # Append metrics
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)
    
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
    )
    
    # Step the learning rate scheduler
    scheduler.step()

# Generate predictions for the test set
model.eval()
test_pred_classes = []

with torch.no_grad():
    for data, _ in tqdm(test_loader, desc="Generating Test Predictions"):
        data = data.to(device)
        output = model(data)
        preds = output.argmax(dim=1)  # Get predicted class (0 or 1)
        test_pred_classes.extend(preds.cpu().numpy())

# Add predictions to the test DataFrame
test['label'] = test_pred_classes

# Save predictions to a CSV file
test[['id', 'label']].to_csv('submission.csv', index=False)
print("Test predictions saved to 'submission.csv'")
