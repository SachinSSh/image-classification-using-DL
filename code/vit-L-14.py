import os
import numpy as np
import pandas as pd
import open_clip
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torchvision.transforms import AutoAugmentPolicy, AutoAugment

# Install required packages
# !pip install open-clip-torch transformers
# !pip install timm opencv-python

# Set paths
path = ''
train_csv = '/train.csv'
test_csv = '/test.csv'

# Load and preprocess data
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

train = train[['file_name', 'label']]
train.columns = ['id', 'label']

# Split data
train_df, val_df = train_test_split(
    train, 
    test_size=0.05, 
    random_state=42,
    stratify=train['label']
)

# CLIP preprocessing
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))
])

# Training transforms
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))
])
train_transforms.transforms.insert(2, AutoAugment(AutoAugmentPolicy.IMAGENET))

# Validation/Test transforms
val_test_transforms = clip_preprocess

class CLIPDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use ViT-L/14 model
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        
        # Freeze most CLIP layers
        for name, param in self.clip_model.named_parameters():
            if "layer.23" not in name:  # Only fine-tune last transformer block
                param.requires_grad = False
            
        # Enhanced classifier with file size feature
        self.classifier = nn.Sequential(
            nn.Linear(770, 512),  # 770 = 768 (CLIP features) + 2 (file size features)
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # File size processing layers
        self.file_size_layers = nn.Sequential(
            nn.Linear(1, 2),
            nn.LayerNorm(2),
            nn.GELU()
        )

    def forward(self, x, file_size):
        # Process image through CLIP
        image_features = self.clip_model.encode_image(x).float()
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        
        # Process file size
        file_size = file_size.unsqueeze(1).float()
        file_size_features = self.file_size_layers(file_size)
        
        # Concatenate features
        combined_features = torch.cat([image_features, file_size_features], dim=1)
        return self.classifier(combined_features)

class EnhancedAIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get file size in MB
        file_size = os.path.getsize(img_name) / (1024 * 1024)
        
        if self.transform:
            image = self.transform(image)
            
        label = self.dataframe.iloc[idx, 1]
        return image, torch.tensor(file_size, dtype=torch.float32), label

class EnhancedTestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Get file size in MB
        file_size = os.path.getsize(img_path) / (1024 * 1024)
        
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(file_size, dtype=torch.float32), os.path.basename(img_path)

# Create datasets
train_dataset = EnhancedAIImageDataset(train_df, path, transform=train_transforms)
val_dataset = EnhancedAIImageDataset(val_df, path, transform=val_test_transforms)
test_file_list = [os.path.join(path, fname) for fname in test['id']]
test_dataset = EnhancedTestAIImageDataset(test_file_list, transform=val_test_transforms)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

# Initialize model and training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPDetector().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW([
    {'params': model.clip_model.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 2e-4},
    {'params': model.file_size_layers.parameters(), 'lr': 2e-4}
], weight_decay=0.01)

# Setup scheduler
epochs = 10
num_training_steps = epochs * len(train_loader)
num_warmup_steps = num_training_steps // 10
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Training loop
best_val_acc = 0
best_model_state = None

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, file_sizes, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        images = images.to(device)
        file_sizes = file_sizes.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, file_sizes)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, file_sizes, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            file_sizes = file_sizes.to(device)
            labels = labels.to(device)
            
            outputs = model(images, file_sizes)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')

# Load best model for prediction
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions
predictions = []
file_names = []

with torch.no_grad():
    for images, file_sizes, fnames in tqdm(test_loader, desc="Generating predictions"):
        images = images.to(device)
        file_sizes = file_sizes.to(device)
        
        outputs = model(images, file_sizes)
        _, predicted = outputs.max(1)
        
        predictions.extend(predicted.cpu().numpy())
        file_names.extend(fnames)

# Create submission file
submission = pd.DataFrame({
    'id': file_names,
    'label': predictions
})

submission.to_csv('submission10.csv', index=False)
print("Submission file created successfully!")


########################

# Read the TSV file
#df = pd.read_csv('submission10.csv')

# Add prefix to id column
#df['id'] = 'test_data_v2/' + df['id']

# Save back to CSV
#df.to_csv('tsv7e2.csv', index=False)
