

# ========== SECTION 1: INSTALLS ==========

#!pip install open-clip-torch
#!pip install opencv-python-headless

# ========== SECTION 2: IMPORTS ==========
import os
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ========== SECTION 3: DATA LOADING ==========
path = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
test_csv = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

train = train[['file_name', 'label']]
train.columns = ['id', 'label']

# ========== SECTION 4: DATA SPLITTING ==========
train_df, val_df = train_test_split(
    train, 
    test_size=0.05, 
    random_state=42,
    stratify=train['label']
)

# ========== SECTION 5: TRANSFORMS ==========
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))
])

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))
])

# ========== SECTION 6: MODIFIED DATASET CLASSES ==========
class AIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform or clip_preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Fixed feature extraction with safeguards
        with BytesIO() as buff:
            # 1. Compression ratio with min size protection
            image.save(buff, format='JPEG', quality=90)
            size_90 = max(buff.tell(), 1)
            buff.seek(0)
            buff.truncate(0)
            image.save(buff, format='JPEG', quality=50)
            size_50 = max(buff.tell(), 1)
            compression_ratio = np.log(size_90 / size_50)
        
        # 2. Entropy with normalized pixel values
        img_np = np.array(image).astype(np.float32) / 255.0
        hist = np.histogram(img_np, bins=256, range=(0, 1))[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7)) / 8  # Normalized to 0-1
        
        # 3. Edge density with adaptive thresholds
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        median = np.median(gray)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        edges = cv2.Canny(gray.astype(np.uint8), lower, upper)
        edge_density = np.mean(edges > 0)
        features = torch.tensor([compression_ratio, entropy, edge_density], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.iloc[idx, 1]
        return image, features, label

class TestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Feature extraction for test images
        buff = BytesIO()
        img.save(buff, format='JPEG', quality=90)
        size_90 = buff.tell()
        buff.seek(0)
        buff.truncate(0)
        img.save(buff, format='JPEG', quality=50)
        size_50 = buff.tell()
        
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        features = torch.tensor([
            size_90/max(size_50, 1),
            -np.sum(np.histogram(img_np.ravel(), bins=256)[0] * np.log2(np.histogram(img_np.ravel(), bins=256)[0] + 1e-7)),
            np.mean(edges > 0)
        ], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
        return img, features, os.path.basename(img_path)

# ========== SECTION 7: UPDATED MODEL ARCHITECTURE ==========
class CLIPDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        # Freeze all but last 4 layers
        for name, param in self.clip_model.named_parameters():
            if not name.startswith(('visual.transformer.resblocks.11', 'visual.ln_post')):
                param.requires_grad = False
                
        # Enhanced feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def forward(self, x, features):
        clip_features = self.clip_model.encode_image(x).float()
        clip_features = nn.functional.normalize(clip_features, p=2, dim=1)
        processed_features = self.feature_processor(features)
        combined = torch.cat([clip_features, processed_features], dim=1)
        return self.classifier(combined)

# ========== SECTION 8: LOSS & MIXUP ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

def mixup_data(x, y, features, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_features = lam * features + (1 - lam) * features[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, mixed_features

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========== SECTION 9: TRAINING SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPDetector().to(device)
criterion = FocalLoss()

# 1. Add class balancing
class_counts = train_df['label'].value_counts().to_list()
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.AdamW([
    {'params': model.clip_model.parameters(), 'lr': 1e-5},
    {'params': model.feature_processor.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 3e-4}
], weight_decay=0.01)

# ========== SECTION 10: DATA LOADERS ==========
train_dataset = AIImageDataset(train_df, root_dir=path, transform=train_transforms)
val_file_list = [os.path.join(path, fname) for fname in val_df['id']]
val_labels = val_df['label'].values
val_dataset = TestAIImageDataset(file_list=val_file_list, transform=clip_preprocess)
test_file_list = [os.path.join(path, fname) for fname in test['id']]
test_dataset = TestAIImageDataset(file_list=test_file_list, transform=clip_preprocess)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=4)

# ========== SECTION 11: TRAINING LOOP ==========
epochs = 1
scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for data, features, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data, features, label = data.to(device), features.to(device), label.to(device)
        mixed_data, y_a, y_b, lam, mixed_features = mixup_data(data, label, features)
        
        optimizer.zero_grad()
        with autocast():
            output = model(mixed_data, mixed_features)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    
    # Validation loop
    model.eval()
    val_preds = []
    with torch.no_grad():
        for data, features, _ in val_loader:
            data, features = data.to(device), features.to(device)
            outputs = model(data, features)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

# ========== SECTION 12: INFERENCE ==========
model.eval()
test_preds = []
with torch.no_grad():
    for data, features, _ in tqdm(test_loader):
        data, features = data.to(device), features.to(device)
        outputs = model(data, features)
        test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

test['label'] = test_preds
test[['id', 'label']].to_csv('submissionb0.csv', index=False)
