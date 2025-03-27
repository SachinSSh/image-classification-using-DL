## !pip install open-clip-torch timm

#### ======================
# 1. Environment Setup
# ======================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import open_clip
import timm
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ======================
# 2. Data Preparation
# ======================
class ChannelConverter:
    """Handle mixed channel inputs (1/3 channels)"""
    def __call__(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

# 1. Verify and Correct Paths
dataset_base = '/kaggle/input/ai-vs-human-generated-dataset/'  # Fixed dataset name

# ======================
# 3. Augmentations
# ======================
def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.GaussianBlur(3),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ======================
# 4. Model Architecture
# ======================

class CLIPDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        #self.attention_pool = AttentionPooling(512)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self._freeze_backbone()
        
    def _freeze_backbone(self):
        for param in self.clip.parameters():
            param.requires_grad = False
        for block in self.clip.visual.transformer.resblocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        features = self.clip.encode_image(x)
        #pooled = self.attention_pool(features)
        return self.classifier(features)

class EfficientNetV2Detector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        self.classifier = nn.Linear(self.base.num_features, num_classes)
        
    def forward(self, x):
        features = self.base.forward_features(x)
        return self.classifier(features.mean(dim=[2, 3]))

# ======================
# 5. Training Utilities
# ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[index], y, y[index], lam



class AIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Convert filename to string explicitly
        #filename = str(self.dataframe.iloc[idx, 0])  # <-- Fix here
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        
        image = Image.open(img_path).convert('RGB')
        #print(image)
        #image = self.base_transform(image)
        #label = self.dataframe.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx, 1]
            
        
        return image, label

class TestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, os.path.basename(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), os.path.basename(img_path)

# ======================
# 6. Training Loop
# ======================
def train_model(model, train_loader, val_loader, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Configure optimizer based on model type
    if isinstance(model, CLIPDetector):
        optimizer = torch.optim.AdamW([
            {'params': model.clip.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ], weight_decay=0.05)
    elif isinstance(model, EfficientNetV2Detector):
        optimizer = torch.optim.AdamW([
            {'params': model.base.parameters(), 'lr': 1e-5},  # Changed from backbone to base
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ], weight_decay=0.05)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss()
    scaler = torch.amp.GradScaler()
    
    best_acc = 0
    torch.cuda.empty_cache() 
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # MixUp augmentation
            x, y_a, y_b, lam = mixup_data(x, y)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                temp = 1.5  # Temperature parameter for confidence calibration
                scaled_logits = logits / temp
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        val_acc = 0
        train_loss, train_acc = 0.0, 0.0
        train_preds, train_labels_list = [], []
        #torch.cuda.empty_cache()
        with torch.no_grad():
            preds = torch.softmax(scaled_logits, dim=1).argmax(dim=1)
            acc = (preds == y).float().mean().item()
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(y.cpu().numpy())
                
            # Update progress bar
            train_loss += loss.item()
            train_acc += acc
            avg_loss = train_loss / (batch_idx + 1)
            avg_acc = train_acc / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        val_acc /= len(val_loader)
        scheduler.step()
        
        # Save best model
        #if val_acc > best_acc:
        best_acc = val_acc
        #torch.save(model.state_dict(), 'best_model.pth')
        model_type = 'clip' if isinstance(model, CLIPDetector) else 'efficientnet'
        # Inside the training loop (replace the current torch.save):
        torch.save(model.state_dict(), f'/kaggle/working/best_{model_type}_model.pth')
            
        print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}')
        #torch.cuda.empty_cache()

# ======================


# ======================
# 7. Ensemble & TTA
# ======================
class EnsembleModel:
    def __init__(self, model_paths):
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # <-- Add this line
        for path in model_paths:
            model = CLIPDetector() if 'clip' in path else EfficientNetV2Detector()
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))  # Load state_dict directly
            self.models.append(model.to(self.device))
    
    def predict(self, loader, n_tta=5):
        all_preds = []
        with torch.no_grad():
            for x, _ in loader:
                batch_preds = []
                for img in x:
                    tta_outputs = []
                    for _ in range(n_tta):
                        # Apply TTA transforms
                        aug_img = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ColorJitter(0.2, 0.2, 0.2)
                        ])(transforms.ToPILImage()(img.cpu()))
                        aug_img = get_transforms()(aug_img).to(self.device)
                        
                        # Ensemble predictions
                        outputs = [m(aug_img.unsqueeze(0)) for m in self.models]
                        tta_outputs.append(torch.stack(outputs).mean(0))
                    batch_preds.append(torch.stack(tta_outputs).mean(0).argmax().item())
                all_preds.extend(batch_preds)
        return all_preds

# 8. Execution Flow
# ======================
if __name__ == "__main__":
    # Load data
    

    # Load datasets
    dataset_base = '/kaggle/input/ai-vs-human-generated-dataset'
    train_dir = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
    test_dir = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'

    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    # Preprocess data
    train_df = train[['file_name', 'label']]
    train_df.columns = ['id', 'label']
    
    # Split data
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.05, 
        random_state=42,
        stratify=train['label']
    )

    
    # Create datasets
    train_dataset = AIImageDataset(train_df, root_dir=dataset_base, transform=get_transforms('train'))
    val_file_list = [os.path.join(dataset_base, fname) for fname in val_df['id']]
    val_labels = val_df['label'].values
    val_dataset = TestAIImageDataset(file_list=val_file_list,  transform=get_transforms('val'))
    test_file_list = [os.path.join(dataset_base, fname) for fname in test['id']]
    test_dataset = TestAIImageDataset(file_list=test_file_list, transform=get_transforms('val'))
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Train CLIP model
    clip_model = CLIPDetector()
    train_model(clip_model, train_loader, val_loader, epochs=3)
    
    # Train EfficientNet model
    effnet_model = EfficientNetV2Detector()
    train_model(effnet_model, train_loader, val_loader, epochs=3)
    
    # Create ensemble
    ensemble = EnsembleModel(['/kaggle/working/best_clip_model.pth', '/kaggle/working/best_efficientnet_model.pth'])
    test_preds = ensemble.predict(test_loader)
    
    # Save results
    test_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')
    test_df['label'] = test_preds
    test_df.to_csv('submissionsa7.csv', index=False)





