'''
Модель для проверки гипотезы о достаточности разметки инструментом smart polygon.
Датасет взят из всего 30 размеченных кадров, аугментирован на roboflow с ипользованием только flip horizontal/vertical и доведен до 210 изображений.
Эксперимент доказал, что даже с таким маленьким и несбалансированным датасетом указанный способ разметки приводит к удовлетворительным результатам.
Однако, т.к. данный датасет содержит только зимние виды, модель не смогла предсказать ни одного изолятора на летних изображениях.
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import pad
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class Config:
    DATA_DIR = "/home/mike-pi/Documents/coding/NU/Stazh/antrax/data/My First Project.v3i.png-mask-semantic"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "validation")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    NUM_CLASSES = 4
    BATCH_SIZE = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "./checkpoints"
    SAVE_EVERY = 5
    IMG_SIZE = (640, 640)
    USE_AMP = True
    USE_GRADIENT_CHECKPOINTING = True

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(640, 640)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        
        # Find all valid image-mask pairs
        self.image_files = []
        self.mask_files = []
        
        for f in os.listdir(self.image_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in f:
                base_name = os.path.splitext(f)[0]
                mask_name = f"{base_name}_mask.png"
                mask_path = os.path.join(self.mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.image_files.append(f)
                    self.mask_files.append(mask_name)
                else:
                    print(f"Warning: Missing mask for image {f}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self._resize_with_padding(image, self.target_size)
        mask = self._resize_with_padding(mask, self.target_size, is_mask=True)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Convert mask to numpy array first, then to tensor
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np.copy()).long()  # Use copy() to make array writable
        
        # Ensure proper class range
        mask_tensor = torch.clamp(mask_tensor, 0, Config.NUM_CLASSES-1)
        
        return image, mask_tensor

    def _resize_with_padding(self, img, target_size, is_mask=False):
        """Resize with aspect ratio preservation using padding"""
        original_size = img.size
        ratio = min(target_size[0]/original_size[0], target_size[1]/original_size[1])
        new_size = [int(original_size[0]*ratio), int(original_size[1]*ratio)]
        
        # Resize first
        interpolation = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize(new_size, interpolation)
        
        # Calculate padding
        pad_width = target_size[1] - new_size[1]
        pad_height = target_size[0] - new_size[0]
        padding = (pad_width//2, pad_height//2, 
                  pad_width - pad_width//2, pad_height - pad_height//2)
        
        # Apply padding
        if sum(padding) > 0:
            fill = 0 if is_mask else (0, 0, 0)  # Black for images, 0-class for masks
            img = pad(img, padding, fill=fill)
        
        return img

def collate_fn(batch):
    """Custom collate function to handle variable-sized images"""
    images, masks = zip(*batch)
    
    # Stack images (assumes they've been transformed to tensors)
    images = torch.stack(images)
    
    # Find max dimensions for masks
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    
    # Pad masks to match max dimensions
    padded_masks = []
    for mask in masks:
        pad_h = max_h - mask.shape[0]
        pad_w = max_w - mask.shape[1]
        padded_mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0)  # Pad with 0 (background)
        padded_masks.append(padded_mask)
    
    masks = torch.stack(padded_masks)
    return images, masks

def get_transforms():
    return transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])

def initialize_model(num_classes):
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    if Config.USE_GRADIENT_CHECKPOINTING:
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
                
    return model.to(Config.DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None):
    best_val_loss = float('inf')
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        for images, masks in pbar:
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(images)['out']
                
                # Ensure output matches mask dimensions
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs, 
                        size=masks.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)
            
            if pbar.n % 10 == 0:
                torch.cuda.empty_cache()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'iou': train_iou / (pbar.n + 1),
                'mem': f"{torch.cuda.memory_allocated()/1e6:.1f}MB"
            })
        
        # Validation phase
        val_loss, val_iou = validate_model(model, val_loader, criterion)
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_iou'].append(train_iou / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'best_model.pth'))
        
        # Save periodically
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            torch.save(model.state_dict(), 
                      os.path.join(Config.SAVE_DIR, f'epoch_{epoch+1}.pth'))
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)
    
    torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'final_model.pth'))
    return history

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            outputs = model(images)['out']
            
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, 
                    size=masks.shape[-2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_iou += calculate_iou(outputs, masks)
    
    return val_loss / len(val_loader), val_iou / len(val_loader)

def calculate_iou(preds, labels):
    preds = torch.argmax(preds, dim=1)
    ious = []
    
    for cls in range(Config.NUM_CLASSES):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        iou = (intersection / (union + 1e-6)).item()
        ious.append(iou)
    
    return np.nanmean(ious)

def main():
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Initialize data
    transform = get_transforms()
    train_dataset = SegmentationDataset(Config.TRAIN_DIR, transform)
    val_dataset = SegmentationDataset(Config.VAL_DIR, transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = initialize_model(Config.NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Val')
    plt.title('IoU')
    plt.legend()
    
    plt.savefig(os.path.join(Config.SAVE_DIR, 'training_metrics.png'))
    plt.show()

if __name__ == "__main__":
    main()