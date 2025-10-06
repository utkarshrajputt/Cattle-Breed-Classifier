"""
PyTorch MobileNetV2 Cattle Breed Classifier - Fair Comparison
===========================================================
Using PyTorch + GPU for fair comparison with EfficientNet-B0
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import json
from datetime import datetime
import timm
from tqdm import tqdm

# Configuration - Same as EfficientNet for fair comparison
CONFIG = {
    'data_dir': 'data/train',
    'img_size': 224,
    'batch_size': 16 if torch.cuda.is_available() else 8,
    'epochs': 25,  # Same as EfficientNet
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'val_split': 0.2,
    'model_name': 'mobilenetv2_100',  # MobileNetV2 from timm
    'num_workers': 2 if torch.cuda.is_available() else 0,
}

# Device configuration - Same as EfficientNet
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {DEVICE}")

# GPU Information
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"⚡ CUDA Version: {torch.version.cuda}")
    
    # Enable optimizations for GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    print("⚠️  No GPU detected, using CPU (training will be slower)")

def get_transforms(img_size=224, is_training=True):
    """Create data transforms - Same as EfficientNet"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def create_data_loaders():
    """Create data loaders - Same setup as EfficientNet"""
    print("📁 Loading dataset...")
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(
        CONFIG['data_dir'],
        transform=get_transforms(CONFIG['img_size'], True)
    )
    
    # Create validation dataset with different transforms
    val_dataset = datasets.ImageFolder(
        CONFIG['data_dir'],
        transform=get_transforms(CONFIG['img_size'], False)
    )
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(CONFIG['val_split'] * total_size)
    train_size = total_size - val_size
    
    train_dataset, _ = random_split(full_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, full_dataset.classes

def create_model(num_classes):
    """Create MobileNetV2 model using timm"""
    print(f"🧠 Creating {CONFIG['model_name']} model...")
    
    model = timm.create_model(
        CONFIG['model_name'],
        pretrained=True,
        num_classes=num_classes
    )
    
    model = model.to(DEVICE)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def train_epoch(model, train_loader, optimizer, criterion, epoch):
    """Training loop"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        
        for data, targets in progress_bar:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    """Main training function"""
    print("🚀 Starting PyTorch MobileNetV2 Training")
    print("=" * 60)
    
    # Create data loaders
    train_loader, val_loader, classes = create_data_loaders()
    num_classes = len(classes)
    
    print(f"📊 Dataset Information:")
    print(f"   Classes: {num_classes}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    
    # Create model
    model = create_model(num_classes)
    
    # Loss and optimizer - Same as EfficientNet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler - Same as EfficientNet
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG['epochs'],
        eta_min=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    
    print("\n🚀 Starting Training")
    print("=" * 50)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"GPU Memory Used: {gpu_memory:.2f} GB")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': CONFIG,
                'classes': classes
            }, 'mobilenetv2_best_model.pth')
            print(f"💾 Saved best model (acc: {best_acc:.2f}%)")
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    # Save results
    results = {
        'model_type': 'MobileNetV2-PyTorch',
        'model_name': CONFIG['model_name'],
        'num_classes': num_classes,
        'classes': classes,
        'best_accuracy': best_acc,
        'final_accuracy': history['val_acc'][-1],
        'history': history,
        'config': CONFIG,
        'training_date': datetime.now().isoformat()
    }
    
    with open('mobilenetv2_pytorch_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("💾 Results saved to 'mobilenetv2_pytorch_results.json'")
    print("💾 Best model saved to 'mobilenetv2_best_model.pth'")
    
    print("📈 Training completed successfully! Use compare_models.py to see results.")

if __name__ == "__main__":
    main()