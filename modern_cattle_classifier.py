"""
Modern Cattle Breed Classifier
------------------------------
Advanced cattle breed classifier using PyTorch with:
- Automatic class detection
- Advanced data augmentation
- Transfer learning with modern architectures
- Class imbalance handling
- Two-phase training (warmup + fine-tuning)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import timm

# Configuration
CONFIG = {
    'data_dir': 'data/train',  # Your training data directory
    'img_size': 224,
    'batch_size': 32,
    'test_size': 0.2,  # 20% for validation
    'random_state': 42,
    
    # Model settings
    'model_name': 'efficientnet_b0',  # Modern, efficient architecture
    'drop_path_rate': 0.2,
    
    # Training settings
    'epochs_warmup': 5,      # Train only classifier head
    'epochs_finetune': 20,   # Fine-tune entire model
    'lr_warmup': 1e-3,       # Learning rate for warmup
    'lr_finetune': 1e-4,     # Learning rate for fine-tuning
    'weight_decay': 1e-4,
    'patience': 7,           # Early stopping patience
    
    # Data augmentation
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'label_smoothing': 0.1,
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def get_transforms(img_size=224, is_training=True):
    """Create data transforms with advanced augmentation"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def setup_data():
    """Setup datasets and data loaders with class imbalance handling"""
    print("Setting up data...")
    
    # Load dataset and automatically detect classes
    train_dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=get_transforms(CONFIG['img_size'], True))
    val_dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=get_transforms(CONFIG['img_size'], False))
    
    classes = train_dataset.classes
    num_classes = len(classes)
    
    print(f"📊 Dataset Information:")
    print(f"   - Found {num_classes} classes: {classes}")
    print(f"   - Total samples: {len(train_dataset)}")
    
    # Check class distribution
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_distribution = dict(zip(classes, class_counts))
    
    print(f"   - Class distribution:")
    for class_name, count in class_distribution.items():
        print(f"     {class_name}: {count} images")
    
    # Stratified split to maintain class balance
    train_idx, val_idx = train_test_split(
        list(range(len(train_dataset.targets))),
        test_size=CONFIG['test_size'],
        stratify=train_dataset.targets,
        random_state=CONFIG['random_state']
    )
    
    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(val_dataset, val_idx)
    
    # Handle class imbalance with weighted sampling
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    sample_weights = [class_weights[t] for t in targets[train_idx]]
    
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   - Training samples: {len(train_ds)}")
    print(f"   - Validation samples: {len(val_ds)}")
    
    return train_loader, val_loader, classes, num_classes

def create_model(model_name, num_classes, drop_path_rate=0.2, pretrained=True):
    """Create model with modern architecture"""
    print(f"Creating model: {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    return model.to(DEVICE)

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch_num, phase="Training"):
    """Training loop for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"{phase} Epoch {epoch_num}")
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast(DEVICE.type):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, epoch_num):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validation Epoch {epoch_num}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast(DEVICE.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions for analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', alpha=0.8)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', alpha=0.8)
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1, 0].plot(history['lr'], label='Learning Rate', color='orange', alpha=0.8)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting indicator
    if len(history['train_loss']) == len(history['val_loss']):
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(loss_diff, label='Val - Train Loss', color='red', alpha=0.8)
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('modern_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix(preds, labels, classes):
    """Create and save confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(max(12, len(classes)), max(10, len(classes))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Modern Cattle Breed Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('modern_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🐄 Modern Cattle Breed Classifier")
    print("=" * 50)
    
    # Setup data
    train_loader, val_loader, classes, num_classes = setup_data()
    
    # Update config with detected classes
    CONFIG['num_classes'] = num_classes
    CONFIG['classes'] = classes
    
    # Create model
    model = create_model(CONFIG['model_name'], num_classes, CONFIG['drop_path_rate'])
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 'lr': []
    }
    
    best_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 50)
    print("🔥 PHASE 1: Warmup Training (Head Only)")
    print("=" * 50)
    
    # Phase 1: Train only classifier head
    for name, param in model.named_parameters():
        if 'head' not in name and 'classifier' not in name:
            param.requires_grad = False
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_warmup'],
        weight_decay=CONFIG['weight_decay']
    )
    
    for epoch in range(CONFIG['epochs_warmup']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch + 1, "Warmup"
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, epoch + 1)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs_warmup']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_warmup.pth')
            patience_counter = 0
        else:
            patience_counter += 1
    
    print("\n" + "=" * 50)
    print("🚀 PHASE 2: Fine-tuning (Full Model)")
    print("=" * 50)
    
    # Phase 2: Fine-tune entire model
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr_finetune'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )
    
    for epoch in range(CONFIG['epochs_finetune']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, 
            CONFIG['epochs_warmup'] + epoch + 1, "Fine-tune"
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, CONFIG['epochs_warmup'] + epoch + 1
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {CONFIG['epochs_warmup'] + epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': CONFIG['epochs_warmup'] + epoch,
                'best_acc': best_acc,
                'config': CONFIG,
                'classes': classes
            }, 'best_modern_model.pth')
            print(f"💾 Saved best model (acc: {best_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print("⏹️ Early stopping triggered")
            break
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    
    model.eval()
    final_val_loss, final_val_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, "Final"
    )
    
    print(f"🎉 Training Completed!")
    print(f"   - Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"   - Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"   - Number of Classes: {num_classes}")
    print(f"   - Total Epochs Trained: {len(history['val_acc'])}")
    
    # Generate reports
    print("\nGenerating classification report...")
    report = classification_report(final_labels, final_preds, target_names=classes)
    print(report)
    
    # Plot results
    plot_training_history(history)
    create_confusion_matrix(final_preds, final_labels, classes)
    
    # Save results
    results = {
        'config': CONFIG,
        'history': history,
        'final_accuracy': final_val_acc,
        'best_accuracy': max(history['val_acc']),
        'classes': classes,
        'classification_report': classification_report(
            final_labels, final_preds, target_names=classes, output_dict=True
        ),
        'training_date': datetime.now().isoformat()
    }
    
    with open('modern_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved:")
    print("   - Model: best_modern_model.pth")
    print("   - Results: modern_training_results.json")
    print("   - Plots: modern_training_history.png, modern_confusion_matrix.png")

if __name__ == "__main__":
    main()