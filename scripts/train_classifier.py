import json
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm
import math
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
MANIFESTS_DIR = REPO_ROOT / 'config' / 'manifests'
EXCLUSION_FILE = REPO_ROOT / 'config' / 'exclusion_list.json'
CLASS_MAP_FILE = REPO_ROOT / 'config' / 'class_map_d2.json'
D2_ROOT = REPO_ROOT / 'Dataset_2'

class LeafDataset(Dataset):
    def __init__(self, images_list, d2_root, transform=None, class_map=None):
        self.images = images_list
        self.d2_root = Path(d2_root)
        self.transform = transform
        self.class_map = class_map
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        item = self.images[idx]
        img_path = self.d2_root / item['path']
        label_name = item['class']
        label = self.class_map[label_name]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label

def ece_score(preds, targets, confidences, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (preds[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return logits / self.temperature

def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    all_preds = []
    all_targets = []
    all_confs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    confs = np.array(all_confs)
    
    macro_f1 = f1_score(targets, preds, average='macro')
    ece = ece_score(preds, targets, confs)
    
    # per-class F1
    per_class_f1 = f1_score(targets, preds, average=None)
    
    return macro_f1, ece, per_class_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to run (default 1 for demo)')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class map
    with open(CLASS_MAP_FILE) as f:
        data = json.load(f)
    
    # Invert the mapping: class_name -> int ID
    class_map = {v: int(k) for k, v in data['classes'].items()}
    num_classes = len(class_map)
    print(f"Classes: {num_classes}")

    # Load manifests
    def load_manifest(split):
        with open(MANIFESTS_DIR / f'd2_{split}.json') as f:
            return json.load(f)['images']
            
    train_imgs = load_manifest('train')
    val_imgs = load_manifest('val')
    test_id_imgs = load_manifest('test_id')
    test_ood_imgs = load_manifest('test_ood')
    
    # Filtered Test OOD
    with open(EXCLUSION_FILE) as f:
        excluded_md5s = set(json.load(f)['excluded_md5s'])
        
    test_ood_filtered_imgs = [img for img in test_ood_imgs if img['md5'] not in excluded_md5s]
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test_ID: {len(test_id_imgs)}")
    print(f"Test_OOD Full: {len(test_ood_imgs)}, Filtered: {len(test_ood_filtered_imgs)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = LeafDataset(train_imgs, D2_ROOT, train_transform, class_map)
    val_ds = LeafDataset(val_imgs, D2_ROOT, val_transform, class_map)
    test_id_ds = LeafDataset(test_id_imgs, D2_ROOT, val_transform, class_map)
    test_ood_full_ds = LeafDataset(test_ood_imgs, D2_ROOT, val_transform, class_map)
    test_ood_filt_ds = LeafDataset(test_ood_filtered_imgs, D2_ROOT, val_transform, class_map)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_id_loader = DataLoader(test_id_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_ood_full_loader = DataLoader(test_ood_full_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_ood_filt_loader = DataLoader(test_ood_filt_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
    
    # We wrap it with temperature scaling
    model = ModelWithTemperature(base_model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # Evaluation
    print("\n--- Evaluation ---")
    val_macro_f1, val_ece, _ = evaluate(model, val_loader, device, "Val")
    print(f"Val -> Macro F1: {val_macro_f1:.4f}, ECE: {val_ece:.4f}")
    
    test_id_macro_f1, test_id_ece, _ = evaluate(model, test_id_loader, device, "Test_ID")
    print(f"Test_ID -> Macro F1: {test_id_macro_f1:.4f}, ECE: {test_id_ece:.4f}")
    
    test_ood_f_macro_f1, test_ood_f_ece, _ = evaluate(model, test_ood_full_loader, device, "Test_OOD (Full)")
    print(f"Test_OOD (Full, D16) -> Macro F1: {test_ood_f_macro_f1:.4f}, ECE: {test_ood_f_ece:.4f}")
    
    test_ood_filt_macro_f1, test_ood_filt_ece, filt_pc = evaluate(model, test_ood_filt_loader, device, "Test_OOD (Filtered)")
    print(f"Test_OOD (Filtered) -> Macro F1: {test_ood_filt_macro_f1:.4f}, ECE: {test_ood_filt_ece:.4f}")
    
    print("\nTraining Complete. Model is not saved in this demo run.")

if __name__ == '__main__':
    main()
