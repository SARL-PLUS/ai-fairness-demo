"""
🐱🐶 Katze-oder-Hund Klassifikation — Training-Skript
=====================================================
Trainiert ein vortrainiertes ResNet-18 zur Unterscheidung von Katzen und Hunden.
Verwendet den Oxford-IIIT Pet Dataset (wird automatisch heruntergeladen).

Verwendung:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ─── Konfiguration ────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.001,
    "image_size": 224,
    "num_workers": 0,           # 0 = kein Multiprocessing (sicher auf jedem OS)
    "max_train_samples": 1000,  # Kleines Subset für schnelles Training
    "max_test_samples": 200,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


# ─── 1. Daten vorbereiten ─────────────────────────────────────────────────────

def get_cat_dog_label(original_label: int) -> int:
    """Oxford-IIIT Pet hat 37 Rassen. Klassen 0-11 = Katzen, 12-36 = Hunde."""
    # The dataset class_to_idx maps breed names; breeds are sorted alphabetically.
    # Cat breeds: Abyssinian, Bengal, Birman, Bombay, British_Shorthair,
    #             Egyptian_Mau, Maine_Coon, Persian, Ragdoll, Russian_Blue,
    #             Siamese, Sphynx  → indices 0-11
    return 0 if original_label < 12 else 1   # 0 = Katze, 1 = Hund


class CatDogDataset(torch.utils.data.Dataset):
    """Wrapper um OxfordIIITPet der die 37 Rassen auf 2 Klassen abbildet."""

    def __init__(self, root, split="trainval", transform=None, download=True):
        self.dataset = datasets.OxfordIIITPet(
            root=root, split=split, target_types="category",
            transform=transform, download=download,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, get_cat_dog_label(label)


def build_dataloaders(cfg):
    """Erstellt Train- und Test-DataLoader mit Augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    print("📥 Lade Oxford-IIIT Pet Dataset (wird beim ersten Mal heruntergeladen)...")
    train_full = CatDogDataset(cfg["data_dir"], split="trainval",
                               transform=train_transform, download=True)
    test_full = CatDogDataset(cfg["data_dir"], split="test",
                              transform=test_transform, download=True)

    # Kleines Subset für schnelles Training im Workshop
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(train_full), min(cfg["max_train_samples"], len(train_full)), replace=False)
    test_idx = rng.choice(len(test_full), min(cfg["max_test_samples"], len(test_full)), replace=False)

    train_set = Subset(train_full, train_idx)
    test_set = Subset(test_full, test_idx)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    print(f"✅ Trainingsdaten: {len(train_set)} Bilder | Testdaten: {len(test_set)} Bilder")
    return train_loader, test_loader


# ─── 2. Modell erstellen ──────────────────────────────────────────────────────

def build_model():
    """ResNet-18 mit vortrainierten Gewichten, angepasst auf 2 Klassen."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Alle Schichten einfrieren (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False

    # Letzte Schicht ersetzen (2 Ausgänge: Katze / Hund)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2),
    )
    return model.to(DEVICE)


# ─── 3. Training ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Evaluation", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train(model, train_loader, test_loader, cfg):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=cfg["learning_rate"])
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print(f"\n🚀 Starte Training auf {DEVICE} ({cfg['epochs']} Epochen)...\n")
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"  Epoche {epoch}/{cfg['epochs']}  |  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.1%}  |  "
              f"Test  Loss: {te_loss:.4f}  Acc: {te_acc:.1%}")

    print(f"\n🏁 Fertig! Beste Test-Accuracy: {max(history['test_acc']):.1%}")
    return history


# ─── 4. Visualisierung ────────────────────────────────────────────────────────

def plot_history(history):
    """Zeichnet Loss- und Accuracy-Kurven."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["test_loss"], label="Test")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoche")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["test_acc"], label="Test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoche")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "training_curves.png"), dpi=150)
    plt.show()
    print("📊 Trainings-Kurven gespeichert als training_curves.png")


def show_predictions(model, test_loader, n=8):
    """Zeigt einige Testbilder mit Vorhersagen."""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:n].to(DEVICE), labels[:n]
    with torch.no_grad():
        preds = model(images).argmax(1).cpu()

    # De-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    class_names = ["🐱 Katze", "🐶 Hund"]

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, ax in enumerate(axes):
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"{class_names[preds[i]]}", color=color, fontsize=11)
        ax.axis("off")
    plt.suptitle("Grün = richtig, Rot = falsch", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "predictions.png"), dpi=150)
    plt.show()
    print("🖼️  Vorhersagen gespeichert als predictions.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, test_loader = build_dataloaders(CONFIG)
    model = build_model()

    print(f"\n📐 Modell: ResNet-18 (vortrainiert)")
    print(f"   Trainierbare Parameter: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = train(model, train_loader, test_loader, CONFIG)
    plot_history(history)
    show_predictions(model, test_loader)

    # Modell speichern
    save_path = os.path.join(os.path.dirname(__file__), "cat_dog_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modell gespeichert: {save_path}")
