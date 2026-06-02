"""
📥 Daten herunterladen
=====================
Lädt den Oxford-IIIT Pet Dataset herunter und zeigt einige Beispielbilder.
Verwendung:
    python download_data.py
"""

import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_and_preview():
    """Lädt den Datensatz herunter und zeigt 12 Beispielbilder."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("📥 Lade Oxford-IIIT Pet Dataset herunter...")
    dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", target_types="category",
        transform=transform, download=True,
    )
    print(f"✅ {len(dataset)} Bilder heruntergeladen nach: {DATA_DIR}")
    print(f"   Rassen (Klassen): {len(dataset.classes)}")
    print(f"   Katzen-Rassen (Index 0-11): {', '.join(dataset.classes[:12])}")
    print(f"   Hunde-Rassen (Index 12+):   {', '.join(dataset.classes[12:17])}...")

    # Zeige 12 Beispielbilder
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i * (len(dataset) // 12)]
        breed = dataset.classes[label]
        is_cat = label < 12
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"{'🐱' if is_cat else '🐶'} {breed}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Beispielbilder aus dem Oxford-IIIT Pet Dataset", fontsize=14)
    plt.tight_layout()

    preview_path = os.path.join(os.path.dirname(__file__), "dataset_preview.png")
    plt.savefig(preview_path, dpi=150)
    plt.show()
    print(f"🖼️  Vorschau gespeichert: {preview_path}")


if __name__ == "__main__":
    download_and_preview()
