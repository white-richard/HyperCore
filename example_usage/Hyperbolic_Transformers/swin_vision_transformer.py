import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from geoopt import ManifoldParameter

from hypercore.manifolds import Lorentz
from hypercore.optimizers import RiemannianAdam
from hypercore.models.Swin_LViT import LSwin_base
from hypercore.models.LViT import LViT_base


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def topk_correct(output, target, ks=(1,)):
    """Return number of correct predictions for each k in ks."""
    maxk = max(ks)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]
    res = []
    for k in ks:
        res.append(correct[:k].reshape(-1).float().sum().item())
    return res


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss, correct1, correct5, total = 0.0, 0.0, 0.0, 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        c1, c5 = topk_correct(logits, y, ks=(1, 5))
        correct1 += c1
        correct5 += c5
        total += x.size(0)

    return (
        running_loss / total,
        100.0 * correct1 / total,
        100.0 * correct5 / total,
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct1, correct5, total = 0.0, 0.0, 0.0, 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        c1, c5 = topk_correct(logits, y, ks=(1, 5))
        correct1 += c1
        correct5 += c5
        total += x.size(0)

    return (
        running_loss / total,
        100.0 * correct1 / total,
        100.0 * correct5 / total,
    )


def main(model_name:str):
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    # CIFAR 10
    """
    LSwin Tiny Best 
    Val@1: 79.41
    Testing (final checkpoint on disk)…
    Results: Loss=0.7792, Acc@1=79.41, Acc@5=98.35
    """
    """
    LViT Tiny Best
    5=97.41
    Best Val@1: 70.19
    Testing (final checkpoint on disk)…
    Results: Loss=0.9133, Acc@1=70.19, Acc@5=97.27
    """
    # CIFAR 100
    """LViT
    5=66.05
    Best Val@1: 37.47
    Testing (final checkpoint on disk)…
    Results: Loss=2.6469, Acc@1=37.47, Acc@5=66.21
    """
    """Swin LViT
    Best Val@1: 44.54
    Testing (final checkpoint on disk)…
    Results: Loss=2.6972, Acc@1=44.54, Acc@5=70.84
    """

    # BASE MODELS
    

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(magnitude=14),
        transforms.ToTensor(),
        transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
	])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
    ])

    data_root = "hypercore/data"
    train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True,  num_workers=8, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    manifold = Lorentz(1.0)

    if model_name=='LViT':
        # CIFAR-10: image_size=32, patch_size=4 → 8×8 tokens; window_size=4 is a good default
        model = LSwin_base(
             manifold_in=manifold,
             manifold_hidden=manifold,
             manifold_out=manifold,
             image_size=32,
             patch_size=4,
             num_classes=100,
             window_size=4, # 4×4 windows inside 8×8 grid
             embed_dim=33,
             dropout=0.0,
        ).to(device)
    elif model_name == 'LSwin':
        model = LViT_base(
            manifold_in=manifold,
            manifold_hidden=manifold,
            manifold_out=manifold,
        image_size=32,
        num_classes=100
        ).to(device)
    elif model_name == 'ESwin':
        from euclidean_swin_ViT import swin_base
        model = swin_base(
            image_size=32,
            num_classes=100
        ).to(device)

    for p in model.parameters():
        if isinstance(p, ManifoldParameter):
            p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()

    epochs = 400

    optimizer = RiemannianAdam(model.parameters(),lr = 1e-4, stabilize=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=optimizer.param_groups[0]['lr'] * 1e-3
    )
    print(optimizer)
    print(model)

    best_acc1 = 0.0
    save_path = "Lorentz_SwinViT_cifar10.pt"

    for epoch in range(epochs):
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, device, optimizer, criterion
        )

        scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate(model, test_loader, device, criterion)

        print(f"Epoch {epoch+1:03d}/{epochs}: "
              f"train_loss={train_loss:.4f}  train@1={train_acc1:.2f}  train@5={train_acc5:.2f} | "
              f"val_loss={val_loss:.4f}  val@1={val_acc1:.2f}  val@5={val_acc5:.2f}")

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            torch.save(model.state_dict(), save_path)

    print(f"Best Val@1: {best_acc1:.2f}")
    print("Testing (final checkpoint on disk)…")

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc1, test_acc5 = evaluate(model, test_loader, device, criterion)
    print(f"Results: Loss={test_loss:.4f}, Acc@1={test_acc1:.2f}, Acc@5={test_acc5:.2f}")


if __name__ == "__main__":
    model_names = ['LViT', 'LSwin','ESwin']
    for model_name in model_names:
        print(f"Training model: {model_name}...\n")
        main(model_name)
        print("\n\n")
