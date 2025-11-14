"""
Epoch 60/60: TripletLoss=0.8544, Avg #Triplets=30.7
Testing
Test Results: TripletLoss=0.8544, Avg #Triplets=30.7
Results: TripletLoss=0.8544, Avg #Triplets=30.7, kNN@5 Acc=34.37%
ViT:
Final Test Knn Acc@1 with AdamW: 10.1100
Swin:
Final Test Knn Acc@1 with AdamW: 34.3700
"""
import torch
from hypercore.optimizers.radamw import RiemannianAdamW
from hypercore.optimizers.radamw_old import RiemannianAdamWOld
from hypercore.optimizers.radam import RiemannianAdam
from torch.optim import AdamW

from hypercore.manifolds import Lorentz
from hypercore.models.LViT import LViT_tiny
from hypercore.models.LSwin_ViT import LSwin_tiny
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from geoopt import Manifold, ManifoldParameter, ManifoldTensor

import torch
from hypercore.manifolds import Lorentz


@torch.no_grad()
def compute_knn_accuracy(model, train_loader, test_loader, device, k=5):
    model.eval()

    # 1. Collect all train embeddings and labels
    train_embeddings = []
    train_labels = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        emb = model(x)
        train_embeddings.append(emb)
        train_labels.append(y)

    train_embeddings = torch.cat(train_embeddings, dim=0)  # [N_train, D]
    train_labels = torch.cat(train_labels, dim=0)          # [N_train]

    # 2. For each test batch, compute distances to all train embeddings
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)                    # [B]
        test_emb = model(x)                 # [B, D]

        # pairwise distances between test and all train embeddings
        # result: [B, N_train]
        dists = torch.cdist(test_emb, train_embeddings)

        # get indices of k nearest neighbors in train set
        knn_dists, knn_indices = torch.topk(dists, k=k, dim=1, largest=False)

        # find labels of these neighbors: [B, k]
        knn_labels = train_labels[knn_indices]

        # majority vote among neighbors
        # mode returns (values, counts)
        preds, _ = torch.mode(knn_labels, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total
    return acc


class TangetHead(torch.nn.Module):
    def __init__(self, manifold: Lorentz, backbone: torch.nn.Module):
        super().__init__()
        self.backbone: torch.nn.Module = backbone
        self.manifold: Lorentz = manifold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        proj_feats = self.manifold.projx(feats)
        ambient_tanget_feats = self.manifold.logmap0(proj_feats)
        # remove time dim 
        tanget_feats = ambient_tanget_feats[:, 1:]
        return tanget_feats 


import torch
from pytorch_metric_learning import losses, miners, distances, reducers


class TripletLossWithMiner(torch.nn.Module):
    def __init__(
        self,
        margin=0.05,
        type_of_triplets="batchhard",
        type_of_reducer="AvgNonZeroReducer",
        normalize_embeddings=True,
        use_soft_margin=True,
        swap=False,
    ):
        super().__init__()
        self.margin = float(margin)
        self.miner = None

        print(f"Using Triplet Loss with margin: {self.margin}\n DEBUG: try (0.0,0.2,1.0) margin")

        if type_of_reducer == "MeanReducer":
            reducer = reducers.MeanReducer()
        elif type_of_reducer == "AvgNonZeroReducer":
            reducer = reducers.AvgNonZeroReducer()
        else:
            raise NotImplementedError

        distance = distances.LpDistance(
            p=2, power=1, normalize_embeddings=normalize_embeddings
        )
        self.loss = losses.TripletMarginLoss(
            margin=margin, reducer=reducer, distance=distance, smooth_loss=use_soft_margin, swap=swap
        )
        if type_of_triplets is not None:
            if type_of_triplets == "batchhard":
                self.miner = miners.BatchHardMiner(distance=distance)
            else:
                self.miner = miners.TripletMarginMiner(
                    margin=margin,
                    type_of_triplets=type_of_triplets,
                    distance=distance
                )

    def forward(self, embeddings, labels):
        if self.miner is not None:
            a, p, n = self.miner(embeddings, labels)
            return self.loss(embeddings, labels, (a, p, n)), len(a)
        else:
            return self.loss(embeddings, labels)



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(opt_name: str, model_name: str):
    # import dataset
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ]
    )

    train_set = datasets.CIFAR10(
        "hypercore/data", train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        "hypercore/data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, num_workers=8, pin_memory=True, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, num_workers=8, pin_memory=True, shuffle=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    manifold = Lorentz(1.0)
    if model_name == "ViT":
        model = LViT_tiny(
            manifold, manifold, manifold, num_classes=0, image_size=32, patch_size=8
        ).to(device)
    elif model_name == "Swin":
        model = LSwin_tiny(
            manifold, manifold, manifold, num_classes=0, image_size=32, window_size=4
        ).to(device)
    else:
        raise NotImplementedError

    for name, param in model.named_parameters():
        if isinstance(param, (Manifold, Lorentz, ManifoldParameter, ManifoldTensor)):
            param.requires_grad = False
            print(f"Froze parameter: {name}")

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = TripletLossWithMiner(margin=0.3)

    learning_rate = 0.0005

    if opt_name == "RiemannianAdamW":
        optimizer = RiemannianAdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.05, stabilize=1
        )
    elif opt_name == "RiemannianAdamW_old":
        optimizer = RiemannianAdamWOld(
            model.parameters(), lr=learning_rate, weight_decay=0.05, stabilize=1
        )
    elif opt_name == "RiemannianAdam":
        optimizer = RiemannianAdam(
            model.parameters(), lr=learning_rate, weight_decay=0.05, stabilize=1
        )
    elif opt_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    else:
        raise ValueError("Unknown optimizer name: {}".format(opt_name))

    model = TangetHead(model.manifold, model)

    print(optimizer)
    print(str(model))
    num_epochs = 60
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - 5, eta_min=learning_rate * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[5]
    )

    for epoch in range(0, num_epochs):
        model.train()
        batch_losses = []
        batch_num_triplets = []

        for _, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = x.to(device)
            y = y.to(device)

            # embeddings, not logits
            embeddings = model(x)

            # TripletLossWithMiner returns (loss, num_triplets) when miner is used
            loss, num_triplets = criterion(embeddings, y)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.get_total_norm(model.parameters(), norm_type=2)
            print(f"Gradient Norm: {grad_norm}")
            optimizer.step()

            with torch.no_grad():
                batch_losses.append(loss.item())
                batch_num_triplets.append(num_triplets)

        with torch.no_grad():
            scheduler.step()
            avg_loss = float(np.mean(batch_losses))
            avg_triplets = float(np.mean(batch_num_triplets))
            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"TripletLoss={avg_loss:.4f}, "
                f"Avg #Triplets={avg_triplets:.1f}"
            )

    model_path = "Lorentz_ViT.pt"
    torch.save(model.state_dict(), model_path)

    print("Testing")
    model.eval()
    test_losses = []
    test_num_triplets = []

    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        embeddings = model(x)
        loss, num_triplets = criterion(embeddings, y)

        test_losses.append(loss.item())
        test_num_triplets.append(num_triplets)

    loss_test = float(np.mean(test_losses))
    avg_triplets_test = float(np.mean(test_num_triplets))

    print(
        "Test Results: TripletLoss={:.4f}, Avg #Triplets={:.1f}".format(
            loss_test, avg_triplets_test
        )
    )
    knn_acc = compute_knn_accuracy(model, train_loader, test_loader, device, k=5)
    print(
        "Results: TripletLoss={:.4f}, Avg #Triplets={:.1f}, kNN@5 Acc={:.2f}%".format(
            loss_test, avg_triplets_test, knn_acc
        )
    )
    return knn_acc


if __name__ == "__main__":
    # ViT
    vit_acc1_adamw = train("AdamW", "ViT")
    # Swin
    swin_acc1_adamw = train("AdamW", "Swin")
    # ViT
    print("ViT:")
    print("Final Test Knn Acc@1 with AdamW: {:.4f}".format(vit_acc1_adamw))
    # Swin
    print("Swin:")
    print("Final Test Knn Acc@1 with AdamW: {:.4f}".format(swin_acc1_adamw))
