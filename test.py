import torch
from hypercore.optimizers.radamw import RiemannianAdamW
from hypercore.optimizers.radamw_old import RiemannianAdamWOld
from hypercore.optimizers.radam import RiemannianAdam

from hypercore.manifolds import Lorentz
from hypercore.models.LViT import LViT_tiny
from hypercore.models.LSwin_ViT import LSwin_tiny
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from geoopt import Manifold, ManifoldParameter, ManifoldTensor


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
            manifold, manifold, manifold, num_classes=10, image_size=32, patch_size=8
        ).to(device)
    elif model_name == "Swin":
        model = LSwin_tiny(
            manifold, manifold, manifold, num_classes=10, image_size=32, window_size=4
        ).to(device)
    else:
        raise NotImplementedError

    for name, param in model.named_parameters():
        if isinstance(param, (Manifold, Lorentz, ManifoldParameter, ManifoldTensor)):
            param.requires_grad = False
            print(f"Froze parameter: {name}")

    criterion = torch.nn.CrossEntropyLoss()
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
    else:
        raise ValueError("Unknown optimizer name: {}".format(opt_name))

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
        losses = []
        acc1 = []
        acc5 = []

        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.append(loss.item())
                acc1.append(top1.item())
                acc5.append(top5.item())

        with torch.no_grad():
            scheduler.step()
            print(
                "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                    epoch + 1, num_epochs, np.mean(losses), np.mean(acc1), np.mean(acc5)
                )
            )

    model_path = "Lorentz_ViT.pt"
    torch.save(model.state_dict(), model_path)

    print("Testing")
    model.eval()
    losses = []
    acc1 = []
    acc5 = []

    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.append(loss.item())
        acc1.append(top1.item())
        acc5.append(top5.item())

    loss_test = np.mean(losses)
    acc1_test = np.mean(acc1)
    acc5_test = np.mean(acc5)

    print(
        "Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test
        )
    )
    return acc1_test


if __name__ == "__main__":
    # ViT
    vit_acc1_adamw = train("RiemannianAdamW", "ViT")
    vit_acc1_adamw_old = train("RiemannianAdamW_old", "ViT")
    vit_acc1_adam = train("RiemannianAdam", "ViT")
    # Swin
    swin_acc1_adamw = train("RiemannianAdamW", "Swin")
    swin_acc1_adamw_old = train("RiemannianAdamW_old", "Swin")
    swin_acc1_adam = train("RiemannianAdam", "Swin")
    # ViT
    print("ViT:")
    print("Final Test Acc@1 with RiemannianAdamW: {:.4f}".format(vit_acc1_adamw))
    print("Final Test Acc@1 with RiemannianAdamWOld: {:.4f}".format(vit_acc1_adamw_old))
    print("Final Test Acc@1 with RiemannianAdam: {:.4f}".format(vit_acc1_adam))
    # Swin
    print("Swin:")
    print("Final Test Acc@1 with RiemannianAdamW: {:.4f}".format(swin_acc1_adamw))
    print(
        "Final Test Acc@1 with RiemannianAdamWOld: {:.4f}".format(swin_acc1_adamw_old)
    )
    print("Final Test Acc@1 with RiemannianAdam: {:.4f}".format(swin_acc1_adam))
