import torch
from hypercore.manifolds import Lorentz
from timm.models.vision_transformer import VisionTransformer
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from hypercore.optimizers.initialize import LR_SchedulerCosineWarmup, Optimizer
from example_usage.Hyperbolic_Transformers.vision_transformer import LViT
from calflops import calculate_flops
from torch.utils.tensorboard import SummaryWriter
import shutil


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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(output_dir, lr, is_hyperbolic: bool = False, writer=None):
    set_seed(42)

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
        "data", train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        "data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    manifold = Lorentz(1.0)
    if is_hyperbolic:
        model = LViT(
            manifold_in=manifold,
            manifold_hidden=manifold,
            manifold_out=manifold,
            out_channel=10,
            dropout=0.0,
        )
    else:
        model = VisionTransformer(
            num_classes=10,
            patch_size=4,
            embed_dim=260,
            depth=6,
            num_heads=4,
            mlp_ratio=4,
            img_size=32,
            in_chans=3,
        )

    model.to(device)

    input_shape = (1, 3, 32, 32)
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=2,
        print_detailed=False,
        print_results=False,
    )
    print(f"FLOPs: {flops}, Params: {params}, MACs: {macs}")
    writer.add_text(
        tag="Model_FLOPs_Params",
        text_string=f"FLOPs: {flops}, Params: {params}, MACs: {macs}",
    )

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = lr

    optimizer = Optimizer(
        model,
        euc_optimizer_type="adamW",
        euc_lr=learning_rate,
        hyp_optimizer_type="radam" if is_hyperbolic else None,
        euc_weight_decay=0.01,
        hyp_weight_decay=0.0075,
        hyp_lr=learning_rate,
        stabilize=1,
    )

    num_epochs = 100
    scheduler = LR_SchedulerCosineWarmup(
        optimizer,
        total_steps_euc=len(train_loader) * num_epochs,
        total_steps_hyp=len(train_loader) * num_epochs,
    )

    results = {
        "train_loss": [],
        "train_acc1": [],
        "train_acc5": [],
        "test_loss": [],
        "test_acc1": [],
        "test_acc5": [],
    }

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
            scheduler.step()

            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.append(loss.item())
                acc1.append(top1.item())
                acc5.append(top5.item())

        with torch.no_grad():
            print(
                "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@3={:.4f}".format(
                    epoch + 1, num_epochs, np.mean(losses), np.mean(acc1), np.mean(acc5)
                )
            )
            results["train_loss"].append(np.mean(losses))
            results["train_acc5"].append(np.mean(acc5))
            results["train_acc1"].append(np.mean(acc1))

            model_prefix = "Hyp" if is_hyperbolic else "Euc"
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={f"{model_prefix}_train_loss": np.mean(losses)},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy_train_acc1",
                tag_scalar_dict={f"{model_prefix}_train_acc1": np.mean(acc1)},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy_train_acc5",
                tag_scalar_dict={f"{model_prefix}_train_acc5": np.mean(acc5)},
                global_step=epoch,
            )
            writer.flush()

    if is_hyperbolic:
        model_path = "Hyp_LViT.pt"
    else:
        model_path = "Euc_ViT.pt"
    model_path = f"{output_dir}/{model_path}"
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
    results["test_loss"].append(loss_test)
    results["test_acc1"].append(acc1_test)
    results["test_acc5"].append(acc5_test)

    model_prefix = "Hyp" if is_hyperbolic else "Euc"
    writer.add_scalars(
        main_tag="Loss",
        tag_scalar_dict={f"{model_prefix}_test_loss": loss_test},
        global_step=epoch,
    )
    writer.add_scalars(
        main_tag="Accuracy_test_acc1",
        tag_scalar_dict={f"{model_prefix}_test_acc1": acc1_test},
        global_step=epoch,
    )
    writer.add_scalars(
        main_tag="Accuracy_test_acc5",
        tag_scalar_dict={f"{model_prefix}_test_acc5": acc5_test},
        global_step=epoch,
    )

    writer.flush()

    print(
        "Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test
        )
    )
    return results


if __name__ == "__main__":
    from pathlib import Path

    learning_rates = [3e-4, 1e-3, 3e-3, 1e-4]
    all_results = {}
    output_dir = "./results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for lr in learning_rates:
        print("Training with learning rate =", lr)
        lr_output_dir = f"{output_dir}/LR_{lr}"
        Path(lr_output_dir).mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=lr_output_dir)

        for is_hyp in [True, False]:
            model_type = "Hyp" if is_hyp else "Euc"
            print(f"Training {model_type} model with learning rate = {lr}")

            results = train(lr_output_dir, lr, is_hyperbolic=is_hyp, writer=writer)
            all_results[f"{model_type}_LR_{lr}"] = results

            with open(f"{lr_output_dir}/results_table.txt", "a") as f:
                for key, value in results.items():
                    f.write(f"{model_type}_LViT, LR={lr}, {key}, {value[-1]}\n")

        shutil.copy(__file__, f"{lr_output_dir}/train_script.py")

        writer.close()