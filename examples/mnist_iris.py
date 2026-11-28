import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile

import torch
import torch_iris
from torch import nn, optim
from torch.utils.data import DataLoader


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
CHECKPOINT_PATH = "best_model.pth"
LAST_CHECKPOINT_PATH = "last_model.pth"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        # Iris requires a floating-point input for this reduction.
        correct += (outputs.argmax(dim=1) == labels).float().sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).float().sum().item()
        total += labels.size(0)

    return correct / total


def training_state(device):
    model = CNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return model, optimizer, scheduler


def save_checkpoint(path, checkpoint):
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            torch.save(checkpoint, stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(path, model, optimizer, scheduler, shuffle):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    torch_iris.set_rng_state(checkpoint["iris_rng_state"])
    shuffle.set_state(checkpoint["shuffle_rng_state"])
    return checkpoint


def train_epochs(model, optimizer, scheduler, train_loader, test_loader, device,
                 start_epoch, epochs, best_accuracy, output_dir):
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate(model, test_loader, device)
        scheduler.step()
        print(
            f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | "
            f"Train: {train_accuracy:.2%} | Test: {test_accuracy:.2%}", flush=True
        )
        history.append({"epoch": epoch, "loss": train_loss, "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy, "next_learning_rate": scheduler.get_last_lr()[0]})

        improved = test_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, test_accuracy)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "accuracy": test_accuracy,
            "best_accuracy": best_accuracy,
            "cpu_rng_state": torch.get_rng_state(),
            "iris_rng_state": torch_iris.get_rng_state(),
            "shuffle_rng_state": train_loader.generator.get_state(),
        }
        save_checkpoint(output_dir / LAST_CHECKPOINT_PATH, checkpoint)
        if improved:
            save_checkpoint(output_dir / CHECKPOINT_PATH, checkpoint)
    return history


def main():
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description="Train and evaluate the Iris Xe MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if not torch_iris.is_available():
        raise SystemExit("torch-iris could not find an Intel Iris Xe GPU through OpenCL.")

    device = torch.device("iris:0")
    print(f"Using device: {device} ({torch_iris.device_name()})")
    torch.manual_seed(args.seed)
    torch_iris.manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, optimizer, scheduler = training_state(device)
    start_epoch, best_accuracy = 1, -1.0
    if args.resume:
        checkpoint = load_training_checkpoint(args.resume, model, optimizer, scheduler, shuffle)
        start_epoch, best_accuracy = checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
        best_path = args.output_dir / CHECKPOINT_PATH
        if not best_path.is_file():
            previous_best = args.resume.parent / CHECKPOINT_PATH
            if previous_best.is_file():
                copyfile(previous_best, best_path)
            else:
                best_accuracy = checkpoint["accuracy"]
                checkpoint["best_accuracy"] = best_accuracy
                save_checkpoint(best_path, checkpoint)
    if args.epochs < start_epoch:
        parser.error("--epochs must include at least one new training epoch")
    torch_iris.reset_peak_memory_stats()
    torch_iris.reset_fallback_stats()
    history = train_epochs(model, optimizer, scheduler, train_loader, test_loader, device,
                           start_epoch, args.epochs, best_accuracy, args.output_dir)
    checkpoint = torch.load(args.output_dir / CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image, label = test_dataset[0]
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)

    print("Actual:", label)
    print("Predicted:", prediction.item())
    print(f"Confidence: {confidence.item():.2%}")
    if args.report:
        torch_iris.synchronize()
        report = {
            "torch": str(torch.__version__), "torch_iris": torch_iris.__version__,
            "package": torch_iris.__file__, "extension": torch_iris._C.__file__,
            "device": vars(torch_iris.get_device_properties()), "seed": args.seed,
            "train_examples": len(train_dataset), "test_examples": len(test_dataset),
            "epochs": history, "best_epoch": checkpoint["epoch"],
            "best_accuracy": checkpoint["accuracy"], "memory": torch_iris.memory_stats(),
            "cpu_fallback_enabled": torch_iris.is_cpu_fallback_enabled(),
            "cpu_fallbacks": torch_iris.fallback_stats(),
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
