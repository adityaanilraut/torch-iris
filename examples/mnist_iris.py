import torch
import torch_iris
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
CHECKPOINT_PATH = "best_model.pth"


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


def main():
    if not torch_iris.is_available():
        raise SystemExit("torch-iris could not find an Intel Iris Xe GPU through OpenCL.")

    device = torch.device("iris:0")
    print(f"Using device: {device} ({torch_iris.device_name()})")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_accuracy = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate(model, test_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | "
            f"Train: {train_accuracy:.2%} | Test: {test_accuracy:.2%}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "accuracy": best_accuracy,
            }, CHECKPOINT_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
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


if __name__ == "__main__":
    main()
