import torch
import torchvision
import torchvision.transforms as transforms
import logging
import argparse
import sys
from torch import nn, optim
from torchvision.models import resnet50

#Usage Instructions
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on CIFAR-10 using PyTorch and ROCm.")
    parser.add_argument("--log-file", type=str, default="training.log",
                        help="The file path where logs will be saved. Default: training.log")
    return parser.parse_args()

#Logging configuration
def configure_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout

#Training
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    logging.info("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Log every 100 batches
                logging.info(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    logging.info("Training completed.")

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    logging.info("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    args = parse_args()
    configure_logging(args.log_file)

    logging.info("Starting ResNet-50 training on CIFAR-10 with PyTorch and ROCm.")

    #Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.error("No GPU devices found. Ensure ROCm is properly configured.")
        raise RuntimeError("No GPU devices found.")
    logging.info(f"Using device: {device}")

    #Data prep
    logging.info("Preparing CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        #from cifar10 precomputed values
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 64

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Build the model
    logging.info("Building ResNet-50 model...")
    model = resnet50(pretrained=False, num_classes=10)
    model = model.to(device)

    #Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train and test the model
    train(model, train_loader, criterion, optimizer, device, epochs=10)
    accuracy = test(model, test_loader, device)

    logging.info(f"Training and evaluation completed. Final accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()