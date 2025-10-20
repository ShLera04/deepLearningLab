import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def show_predictions(model, testloader, classes, device, num_images=8):
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    images = images.cpu()
    images = images / 2 + 0.5
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}',
                         color='green' if predicted[i] == labels[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / len(labels)
    print(f"Точность на этом батче: {accuracy:.2f}%")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Обучающих изображений: {len(trainset)}")
    print(f"Тестовых изображений: {len(testset)}")

    net = SimpleCNN().to(device)
    print("\nАрхитектура модели:")
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    train_losses = []
    train_accuracies = []
    test_losses = []

    print("\nОбучение...\n")

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                test_loss += criterion(outputs, labels).item()
        avg_test_loss = test_loss / len(testloader)
        test_losses.append(avg_test_loss)

        print(f"Эпоха [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} "
              f"Train Acc: {epoch_acc:.4f} "
              f"Test Loss: {avg_test_loss:.4f}")

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = correct / total
    print(f"\nТочность: {100 * final_accuracy:.2f}%")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.title('График ошибки (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green', marker='o')
    plt.title('Точность на обучающем наборе')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\nГрафик сохранён, как 'training_results.png'")

    torch.save(net.state_dict(), 'cifar10_cnn.pth')
    print("Модель сохранена, как 'cifar10_cnn.pth'")

    print("\nВизуализация ...")
    show_predictions(net, testloader, classes, device)