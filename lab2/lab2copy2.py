import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchmetrics import Accuracy

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
    
class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class WiderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
def train_model(model, trainloader, criterion, optimizer, device, num_epochs):
    train_losses = []
    train_accuracies = []

    train_acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_acc_metric.reset()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_acc_metric.update(outputs, labels)

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = train_acc_metric.compute().item()

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Эпоха [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    return train_losses, train_accuracies

def show_predictions(model, testloader, classes, device, num_images, save_path):
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
    for i in range(min(num_images, len(images))):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}',
                         color='green' if predicted[i] == labels[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

    acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)
    batch_accuracy = acc_metric(outputs, labels).item()
    print(f"Точность на этом батче: {100 * batch_accuracy:.2f}%")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            test_acc_metric.update(outputs, labels)

    avg_test_loss = test_loss / len(dataloader)
    test_acc = test_acc_metric.compute().item()
    return avg_test_loss, test_acc

def plot_training_history(train_losses, train_accuracies,  test_loss, test_acc, save_path):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.title('График ошибки (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.title('Точность на обучающем и тестовом наборах')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    test_info = f"Точность на тесте: {100 * test_acc:.2f}%"
    plt.figtext(0.5, 0.01, test_info, ha="center",fontsize=12,
        linespacing=1.8, bbox={"facecolor": "lightgray", "alpha": 0.7, "pad": 6, "edgecolor": "black"})
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"График сохранён как '{save_path}'")

def load_classes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def load_model_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        model_names = [line.strip() for line in f if line.strip()]
    return model_names

def get_model(name: str) -> nn.Module:
    if name == 'simple':
        return SimpleCNN()
    elif name == 'shallow':
        return ShallowCNN()
    elif name == 'wider':
        return WiderCNN()
    else:
        raise ValueError(f"Неизвестная модель: {name}")

def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Обучение и оценка CNN-модели на датасете CIFAR-10."
    )

    parser.add_argument('-dd', '--data_dir',
                        help='Директория для загрузки/хранения датасета CIFAR-10',
                        type=str,
                        dest='data_dir',
                        default='./data')

    parser.add_argument('-cf', '--classes_file',
                        help='Путь к файлу со списком классов (по одному на строку)',
                        type=str,
                        dest='classes_file',
                        default='classes.txt')

    parser.add_argument('-e', '--epochs',
                        help='Количество эпох обучения',
                        type=int,
                        dest='epochs',
                        default=10)

    parser.add_argument('-bs', '--batch_size',
                        help='Размер батча для DataLoader',
                        type=int,
                        dest='batch_size',
                        default=64)
    parser.add_argument('-lr', '--learning_rate',
                        help='Скорость обучения для оптимизатора SGD',
                        type=float,
                        dest='learning_rate',
                        default=0.01)

    parser.add_argument('-mom', '--momentum',
                        help='Моментум для оптимизатора SGD',
                        type=float,
                        dest='momentum',
                        default=0.9)

    parser.add_argument('-msp', '--model_save_path',
                        help='Путь для сохранения весов модели',
                        type=str,
                        dest='model_save_path',
                        default='cifar10_cnn.pth')

    parser.add_argument('-hsp', '--history_save_path',
                        help='Путь для сохранения графика истории обучения',
                        type=str,
                        dest='history_save_path',
                        default='training_results.png')

    parser.add_argument('-psp', '--predictions_save_path',
                        help='Путь для сохранения изображения с предсказаниями',
                        type=str,
                        dest='predictions_save_path',
                        default='predictions.png')
    parser.add_argument('-ni', '--num_images',
                        help='Количество изображений для визуализации предсказаний (макс. размер батча)',
                        type=int,
                        dest='num_images',
                        default=8)
    parser.add_argument('--model',
                        help='Архитектура CNN для обучения',
                        type=str,
                        default='simple')
    parser.add_argument('--compare',
                        help='Обучить и сравнить все модели',
                        action='store_true')
    parser.add_argument('--model_names_file',
                        help='Путь к файлу со списком доступных моделей (по одной на строку)',
                        type=str,
                        default='model_names.txt')

    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    available_models = load_model_names(args.model_names_file)
    print(f"Доступные модели: {available_models}")

    classes = load_classes(args.classes_file)
    print(f"Классы: {classes}")
    print(f"Обучающих изображений: {len(trainset)}")
    print(f"Тестовых изображений: {len(testset)}")

    if args.compare:
        model_names = available_models
    else:
        model_names = [args.model]

    results = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"Обучение модели: {name}")
        print(f"{'='*60}")

        model = get_model(name).to(device)
        print(model)

        model_save_path = f"{name}_cifar10.pth"
        history_save_path = f"{name}_training_results.png"
        predictions_save_path = f"{name}_predictions.png"

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        print("\nОбучение...\n")
        train_losses, train_accuracies = train_model(model, trainloader, criterion, optimizer, device, args.epochs)

        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        results[name] = test_acc
        print(f"\nФинальная точность модели '{name}': {100 * test_acc:.2f}%")


        plot_training_history(train_losses, train_accuracies, test_loss, test_acc, save_path=history_save_path)
        torch.save(model.state_dict(), model_save_path)
        print(f"Модель сохранена как '{model_save_path}'")

        print("\nВизуализация предсказаний...")
        show_predictions(model, testloader, classes, device,num_images=args.num_images, save_path=predictions_save_path)

    if args.compare:
        print("\n" + "="*50)
        print("Сравнение моделей")
        print("="*50)
        for name in model_names:
            print(f"{name:10} → точность: {100 * results[name]:.2f}%")
        print("="*50)

if __name__ == '__main__':
    args = cli_argument_parser()
    main(args)