import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchviz
import os

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_test_loss = test_loss / len(dataloader)
    test_acc = correct / total
    return avg_test_loss, test_acc

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def create_transfer_model(model_name, num_classes=10, pretrained=True, freeze_backbone=True):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        backbone_params = model.features.parameters()

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        backbone_params = model.features.parameters()

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    if freeze_backbone:
        for param in backbone_params:
            param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        init_info = "ImageNet веса, backbone заморожен"
    else:
        trainable_params = model.parameters()
        init_info = "ImageNet веса, полное обучение"

    return model, trainable_params, init_info

def run_experiment(model_name, freeze_backbone, trainloader, testloader, device, num_epochs):
    model, trainable_params, init_info = create_transfer_model(
        model_name, num_classes=10, pretrained=True, freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    if model_name == 'vgg16':
        modified_layer_info = f"VGG16 Classifier[6]: {model.classifier[6].in_features} -> {model.classifier[6].out_features}"
    elif model_name in ['resnet18', 'resnet50']:
        modified_layer_info = f"ResNet FC: {model.fc.in_features} -> {model.fc.out_features}"
    elif model_name == 'efficientnet_b0':
        modified_layer_info = f"EfficientNet Classifier[1]: {model.classifier[1].in_features} -> {model.classifier[1].out_features}"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trainable_params, lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

    final_acc = calculate_accuracy(model, testloader, device)
    exp_type = "classifier_only" if freeze_backbone else "full_finetune"
    return {
        'model': model_name,
        'experiment': exp_type,
        'accuracy': final_acc,
        'init_info': init_info,
        'optimizer': 'SGD (lr=0.01, momentum=0.9)',
        'modified_layer': modified_layer_info,
        'epochs': num_epochs
    }

def plot_comparison_histogram(results, save_path='transfer_learning_comparison.png'):
    labels = []
    accuracies = []
    for r in results:
        exp_label = "clf" if r['experiment'] == 'classifier_only' else "full"
        labels.append(f"{r['model']}\n({exp_label})")
        accuracies.append(r['accuracy'])

    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, accuracies, color='steelblue')
    plt.ylabel('Точность (Accuracy)', fontsize=12)
    plt.title('Сравнение моделей и стратегий переноса обучения на CIFAR-10', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_detailed_results(results):
    print("Итоговые результаты")

    models_dict = {}
    for result in results:
        model_name = result['model']
        if model_name not in models_dict:
            models_dict[model_name] = []
        models_dict[model_name].append(result)
    
    for model_name, experiments in models_dict.items():
        print(f"\nМОДЕЛЬ: {model_name.upper()}")
        
        modified_layer = experiments[0]['modified_layer']
        print(f"   Модифицированный слой: {modified_layer}")
        print(f"   Эпох обучения: {experiments[0]['epochs']}")
        print(f"   Оптимизатор: {experiments[0]['optimizer']}")
        print()
        
        for exp in experiments:
            exp_type = "Только классификатор" if exp['experiment'] == 'classifier_only' else "Полное дообучение"
            accuracy = exp['accuracy']
            init_info = exp['init_info']
            
            print(f"   - {exp_type}:")
            print(f"      Инициализация: {init_info}")
            print(f"      Точность: {100 * accuracy:.2f}%")

def print_best_result(results):
    if not results:
        return
        
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("Наилучшая модель")
    print(f"Исходная архитектура: {best_result['model']}")
    print(f"Модифицированный слой: {best_result['modified_layer']}")
    exp_desc = "Обучение только классификатора" if best_result['experiment'] == 'classifier_only' else "Полное дообучение"
    print(f"Тип эксперимента:     {exp_desc}")
    print(f"Инициализация весов:  {best_result['init_info']}")
    print(f"Оптимизатор:          {best_result['optimizer']}")
    print(f"Эпох обучения:        {best_result['epochs']}")
    print(f"Точность на тесте:    {100 * best_result['accuracy']:.2f}%")

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

    print(f"Обучающих изображений: {len(trainset)}")
    print(f"Тестовых изображений: {len(testset)}")

    model_names = ['vgg16', 'resnet50', 'resnet18', 'efficientnet_b0']
    results = []

    for model_name in model_names:
        print(f"\nЗапуск экспериментов для {model_name}...")
        try:
            res1 = run_experiment(model_name, freeze_backbone=True,
                                    trainloader=trainloader, testloader=testloader,
                                    device=device, num_epochs=1)
            results.append(res1)
            print(f"classifier_only: {res1['accuracy']:.4f}")

            res2 = run_experiment(model_name, freeze_backbone=False,
                                    trainloader=trainloader, testloader=testloader,
                                    device=device, num_epochs=1)
            results.append(res2)
            print(f"full_finetune:   {res2['accuracy']:.4f}")
        except Exception as e:
            print(f"Ошибка при обучении {model_name}: {e}")

    if not results:
        print("Ни одна модель не была успешно обучена.")
    else:
        best_result = max(results, key=lambda x: x['accuracy'])

        plot_comparison_histogram(results, save_path='transfer_learning_comparison.png')

        print_detailed_results(results)
        
        print_best_result(results)
