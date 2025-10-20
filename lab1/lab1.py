import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import time
import sys
import os


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.W1 = np.random.uniform(-0.1, 0.1, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.output_size))

        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = self.relu(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.output = self.softmax(self.output_input)
        
        return self.hidden_output, self.output
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        dZ2 = y_pred - y_true         

        dW2 = (1/m) * np.dot(self.hidden_output.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.hidden_input)
        
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def predict(self, X):
        _, output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

def load_mnist(data_dir='data'):
    train_path = os.path.join(data_dir, 'mnist_train.csv')
    test_path = os.path.join(data_dir, 'mnist_test.csv')
    
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    y_train = train_df.iloc[:, 0].values.astype(np.int32)
    X_train = train_df.iloc[:, 1:].values.astype(np.float32)
    
    y_test = test_df.iloc[:, 0].values.astype(np.int32)
    X_test = test_df.iloc[:, 1:].values.astype(np.float32)
    
    print("Данные успешно загружены ...")
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train_onehot = lb.transform(y_train)
    y_test_onehot = lb.transform(y_test)
    
    return X_train, X_test, y_train_onehot, y_test_onehot


def train_neural_network(X_train, y_train_onehot, X_test, y_test_onehot, 
                        hidden_size=300, learning_rate=0.1, 
                        batch_size=32, epochs=20):
    
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    
    print("Начало обучения...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            _, y_pred = nn.forward(X_batch)
            
            batch_loss = nn.compute_loss(y_batch, y_pred)
            epoch_loss += batch_loss
            num_batches += 1
            
            dW1, db1, dW2, db2 = nn.backward(X_batch, y_batch, y_pred)
            
            nn.update_parameters(dW1, db1, dW2, db2)
        
        avg_loss = epoch_loss / num_batches
        
        train_accuracy = nn.accuracy(X_train, y_train_onehot)
        test_accuracy = nn.accuracy(X_test, y_test_onehot)
        
        train_loss_history.append(avg_loss)
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Эпоха {epoch+1}/{epochs}")
        print(f"  Потери: {avg_loss:.4f}")
        print(f"  Точность на обучающих данных: {train_accuracy:.4f}")
        print(f"  Точность на тестовых данных: {test_accuracy:.4f}")
        print(f"  Время эпохи: {epoch_time:.2f} сек")

    return nn, train_loss_history, train_accuracy_history, test_accuracy_history

def plot_results(train_loss_history, train_accuracy_history, test_accuracy_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_loss_history)
    ax1.set_title('Функция потерь во время обучения')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.grid(True)
    
    ax2.plot(train_accuracy_history, label='Обучающая выборка')
    ax2.plot(test_accuracy_history, label='Тестовая выборка')
    ax2.set_title('Точность во время обучения')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def display_sample_predictions(nn, X_test, y_test, num_samples=10):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        sample = X_test[idx].reshape(1, -1)
        prediction = nn.predict(sample)[0]
        true_label = np.argmax(y_test[idx])
        
        axes[i].imshow(sample.reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Предсказано: {prediction}\nИстинно: {true_label}')
        axes[i].axis('off')
        
        if prediction != true_label:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.show()

def main():
    sys.stdout.reconfigure(encoding='utf8')
    
    print("Загрузка данных ...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    print("Предобработка данных...")
    X_train, X_test, y_train_onehot, y_test_onehot = preprocess_data(
        X_train, X_test, y_train, y_test
    )


    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(10):
        class_indices = np.where(y_train == i)[0]
        if len(class_indices) > 0:
            img = X_train[class_indices[0]].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Класс: {i}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    hidden_size = 300
    learning_rate = 0.1
    batch_size = 32
    epochs = 20
    
    print(f"\nПараметры обучения:")
    print(f"  Количество скрытых нейронов: {hidden_size}")
    print(f"  Скорость обучения: {learning_rate}")
    print(f"  Размер пачки: {batch_size}")
    print(f"  Количество эпох: {epochs}")
    
    nn, train_loss_history, train_accuracy_history, test_accuracy_history = train_neural_network(
        X_train, y_train_onehot, X_test, y_test_onehot,
        hidden_size, learning_rate, batch_size, epochs
    )
    
    final_train_accuracy = nn.accuracy(X_train, y_train_onehot)
    final_test_accuracy = nn.accuracy(X_test, y_test_onehot)
    
    print("Результаты:")
    print(f"Точность на обучающих данных: {final_train_accuracy:.4f}")
    print(f"Точность на тестовых данных: {final_test_accuracy:.4f}")
    
    print("\nПостроение графиков...")
    plot_results(train_loss_history, train_accuracy_history, test_accuracy_history)
    
    display_sample_predictions(nn, X_test, y_test_onehot)
    
    predictions = nn.predict(X_test)
    true_labels = np.argmax(y_test_onehot, axis=1)
    
    cm = confusion_matrix(true_labels, predictions)
    
    print("\nМатрица ошибок:")
    print(cm)
    
    print("\nОтчет по классификации:")
    print(classification_report(true_labels, predictions))
    
    return nn

if __name__ == "__main__":
    trained_model = main()
