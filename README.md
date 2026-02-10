# 🚀 Fashion MNIST Image Classification using PyTorch

This repository contains a simple Neural Network implementation using PyTorch to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset is a collection of Zalando's fashion article images, consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## ✨ Features

*   **Simple Neural Network:** A straightforward feed-forward neural network architecture.
*   **PyTorch Implementation:** Built entirely using the PyTorch deep learning framework.
*   **Fashion MNIST Dataset:** Utilizes the popular Fashion MNIST dataset for image classification.
*   **Training & Evaluation:** Includes functions for training the model and evaluating its performance.
*   **Model Saving/Loading:** Demonstrates how to save and load the trained model's weights.
*   **Visualization:** Example of predicting a random test image and comparing it with the true label.

## 🛠️ Setup

To get this project running locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/amey1942007/Fashion_MNIST-Pytorch-Model
cd Fashion_MNIST-Pytorch-Model
```

### 2. Install Dependencies

Ensure you have Python 3.x installed. Then, install the required libraries:

```bash
pip install torch torchvision numpy pandas matplotlib random
```

## 🏃 Usage

### 1. Run the Notebook

Open the provided Jupyter Notebook (e.g., `fashion_mnist_classifier.ipynb` or your Colab Notebook) to execute the code step-by-step.

### 2. Training the Model

The notebook includes cells for defining the model, loss function, optimizer, and the training/testing loops. Simply run these cells in order.

### 3. Making Predictions

After training, you can use the model to make predictions on new images or the test set.

## 🧠 Model Architecture

The Neural Network consists of a simple feed-forward architecture:

*   **Input Layer:** Flattens the 28x28 input image to a 784-dimensional vector.
*   **Hidden Layer 1:** Linear transformation with 512 output features, followed by a ReLU activation.
*   **Hidden Layer 2:** Linear transformation with 512 output features, followed by a ReLU activation.
*   **Output Layer:** Linear transformation with 10 output features (corresponding to the 10 fashion classes).

```python
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        flatten_input = self.flatten(x)
        network = self.linear_relu_stack(flatten_input)
        return network
```

## 📊 Training Details

*   **Loss Function:** `nn.CrossEntropyLoss()` is used, suitable for multi-class classification.
*   **Optimizer:** `torch.optim.SGD()` (Stochastic Gradient Descent) with a learning rate of `0.01`.
*   **Epochs:** The model is trained for `25` epochs.
*   **Batch Size:** Data is processed in batches of `100`.

## ✅ Evaluation

After each epoch, the model is evaluated on the test set to monitor its performance. The `Test` function calculates the average loss and accuracy.

Example Output:

```
   Epoch 25  -------------------- 

current loss: 0.16527  , batch number: 0
...
 Test Evaluation result: 
 Accuracy: 88.00,
 Avg loss:0.350
```

## 🖼️ Visualization

The notebook includes a section to visualize a random image from the test set, display it, and show the model's prediction alongside the true label.

```python
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

random_idx = random.randint(0, 9999)
x = test[random_idx][0]
y = test[random_idx][1]

with torch.no_grad():
    pred = model(x)
    prediction = labels[pred[0].argmax(0)]
    truth      = labels[y]

print(f"simpleNN predict as {prediction} ; truth is {truth}")
plt.imshow(test.data[random_idx], cmap = "Greys")
plt.show()
```

## 💾 Saving and Loading the Model

The trained model's `state_dict` (containing learned parameters) is saved to a `.pth` file and can be loaded back into a new model instance.

### Saving:
```python
model_save_path = "fashion_mnist_model_weights.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")
```

### Loading:
```python
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load("fashion_mnist_model_weights.pth"))
loaded_model.eval()
print("Model weights loaded successfully into 'loaded_model'.")
```

---
