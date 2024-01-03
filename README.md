# Neural-Network-Keras_Dataset
This repository contains two closely related codes for implementing neural networks using TensorFlow and Keras. The primary difference between the two codes lies in the dataset used for training and testing. One code utilizes the Fashion MNIST dataset, while the other employs the Numbers MNIST dataset.

# Code Overview:

Both codes follow a similar structure, with the only variation being the dataset import line. The provided code demonstrates the usage of the Fashion MNIST dataset. To switch to the Numbers MNIST dataset, you only need to comment/uncomment the relevant lines.

# Instructions:

Ensure you have TensorFlow and Keras installed in your Python environment.

Clone the repository to your local machine using the following command:

git clone https://github.com/your-username/your-repository.git

Open the cloned directory:

cd your-repository

Execute the chosen script, either fashion_mnist_script.py or numbers_mnist_script.py, depending on the dataset you want to work with.

# Code Features:

Dataset Loading: Both scripts load the respective datasets (Fashion MNIST or Numbers MNIST) for training and testing.

Neural Network Architecture: The code defines a simple neural network architecture using Keras, consisting of a flatten layer, two hidden layers with ReLU activation, and an output layer with sigmoid activation.

Training: The neural network is trained using Stochastic Gradient Descent (SGD) as the optimizer, and Sparse Categorical Crossentropy as the loss function.

Evaluation: The script evaluates the performance of the trained model on both the training and test sets, providing metrics such as cost and accuracy.

Visualization: The code includes visualization plots depicting the evolution of the cost and accuracy during training. Additionally, it displays random images from the training and test sets, along with their true and predicted classes.
