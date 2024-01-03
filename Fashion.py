# Modules for neural networks
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Sequential  # Executes layer by layer
from keras.optimizers import SGD, RMSprop, Adam  # Algorithms to find optimal weights and biases
from keras.losses import SparseCategoricalCrossentropy  # Defines how the network behaves, provides information on obtained results

# Module for generating plots
import matplotlib.pyplot as plt

# Module for array manipulation and random number generation
import numpy as np

# Load the dataset. Training and test sets, along with their respective labels.
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Image resolution
resolution = (x_train.shape[1], x_train.shape[2])
inp_dim = resolution[0] * resolution[1]
out_dim = np.unique(y_train).shape[0]

print(f'Number of Classes {out_dim}, Classes {np.unique(y_train)}')
print(f'Training Records {x_train.shape[0]}, Resolution {resolution}, Dimension {inp_dim}, Labels {y_train.shape[0]}')
print(f'Test Records {x_test.shape[0]}, Resolution {resolution}, Dimension {inp_dim}, Labels {y_test.shape[0]}')

# Display random images from the training set
for i in range(9):
    r = np.random.randint(0, x_train.shape[0])
    plt.subplot(330 + i + 1)
    plt.imshow(x_train[r], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f'Class: {y_train[r]}')
    plt.axis('off')
plt.show()

# Define the neural network architecture
ann = Sequential([
    Flatten(input_shape=resolution, name='LinearLayer'),
    Dense(units=5, activation='relu', name='HiddenLayer1'),
    Dense(units=5, activation='relu', name='HiddenLayer2'),
    Dense(units=out_dim, activation='sigmoid', name='OutputLayer')
])
ann.summary()

# Set the learning rate, which determines how much the weights change in each iteration
tr_history = ann.compile(optimizer=SGD(learning_rate=0.01),
                         loss=SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

# Scale the data between 0 and 1
x_tr = x_train / 255
x_te = x_test / 255

# Set training parameters
epoch = 50
batch_size = 500

# Train the neural network
tr_history = ann.fit(x=x_tr, y=y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1, shuffle=True, verbose=True)

# Plot the evolution of the cost during training
plt.plot(tr_history.history['loss'])
plt.plot(tr_history.history['val_loss'])
plt.title('Neural Network Cost Evolution')
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Plot the evolution of accuracy during training
plt.plot(tr_history.history['accuracy'])
plt.plot(tr_history.history['val_accuracy'])
plt.title('Neural Network Accuracy Evolution')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Evaluate the performance on the training and test sets
tr_perf = ann.evaluate(x=x_tr, y=y_train, batch_size=batch_size, verbose=False)
te_perf = ann.evaluate(x=x_te, y=y_test, batch_size=batch_size, verbose=False)
print(f'Training Cost {tr_perf[0]}, Test Cost {te_perf[0]}')
print(f'Training Accuracy {tr_perf[1]}, Test Accuracy {te_perf[1]}')

# Make predictions on the test set
pred = ann.predict(x=x_te, batch_size=batch_size).argmax(axis=-1)

# Display random images from the test set along with their true and predicted classes
for i in range(9):
    r = np.random.randint(0, pred.shape[0])
    plt.subplot(330 + i + 1)
    plt.imshow(x_test[r], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f'Cls: {y_test[r]}, Prd: {pred[r]}')
    plt.axis('off')
plt.show()