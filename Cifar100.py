#Modulos para redes neuronales
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Sequential # EJECUTA CAPA POR CAPA
from keras.optimizers import SGD, RMSprop, Adam # ALGORITMOS QUE ENCUENTRAN LOS PESOS Y SESGOS DE UNA FORMA OPTIMA
from keras.losses import SparseCategoricalCrossentropy  # COMO SE VA A COMPORTAR NUESTRA RED, DA INFORMACIÓN SOBRE LOS RESULTADOS QUE SE ESTAN OBTENIENDO
#Modulo para generar gráficos
import matplotlib.pyplot as plt
#Modulo para manejo de arreglos y numeros aleatorios
import numpy as np
import cv2

# Carga de la base de datos. Conjuntos (disjuntis) de entrenamiento y prueba, y sus respectivas etiquetas.
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # --- IMAGENES A COLOR
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()  # --- IMAGENES A COLOR

x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train])
x_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test])

# Resolución delas imagenes
resolution = (x_train.shape[1], x_train.shape[2])

inp_dim = resolution[0] * resolution[1]

out_dim = np.unique(y_train).shape[0]

print(f'Numero Clases {out_dim}, Clases {np.unique(y_train)}')
print(f'Registros de entrenamiento  {x_train.shape[0]}, Resolución {resolution}, Dimensión {inp_dim}, Etiquetas {y_train.shape[0]}')
print(f'Registros de entrenamiento  {x_test.shape[0]}, Resolución {resolution}, Dimensión {inp_dim}, Etiquetas {y_test.shape[0]}')

for i in range(9):
  r = np.random.randint(0,x_train.shape[0])
  plt.subplot(330 + i+1)
  plt.imshow(x_train[r], cmap=plt.cm.gray_r, interpolation="nearest")
  plt.title(f'Cls: {y_train[r]}')
  plt.axis('off')
plt.show()

ann = Sequential([
Flatten(input_shape=resolution, name='CapaLineal'),
Dense(units=5,activation='relu', name='CapaOculta1'),
Dense(units=5,activation='sigmoid', name='CapaOculta2'),
Dense(units=5,activation='tanh', name='CapaOculta3'),
Dense(units=5,activation='sigmoid', name='CapaOculta4'),
Dense(units=5,activation='relu', name='CapaOculta5'),
Dense(units=5,activation='relu', name='CapaOculta6'),
Dense(units=out_dim, name='CapaSalida')
])
ann.summary()

# Los parametros son los pesos y sesgos(b) que entran en cada neurona, 784(entradas) * 5(neuronas) + 5,sesgos(b, 1 por neurona)

# Learning rate, es cuanto varian los pesos para cada iteración,
#tr_history = ann.compile(optimizer=RMSprop(learning_rate=0.01),loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tr_history = ann.compile(optimizer=Adam(learning_rate=0.01),loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#tr_history = ann.compile(optimizer=SGD(learning_rate=0.01),loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Escalamiento de los datos entre 0 y 1 - esto se hace porque se quiere manejar unos rangos de valores entre 0 y 1
x_tr = x_train / 255
x_te = x_test / 255
# Definición de parámetros en el entrenamiento
epoch = 50
batch_size = 500
# Entrenamiento
tr_history = ann.fit(x=x_tr, y=y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1, shuffle=True, verbose=True)

plt.plot(tr_history.history['loss'])
plt.plot(tr_history.history['val_loss'])
plt.title(' Evolución del costo en la Red Neuronal')
plt.ylabel('Costo')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

plt.plot(tr_history.history['accuracy'])
plt.plot(tr_history.history['val_accuracy'])
plt.title(' Evolución de la exactitud de la Red Neuronal')
plt.ylabel('Exactitud')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

tr_perf = ann.evaluate(x = x_tr, y = y_train, batch_size= batch_size, verbose = False)
te_perf = ann.evaluate(x = x_te, y = y_test, batch_size= batch_size, verbose = False)
print(f'Costo entrenamiento {tr_perf[0]}, Costo prueba {te_perf[0]}')
print(f'Exactitud entrenamiento {tr_perf[1]}, Exactitud prueba {te_perf[1]}')
pred = ann.predict(x = x_te, batch_size = batch_size).argmax(axis=-1)

for i in range(9):
  r = np.random.randint(0,pred.shape[0])
  plt.subplot(330 + i+1)
  plt.imshow(x_test[r], cmap=plt.cm.gray_r, interpolation="nearest")
  plt.title(f'Cls: {y_test[r]}, Prd: {pred[r]}')
  plt.axis('off')
plt.show()