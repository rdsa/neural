import os
import cv2
import tflearn
import tensorflow as tf
import numpy as np
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

'''
DNN que classifica imagens de carros ou motos (ou quaisquer duas classes, é só treinar)
    Para testar colocar imagens na pasta datasets/test/
    Para treinar colocar imagens dentro de pastas com o nome de sua classe (e.g. "car", "motorcycle") na pasta datasets/train/

Datasets usados:
    Carros: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
    Motos: http://www.vision.caltech.edu/html-files/archive.html
    + algumas imagens aleatórias da internet

Dependências:
    Numpy
    OpenCV
    TensorFlow
    TFLearn
'''

# Pasta com as imagens para treino
TRAIN_DIR = './datasets/train/'
# Pasta com as imagens para teste
TEST_DIR = './datasets/test/'
# Tamanho da imagem (para o resize)
IMG_SIZE = 75
# Nome do modelo que será salvo
MODEL_NAME = 'CarsVSMotorcycles'

# Abre todas as imagens de treinamento
def preprocess_training_data():
    training_data = []
    for label in os.listdir(TRAIN_DIR):
        dir = os.path.join(TRAIN_DIR, label)
        for item in os.listdir(dir):
            path = os.path.join(dir, item)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            c = [1,0] if label == 'car' else [0,1]
            training_data.append([img, np.array(c)])
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data

# Abre as imagens que serão testadas após o treinamento
def preprocess_testing_data():
    testing_data = []
    original_images = []
    for item in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, item)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([img, item])
        original_images.append(cv2.imread(path, 1))
    return testing_data, original_images

########## Leitura das imagens para treinamento ##########
# Para usar novas imagens:
#training_data = preprocess_training_data()
# Para carregar imagens utilizadas anteriormente:
training_data = np.load('training_data.npy')

tf.reset_default_graph()

# Parâmetros da Deep Neural Network usando TFLearn (TensorFlow)
# 'AlexNet' adaptado de:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2098, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum',\
                    loss='categorical_crossentropy',\
                    learning_rate=0.001)

model = tflearn.DNN(network)

path = "./" + MODEL_NAME + ".meta"
if os.path.exists(path):
    model.load(MODEL_NAME)
    print('Model loaded')

#################### Parte do treinamento ####################
'''
train = training_data

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

model.fit(X, Y, n_epoch=10, validation_set=0.2, shuffle=True,\
            show_metric=True, batch_size=32, run_id=MODEL_NAME)

# Salva o modelo
model.save(MODEL_NAME)
'''

# Lê as imagens para teste pós-treinamento
test_data, original_images = preprocess_testing_data()

# Testa a rede neural com as imagens da pasta datasets/test/
# A predição da rede neural é o título da janela
# Apertar qualquer botão para ir para a próxima imagem
for data, orig in zip(test_data,original_images):
    img_data = data[0]

    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    label = 'Motorcycle' if np.argmax(model_out) == 1 else 'Car'

    cv2.imshow(label, orig)
    cv2.waitKey(0)
