import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from google.colab import drive
drive.mount('/content/drive')
from keras import backend as K

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ALGORITHM = "guesser"
ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

DATASET = "mnist_d"
DATASET = "mnist_f"
DATASET = "cifar_10"
DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072

#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6):
    #TODO: Implement a standard ANN here.
    model = tf.keras.models.Sequential()
    model.add(Dense(128, input_shape=(IS,), activation=tf.nn.relu))
    model.add(Dense(64, input_shape=(128,), activation=tf.nn.relu))
    model.add(Dense(NUM_CLASSES, input_shape=(64,), activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    model.fit( x, y , epochs = eps)
    model_json = model.to_json()
    s = '/content/drive/My Drive/CS390/' + ALGORITHM +'_' + DATASET
    with open(s, 'w') as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    return model



def buildTFConvNet(x, y, eps = 12, dropout = True, dropRate = 0.25):
    #TODO: Implement a CNN here. dropout option is required.
    model = tf.keras.models.Sequential()
    inShape = (IH, IW, IZ)

    model.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", input_shape = inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu"))

    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 


    model.add(keras.layers.Conv2D(128, kernel_size = (3,3), activation = "relu"))
    model.add(keras.layers.Conv2D(128, kernel_size = (3,3), activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization()) 

    model.add(keras.layers.Conv2D(256, kernel_size = (3,3), activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.4))

    model.add(keras.layers.Flatten())
    model.add(BatchNormalization()) 
    
    model.add(keras.layers.Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax"))

     ##Improving accuracy for cifar_100_f
     # optimizer = keras.optimizers.Adam(lr=0.001) 
     # epochs = 20

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x, y, epochs = 12, batch_size = 256)
    model_json = model.to_json()
    s = '/content/drive/My Drive/CS390/' + ALGORITHM +'_' + DATASET
    with open(s, 'w') as json_file:
      json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    elif DATASET == "cifar_100_f":
        # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    # model = trainModel(data[0])
    
    if ALGORITHM == "tf_net":

      if DATASET == "mnist_d":
         json_file = open('/content/drive/My Drive/CS390/tf_net_mnist_d', 'r')

      elif DATASET == "mnist_f":
        json_file = open('/content/drive/My Drive/CS390/tf_net_mnist_f', 'r')
          
      elif DATASET == "cifar_10":
          json_file = open('/content/drive/My Drive/CS390/tf_net_cifar_10', 'r')

      elif DATASET == "cifar_100_f":
          json_file = open('/content/drive/My Drive/CS390/tf_net_cifar_100_f', 'r')

      elif DATASET == "cifar_100_c":
        json_file = open('/content/drive/My Drive/CS390/tf_net_cifar_100_c', 'r')
         
  

    elif ALGORITHM == "tf_conv":
      if DATASET == "mnist_d":
         json_file = open('/content/drive/My Drive/CS390/tf_conv_mnist_d', 'r')

      elif DATASET == "mnist_f":
        json_file = open('/content/drive/My Drive/CS390/tf_conv_mnist_f', 'r')
          
      elif DATASET == "cifar_10":
          json_file = open('/content/drive/My Drive/CS390/tf_conv_cifar_10', 'r')

      elif DATASET == "cifar_100_f":
          json_file = open('/content/drive/My Drive/CS390/tf_conv_cifar_100_f', 'r')
          
      elif DATASET == "cifar_100_c":
        json_file = open('/content/drive/My Drive/CS390/tf_conv_cifar_100_c', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    load weights into new model
    loaded_model.load_weights("model.h5")
    model = loaded_model
    print("Loaded model from disk")
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

if __name__ == '__main__':
    main()
