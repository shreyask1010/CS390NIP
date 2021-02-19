import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from sklearn import metrics



# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784
neurons = 32

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):  
        return(1/(1+np.exp(-x)))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return(x*(1-x))
    
    def __relu(self, x):
        return max(0, x)
    
    def __reluDerivative(self, x):
        if x > 0:
            x =1
        else:
            x = 0
        return x

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs, minibatches = True, mbs = 64):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        for i in range(epochs):
            xBatches = self.__batchGenerator(xVals, mbs)
            yBatches = self.__batchGenerator(yVals, mbs)
            for xBatch, yBatch in zip(xBatches, yBatches):
         
                #feed forward
                l1out, l2out = self.__forward(xBatch)

                #Backpropagation
                l2e = (l2out - yBatch)/len(yBatch)
                l2d = l2e * self.__sigmoidDerivative(l2out)
                l1e = np.dot(l2d,self.W2.T)
                l1d = l1e * self.__sigmoidDerivative(l1out)
                l1a = np.dot(xBatch.T,l1d) *self.lr
                l2a = np.dot(l1out.T, l2d) *self.lr
                self.W1 = self.W1 - l1a
                self.W2 = self.W2 - l2a

        

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = xTrain*1.0 / 255
    xTest = xTest*1.0 / 255
    xTrain = np.reshape(xTrain, (60000, 784))
    xTest = np.reshape(xTest,(10000, 784))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
            
        #TODO: Write code to build and train your custon neural net.
        custom_model = NeuralNetwork_2Layer(inputSize=IMAGE_SIZE, outputSize=NUM_CLASSES, 
                                            neuronsPerLayer=neurons, learningRate = 0.5)
        custom_model.train(xTrain, yTrain, epochs = 50)
        return custom_model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        #TODO: Write code to build and train your keras neural net.
        model = tf.keras.models.Sequential()
        model.add(Dense(128, input_shape=(IMAGE_SIZE,), activation=tf.nn.relu))
        model.add(Dense(64, input_shape=(512,), activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(32, input_shape=(64,), activation=tf.nn.relu))
        model.add(Dropout(0.25))
        model.add(Dense(NUM_CLASSES, input_shape=(32,), activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        model.fit( x=xTrain , y=yTrain , epochs=20)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        #TODO: Write code to run your custon neural net.
        return(model.predict(data))
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        #TODO: Write code to run your keras neural net.
        return(model.predict(data))
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    y_act = []
    y_pred = []
    for i in range(preds.shape[0]):
        predicted = np.argmax(preds[i])
        y_pred.append(predicted)
        testing = np.argmax(yTest[i])
        y_act.append(testing)
        if predicted == testing:
            acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(y_act,y_pred))
    print()
    print("F1 score:")
    print(metrics.f1_score(y_act, y_pred, average=None))
          


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
