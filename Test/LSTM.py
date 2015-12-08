__author__ = 'JoJo'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.datasets import mnist
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import ModuleValidator

path = "/Users/JoJo/Desktop/Macalester-Sophomore/COMP221/Final/MNIST"
(train, test) = mnist.makeMnistDataSets("/Users/JoJo/Desktop/Macalester-Sophomore/COMP221/Final/MNIST")

#Build the networks with increasing complexity in layers in order to test if number of layers affects error rate
#Trainers are created for each network and each network is trained once
# TODO add timing in seconds/minutes to
baselineNumberOfLayers = 10
LSTMNetworkNumber = 1
LSTMNetworks = []
multipliers = [1,2,5,10,100]
overallError = []
for i in range (0, len(multipliers)):
    print("LSTMNetwork" + str(LSTMNetworkNumber) + " created")
    LSTMNetwork = buildNetwork(28*28, multipliers[i] * baselineNumberOfLayers, 10, hiddenclass = LSTMLayer, recurrent = True)
    LSTMNetworks.append(LSTMNetwork)
    print("Trainer created")
    trainer = BackpropTrainer(LSTMNetwork, dataset=train)
    error = trainer.train()
    print("LSTMNetwork" + str(LSTMNetworkNumber) + " has an error rate of: " + str(error))
    overallError.append(error)
    LSTMNetworkNumber += 1

'''
Perhaps I want to add more layers with the same number of cells instead of more cells to a single layer...
'''

'''
#For testing purposes...
print("Validating the trained network to the testing one")
validation = ModuleValidator.MSE(LSTMNetwork, test)
print("Backpropagation Trainer " + str(LSTMNetwork) + " has test error rate of: " + str(validation))

or

activate(for one specific input from the file) or activateOnDataset(for the test dataset)

or

testOnClassData(dataset = test)
'''
# TODO plot error and whatnot on a graph


#Build a network and then test to see if more training affects the error rate
# TODO implement this
# TODO plot error and whatnot on a graph