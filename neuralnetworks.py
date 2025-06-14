import numpy as np
import random



inputs = [[0.1,0.2,-0.4],[0.3,-0.1,5],[0.3,0.5,0.6]]

class nn:
    def __init__(self,inputs,neurons):
        self.weights = 0.1*np.random.rand(inputs,neurons)
        self.biases = 0.0001*np.random.rand(neurons)
    
    def forward_nn(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases


layer1 = nn(3,4)
layer1.forward_nn(inputs)
print(layer1.outputs)

layer2 = nn(4,3)
layer2.forward_nn(layer1.outputs)
print(layer2.outputs)
