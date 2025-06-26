import numpy as np

class nn:
    def __init__(self,inputs,neurons):
        self.weights = 0.1*np.random.rand(inputs,neurons)
        self.biases = 0.0001*np.random.rand(neurons)
    
    def forward_nn(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases

class activation_relu:
    def forward(self,inputs):
        self.outputs = np.maximum(0,inputs)



class activation_softmax:
    def forward(self,inputs):
        self.outputs = np.exp(inputs)/np.sum(np.exp(inputs),axis=1,keepdims=True)


inputs = [[0.1,0.2,-0.4],[0.3,-0.1,5],[0.3,0.5,0.6]]   

layer1 = nn(3,4)
layer1.forward_nn(inputs)
print("layer1_outputs >> ",layer1.outputs)

layer2 = nn(4,3)
layer2.forward_nn(layer1.outputs)
print("layer2_outputs >> ",layer2.outputs)


activation1 = activation_relu()
activation1.forward(layer2.outputs)
print("activation1_outputs>> ",activation1.outputs)

activation2 = activation_softmax()
activation2.forward(activation1.outputs)
print("activation2_outputs >> ",activation2.outputs)