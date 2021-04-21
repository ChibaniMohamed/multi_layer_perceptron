import numpy as np
class Fc():
    def __init__(self,layer):
         self.layer = layer
         self.weights = np.asarray(np.random.randn(self.layer))
         self.biases = np.asarray(np.random.randn(1))
         self.neurons_outputs = np.zeros(self.layer)
    def forward(self,input):
        self.input = input
        out = []
        dot = np.dot(np.squeeze(self.layer),np.squeeze(self.weights))
        sum = dot+self.biases
        self.neurons_outputs += sum
        out.append(self.activation(sum))
        return out
    def backward(self,y_pred,y):
       cost_func = np.dot((y_pred-y)*self.sigmoid_derivative(y_pred),self.neurons_outputs)
       for w in range(len(self.weights)):
        self.weights[w] = self.weights[w] - cost_func

    def sigmoid_derivative(self,sig):
        return sig*(1-sig)
    def activation(self,value):
        return self.sigmoid(value)

    def sigmoid(self,x_):
        return 1.0/(1.0+np.exp(-x_))

class Network():
    def __init__(self):
       self.layers = []
      # self.n_input = n_input
    def add(self,layers):
        return self.layers.append(layers)
    def prediction(self,input_):
        self.input_ = input_
        output = []
        for i in range(len(input_)):
         for layer in self.layers:
            output.append(layer.forward(self.input_[i]))

        return output[len(output)-1]
    def fit(self,x,y,epoch):

        error = []
        for ep in range(epoch):
         for i in range(len(y)):
            for j in range(1):
             y_pred = self.prediction(x)
             error.append(0.5*(y[i] - y_pred[j])*(y[i] - y_pred[j]))

             for layer in self.layers:
               layer.backward(y_pred[0][0],y[i])


        return y_pred,error









x_train = [2.7810836,2.550537003,1.465489372,2.362125076,3.396561688
,4.400293529,1.38807019,1.850220317,3.06407232,3.005305973,
7.627531214,2.759262235,5.332441248,2.088626775,
6.922596716,1.77106367,8.675418651,-0.242068655,7.673756466,3.508563011]
y_train = [0,0,0,0,0,1,1,1,1,1]
model = Network()
model.add(Fc(len(x_train)))
model.add(Fc(20))
model.add(Fc(10))
model.add(Fc(1))
model.fit(x_train,y_train,epoch=100)
predict = model.prediction([-2.458797978])
print("test : ",predict)
'''
print(f"err : {error}")
print(f"pred : {pred}")

#pred = model.prediction(x_train)
'''
