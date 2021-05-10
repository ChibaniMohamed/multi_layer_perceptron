import numpy as np
class Perceptron():
 def __init__(self):
     self.input_ = []
     if self.input_:
         print(self.input_)
       
         self.weights = np.asarray(np.random.randn(len(self.input)))
         self.biases = np.asarray(np.random.randn(1))
         #self.input = 0
        # self.w = np.asarray(np.random.randn(len(self.input)))
         '''
         self.out_n = []
         self.in_n = []
         self.s = 0
         '''

     #self.t = [x for xs in self.in_n for x in xs]

 def input(self,input_layer,size):
     self.h1_size = size
     self.input_layer = input_layer
     return self.input_layer
 def add_params(self,in_n):
     self.input_.append(in_n)

     self.w.append(w)
     self.b.append(b)
     self.in_n.append(in_n)
     self.out_n.append(out_n)

     if len(self.out_n) >29:
      print(self.out_n[5],self.w[5][1])

 def fc(self,layer,h_size):
     self.input = layer
     self.h1_size = h_size
     self.input_.append(h_size)
     neuron = []
     for hidden in range(self.h1_size):
         dot = np.dot(np.squeeze(self.input),np.squeeze(self.weights))
         sum = dot+self.biases
         neuron.append(self.activation(sum))
         #self.add_params(sum,self.activation(sum),self.weights,self.biases)
         #self.add_params(self.input)

     return neuron

 def sigmoid(self,x):
     return 1.0/(1.0+np.exp(-x))
 def activation(self,out):
        return self.sigmoid(out)
'''
 def dnn(self,x_train,y_train):
     self.x_train = x_train
     self.y_train = y_train
     self.inp = self.input(self.x_train,len(self.x_train))
     self.f1 = self.fc(self.inp,10)
     self.f2 = self.fc(self.f1,10)
     self.output = self.fc(self.f2,1)
     #self.forward()
     return self.output

 def back_propagation(self):
     for y in range(len(self.y_train)):
      total_error = (self.y_train[y] - self.output)*(self.y_train[y] - self.output) #y is the expected output
     for i in range(len(self.out_n)):
         derivative = self.out_n[i]*(1-self.out_n[i])*(self.out_n[i]-y)*self.in_n[i]
         for j in range(len(self.w)):
         self.update_weights(self.w[i][j],i,j,0.5,derivative)
 def update_weights(self,weight,idx,idy,learning_rate,deriv):
     self.weight = weights
     self.idx = idx
     self.idy = idy
     self.learning_rate = learning_rate
     self.deriv = deriv
     self.w[idx][idy] = self.weight - self.learning_rate*self.deriv
'''

model = Perceptron()
x_train = [2.7810836,2.550537003,1.465489372,2.362125076,3.396561688
,4.400293529,1.38807019,1.850220317,3.06407232,3.005305973,
7.627531214,2.759262235,5.332441248,2.088626775,
6.922596716,1.77106367,8.675418651,-0.242068655,7.673756466,3.508563011]
y_train = [0,0,0,0,0,1,1,1,1,1]

inp = model.input(x_train,len(x_train))
f1 = model.fc(inp,10)
f2 = model.fc(f1,10)
output = model.fc(f2,1)
#print(model.dnn(x_train,y_train))
#model.back_propagation()
