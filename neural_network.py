import numpy as np
class Network():
 def __init__(self):
     self.w = []
     self.b = []
     self.out_n = []
     self.in_n = []


     self.t = [x for xs in self.in_n for x in xs]

 def input(self,input_layer,size):
     self.h1_size = size
     self.input_layer = input_layer
     return self.input_layer
 def add_params(self,in_n,out_n,w,b):
     self.w.append(w)
     self.b.append(b)
     self.in_n.append(in_n)
     self.out_n.append(out_n)
     if len(self.out_n) >29:
      print(self.out_n[5],self.w[5])

 def fn(self,layer,h_size):
     self.input = layer
     self.h1_size = h_size
     neuron = []
     for hidden in range(self.h1_size):

         self.weights = np.asarray(np.random.randn(len(self.input)))
         self.biases = np.asarray(np.random.randn(1))
         dot = np.dot(np.squeeze(self.input),np.squeeze(self.weights))
         sum = dot+self.biases
         neuron.append(self.activation(sum))
         self.add_params(sum,self.activation(sum),self.weights,self.biases)

     return neuron

 def sigmoid(self,x):
     return 1.0/(1.0+np.exp(-x))
 def activation(self,out):
        return self.sigmoid(out)

 def forward(self):
     train = [1,2,3,4,0.5,0.6,0.8,0.6,0.1]
     self.inp = self.input(train,len(train))
     self.f1 = self.fn(self.inp,10)
     self.f2 = self.fn(self.f1,10)
     self.output = self.fn(self.f2,10)
     return self.output
model = Network()
print(model.forward())
