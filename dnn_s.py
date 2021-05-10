from keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# TAKE JUST TWO CLASSES 0 AND 1
indices_1 = [i for i in range(len(y_train)) if y_train[i] == 1]
indices_0 = [i for i in range(len(y_train)) if y_train[i] == 0]
indices = np.concatenate((indices_1,indices_0))
rand = np.random.choice(indices,size = len(indices),replace=False)
x_train = [x_train[x] for x in rand]
y_train = [y_train[y] for y in rand]

test_indices_1 = [i for i in range(len(y_test)) if y_test[i] == 1]
test_indices_0 = [i for i in range(len(y_test)) if y_test[i] == 0]
test_indices = np.concatenate((test_indices_1,test_indices_0))
test_rand = np.random.choice(test_indices,size = len(test_indices),replace=False)
x_test = [x_test[x] for x in test_rand]
y_test = [y_test[y] for y in test_rand]

x_train = np.array(x_train).reshape(-1, 28*28)
x_test = np.array(x_test).reshape(-1, 28*28)
y_train = np.array(y_train)

#MULTI LAYER PERCEPTRON CLASS

class Network():
    def init_params(self):

        np.random.seed(10)
        self.w1_l1 = np.random.randn(10,784)
        self.w2_l2 = np.random.randn(1,10)


        self.biases_l1 = np.random.randn(1,10)
        self.biases_l2 = np.random.randn(1)

        return self.w1_l1,self.w2_l2,self.biases_l1,self.biases_l2

    def forward(self,x,w1_l1,w2_l2,b1,b2):
           

            dot_product_l1 = np.dot(x,w1_l1.T)
            n_in1 = dot_product_l1
            n_out1 = self.sigmoid(n_in1)

            dot_product_l2 = np.dot(n_out1,w2_l2.T)
            n_in2 = dot_product_l2 + b2
            n_out2 = self.sigmoid(n_in2)

            return n_in1,n_out1,n_in2,n_out2


    def sigmoid(self,x_):
        return 1/(1+np.exp(-x_))
    def sigmoid_derivative(self,sig):
        return sig*(1-sig)
    def ReLU(self,Z):
      return np.maximum(Z, 0)

    def softmax(self,z):
      exps = np.exp(z - z.max())
      return exps / np.sum(exps, axis=0)

    def relu_deriv(self,z):
      return z > 0

    def predict(self,input,w1_l1,w2_l2,b1,b2):
        n_in1,n_out1,n_in2,n_out2 = self.forward(input,w1_l1,w2_l2,b1,b2)


        return n_in1,n_out1,n_in2,n_out2


    def backward(self,X,Y,w1_l1,w2_l2,n_in1,n_out1,n_in2,n_out2):
       #w2

       error_2 = n_out2 - Y
       d_out_2 = self.sigmoid(n_out2)
       cost_2 = np.dot(n_in1.reshape(10,1),(error_2*d_out_2))
       w2_l2 -= 0.5*cost_2.T

       #w1
       error_1 = n_out1 - Y
       d_out_1 = self.sigmoid_derivative(n_out1)
       d = d_out_1*error_1
       cost_1 = np.dot(X.reshape(784,1),d.reshape(1,10))
       w1_l1 -= 0.5*cost_1.T

       return w1_l1,w2_l2


    def fit(self,x_train,y_train):
      w1_l1,w2_l2,b1,b2 = self.init_params()
      for i in range(500):

        for input,output in zip(x_train,y_train):

         #print(input)
         #all in one dimension

         n_in1,n_out1,n_in2,n_out2 = self.predict(input,w1_l1,w2_l2,b1,b2)
         #error,y_pred = self.compute_error(n_out2,y_train)

         w1_l,w2_l = self.backward(input,output,w1_l1,w2_l2,n_in1,n_out1,n_in2,n_out2)
         w1_l1,w2_l2 = w1_l,w2_l
         if i % 10 == 0:
          accuracy = np.sum(n_out2 == y_train)/len(y_train)


      return w1_l1,w2_l2,b1,b2


model = Network()
w1_l1,w2_l2,b1,b2 = model.fit(x_train[:100],y_train[:100])
fig ,ax = plt.subplots(2,4,figsize=(15,5))
for x in range(2):
  for j in range(4):
      r = np.random.randint(0,1000,1)[0]
      a,b,c,d = model.forward(x_test[r],w1_l1,w2_l2,b1,b2)
      ax[x,j].set(title =f"Predicted : {d} / Expected : {y_test[r]}")
      ax[x,j].imshow(x_test[r].reshape(28,28))
      fig.tight_layout()


plt.show()
