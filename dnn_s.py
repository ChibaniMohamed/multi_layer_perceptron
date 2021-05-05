import numpy as np
class Network():
    def init_params(self):

        self.w1_l1 = np.random.randn(10,784) * np.sqrt(1. / 10)

        self.w2_l2 = np.random.randn(10,1) * np.sqrt(1. / 1)

        self.biases_l1 = np.random.randn(10,1)
        self.biases_l2 = np.random.randn(10,1)
        '''
        self.neurons_inputs_l1 = np.zeros(10)
        self.neurons_outputs_l1 = np.zeros(10)
        self.neurons_inputs_l2 = np.zeros(10)
        self.neurons_outputs_l2 = np.zeros(10)
        '''
        return self.w1_l1,self.w2_l2,self.biases_l1,self.biases_l2

    def forward(self,x,w1_l1,w2_l2,b1,b2):
           # print(len(x_train),len(self.w1_l1))

            dot_product_l1 = np.dot(w1_l1,x)

            n_in1 = dot_product_l1+ b1
            n_in1 = n_in1.reshape(-1,10)[0]
            n_out1 = self.sigmoid(n_in1)



            dot_product_l2 = np.dot(n_out1,w2_l2)
            n_in2 = dot_product_l2 + b2
            n_in2 = n_in2.reshape(-1,10)[0]
            n_out2 = self.softmax(n_in2)
            n_out2 = n_out2.reshape(-1,10)[0]
            print(n_in1.shape,n_in2.shape,n_out2.shape)
            return n_in1,n_out1,n_in2,n_out2

    def activation(self,x):
        return self.sigmoid(x)
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
    def softmax_deriv(self,z):
      exps = np.exp(z - z.max())
      return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def predict(self,input,w1_l1,w2_l2,b1,b2):
        n_in1,n_out1,n_in2,n_out2 = self.forward(input,w1_l1,w2_l2,b1,b2)

        return n_in1,n_out1,n_in2,n_out2
    '''
    def compute_error(self,y_pred,y_true):
        # print(np.argmax(y_pred))
         #print('pred : ',y_pred[0])
        # print('true : ',y_pred.size)
         error = y_pred - self.one_hot(y_true)
         print("err : ",error.shape)
         return error,y_pred
    '''

    def backward(self,X,Y,w1_l1,w2_l2,n_in1,n_out1,n_in2,n_out2):
       print(n_out2-Y)
       error = 2*(n_out2-Y)*self.sigmoid_derivative(n_in2)*n_out2.shape
       w2 = np.outer(error, n_out1)
       error = np.dot(w2_l2.T, error) * self.relu_deriv(n_in2)
       print('err : ',w2_l2.T.shape ,error.shape,self.relu_deriv(n_in2).shape)
       w1 = np.outer(error, X)

       print('w2 : ',w2.shape)
       print('pre w2 : ',w2_l2.shape)

       #print(error.shape)
       w2 = w2.reshape(-1,10)[0]
       w1_l1 -=0.5*w1
       #print(self.w1_l1[0][0])
       w2_l2 -=0.5*w2.T
       return w1_l1,w2_l2

    def one_hot(self,y):
     #print("size : ",y.max())
     one_hot_Y = np.zeros((y.size, y.max() + 1))

     one_hot_Y[np.arange(y.size), y] = 1
     one_hot_Y = one_hot_Y.T
     return one_hot_Y

    def fit(self,x_train,y_train):
      w1_l1,w2_l2,b1,b2 = self.init_params()
      for i in range(10):
        for input,output in zip(x_train,y_train):
         #print(input)
         #all in one dimension
         n_in1,n_out1,n_in2,n_out2 = self.predict(input,w1_l1,w2_l2,b1,b2)
         #error,y_pred = self.compute_error(n_out2,y_train)

         w1_l,w2_l = self.backward(input,output,w1_l1,w2_l2,n_in1,n_out1,n_in2,n_out2)
         w1_l1,w2_l2 = w1_l,w2_l


x_train = x_train.reshape(-1, 28*28)
model = Network()
model.fit(x_train[:100],y_train[:100])
