import numpy as np
class Network():
    def init_params(self):

        '''
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.input = x
        self.output = y
        '''
        self.w1_l1 = np.random.randn(10,784)

        self.w2_l2 = np.random.randn(10,10)

        self.biases_l1 = np.random.randn(10,1)
        self.biases_l2 = np.random.randn(10,1)
        self.neurons_inputs_l1 = np.zeros(10)
        self.neurons_outputs_l1 = np.zeros(10)
        self.neurons_inputs_l2 = np.zeros(10)
        self.neurons_outputs_l2 = np.zeros(10)
        return self.w1_l1,self.w2_l2,self.biases_l1,self.biases_l2

    def forward(self,x,w1_l1,w2_l2,b1,b2):
           # print(len(x_train),len(self.w1_l1))

            dot_product_l1 = np.dot(w1_l1,x.T)

            n_in1 = sum(dot_product_l1) + b1
            n_out1 = self.ReLU(n_in1)


            dot_product_l2 = np.dot(w2_l2,n_out1)
            n_in2 = sum(dot_product_l2) + b2
            n_out2 = self.softmax(n_in2)
            return n_in1,n_out1,n_in2,n_out2

    def activation(self,x):
        return self.sigmoid(x)
    def sigmoid(self,x_):
        return 1/(1+np.exp(-x_))
    def sigmoid_derivative(self,sig):
        return sig*(1-sig)
    def ReLU(self,Z):
      return np.maximum(Z, 0)

    def softmax(self,Z):
      return np.exp(Z) / sum(np.exp(Z))

    def relu_deriv(self,z):
      return z > 0

    def predict(self,input,w1_l1,w2_l2,b1,b2):
        n_in1,n_out1,n_in2,n_out2 = self.forward(input,w1_l1,w2_l2,b1,b2)

        return n_in1,n_out1,n_in2,n_out2

    def compute_error(self,y_pred,y_true):
        # print(np.argmax(y_pred))
         #print('pred : ',y_pred[0])
        # print('true : ',y_pred.size)
         error = y_pred - self.one_hot(y_true)
         print("err : ",error.shape)
         return error,y_pred

    def backward(self,X,error,y_pred,w1_l1,w2_l2,n_in1,n_out1,n_out2):

       cost_func_1 = np.dot(error,n_out1.T)*self.relu_deriv(y_pred)
       print(cost_func_1.shape)
       cost_func_2 = np.dot(error,n_out2.T)*self.relu_deriv(y_pred)

       #print(error.shape)
       w1_l1 -=0.5*cost_func_1
       #print(self.w1_l1[0][0])
       w2_l2 -=0.5*cost_func_2
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
        #for input,output in zip(x_train,y_train):
         #print(input)
         #all in one dimension
         n_in1,n_out1,n_in2,n_out2 = self.predict(x_train,w1_l1,w2_l2,b1,b2)
         error,y_pred = self.compute_error(n_out2,y_train)

         w1_l,w2_l = self.backward(x_train,error,y_pred,w1_l1,w2_l2,n_in1,n_out1,n_out2)
         w1_l1,w2_l2 = w1_l,w2_l


x_train = x_train.reshape(-1, 28*28)
model = Network()
model.fit(x_train[:100],y_train[:100])
