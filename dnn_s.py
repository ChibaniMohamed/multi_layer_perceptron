import numpy as np
class Network():
    def __init__(self):
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

    def forward(self,x_train):
            print(len(x_train),len(self.w1_l1))
            dot_product_l1 = self.w1_l1.dot(x_train)
            self.neurons_inputs_l1 = dot_product_l1 + self.biases_l1
            self.neurons_outputs_l1 = self.activation(self.neurons_inputs_l1)
            #print(len(self.neurons_outputs_l1))
            dot_product_l2 = np.dot(self.w2_l2,self.neurons_outputs_l1)
            self.neurons_inputs_l2 = dot_product_l2 + self.biases_l2
            self.neurons_outputs_l2 = self.activation(self.neurons_inputs_l2)
            return self.neurons_outputs_l2

    def activation(self,x):
        return self.sigmoid(x)
    def sigmoid(self,x_):
        return 1/(1+np.exp(-x_))
    def sigmoid_derivative(self,sig):
        return sig*(1-sig)

    def predict(self,input):
        prediction = self.forward(input)
        return prediction

    def compute_error(self,y_pred,y_true):
         error = y_pred - y_true
         print(error)
         return error,y_pred

    def backward(self,error,y_pred):

       cost_func_1 = np.dot(self.neurons_outputs_l1.T,error*self.sigmoid_derivative(y_pred))
       cost_func_2 = np.dot((error)*self.sigmoid_derivative(y_pred),self.neurons_outputs_l2.T)
       print(len(error),len(self.neurons_outputs_l1))
       self.w1_l1 = self.w1_l1-0.5*cost_func_1
       self.w2_l2 = self.w2_l2-0.5*cost_func_2

    def fit(self,x_train,y_train):
        for input,output in zip(x_train,y_train):
         #print(input)
         #all in one dimension
         pred = self.predict(input)
         error,y_pred = self.compute_error(pred,output)

         self.backward(error,y_pred)

x_train = x_train.reshape(-1, 28*28)
model = Network()
model.fit(x_train[:100],y_train[:100])

#computing every error and every forward with every input matrix
#input should be 1x28*28 not 28x28 dimension
