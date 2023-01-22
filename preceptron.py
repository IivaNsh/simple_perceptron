from activate_functions import *
 
class Perceptron: 
 
    def __init__(self, learning_rate=0.01, n_iters = 100): 
        self.lr = learning_rate  
        self.n_iters = n_iters 
        self.activation_func = sigmod_func 
        self.weights = None
        self.bias = None
 
    def learn(self, X, types): 
        

        print("-------<start learning>-------")

        self.init_learn(X, types)

        type_ = np.where(types > 0, 1, 0) 
 
        #learn weights 
        for _ in range(self.n_iters): 
            print("iteration ", _, ">-----------")
            for idx, x_i in enumerate(X):
                self.learn_iteration(x_i, type_[idx])

        print("-------<end learning>-------")
        

    def init_learn(self, X, types):
        n_samples, n_features = X.shape 
        
        #init paramters
        self.weights = np.zeros(n_features) # make randome 
        self.bias = 0





    def learn_iteration(self, X, type):
        linear_output = np.dot(X, self.weights) + self.bias 
        y_predicted  = self.activation_func(linear_output) 

        #apply percepton update rule
        update = self.lr * (type - y_predicted) 
        self.weights += update * X
        self.bias += update

        print("----------------------------")
        print(self.weights, self.bias)
 


    def predict(self, X): 
        linear_output = np.dot(X, self.weights) + self.bias 
        y_predicted = self.activation_func(linear_output) 
        return y_predicted
