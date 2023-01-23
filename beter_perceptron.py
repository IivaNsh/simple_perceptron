from activate_functions import *

class Node:
    def __init__(self, n):
        self.activation_func = sigmod_func
        self.weights = np.array([(random.random()-0.5)*3 for i in range(n)])
    

    def get(self, X):
        linear_out = np.dot(self.weights, X)
        #return linear_out
        return self.activation_func(linear_out)


    def disp(self):
        print(self.weights)


class Beter_Perceptron: 
 
    def __init__(self, learning_rate=0.01, n_iters = 100): 
        self.lr = learning_rate  
        self.n_iters = n_iters 
        self.activation_func = sigmod_func
        self.nodes = [
            [Node(3),Node(3),Node(3)],
            [Node(4),Node(4),Node(4)],
            [Node(4)]
            ]
 
    def learn(self, X, types): 
        
        print("-------<start learning>-------")

        type_ = np.where(types > 0, 1, 0)
 
        #learn weights 
        for _ in range(self.n_iters):
            #print("iteration ", _, ">-----------")
            for idx, x_i in enumerate(X):
                self.learn_iteration(x_i, type_[idx])
        

        print("-------<end learning>-------")

        for _, nodes in enumerate(self.nodes):
            #print("layer ", _)
            for i, node in enumerate(nodes):
                #print("node ", i)   
                node.disp()




    def learn_iteration(self, X, type):


        l1 = X

        a1 = self.nodes[0][0].get(X)
        a2 = self.nodes[0][1].get(X)
        a3 = self.nodes[0][2].get(X)

        l2 = np.array([a1, a2, a3, 1])

        b1 = self.nodes[1][0].get(l2)
        b2 = self.nodes[1][1].get(l2)
        b3 = self.nodes[1][2].get(l2)

        l3 = np.array([b1, b2, b3, 1])

        c1 = self.nodes[2][0].get(l3)

        values = []
        values.append(X)
        for i, nodes in enumerate(self.nodes):
            values.append([])
            for j, node in enumerate(nodes):
                values[i+1].append(0)
            values[i+1].append(1)
        #print(values)

        for i, nodes in enumerate(self.nodes):
            for j, node in enumerate(nodes):
                values[i+1][j] = node.get(values[i])


        k1 = (type - c1) * c1 * (1 - c1)
        self.nodes[2][0].weights += k1 * l3 * self.lr

        #err_values = []
        #err_values.append([ [ type - values[-1][0] ] ])
        #for i, nodes in enumerate(self.nodes, -1):
        #    err_values.append([])
        #    for j, node in enumerate(nodes):
        #        err_values[i+1].append([])
        #        for k, w in enumerate(node.weights):
        #            #c = err_values[i][j][k]
        #            err_values[i+1][j+1].append()
        #            self.nodes[i][j].weights[k] += err_values[i]
        
        k21 = (k1 * self.nodes[2][0].weights[0]) * b1 * (1 - b1)
        self.nodes[1][0].weights += k21 * l2 * self.lr
        
        k22 = (k1 * self.nodes[2][0].weights[1]) * b2 * (1 - b2)
        self.nodes[1][1].weights += k22 * l2 * self.lr

        k23 = (k1 * self.nodes[2][0].weights[2]) * b3 * (1 - b3)
        self.nodes[1][1].weights += k23 * l2 * self.lr


        k31 = (k21 * self.nodes[1][0].weights[0] + k22 * self.nodes[1][1].weights[0] + k23 * self.nodes[1][2].weights[0]) * a1 * (1 - a1)
        self.nodes[0][0].weights += k31 * l1 * self.lr
        
        k32 = (k21 * self.nodes[1][0].weights[1] + k22 * self.nodes[1][1].weights[1] + k23 * self.nodes[1][2].weights[1]) * a2 * (1 - a2)
        self.nodes[0][1].weights += k32 * l1 * self.lr
        
        k33 = (k21 * self.nodes[1][0].weights[2] + k22 * self.nodes[1][1].weights[2] + k23 * self.nodes[1][2].weights[2]) * a3 * (1 - a3)   
        self.nodes[0][2].weights += k33 * l1 * self.lr
        

    
    def get(self, X):


        values = []
        values.append(X)
        for i, nodes in enumerate(self.nodes):
            values.append([])
            for j, node in enumerate(nodes):
                values[i+1].append(0)
            values[i+1].append(1)
        #print(values)

        for i, nodes in enumerate(self.nodes):
            for j, node in enumerate(nodes):
                values[i+1][j] = node.get(values[i])

        return values[-1][0]

