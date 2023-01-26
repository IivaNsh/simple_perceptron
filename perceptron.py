from help_functions import *



class Perceptron:

    def __init__(self, eta, n_inputs):
        self.eta = eta

        #network signatutre
        self.w = [
            [[], [], [], [], [], [], []],
            [[], [], [], [], [], [], []],
            [[], [], [], [], [],],
            [[], [], [], []],
            [[]]
        ]
        self.b = [np.random.rand(len(e))-0.5 for e in self.w]

        self.w[0] = np.random.rand(len(self.w[0]), n_inputs)-0.5
        for i in range(1, len(self.w)):
            self.w[i] = np.random.rand(len(self.w[i]), len(self.w[i-1])) -0.5

    def predict(self, input_vec):

        r = [[0]*len(e) for e in self.b]
        a = [[0]*len(e) for e in self.b]

        #print(r)
        #print(a)

        r[0] = [np.dot(input_vec, self.w[0][i])+self.b[0][i] for i in range(len(r[0]))]
        a[0] = [sigmoida(e) for e in r[0]]
        for i in range(1, len(r)):
            r[i] = [np.dot(a[i-1], self.w[i][j])+self.b[i][j] for j in range(len(r[i]))]
            a[i] = [sigmoida(e) for e in r[i]]

        return a[-1]


    def fit(self, input_vec, target_vec):


        r = [[0]*len(e) for e in self.b]
        a = [[0]*len(e) for e in self.b]

        #print(r)
        #print(a)

        r[0] = [np.dot(input_vec, self.w[0][i])+self.b[0][i] for i in range(len(r[0]))]
        a[0] = [sigmoida(e) for e in r[0]]
        for i in range(1, len(r)):
            r[i] = [np.dot(a[i-1], self.w[i][j])+self.b[i][j] for j in range(len(r[i]))]
            a[i] = [sigmoida(e) for e in r[i]]


        #print(cost(array(target_vec),array(a[-1])))
        #print(sum(cost(array(target_vec),array(a[-1])))/len(target_vec))



        delta = cost_prime(np.array(target_vec),np.array(a[-1])) * sigmoida_prime(np.array(r[-1]))
        #print(delta)
        #print()

        nabla_w = [np.zeros(e.shape) for e in self.w]
        nabla_b = [np.zeros(e.shape) for e in self.b]
        #print(nabla_w)
        #print(nabla_b)

        nabla_b[-1] = delta
        nabla_w[-1] = [np.array(a[-2])*e for e in delta]
        for l in range(2, len(self.w)):
            delta = np.dot(np.array(self.w[-l+1]).transpose(), delta) * sigmoida_prime(np.array(r[-l]))
            nabla_b[-l] = delta
            nabla_w[-l] = [np.array(a[-l-1])*e for e in delta]

        self.w = [np.array(ew) + self.eta*np.array(enw)
                        for ew, enw in zip(self.w, nabla_w)]
        self.b = [np.array(eb) + self.eta*np.array(enb) 
                        for eb, enb in zip(self.b, nabla_b)]

    def get_cost(self, input_vec, target_vec):

        r = [[0]*len(e) for e in self.b]
        a = [[0]*len(e) for e in self.b]

        #print(r)
        #print(a)

        r[0] = [np.dot(input_vec, self.w[0][i])+self.b[0][i] for i in range(len(r[0]))]
        a[0] = [sigmoida(e) for e in r[0]]
        for i in range(1, len(r)):
            r[i] = [np.dot(r[i-1], self.w[i][j])+self.b[i][j] for j in range(len(r[i]))]
            a[i] = [sigmoida(e) for e in r[i]]

        return sum(cost(np.array(target_vec),np.array(a[-1])))/len(target_vec)

