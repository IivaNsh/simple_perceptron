from preceptron import *
from beter_perceptron import *
import matplotlib.pyplot as plt


print("--------<prepating data>--------")

xs = [-7, -3, -2, -5, -2,  1,  3,  3,  1,  5]
ys = [ 2,  3,  1, -2, -4,  2,  5,  1, -1, -2]
ts = [ 1,  1,  0,  0,  0,  1,  1,  0,  0,  1]
N = len(xs)

data_ps = np.array([[xs[i],ys[i]] for i in range(N)])
data_ts = np.array(ts)


b_data_ps = np.array([[xs[i],ys[i],1] for i in range(N)])
b_data_ts = np.array(ts)

print("--------<training data>--------")
print("points:")
print(data_ps)
print()
print("types fo points:")
print(data_ts)
print()

#p = Perceptron(0.01, 10)
#p.learn(data_ps, data_ts)

bp = Beter_Perceptron(0.2, 1000)
bp.learn(b_data_ps, b_data_ts)


print()
print("------<pedictions>------")


fig, ax = plt.subplots()




#
#x1,y1 = -9,-1
#print(x1, y1, " => ", p.predict([x1,y1]))
#x2,y2 = 3,-3
#print(x2, y2, " => ", p.predict([x2,y2]))
#

xs_o = [xs[i] for i in range(len(xs)) if ts[i]==0]
ys_o = [ys[i] for i in range(len(ys)) if ts[i]==0]

xs_t = [xs[i] for i in range(len(xs)) if ts[i]==1]
ys_t = [ys[i] for i in range(len(ys)) if ts[i]==1]

ax.plot(xs_o, ys_o, "o", color="red")

ax.plot(xs_t, ys_t, "^", color="blue")
#
#lx1, lx2 = -7.5, 7.5
#ly1, ly2 = (-p.bias-p.weights[0]*lx1)/p.weights[1], (-p.bias-p.weights[0]*lx2)/p.weights[1]

#ax.plot([lx1, lx2], [ly1, ly2], "-k")




resolution = 100

y, x = np.meshgrid(np.linspace(-10, 10, resolution), np.linspace(-10, 10, resolution))

#z = [[p.predict([x[y_i][x_i], y[y_i][x_i]]) for x_i in range(resolution)] for y_i in range(resolution)] 
z = [[bp.get([x[y_i][x_i], y[y_i][x_i], 1]) for x_i in range(resolution)] for y_i in range(resolution)] 

#z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

ax.pcolormesh(x, y, z, cmap='BuPu', vmin=z_min, vmax=z_max)



plt.show()