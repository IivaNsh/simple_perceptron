from perceptron import *
import point_generator
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

print("--------<prepating data>--------")


ps1, ps2 = point_generator.generate_points(50, 50, (-1,-1), (1,1))

ps = np.concatenate((ps1, ps2))
ts = np.concatenate((np.array([[0]]*len(ps1)),np.array([[1]]*len(ps2))))


#xs = [-7, -3, -2, -5, -2,  1,  3,  3,  1,  5, -7.1, -3.1, -2.1, -5.1, -2.1,  1.1,  3.1,  3.1,  1.1,  5.1]
#ys = [ 2,  3,  1, -2, -4,  2,  5,  1, -1, -2,  2.1,  3.1,  1.1, -2.1, -4.1,  2.1,  5.1,  1.1, -1.1, -2.1]
#ts = [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  1,  0,  0,  1]
#N = len(xs)




metadata = dict(title="learing_animation", atrist="ivn")
write = PillowWriter(fps=25, metadata=metadata)


fig, ax = plt.subplots()

xs_o = ps1.transpose()[0]
ys_o = ps1.transpose()[1]

xs_t = ps2.transpose()[0]
ys_t = ps2.transpose()[1]

ax.plot(xs_o, ys_o, "o", color="red")

ax.plot(xs_t, ys_t, "^", color="blue")



resolution = 100
y, x = np.meshgrid(np.linspace(-10, 10, resolution), np.linspace(-10, 10, resolution))



model1 = Perceptron(2, 2)

print("-------<start learning>--------")

with write.saving(fig, "learing_animation.gif", 100):
    iterations = 2500
    for _ in range(iterations):
        for i in range(len(ps)):
            model1.fit(ps[i], ts[i])

        if (_%10==0):
            z = [[model1.predict([x[y_i][x_i], y[y_i][x_i]])[0] for x_i in range(resolution)] for y_i in range(resolution)] 
            z_min, z_max = -np.abs(z).max(), np.abs(z).max()
            ax.pcolormesh(x, y, z, cmap='RdYlGn')
            write.grab_frame()

    

print("-------<end learning>--------")





plt.show()