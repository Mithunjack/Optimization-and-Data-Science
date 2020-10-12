import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# interval between values
s = 0.5
def draw_function(x,y,f,extrema_coord):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca(projection='3d')
    x,y = np.meshgrid(x,y)
    # print(x,y)
    z= f(x,y)
    # print(z)
    ax.plot_surface(x, y, z)
    ax.scatter(extrema_coord[0],extrema_coord[1],extrema_coord[2],c='green',s=5000)
    plt.show()

if __name__ == '__main__':
    # Roosenbork Funtion
    roosenbork = lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    minima_roosenbork = (1, 1, 0)
    x = np.arange(-2, 2. + s, s)
    y = np.arange(-2, 2. + s, s)
    draw_function(x, y, roosenbork, minima_roosenbork)

    # Bazara Shetty Function
    bazara_shetty = lambda x, y: (x - 2) ** 4 + (x - 2 * y) ** 2
    x = np.arange(-2, 2. + s, s)
    y = np.arange(-2, 2. + s, s)
    minima_bazara_shetty = (2, 1, 0)
    draw_function(x, y, bazara_shetty, minima_bazara_shetty)