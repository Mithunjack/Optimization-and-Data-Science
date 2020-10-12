import numpy as np
from scipy import stats
from numpy import random, sqrt, log, sin, cos, pi
from pylab import show,hist,subplot,figure
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def hypothesis_test(data, significance_level):
    k2, p = stats.normaltest(data)
    if p > significance_level:
        return True
    else:
        return False


# transformation function
def gaussian(n, mu, sigma):
    # uniformly distributed values between 0 and 1
    np.random.seed(28041990)
    u1 = random.uniform(size=n)
    u2 = random.uniform(size=n)

    z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    z2 = sqrt(-2 * log(u1)) * sin(2 * pi * u2)
    return z1 * sigma + mu, z2 * sigma + mu


def set_species_name(coulum):
    if coulum['species'] == '0.0':
        val = 'setosa'
    elif coulum['species'] == '1.0':
        val = 'versicolor'
    else:
        val = 'virginica'
    return val


if __name__ == "__main__":
    # 1
    data_1 = np.loadtxt('data1.txt', delimiter='\n')
    if hypothesis_test(data_1, 0.05):
        print('Sample looks Normal distribution')
    else:
        print('Sample does not look Normal disrtibution')

    data_2 = np.loadtxt('data2.txt', delimiter='\n')
    if hypothesis_test(data_2, 0.05):
        print('Sample looks Normal distribution')
    else:
        print('Sample does not look Normal disrtibution')

    # # 2
    # z1, z2 = gaussian(1000, 10, 2.5)
    # x = np.concatenate((z1, z2))
    # k2, p = stats.normaltest(x)
    # if hypothesis_test(x, 0.05):
    #     print('Sample looks Normal distribution')
    # else:
    #     print('Sample does not look Normal distribution')
    #
    # plt.hist(x, density=True, bins=30, label="Data")
    # plt.scatter(x, norm.pdf(x, 10, 2.5), color='red', label="PDF")
    # plt.legend(loc="upper left")
    # plt.show()

    # 3
    iris_data = pd.read_csv("iris_data.csv")
    iris_data.head()
    #
    features = ['# sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    x = iris_data.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    # pd.DataFrame(data=x, columns=features).head()
    #
    # # PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2', 'principal component 3'])

    # principalDf.head()

    finalDf = pd.concat([principalDf, iris_data.iloc[:, -3]], axis=1)
    finalDf = finalDf.rename(columns={'species (0: setosa': 'species'})
    finalDf.head()
    #
    # Plot PCA in 2D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    species = [0.0, 1.0, 2.0]
    colors = ['y', 'g', 'r']
    for specie, color in zip(species, colors):
        indicesToKeep = finalDf['species'] == specie
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(['setosa', 'Versicolour', 'Virginica'])
    ax.grid()
    plt.show()
    #
    # #plot PCA in 3D
    variance = pca.explained_variance_ratio_
    y = finalDf['species'].values
    # y = np.int_(y)
    centers = [[1, 1], [-1, -1], [1, -1]]
    fig = plt.figure(1, figsize=(8, 7))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()

    X = principalComponents
    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()