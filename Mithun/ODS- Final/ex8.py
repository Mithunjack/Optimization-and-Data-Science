import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def frequency_matrix(data=0):
    frequency = np.unique(data, return_counts=True)
#     print('test',np.array(frequency).T)
    return np.array(frequency).T

def expectedvalue(data=0):
    size = len(data)
    print(size)
    probability = frequency_matrix(data)
#     print(probability[:,1])
    probability[:,1] = probability[:,1]/size
#     print('first_test: ',probability[:,0])
#     print('second_test: ',probability[:,1])
    return np.sum(probability[:,0]*probability[:,1])

def variance(data=0):
    size = len(data)
    frequency_mat = frequency_matrix(data)
    exp_value = expectedvalue(data)
    return (1/size)* np.sum(frequency_mat[:,1]*((frequency_mat[:,0] - exp_value)**2))


def interval(data):
    exp_value = expectedvalue(data)
    sd = np.std(data)
    return (exp_value - sd), (exp_value + sd)

def confidence_interval(data,confidence):
    exp_value = expectedvalue(data)
    return norm.interval(confidence, loc=exp_value, scale=np.std(data))

if __name__ == "__main__":
    data_1 = np.loadtxt('data1.txt', delimiter='\n')
    data_2 = np.loadtxt('data2.txt', delimiter='\n')

    # # 1a
    # expectedvalue(data_1)
    # variance(data_1)

    # 1b
    exp_value = expectedvalue(data_1)
    var = variance(data_1)
    x = np.random.normal(exp_value, var, 1000)
    # print(x)
    plt.hist(data_1, density=True, bins=30, color='green', label="Data")
    plt.scatter(x, norm.pdf(x, exp_value, var), color='red', label="PDF")
    plt.legend(loc="upper left")
    plt.show()

    # 1c
    data = np.random.normal(size=1000)
    hist, bins = np.histogram(data, bins=50)

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(10000)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]

    plt.subplot(121)
    plt.hist(data, 50, color='green', )
    plt.subplot(122)
    plt.hist(random_from_cdf, 50, color='green', )
    plt.show()

    # 2a
    data_ud = np.random.uniform(0, 4, 50)
    count, bins, ignored = plt.hist(data_ud, 30, density=True, color='green', )
    plt.show()

    # 2b
    mu, sigma = 0, 4  # mean and standard deviation
    nd_size = 1000
    normal_data = np.random.normal(mu, sigma, nd_size)
    count, bins, ignored = plt.hist(normal_data, 30, density=True, color='green', )
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()


    # 2c
    mu, sigma = 0, 4  # mean and standard deviation for best value .25
    s = np.random.lognormal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
    x = np.linspace(min(bins), max(bins), 10000)
    pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
           / (x * sigma * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, linewidth=2, color='r')
    plt.axis('tight')
    plt.show()

    # # Expected Value of Uniform Distibution
    # data_ud = np.random.uniform(0, 4, 50)
    # expectedvalue(data_ud)
    #
    # # Expected Value of Normal Distibution
    # mu, sigma = 0, 4
    # nd_size = 1000
    # normal_data = np.random.normal(mu, sigma, nd_size)
    # np.var(normal_data)
    # variance(normal_data)

    # #Confidence Interval
    # mu, sigma = 0, 4
    # nd_size = 1000
    # normal_data = np.random.normal(mu, sigma, nd_size)
    # confidence_interval(normal_data, 0.95)
    # interval(normal_data)