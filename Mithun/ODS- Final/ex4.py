import pandas as pnd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ex2 import ifft_ncoeff

def extract_data():
    csv = pnd.read_csv("WHO-COVID-19-global-data.csv", names = 
                       ['day', 'country', 'country_name', 'region',
                        'deaths', 'cumulative_deaths',
                        'confirmed', 'cumulative_confirmed'])
    ger = csv[csv.country == 'DE'][csv.day >= '2020-03-01']
    return ger['cumulative_confirmed'].to_list()

def lin_reg(x,y):
    mod = LinearRegression()
    x_rsp = x.reshape(-1,1)
    mod.fit(x_rsp,y)
    x_pred = np.linspace(0, np.size(y), np.size(y))
    y_pred = mod.predict(x_rsp)
    
    return x_pred, y_pred

def draw_1(x,y):
    x_1, y_1 = lin_reg(x[0:7], y[0:7])
    x_2, y_2 = lin_reg(x[0:13], y[0:13])
    x_3, y_3 = lin_reg(x[0:20], y[0:20])
    x_4, y_4 = lin_reg(x[0:27], y[0:27])
    x_all, y_all = lin_reg(x, y)
    
    plt.figure(1)
    plt.title("Linear regression with 7, 13, 20 and 27")
    plt.xlabel('days since 1 March 2020')
    plt.ylabel('cumulative deaths') 
    plt.scatter(x, y)
    plt.plot(x_1, y_1, 'g')
    plt.plot(x_2, y_2, 'r')
    plt.plot(x_3, y_3, 'y')
    plt.plot(x_4, y_4, 'g')
    plt.plot(x_all, y_all, 'b')
    
def draw_2a(x, y, fft_y):
    fft1 = ifft_ncoeff(fft_y, 1)
    fft5 = ifft_ncoeff(fft_y, 5)
    fft10 = ifft_ncoeff(fft_y, 10)
    
    plt.figure(2)
    plt.title("FFTs with 1, 5 and 10 coeffs")
    plt.xlabel('days since 1 March 2020')
    plt.ylabel('cumulative deaths') 
    plt.scatter(x, y)
    plt.plot(fft1, 'y')
    plt.plot(fft5, 'g')
    plt.plot(fft10, 'r')

def draw_2b(x, y):
    y_1 = np.poly1d(np.polyfit(x, y, 1))
    y_5 = np.poly1d(np.polyfit(x, y, 5))
    y_10 = np.poly1d(np.polyfit(x, y, 10))
    
    plt.figure(3)
    plt.title("Poly regression with deg=1, 5, 10")
    plt.xlabel('days since 1 March 2020')
    plt.ylabel('cumulative deaths') 
    plt.scatter(x, y)
    plt.plot(x, y_1(x))
    plt.plot(x, y_5(x))
    plt.plot(x, y_10(x))

def draw_2c(x, y, fft_y):
    fft1 = ifft_ncoeff(fft_y, 1)
    fft5 = ifft_ncoeff(fft_y, 5)
    fft10 = ifft_ncoeff(fft_y, 10)
    y_1 = np.poly1d(np.polyfit(x, y, 1))
    y_5 = np.poly1d(np.polyfit(x, y, 5))
    y_10 = np.poly1d(np.polyfit(x, y, 10))
    mse_fft = [mean_squared_error(y[1:], fft1), 
               mean_squared_error(y[1:], fft5),
               mean_squared_error(y[1:], fft10)
               ]
    mse_poly = [mean_squared_error(y, y_1(x)),
                mean_squared_error(y, y_5(x)),
                mean_squared_error(y, y_10(x))
                ]
    plt.figure(4)
    plt.subplot(211)
    plt.title ("MSE from FFT")
    plt.plot(mse_fft)
    plt.subplot(212)
    plt.title ("MSE from poly reg")
    plt.plot(mse_poly)

def main():
    y = np.array(extract_data()).astype(np.float)
    x = np.arange(np.size(y))
    # draw_1(x, y)
    # draw_2a(x, y, np.fft.rfft(y))
    # draw_2b(x, y)
    draw_2c(x, y, np.fft.rfft(y))
    plt.show()

if __name__ == "__main__":
    main()
