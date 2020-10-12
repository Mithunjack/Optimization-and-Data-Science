import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from matplotlib.image import imread
from matplotlib.pyplot import imshow

def get_difference(original, filtered):
    differences = (np.subtract(original, filtered)).reshape(-1)
    avg_difference = (np.sum(differences)) / differences.size
    return avg_difference


def get_minimum_number_of_coef(image):
    coef_array = fft.fft2(image)
    sorted_coef_array = np.sort(np.abs(coef_array.reshape(-1)))  # sort coef and #reshape convert 2D array into a Vector
    i = 237100
    while i <= 228000:
        thresh = sorted_coef_array[i]
        ind = np.abs(coef_array) > thresh
        filtered_coef = coef_array * ind
        real_valued_compressed_data = fft.ifft2(filtered_coef).real

        plt.figure(i)
        plt.title('Convert ' + str(i) + ' %')
        plt.imshow(real_valued_compressed_data)
        i += 10


if __name__ == '__main__':
    A = imread('easter.png')

    np.shape(A)  # for getting the dimention
    # imshow(A)
    coef_array = fft.fft2(A)
    # print('coef array ', coef_array)

    sorted_coef_array = np.sort(np.abs(coef_array.reshape(-1)))  # sort coef and #reshape convert 2D array into a Vector
    # print('sorted coef array ', sorted_coef_array)
    i = 0
    # print('sorted coef array size', sorted_coef_array.size)
    for keep in (1, 0.1, 0.05, 0.02, 0.01, 0.001):
        minimum_coef_index = int(np.floor((1 - keep) * len(sorted_coef_array)))
        # print('minimum coef index', minimum_coef_index)
        thresh = sorted_coef_array[minimum_coef_index]  # minimum coef value
        # print('text', np.abs(coef_array))
        ind = np.abs(coef_array) > thresh
        # print('text2', ind)
        filtered_coef = coef_array * ind
        real_valued_compressed_data = fft.ifft2(filtered_coef).real
        #     print('text3', real_valued_compressed_data)

        plt.figure(i)
        plt.title('Convertion ' + str(keep * 100) + ' %')
        plt.imshow(real_valued_compressed_data)
        plt.show()
        i += 1
        d = get_difference(A, real_valued_compressed_data)
        print(d)

    # get_minimum_number_of_coef(A)