import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread,imshow
from PIL import Image

if __name__ == "__main__":

    plt.style.use('classic')
    img = Image.open('easter.png')
    # convert image to grayscale
    imggray = img.convert('LA')
    # convert to numpy array
    imgmat = np.array(list(imggray.getdata(band=0)), float)
    # Reshape according to orginal image dimensions
    imgmat.shape = (imggray.size[1], imggray.size[0])

    plt.figure(figsize=(9, 6))
    plt.imshow(imgmat, cmap='gray')
    plt.show()

    U, D, V = np.linalg.svd(imgmat)
    U.shape
    D.shape
    V.shape
    imgmat.shape

    size_d = len(D)
    original_img_size = imgmat.size
    exp_singularvalues = np.array([20 / 100 * size_d, 10 / 100 * size_d, 2 / 100 * size_d]).astype(int)
    for i in exp_singularvalues:
        percent = i * 100 / size_d
        reconstimg = np.matrix(U[:, :i]) * np.diag(D[:i]) * np.matrix(V[:i, :])
        real_Value_size = (reconstimg.shape[0] + reconstimg.shape[1] + 1) * i
        reduce_img_size = ((original_img_size - real_Value_size) / (original_img_size)) * 100
        plt.imshow(reconstimg, cmap='gray')
        title = f"{percent}% singular values\n {real_Value_size} real valued numbers\n reduced size {reduce_img_size}%"
        plt.title(title)
        plt.show()

    exp_singularvalues
    reconstimg = np.matrix(U[:, :200]) * np.diag(D[:200]) * np.matrix(V[:200, :])
    reconstimg.shape