import numpy as np
import cv2 as cv
import scipy.signal as sp
import matplotlib.pyplot as plt


def gauss(dim, sigma):
    # Takes the dimensions of the desired gauss filter and the sigma and returns the filter
    # Dim must be odd
    # Filter has been scaled to add to 1 to preserve image brightness
    ret = np.zeros(shape=(dim, dim))
    start = -(dim // 2)
    for i in range(0, dim):
        for j in range(0, dim):
            ret[i][j] = (1 / (2 * np.pi * (np.square(sigma)))) * (
                np.power(np.e, (-(np.square(start + i) + np.square(start + j)) / (2 * np.square(sigma)))))
    return ret * 1/(np.sum(ret))


def laplace_gauss(dim, sigma):
    # Takes the dimensions of the desired Laplacian of Gaussian filter and returns the filter
    # Dim must be odd
    # Source of LoG function: http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    ret = np.zeros(shape=(dim, dim))
    start = -(dim // 2)
    for i in range(0, dim):
        for j in range(0, dim):
            ret[i][j] = (np.square(start + i) + np.square(start + j) - 2*np.square(sigma))/(np.power(sigma, 4)) * \
                        (np.power(np.e, (-(np.square(start + i) + np.square(start + j))/(2*np.square(sigma)))))
    return ret


def detect_zero(img):
    # Takes in the Laplacian filtered image and detects the zero crossings
    # Algorithm sources:
    # https://stackoverflow.com/questions/22050199/python-implementation-of-the-laplacian-of-gaussian-edge-detection
    # https://theailearner.com/tag/zero-crossings/
    # Code was written by me after looking at these algorithms, and my algorithm uses parts of both
    thresh = np.absolute(img).mean()
    height = len(img)
    width = len(img[0])
    ret = np.zeros(shape=(height, width))
    for i in range(1, height-1):
        for j in range(1, width-1):
            patch = img[i-1:i+2, j-1:j+2]
            minp = patch.min()
            maxp = patch.max()
            p = img[i][j]
            if p < 0 < maxp and maxp - minp > thresh:
                ret[i][j] = np.abs(p) + maxp if np.abs(p) + maxp > thresh else 0
            elif minp < 0 < p and maxp - minp > thresh:
                ret[i][j] = p + np.abs(minp) if p + np.abs(minp) > thresh else 0
    return ret.astype(np.uint8)


def plot_array(kernel, title):
    # Plot a 2D kernel as a colour map
    # Source of function:
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots()
    im = ax.imshow(kernel)

    for i in range(len(kernel)):
        for j in range(len(kernel)):
            text = ax.text(j, i, round(kernel[i, j], 2),
                           ha="center", va="center", color="w")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def convolve(img, kernel):
    # Convolve and image with the given kernel
    return sp.convolve2d(img.astype(float), kernel)


def convolve_display(img, kernel, title):
    # Perform the convolution and display the image
    cv.imshow(title, convolve(img, kernel).astype(np.uint8))



if __name__ == '__main__':
    paolina = cv.cvtColor(cv.imread('images/Paolina.jpg'), cv.COLOR_BGR2GRAY)
    dog = cv.cvtColor(cv.imread('images/dog.png'), cv.COLOR_BGR2GRAY)
    # Image source for dog picture:
    # https://www.google.com/url?sa=i&url=http%3A%2F%2Fpets.university%2Fi-love-dogs%2F&psig=AOvVaw2-Xgkpnat2Z2c3wkaX4pNv&ust=1590948869508000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKikmMeY3OkCFQAAAAAdAAAAABAW

    # Question 4
    # convolve_display(paolina, gauss(5, 1), "Gaussian sigma=1")
    # convolve_display(paolina, gauss(11, 2), "Gaussian sigma=2")
    plot_array(gauss(5, 1), "Gaussian 5x5, sigma=1")
    plot_array(gauss(11, 2), "Gaussian 11x11, sigma=2")

    # Question 5
    log1 = laplace_gauss(7, 1)
    log2 = laplace_gauss(13, 1.5)
    log3 = laplace_gauss(21, 3)
    plot_array(log1, "Laplacian of Gaussian 7x7, sigma=1")
    plot_array(log2, "Laplacian of Gaussian 13x13, sigma=1.5")

    # Question 6
    paolina_log1 = convolve(paolina, log1)
    paolina_log3 = convolve(paolina, log3)
    dog_log1 = convolve(dog, log1)
    dog_log3 = convolve(dog, log3)

    # Question 7
    t = detect_zero(paolina_log1)
    cv.imshow("Paolina", paolina)
    cv.imshow("dog", dog)
    cv.imshow("Paolina with sigma=1", detect_zero(paolina_log1))
    cv.imshow("Paolina with sigma=3", detect_zero(paolina_log3))
    cv.imshow("Dog with sigma=1", detect_zero(dog_log1))
    cv.imshow("Dog with sigma=3", detect_zero(dog_log3))

    k = cv.waitKey(0)
