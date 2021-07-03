import numpy as np
import cv2 as cv
from scipy import ndimage, signal
import matplotlib.pyplot as plt


def diff_x(img):
    return signal.convolve2d(img, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) * 1 / 3, mode='same')


def diff_y(img):
    return signal.convolve2d(img, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) * 1 / 3, mode='same')


def get_corners(img, sigma=1, alpha=0.05, thresh=1000):
    """ Returns the detected corners as a list of tuples """
    ret = []
    i_x = diff_x(img)
    i_y = diff_y(img)
    i_xx = ndimage.gaussian_filter(i_x ** 2, sigma=sigma)
    i_yy = ndimage.gaussian_filter(i_y ** 2, sigma=sigma)
    i_xy = ndimage.gaussian_filter(i_x * i_y, sigma=sigma)
    height, width = img.shape[:2]
    det = i_xx * i_yy - i_xy ** 2
    trace = i_xx + i_yy
    r_val = det - alpha * trace ** 2
    for i in range(2, height - 3):
        for j in range(2, width - 3):
            if r_val[i, j] > thresh and r_val[i, j] == np.amax(r_val[i - 1:i + 2, j - 1:j + 2]):
                ret.append((i, j))
    return ret


def plot_harris(img_list, name, sigma=1):
    """ Plot the Harris Corner results using matplotlib for three image sizes """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
    ax[0].set_title("Harris Corner on {} (Full Size)".format(name))
    ax[1].set_title("Harris Corner on {} (Half Size)".format(name))
    ax[2].set_title("Harris Corner on {} (Quarter Size)".format(name))
    size_up = [1, 2, 4]
    for i in range(0, 3):
        ax[i].imshow(cv.cvtColor(img_list[i], cv.COLOR_GRAY2RGB))
        corners = get_corners(img_list[i], sigma=sigma)
        for corner in corners:
            ax[i].scatter([corner[1]], [corner[0]], c='r')
    plt.show()


def gauss_pyramid(img, num_levels=7, orig_size=1024):
    """ Generate the images that create the Gaussian pyramid"""
    ret = []
    for level in range(0, num_levels):
        sigma = 2 ** level
        size = orig_size // sigma
        ret.append(cv.resize(ndimage.gaussian_filter(img, sigma=sigma), (size, size)))
    return ret


def dog_pyramid(img, num_levels=6, orig_size=1024):
    """ Generate the difference of Gaussian pyramid """
    ret = []
    gauss = gauss_pyramid(img, num_levels=num_levels + 1, orig_size=orig_size)
    for level in range(0, num_levels):
        size = orig_size // 2 ** level
        ret.append(gauss[level].astype(float) - cv.resize(gauss[level + 1], (size, size)).astype(float))
    return ret


def keypoint(img, num_levels=5, orig_size=1024, thresh=-24):
    """ Find keypoint locations on an image using DoG (pass in BGR image)
        Returns (matrix, image overlay, number points) where matrix is a Nx3 matrix with 3-tuple (x, y, sigma), image
        overlay is the image with keypoints for displaying, and number points is the number of keypoints """
    tracker = [{}, {}, {}, {}, {}]
    img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dog = dog_pyramid(img_bw, num_levels=num_levels, orig_size=orig_size)
    dog_local_min = np.zeros((orig_size, orig_size), dtype=int)

    for scale in range(0, num_levels):  # Find keypoints at different scales
        size = orig_size // 2 ** scale
        for level in range(0, scale + 1):
            curr = cv.resize(dog[level], (size, size))  # resize the image to current size
            for i in range(2, size - 3):
                for j in range(2, size - 3):
                    val = curr[i, j]
                    if val < thresh \
                            and val == np.amin(curr[i - 1:i + 2, j - 1:j + 2]) \
                            and val < dog_local_min[i, j]:  # Find local minima, compare to local minima at each level
                        a = np.hstack((curr[i - 2:i + 3, j - 2], curr[i - 2:i + 3, j + 2], curr[i - 2, j - 1:j + 2],
                                       curr[i + 2, j - 1:j + 2]))
                        # if not a[a < 0].any():
                        if np.sum(a) > (-1 * thresh):
                            dog_local_min[i, j] = val
                            tracker[scale][(i, j)] = level

    ret_image = multi_overlay(img, tracker)
    return ret_image, len(tracker[0]) + len(tracker[1]) + len(tracker[2]) + len(tracker[3]) + len(tracker[4])


def multi_overlay(display, tracker, orig_size=1024):
    """ Multi-colour overlay for Q7 """
    # blue (level 1), green (level 2), yellow (level 3), magenta (level 4), red (level 5)
    colours = [[0, 0, 255], [0, 255, 0], [225, 225, 0], [225, 0, 225], [255, 0, 0]]
    ret = cv.cvtColor(cv.cvtColor(display, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
    over_list = []

    for scale in range(0, 5):
        size = orig_size // 2 ** scale
        over = np.zeros((size, size, 3), np.uint8)
        curr_level = tracker[scale]
        for coord in curr_level:
            over[coord[0], coord[1]] = colours[curr_level[coord]]
        over_list.append(cv.resize(over, (orig_size, orig_size)))
    over_list.reverse()  # Reverse list so that larger interest points are prioritized
    for i in range(0, orig_size):
        for j in range(0, orig_size):
            for over in over_list:
                if not np.array_equal(over[i, j], [0, 0, 0]) and np.sum(over[i, j]) > 50:
                    ret[i, j] = over[i, j]
                    break
    return ret


if __name__ == '__main__':
    # Toggle which parts are shown.
    test_part_2 = True
    test_part_3 = True

    shapes = cv.cvtColor(cv.imread('images/shapes.png'), cv.COLOR_BGR2RGB)
    cnTower = cv.cvtColor(cv.imread('images/cnTower.jpg'), cv.COLOR_BGR2RGB)
    sunflower = cv.cvtColor(cv.imread('images/sunflower.jpg'), cv.COLOR_BGR2RGB)

    if test_part_2:
        plot_shapes = True
        plot_cnTower = True

        if plot_shapes:
            shapes_large = cv.cvtColor(shapes, cv.COLOR_BGR2GRAY)
            shapes_med = cv.resize(shapes_large, (int(shapes.shape[1] * 0.5), int(shapes.shape[0] * 0.5)))
            shapes_small = cv.resize(shapes_large, (int(shapes.shape[1] * 0.25), int(shapes.shape[0] * 0.25)))

            img_list = [shapes_large, shapes_med, shapes_small]
            plot_harris(img_list, "Shapes")

        if plot_cnTower:
            cnTower_large = cv.cvtColor(cnTower, cv.COLOR_BGR2GRAY)
            cnTower_med = cv.resize(cnTower_large, (int(cnTower.shape[1] * 0.5), int(cnTower.shape[0] * 0.5)))
            cnTower_small = cv.resize(cnTower_large, (int(cnTower.shape[1] * 0.25), int(cnTower.shape[0] * 0.25)))

            img_list = [cnTower_large, cnTower_med, cnTower_small]
            plot_harris(img_list, "CN Tower", sigma=3)

    if test_part_3:
        test_q5 = True
        test_q6 = True
        test_q7 = True

        if test_q5:
            pyramid = gauss_pyramid(sunflower)

            fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(sunflower)
            for i in range(0, 7):
                row = 0 if i < 3 else 1
                col_offset = 1 if i < 3 else -3
                ax[row, i + col_offset].set_title("Gaussian Pyramid Level {}".format(i))
                ax[row, i + col_offset].imshow(pyramid[i])
            plt.show()

        if test_q6:
            dog = dog_pyramid(cv.cvtColor(sunflower, cv.COLOR_BGR2GRAY))
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
            for i in range(0, 6):
                row = 0 if i < 3 else 1
                col_offset = 0 if i < 3 else -3
                ax[row, i + col_offset].set_title("DoG Pyramid Level {}".format(i + 1))
                ax[row, i + col_offset].imshow(dog[i])
            plt.show()

        if test_q7:
            result = keypoint(sunflower)
            fig, ax = plt.subplots()
            ax.set_title("Keypoint Detector ({} Points)".format(result[1]))
            ax.imshow(result[0])
            plt.show()
