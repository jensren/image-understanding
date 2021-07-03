import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

M = np.array([[-0.035/0.00002611, 0, 900/2],
              [0, -0.035/0.00002611, 600/2],
              [0, 0, 1]])

FOCAL_LENGTH = 0.035


def load_matches(file):
    """ Load a set of matching points from the matches.txt file and reformat as list of tuples (left point, right point)
        matches.txt is space delimited in format x y x' y' """
    mat = np.loadtxt(file, dtype=int)
    ret_lst = []
    for row in range(mat.shape[0]):
        ret_lst.append(((mat[row, 0], mat[row, 1]), (mat[row, 2], mat[row, 3])))
    return ret_lst


def plot_epilines(img1, img2, matches, epip_tup, fundamental, name, plot_f=False):
    """ Plot the epilines for the two images
        If plot_f, also plot the fundamental matrix """
    # Source of heatmap plotting code for displaying the fundamental matrix:
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 15)) if plot_f \
        else plt.subplots(nrows=1, ncols=2, figsize=(40, 11))
    fig.suptitle("Epilines ({})".format(name))
    ax[0].imshow(img1)
    ax[0].set_title("Left Image")
    ax[1].imshow(img2)
    ax[1].set_title("Right Image")

    colour_list = ['r', 'g', 'b', 'c', 'm', 'y']
    e_l, e_r = epip_tup

    for p_l, p_r in matches:
        colour = random.randint(0, len(colour_list) - 1)
        ax[0].plot((e_l[0], p_l[0]), (e_l[1], p_l[1]), marker='o', ls='-', c=colour_list[colour])
        ax[1].plot((e_r[0], p_r[0]), (e_r[1], p_r[1]), marker='o', ls='-', c=colour_list[colour])

    if plot_f:
        ax[2].imshow(fundamental)
        ax[2].set_title("Fundamental Matrix")
        for i in range(len(fundamental)):
            for j in range(len(fundamental)):
                ax[2].text(j, i, round(fundamental[i, j], 5), ha="center", va="center", color="w")

    plt.show()


def plot_poly_3d(points_sets, point_matches, name, img1, img2):
    """ Takes 3d points and plots them as polygons to show depth
        Each item in the points_sets is a set of points that create one polygon """
    # source for code used to plot:
    # https://stackoverflow.com/questions/4622057/plotting-3d-polygons-in-python-matplotlib
    # https://stackoverflow.com/questions/18897786/transparency-for-poly3dcollection-plot-in-matplotlib

    colour_list = ['r', 'g', 'b', 'c', 'm', 'y']

    # plot of matching points
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 11))
    fig.suptitle("{}".format(name))
    ax[0].imshow(img1)
    ax[0].set_title("Left Image")
    ax[1].imshow(img2)
    ax[1].set_title("Right Image")

    i = 0  # tracks the corresponding point in point_matches
    for s in range(len(points_sets)):
        for p in range(len(points_sets[s])):
            ax[0].scatter(point_matches[i, 0, 0], point_matches[i, 0, 1], c=colour_list[s])
            ax[1].scatter(point_matches[i, 1, 0], point_matches[i, 1, 1], c=colour_list[s])
            i += 1

    plt.show()

    # plot of recovered depth
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Recovered Depth ({})".format(name))
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    for s in range(len(points_sets)):
        pts = points_sets[s]

        x, y, z = np.array(pts)[:, 0], np.array(pts)[:, 1], np.array(pts)[:, 2]
        # x = [0, 1, 1, 0]
        # y = [0, 0, 1, 1]
        # z = [1, 1, 1, 1]

        ax.scatter(x, y, z, c=colour_list[s])

        vertices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        tupleList = list(zip(x, y, z))

        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]

        collection = Poly3DCollection(poly3d, linewidths=1, alpha=0.2)
        collection.set_facecolor(colour_list[s])
        collection.set_alpha(0.3)
        ax.add_collection3d(collection)
        ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.2, linestyles=':'))

    plt.show()


def get_h(point_lst):
    """ Calculate H, as explained in Exercise 7.6 of the book """
    homogeneous_pts = np.array([(p[0], p[1], 1) for p in point_lst])
    mean = np.mean(homogeneous_pts, axis=0)
    dist = np.array([np.linalg.norm(p - mean) for p in homogeneous_pts])
    mean_dist = np.mean(dist)
    return np.array([[np.sqrt(2) / mean_dist, 0, -np.sqrt(2) / mean_dist * mean[0]],
                     [0, np.sqrt(2) / mean_dist, np.sqrt(2) / mean_dist * mean[1]],
                     [0, 0, 1]])


def eight_point(points_lst):
    """ Eight-point algorithm that returns the estimate of the fundamental matrix
        points_lst is a list of tuples (left point, right point) in image coordinates
        max_dim is the max value of (height, width) used to scale points to prevent numerical instabilities """

    # get H for normalization and produce normalized points
    points_lst = np.array(points_lst)
    h_l = get_h(points_lst[:, 0])
    h_r = get_h(points_lst[:, 1])
    p_l_norm = [h_l @ np.array([p[0], p[1], 1]) for p in points_lst[:, 0]]
    p_r_norm = [h_r @ np.array([p[0], p[1], 1]) for p in points_lst[:, 1]]

    # create A using normalized points
    a = []
    for p_l, p_r in zip(p_l_norm, p_r_norm):
        x_l, y_l = p_l[0], p_l[1]
        x_r, y_r = p_r[0], p_r[1]
        a.append([x_r * x_l, x_r * y_l, x_r, y_r * x_l, y_r * y_l, y_r, x_l, y_l, 1])
    a = np.array(a)

    u, s, vh = np.linalg.svd(a)
    f_mat = np.reshape(vh[-1, :], (3, 3))

    # enforce singularity constraint
    u, s, vh = np.linalg.svd(f_mat)
    s[-1] = 0
    f_unscaled = (u * s) @ vh

    # rescale F
    return np.linalg.inv(h_r) @ f_unscaled @ np.linalg.inv(h_l)


def epipoles_location(f_mat):
    """ Computer the location of the epipoles from the fundamental matrix
        Returns (left epipole, right epipole) """
    u, s, vh = np.linalg.svd(f_mat)
    e_l = vh[-1, :]
    e_r = u[:, -1]
    # get x, y by dividing by w
    e_l = (e_l[0] / e_l[2], e_l[1] / e_l[2])
    e_r = (e_r[0] / e_r[2], e_r[1] / e_r[2])
    return e_l, e_r


def compute_e(f_mat, m_mat):
    """ Compute the essential matrix given F and M, assuming M_r = M_l """
    return m_mat.T @ f_mat @ m_mat


def compute_r_t(e_mat):
    """ Compute R, t_hat from the essential matrix """
    e_hat = e_mat / np.sqrt(np.trace(e_mat.T @ e_mat) / 2)
    et_e_hat = e_hat.T @ e_hat

    # using 7.26 from the book
    t_hat = np.array([np.sqrt(1 - et_e_hat[0, 0]), np.sqrt(1 - et_e_hat[1, 1]), np.sqrt(1 - et_e_hat[2, 2])])

    w = np.array([np.cross(e_hat[i, :], t_hat) for i in range(3)])  # [w_i, w_j, w_k]
    r = np.array([w[0] + np.cross(w[1], w[2]), w[1] + np.cross(w[2], w[0]), w[2] + np.cross(w[0], w[1])])
    return r, t_hat


def add_z(point, focal_len):
    """ Given a point (x, y) and the focal length, returns (x, y, f) """
    return np.array([point[0], point[1], focal_len])


def triang_point(r_mat, t_vec, focal_len, p_l, p_r):
    """ Compute P' given p_l, p_r, the rotation matrix, and the translation vector
        p_l, p_r are in the form (x, y) in image coordinates
        units are in metres """

    # convert p_l, p_r into 3D coordinates (according to the book, z = focal length)
    p_l = add_z(p_l, focal_len)
    p_r = add_z(p_r, focal_len)

    # formulate the linear system and solve for a, b, c
    a = [p_l, r_mat.T @ p_r, (np.cross(p_l, r_mat.T @ p_r))]
    sol = np.linalg.inv(a).dot(t_vec)

    # return the midpoint of line segment joining a * p_l and T + b * R_T @ p_r
    return (np.reshape(sol[0] * p_l, 3) + (np.reshape(t_vec, 3) + sol[1] * r_mat.T @ p_r)) / 2


def triang_four(matches):
    # takes 4 matches and returns a set of 4 3d points
    pts_set = []
    for tup in matches[:4]:
        p_3d = triang_point(R, t, FOCAL_LENGTH, tup[0], tup[1])
        pts_set.append(p_3d)
    return pts_set


if __name__ == '__main__':
    test_q4 = True
    test_q5 = True
    test_q6 = True

    p1 = (cv.cvtColor(cv.imread("first_pair/p11.jpg"), cv.COLOR_BGR2RGB),
          cv.cvtColor(cv.imread("first_pair/p12.jpg"), cv.COLOR_BGR2RGB))
    p2 = (cv.cvtColor(cv.imread("second_pair/p21.jpg"), cv.COLOR_BGR2RGB),
          cv.cvtColor(cv.imread("second_pair/p22.jpg"), cv.COLOR_BGR2RGB))

    p1_matches = load_matches("first_pair/matches.txt")
    p2_matches = load_matches("second_pair/matches.txt")
    max_dim = max(p1[0].shape[:-1])

    if test_q4:
        compare_f = True

        plot_lst_img1 = []
        plot_lst_img2 = []

        f1 = eight_point(p1_matches)
        ep1 = epipoles_location(f1)
        plot_lst_img1.append((f1, ep1, "First Pair, Test F"))

        f2 = eight_point(p2_matches)
        ep2 = epipoles_location(f2)
        plot_lst_img2.append((f2, ep2, "Second Pair, Test F"))

        if compare_f:
            p1_matches_subset_1 = p1_matches[5:]
            p1_matches_subset_2 = p1_matches[:-5]

            f3 = eight_point(p1_matches_subset_1)
            plot_lst_img1.append((f3, epipoles_location(f3), "First Pair, Test F Less First 5 Matches"))

            f4 = eight_point(p1_matches_subset_2)
            plot_lst_img1.append((f4, epipoles_location(f4), "First Pair, Test F Less Last 5 Matches"))

        for f, ep, name in plot_lst_img1:
            plot_epilines(p1[0], p1[1], p1_matches, ep, f, name=name, plot_f=True)
        for f, ep, name in plot_lst_img2:
            plot_epilines(p2[0], p2[1], p2_matches, ep, f, name=name, plot_f=True)

    if test_q5:
        p1_matches_arr = np.array(p1_matches)
        k1, k2 = p1_matches_arr[:, 0, :], p1_matches_arr[:, 1, :]
        f1 = cv.findFundamentalMat(k1, k2)[0]

        p2_matches_arr = np.array(p2_matches)
        k1, k2 = p2_matches_arr[:, 0, :], p2_matches_arr[:, 1, :]
        f2 = cv.findFundamentalMat(k1, k2)[0]

        epipoles = epipoles_location(f1)
        epipoles = epipoles_location(f2)

        plot_epilines(p1[0], p1[1], p1_matches, epipoles, f1, name="First Pair, Test Epipoles", plot_f=True)
        plot_epilines(p2[0], p2[1], p2_matches, epipoles, f2, name="Second Pair, Test Epipoles", plot_f=True)

    if test_q6:
        # first pair
        p1_matches_arr = np.array(p1_matches)
        k1, k2 = p1_matches_arr[:, 0, :], p1_matches_arr[:, 1, :]
        essential = cv.findEssentialMat(k1, k2, M)[0]
        points, R, t, mask = cv.recoverPose(essential, k1, k2, M)

        s1 = []
        for i in range(0, len(p1_matches), 4):
            s1.append(triang_four(p1_matches_arr[i:i+4]))

        plot_poly_3d(s1, p1_matches_arr, "First Pair", p1[0], p1[1])

        # second pair
        p2_matches_arr = np.array(p2_matches)
        k1, k2 = p2_matches_arr[:, 0, :], p2_matches_arr[:, 1, :]
        essential = cv.findEssentialMat(k1, k2, M)[0]
        points, R, t, mask = cv.recoverPose(essential, k1, k2, M)

        s2 = []
        for i in range(0, len(p2_matches), 4):
            s2.append(triang_four(p2_matches_arr[i:i + 4]))

        plot_poly_3d(s2, p2_matches_arr, "Second Pair", p2[0], p2[1])



