import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def load_matrix(file_num, ex):
    """ Load the extrinsic or intrinsic matrix from a relative path
        True for extrinsic, false for intrinsic """
    return np.loadtxt("{}/extrinsic.txt".format(file_num)) if ex \
        else np.loadtxt("{}/intrinsics.txt".format(file_num))


def save_cloud(file_num, cloud_matrix):
    """ Save a point cloud matrix to a file at a relative path """
    np.savetxt("{}/pointCloud.txt".format(file_num), cloud_matrix, delimiter=',',
               fmt=['%.2f', '%.2f', '%.2f', '%d', '%d', '%d'])


def compute_point_cloud(img_rgb, img_depth, file_num, world=True, rot_matrix=None):
    """ Compute a point cloud using the rgb and depth images and save it at a relative path
        Computes the world coordinate point cloud if world, else camera coordinate with given rotation """
    intrinsic_inv = np.linalg.inv(load_matrix(file_num, False))
    extrinsic = load_matrix(file_num, True)
    ex_rotation_inv = np.linalg.inv(extrinsic[:, :3])
    ex_translation = extrinsic[:, 3]
    height, width = np.shape(img_depth)

    point_cloud = []
    for x in range(width):
        for h in range(height):
            y = height - h - 1  # y axis pointing up from bottom left corner
            w = img_depth[y, x]
            coord = np.matmul(intrinsic_inv, np.array([w * x, w * y, w]))
            if world:
                coord = np.matmul(ex_rotation_inv, coord) + ex_translation
            else:
                coord = np.matmul(rot_matrix, coord)
            point_cloud.append(np.concatenate((coord, img_rgb[y, x, :])))

    return np.array(point_cloud)


def camera_cloud_to_img(camera_cloud, height, width, file_num):
    """ Create an rgb and a depth image from a camera coordinate point cloud """
    intrinsic = load_matrix(file_num, False)
    depth_img = np.zeros((height, width), dtype=int)
    rgb_img = np.zeros((height, width, 3), dtype=int)

    for pixel in camera_cloud:
        q_w = np.matmul(intrinsic, pixel[:3])
        w = q_w[2]
        if w != 0:
            x = int(round(q_w[0] / w))
            y = int(round(q_w[1] / w))
            if 0 <= x < width and 0 <= y < height:
                depth_img[y, x] = w
                rgb_img[y, x, :] = pixel[3:]

    return rgb_img, depth_img


def rotate(img_rgb, img_depth, file_num, axis, theta):
    """ Rotate the camera (theta is in radians) and return the new rgb and depth images
        dir in ['x', 'y', 'z'] """
    # source for getting the rotation matrices: http://planning.cs.uiuc.edu/node102.html

    assert axis in ['x', 'y', 'z']
    if axis == 'x':
        rot_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    else:  # axis == 'z'
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    height, width = np.shape(img_depth)
    camera_point_cloud = compute_point_cloud(img_rgb, img_depth, file_num, world=False, rot_matrix=rot_matrix)
    return camera_cloud_to_img(camera_point_cloud, height, width, file_num)


def rot_sequence(img_rgb, img_depth, file_num, axis, step):
    """ Generate a sequence of step rotated images from 0 to pi/2 """
    seq = []
    for s in range(step):
        theta = s * (np.pi / 2) / step
        seq.append(rotate(img_rgb, img_depth, file_num, axis, theta))
    return seq


def create_vid(seq, name, path):
    """ Create a video using the sequence provided """
    # code partially taken from https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    two = [np.concatenate((cv.cvtColor(tup[0].astype(np.uint8), cv.COLOR_RGB2BGR),
                           cv.cvtColor(tup[1].astype(np.uint8), cv.COLOR_GRAY2BGR))) for tup in seq]
    size = (two[0].shape[1], two[0].shape[0])  # width, height

    two_vid = cv.VideoWriter("{}/{}.avi".format(path, name), cv.VideoWriter_fourcc(*'DIVX'), 15, size)
    for img in two:
        for i in range(8):
            two_vid.write(img)
    two_vid.release()


if __name__ == '__main__':
    test_q1 = True
    test_q2 = True
    test_q3 = True

    # load the images
    rgb_imgs = [cv.cvtColor(cv.imread("{}/rgbImage.jpg".format(i)), cv.COLOR_BGR2RGB) for i in range(1, 4)]
    depth_imgs = [cv.imread("{}/depthImage.png".format(i), cv.IMREAD_GRAYSCALE) for i in range(1, 4)]

    axis_choices = ['x', 'y', 'z']

    if test_q1:
        for i in range(1, 4):
            save_cloud(i, compute_point_cloud(rgb_imgs[i - 1], depth_imgs[i - 1], i))

    if test_q2:
        img_lst = [rotate(rgb_imgs[i], depth_imgs[i], i + 1, axis_choices[i], np.pi / 16) for i in range(3)]

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))
        fig.suptitle("Rotation by pi/16")
        for r in range(3):
            ax[r, 0].imshow(img_lst[r][0])
            ax[r, 0].set_title("RGB ({} axis)".format(axis_choices[r]))
            ax[r, 1].imshow(img_lst[r][1])
            ax[r, 1].set_title("Depth ({} axis)".format(axis_choices[r]))
        plt.show()

    if test_q3:
        test_full = True
        # if test_full produces x, y, z axis videos for all three images
        # otherwise only produce one example video (image 1 x-axis)

        if not test_full:
            sequence_1 = rot_sequence(rgb_imgs[0], depth_imgs[0], 1, 'x', 15)
            create_vid(sequence_1, "image1_x-axis", 1)
        else:
            for i in range(3):
                for a in axis_choices:
                    seq = rot_sequence(rgb_imgs[i], depth_imgs[i], i + 1, a, 15)
                    create_vid(seq, "image{}_{}-axis".format(i + 1, a), i + 1)
