import matplotlib.pyplot as plt
import numpy as np
from mpldatacursor import datacursor
import cv2
import os
import sys


# Data cursor code
def Q1():
    fig, axes = plt.subplots(nrows=3)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway1.jpg")
    data = cv2.imread(path)
    axes[0].imshow(data)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway2.jpg")
    data = cv2.imread(path)
    axes[1].imshow(data)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway3.jpg")
    data = cv2.imread(path)
    axes[2].imshow(data)
    datacursor(display='single')
    plt.show()


"""
Question 2 : FINDING HOMOGRAPHY  
"""


def find_homography(left, right):
    # the sizes are expected to be the same
    N, _ = right.shape
    left = np.hstack((left, np.ones(shape=(N, 1), dtype=np.int)))
    right = np.hstack((right, np.ones(shape=(N, 1), dtype=np.int)))

    A = np.zeros(shape=(N * 2, 9), dtype=int)

    j = 0

    while j < N:
        i = j * 2

        row = left[j].reshape((1, 3))
        row = np.hstack((row, np.zeros(shape=(1, 3), dtype=int)))
        row = np.append(row, - right[j, 0] * left[j, 0])
        row = np.append(row, - right[j, 0] * left[j, 1])
        row = np.append(row, -right[j, 0])
        A[i, :] = row

        i += 1
        row = left[j].reshape((1, 3))
        row = np.hstack((np.zeros(shape=(1, 3), dtype=int), row))
        row = np.append(row, - right[j, 1] * left[j, 0])
        row = np.append(row, - right[j, 1] * left[j, 1])
        row = np.append(row, -right[j, 1])
        A[i, :] = row

        j += 1

    values, vectors = np.linalg.eig(A.T @ A)
    value = np.argmin(values)
    return_value = vectors[:, value].reshape((3, 3))

    print(return_value)
    return return_value


"""
Question 3: PLOT POINTS  
"""


def plot_points():
    right_hallway1 = np.array([[1064, 12], [882, 234], [929, 144], [856, 214], [1096, 177], [879, 444]])

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway1.jpg")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for point in right_hallway1:
        plt.plot(point[0], point[1], 'rs', markersize=3)
    plt.title("hallway 1 ")

    plt.imshow(data, cmap='gray')
    plt.show()

    right_hallway2 = np.array([[906, 327], [739, 549], [780, 461], [709, 532], [942, 488], [747, 760]])

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway2.jpg")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for point in right_hallway2:
        plt.plot(point[0], point[1], 'rs', markersize=3)
    plt.title("hallway 2 ")

    plt.imshow(data, cmap='gray')
    plt.show()

    right_hallway3 = np.array([[939, 202], [840, 427], [866, 333], [824, 406], [961, 365], [844, 635]])

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway3.jpg")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for point in right_hallway3:
        plt.plot(point[0], point[1], 'rs', markersize=3)

    plt.title("hallway 3 ")

    plt.imshow(data, cmap='gray')
    plt.show()

    floor1 = np.array([[485, 778], [836, 660], [653, 545], [575, 494], [597, 494]])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway1.jpg")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for point in floor1:
        plt.plot(point[0], point[1], 'rs', markersize=3)

    plt.title("hallway 3 ")

    plt.imshow(data, cmap='gray')
    plt.show()

    floor3 = np.array([[416, 983], [824, 854], [687, 740], [630, 690], [651, 690]])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway3.jpg")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for point in floor3:
        plt.plot(point[0], point[1], 'rs', markersize=3)

    plt.title("hallway 3 ")

    plt.imshow(data, cmap='gray')
    plt.show()


'''
Note that the code was edited here to produce all the plots, this code is for the last plot 
I didnt make reusable code here since its only 3 plots 
'''


def plot_predictions():
    right_hallway1 = np.array([[1064, 12], [882, 234], [929, 144], [856, 214], [1096, 177], [879, 444]])
    right_hallway2 = np.array([[906, 327], [739, 549], [780, 461], [709, 532], [942, 488], [747, 760]])
    right_hallway3 = np.array([[939, 202], [840, 427], [866, 333], [824, 406], [961, 365], [844, 635]])
    floor1 = np.array([[485, 778], [836, 660], [653, 545], [575, 494], [597, 494]])
    floor3 = np.array([[416, 983], [824, 854], [687, 740], [630, 690], [651, 690]])

    # Find the homography
    homography = find_homography(right_hallway1, right_hallway3)

    right_hallway1 = np.hstack((right_hallway1, np.ones(shape=(6, 1), dtype=np.int)))

    est_right_hallway2 = np.zeros(shape=(6, 2))

    for i in range(0, 6):
        est_point = homography @ right_hallway1[i]
        est_right_hallway2[i, :] = est_point[:2] / est_point[2]

    for i in range(0, 6):
        plt.plot(est_right_hallway2[i, 0], est_right_hallway2[i, 1], 'rs', markersize=3)
        plt.plot(right_hallway3[i, 0], right_hallway3[i, 1], 'gs', markersize=1)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_images/hallway3.jpg")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(data, cmap='gray')
    plt.show()


'''
Here it is important to note that the values are being flipped since matplotlib x,y are different 
from image x,y
'''


def find_homographyA():
    right_hallway1 = np.array([[1064, 12], [882, 234], [929, 144], [856, 214], [1096, 177], [879, 444]])
    right_hallway2 = np.array([[906, 327], [739, 549], [780, 461], [709, 532], [942, 488], [747, 760]])
    right_hallway2 = np.flip(right_hallway2, axis=1)
    right_hallway1 = np.flip(right_hallway1, axis=1)
    return find_homography(right_hallway1, right_hallway2)


def find_homographyB():
    right_hallway1 = np.array([[1064, 12], [882, 234], [929, 144], [856, 214], [1096, 177], [879, 444]])
    right_hallway3 = np.array([[939, 202], [840, 427], [866, 333], [824, 406], [961, 365], [844, 635]])
    right_hallway3 = np.flip(right_hallway3, axis=1)
    right_hallway1 = np.flip(right_hallway1, axis=1)
    return find_homography(right_hallway1, right_hallway3)


def find_homographyC():
    floor1 = np.array([[485, 778], [836, 660], [653, 545], [575, 494], [597, 494]])
    floor3 = np.array([[416, 983], [824, 854], [687, 740], [630, 690], [651, 690]])
    floor1 = np.flip(floor1, axis=1)
    floor3 = np.flip(floor3, axis=1)
    return find_homography(floor1, floor3)

find_homographyA()
find_homographyB()
find_homographyC()