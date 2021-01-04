import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os


def lucas_kanade(filepath, w_size, overlap, picture_index, increment):
  # get the the files in the path and ensure they are in order
  files = os.listdir(filepath)
  num_files = len(files)
  files.sort()

  # Get the base image
  print(files[0])
  first_image = files[0]
  first_image = cv2.imread(filepath + "/" + first_image, cv2.IMREAD_GRAYSCALE)
  H, W = first_image.shape
  image = np.zeros(shape=(H, W, num_files))
  image[:, :, 0] = first_image

  # read all the images
  for i in range(1, num_files):
    image[:, :, i] = cv2.imread(filepath + "/" + files[i], cv2.IMREAD_GRAYSCALE)

  # Smooth the image
  image_1 = image[:, :, picture_index].copy()
  image_2 = image[:, :, picture_index + 1].copy()

  return_image = np.ndarray.copy(image_1)
  image_1 = cv2.GaussianBlur(image_1, (w_size, w_size), 3, 3)
  image_2 = cv2.GaussianBlur(image_2, (w_size, w_size), 3, 3)

  # Copy so that we do not do some of the same calculations on this
  image_1 = np.ndarray.copy(image_1)
  image_2 = np.ndarray.copy(image_2)

  # Derivative with respect to time w/ 1D normalization
  if num_files % 2 == 0:
    gaussian_filter = gaussian1D(num_files - 1)
  else:
    gaussian_filter = gaussian1D(num_files)

  for i in range(0, H):
    for j in range(0, W):
      image[i, j, :] = cv2.filter2D(image[i, j, :], -1, gaussian_filter).reshape((8,))

  dims = image_1.shape
  H, W = dims[0], dims[1]

  # Take the absolute values since w
  sobel_imagex = np.gradient(image_1)[0]
  sobel_imagey = np.gradient(image_1)[1]

  print(np.max(sobel_imagex))
  print(np.max(sobel_imagey))

  # calculate I_xx I_yy and I_xy and  clip
  sobel_imagexx = np.multiply(sobel_imagex, sobel_imagex)
  sobel_imageyy = np.multiply(sobel_imagey, sobel_imagey)
  sobel_imagexy = np.multiply(sobel_imagex, sobel_imagey)

  # if we do want overlap, specify the overlap
  if overlap:
    increment = increment
  else:
    # change w_size to the padding required per image
    increment = w_size
    w_size = w_size // 2

  # declare the U and V quiver plots
  quiver = np.zeros(shape=(H, W, 2))

  x = w_size // 2
  y = w_size // 2

  while x + w_size < H and y + w_size < W:

    # iterative refinement
    x_prime = x
    y_prime = y

    uv = np.zeros(shape=(2,))
    uv[0] = math.inf
    uv[1] = math.inf
    counter = 0

    # exit when one of the estimates goes out of the picture
    # or when the difference is very small
    # Adding a last ditch effort to terminate the loop when noise causes issues
    while math.sqrt((uv[0] ** 2) + (uv[1] ** 2)) > 2 and \
        x_prime + w_size <= H - 1 and x_prime - w_size >= 0 and \
        y_prime + w_size <= W - 1 and y_prime - w_size >= 0 and \
        counter <= 10:

      counter += 1
      ATA = np.zeros(shape=(1, 4))
      b = np.zeros(shape=(1, 2))
      derives = np.zeros(shape=(1, 2))
      print(x, y)
      for i in range(-w_size, w_size + 1):
        for j in range(-w_size, w_size + 1):
          # print(f"xx {sobel_imagexx[x+i, y + j]}")
          ATA += np.array([sobel_imagexx[x + i, y + j], sobel_imagexy[x + i, y + j],
                           sobel_imagexy[x + i, y + j], sobel_imageyy[x + i, y + j]])

          # billinear interpolation to get decimal values
          x1 = math.floor(x_prime + i)
          x2 = math.ceil(x_prime + i)
          y1 = math.floor(y_prime + j)
          y2 = math.ceil(y_prime + j)
          # Use the time_smoothed version of this

          Q11_2 = image[x1, y1, picture_index + 1]
          Q12_2 = image[x1, y2, picture_index + 1]
          Q21_2 = image[x2, y1, picture_index + 1]
          Q22_2 = image[x2, y2, picture_index + 1]

          It = bilinear_interpolate(x_prime + i, y_prime + j, x1, x2, y1, y2, Q11_2, Q12_2, Q21_2, Q22_2) \
               - image[x + i, y + j, picture_index]

          # print(f"x:{sobel_imagex[x+i, y+j]}")
          derives += np.array([sobel_imagex[x + i, y + j], sobel_imagey[x + i, y + j]])
          b += np.array([It * sobel_imagex[x + i, y + j], It * sobel_imagey[x + i, y + j]])

      ATA = np.reshape(ATA, (2, 2))

      # add some conditions where this value should be ignored
      w, v = np.linalg.eig(ATA)

      if np.linalg.det(ATA) != 0:
        uv = np.linalg.inv(ATA) @ (-b.T)
      else:
        uv = [0, 0]

      # eigenvalues cannot be too small
      if np.sum(w) < 5:
        uv = [0, 0]
      # ATA needs to be well conditioned
      if np.max(w) / np.min(w) > 5:
        uv = [0, 0]

      x_prime = x_prime + uv[0]
      y_prime = y_prime + uv[1]

    # In the x direction, if we have a higher x_prime, the line has to go lower (-)
    u = x - x_prime
    # in the y direction (column), if we have a higher y_prime, the line has to go right (+)
    v = y_prime - y

    if x == 100 and y == 33:
      print(f"x:{x}, y:{y}")
      print(f"x_prime:{x_prime}, y_prime{y_prime}")
      print(f"x - x_prime :{x_prime}, y_prime - y: {y_prime}")
      print(f"u,v: {u, v}")
      print([u, v])

    elif x == 60 and y == 533:
      print(f"x:{x}, y:{y}")
      print(f"x_prime:{x_prime}, y_prime{y_prime}")
      print([u, v])

    if math.sqrt(u ** 2 + v ** 2) < 0.5:
      u = 0
      v = 0

    quiver[x, y] = np.array([u, v]).reshape((2,))
    # Increment the values
    x += increment
    if x + w_size >= H - 1:
      x = 0
      y += increment
      print(x, y)
  return return_image, image_2, quiver, H, W


'''
bilinear interpolation for filling in pixel value.
formula from https://structx.com/Bilinear_interpolation.html
'''


def bilinear_interpolate(x, y, x1, x2, y1, y2, Q11, Q12, Q21, Q22):
  # both values are same means we do not need to care
  if x1 == x2 and y1 == y2:
    return Q11
  # One of the values are the same entails we need to do some linear interpolation
  elif x1 == x2:
    return Q11 + (y - y1) * (Q12 - Q11) / (y2 - y1)
  elif y1 == y2:
    return Q11 + (x - x1) * (Q21 - Q11) / (x2 - x1)

  # If neither values are the same, we need to do bilinear interpollation
  else:
    T1 = (x2 - x) * (y2 - y) / (x2 - x1) / (y2 - y1) * Q11
    T2 = (x - x1) * (y2 - y) / (x2 - x1) / (y2 - y1) * Q12
    T3 = (x2 - x) * (y - y1) / (x2 - x1) / (y2 - y1) * Q21
    T4 = (x - x1) * (y - y1) / (x2 - x1) / (y2 - y1) * Q22
    return (T1 + T2 + T3 + T4)[0]


def gaussian1D(size):
  sigma = 1
  filter_size = size // 2
  output_list = list()
  for x in range(-filter_size, filter_size):
    value = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-1 / 2 * (x ** 2) / (sigma ** 2))
    output_list.append(value)

  output_list = np.array(output_list)
  output_list = output_list / sum(output_list)
  return np.array(output_list)


if __name__ == "__main__":
  image_1, image_2, quiver, H, W = lucas_kanade(os.getcwd() + "/../../A3/Q3_optical_flow/Q3_optical_flow/Dumptruck",
                                                w_size=15,
                                                overlap=True, picture_index=3, increment=10)

  # Mesh grid and picture size, show the picture size
  x, y = np.meshgrid(np.arange(0, W, 1), np.arange(0, H, 1))
  figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
  plt.imshow(image_1, cmap='gray')

  plt.quiver(x, y, quiver[:, :, 1], quiver[:, :, 0], scale_units='xy', scale=1, width=0.001, minshaft=1,
             minlength=0, color='red')
  plt.show()
