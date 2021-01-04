import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import os


def histogram_of_oriented_gradients(image):
  image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
  H, W = image.shape
  original_image = image

  # gradient magnitudes and directions
  image = cv2.GaussianBlur(image, (9, 9), 6, 6)
  sobel_imagex = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
  sobel_imagey = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
  sobel_imagex = (sobel_imagex / np.ndarray.max(sobel_imagex)) * 255
  sobel_imagey = (sobel_imagey / np.ndarray.max(sobel_imagey)) * 255

  grad_magnitude = np.sqrt(sobel_imagex ** 2 + sobel_imagey ** 2)
  grad_direction = np.arctan2(sobel_imagey, sobel_imagex) * 180 / np.pi

  # Any direction less than 0 is converted to its positive counterpart
  grad_direction = np.where(grad_direction < 0, 180 + grad_direction, grad_direction)

  # Turn values greater than 165 into their negative counterpart
  grad_direction = np.where(grad_direction >= 165, grad_direction - 180, grad_direction)

  # Convert to their bin numbers
  bin_number = np.where(grad_direction < 0, 1, grad_direction)
  bin_number = np.where(bin_number > 0, bin_number // 15 + 1, bin_number)

  tao = 8

  ceiledH = math.ceil(H / 8)
  ceiledW = math.ceil(W / 8)

  # Use this to convert our angles to u and v
  U = np.zeros(shape=(ceiledH, ceiledW, 6))
  V = np.zeros(shape=(ceiledH, ceiledW, 6))
  HoG = np.zeros(shape=(ceiledH, ceiledW, 6))

  x = 0
  y = 0

  while x + tao < H and y + tao < W:
    for i in range(0, tao):
      for j in range(0, tao):
        u = x + i
        v = y + j

        # assume that each angle is just
        final_bin = bin_number[u, v] // 2
        angle = final_bin * 30

        # calculate the length for a unit vector
        # should compensate for negatives automatically as well
        u_comp = np.sin(np.deg2rad(angle))
        v_comp = np.cos(np.deg2rad(angle))

        # multiply the unit vector by the magnitude to show proportion
        # this is done to help visualize

        U[x // 8, y // 8, int(bin_number[u, v] // 2)] += u_comp * grad_magnitude[u, v]
        V[x // 8, y // 8, int(bin_number[u, v] // 2)] += v_comp * grad_magnitude[u, v]
        HoG[x // 8, y // 8, int(bin_number[u, v] // 2)] += grad_magnitude[u, v]

        # Calculate the actual matrix, for part 2

    # ignoring last blocks for convenience
    x += tao
    if x > H - tao - 1:
      x = 0
      y += tao
  return original_image, HoG, U, V, ceiledH, ceiledW


def visualize_HOG(original_image, U, V, ceiledH, ceiledW):
  # U and V are the actual values, everything done to U_norm and V_norm are to help visualize it properly
  U_norm = U.copy()
  V_norm = V.copy()
  H, W = original_image.shape


  # Normalize U, V with respect to their bins for visualization
  for x in range(0, ceiledH):
    for y in range(0, ceiledW):
      U_norm[x, y, :] = U[x, y, :] / np.sum(np.abs(U[x, y, :]))
      V_norm[x, y, :] = V[x, y, :] / np.sum(np.abs(V[x, y, :]))

  # Normalization might lead to divide by 0
  U_norm = np.nan_to_num(U_norm)
  V_norm = np.nan_to_num(V_norm)

  # scale it up to image size
  U_scaled = np.zeros(shape=(H, W, 6))
  V_scaled = np.zeros(shape=(H, W, 6))

  for x in range(0, ceiledH -1):
    for y in range(0, ceiledW - 1):
      # lets just put it at the 3rd value for visual purposes
      # slighly off centre but it works
      U_scaled[x * 8 + 4, y * 8 + 4, :] = U_norm[x, y, :]
      V_scaled[x * 8 + 4, y * 8 + 4, :] = V_norm[x, y, :]

  # plot the quiver plots
  for t in range(0, 6):
    plt.quiver(U_scaled[:, :, t], V_scaled[:, :, t], scale=60, width=0.0012, color='red', minshaft=1, minlength=0, )

  # just to make its symmetric
  for t in range(0, 6):
    plt.quiver(np.negative(U_scaled[:, :, t]), np.negative(V_scaled[:, :, t]),
               scale=60, width=0.0012, minshaft=1, minlength=0, color='red')
  plt.imshow(original_image, cmap='gray')

  # save a high res copy
  plt.savefig('3_quiver.png', dpi=300)
  plt.show()


def L2_normalization(HoG, filename):
  H, W, D = HoG.shape
  block_size = 2

  normalized = np.zeros(shape=(H - 1, W - 1, block_size ** 2 * D))

  x = 0
  y = 0
  while y < W - 1:

    intermediate = np.array([])

    for i in range(0, block_size):
      for j in range(0, block_size):
        u = x + i
        v = y + j
        intermediate = np.append(intermediate, HoG[u, v, :])

    # Flatten and normalizing
    intermediate = intermediate.flatten()
    denominator = np.sqrt(np.sum(intermediate ** 2))

    intermediate = intermediate / denominator
    normalized[x, y, :] = intermediate
    x += 1
    if x >= H - 1:
      x = 0
      y += 1

  try:
    os.remove(filename + ".txt")
  except OSError as error:
    print("")
    # do nothing

  f = open(filename + ".txt", "x")

  np.set_printoptions(linewidth=np.inf)
  # start with each depth
  for d in range(0, D):
    f.write(f"D={d}")
    # go through each row
    for i in range(0, H - 1):
      f.write(f"|Row={i}|")
      # add all the contents for a specific row at a certain depth
      f.write(str(normalized[i, :, d]))


filename = "3"
original_image, HoG, U, V, ceiledH, ceiledW = histogram_of_oriented_gradients(
  "/home/john/Documents/CSC420/A3/Q4/Q4/" + filename + ".jpg")
visualize_HOG(original_image, U, V, ceiledH, ceiledW)
L2_normalization(HoG, filename)
