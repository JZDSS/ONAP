import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

root = 'images2/0/'
for name in os.listdir(root):
    img = cv2.imread(os.path.join(root, name), 0)
    filtered = cv2.medianBlur(img, 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    filtered = cv2.filter2D(filtered, -1, kernel=kernel)

    cv2.imshow("", filtered)
    cv2.waitKey(0)
    # filtered = cv2.bilateralFilter(img, 3, 75, 75)
    # plt.subplot(2, 2, 2)
    # plt.imshow(filtered, cmap='gray')
    # plt.title('bilater')
    #
    # filtered = cv2.medianBlur(filtered, 3)
    # plt.subplot(2, 2, 4)
    # plt.imshow(filtered, cmap='gray')
    # plt.title('both')
    #
    # filtered = cv2.medianBlur(img, 3)
    # plt.subplot(2, 2, 3)
    # plt.imshow(filtered, cmap='gray')
    # plt.title('med')
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('ori')
    # plt.show()