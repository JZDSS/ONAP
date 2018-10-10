import cv2
import numpy as np
import math


img = cv2.imread('images/0/8.jpg', 0)
def add_noise(img):
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            a = 255 - img[w, h]
            p = 1 / (1 + math.exp(-(a - 220)))
            dice = np.random.uniform(0, 1, ())
            if dice < p:
                if np.random.uniform(0, 1, ()) > 0.93:
                    img[w, h] = 255
    n1 = (np.random.randn(256, 256)*15).astype(np.uint8)
    n2 = (np.random.randn(256, 256) * 15).astype(np.uint8)
    img = img + n1 - n2
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img

img = add_noise(img)