import cv2
import numpy as np

image = cv2.imread("../resources/monkey.png")

cv2.imshow("Original", image)
cv2.waitKey(0)

kernel = np.array([[-0.1, 0.2, -0.1],
                   [0.2, 3.0, 0.2],
                   [-0.1, 0.2, -0.1]
                   ])

brightened_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
cv2.imshow('Bright', brightened_image)
cv2.waitKey(0)

border_image = cv2.copyMakeBorder(src=image, top=10, right=10, bottom=10, left=10,
                                  borderType=cv2.BORDER_REPLICATE)

cv2.imshow('Border', border_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
