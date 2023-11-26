import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("../resources/monkey.png")
cv.imshow("img", img)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

colors = ('b', 'g', 'r')
plt.figure(figsize=(16, 9))
for i, color in enumerate(colors):
    histogram = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
