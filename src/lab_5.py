import cv2
import numpy as np
import time

image1 = cv2.imread("../resources/monkey.png")
image2 = cv2.imread("../resources/eye.png")

image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

w, h = image2_gray.shape[::-1]

res = cv2.matchTemplate(image1_gray, image2_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(image1, top_left, bottom_right, (0, 0, 255), 10)
cv2.imshow('Detected', image1)
cv2.waitKey(0)


def freq_filter(shape, d0, n=None, type='gaussian'):
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    u, v = np.meshgrid(x, y)
    d = np.sqrt(u ** 2 + v ** 2)
    if type == 'butterworth':
        h = 1 / (1 + (d / d0) ** (2 * n))
    else:
        h = np.exp(-(d ** 2) / (2 * (d0 ** 2)))
    return h


F = np.fft.fft2(image1_gray)
Fshift = np.fft.fftshift(F)

d0_values = [0.05, 0.1, 0.15, 0.2, 0.25]

cv2.imshow('orig', image1_gray)
cv2.waitKey(0)

for d0 in d0_values:
    print("-" * 50 + f"d0 = {d0}" + "-" * 50)

    start_time = time.time()
    butterworth = freq_filter(image1_gray.shape, d0=d0, n=2, type='butterworth')
    filtered_image_bw = np.abs(np.fft.ifft2(np.fft.ifftshift(Fshift * butterworth)))
    end_time = time.time()
    print("Butterworth time:", d0, end_time - start_time)

    start_time = time.time()
    gaussian = freq_filter(image1_gray.shape, d0=d0)
    filtered_image_gauss = np.abs(np.fft.ifft2(np.fft.ifftshift(Fshift * gaussian)))
    end_time = time.time()
    print("Gaussian time:", d0, end_time - start_time)

    cv2.imshow(f"Butterworth d0 = {d0}", np.uint8(filtered_image_bw))
    cv2.imshow(f"Gaussian d0 = {d0}", np.uint8(filtered_image_gauss))
    cv2.waitKey(0)
cv2.destroyAllWindows()
