import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取灰度图像
image = cv2.imread('resources_labwork8/2fingerprint.png', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊
# 获取直方图
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
# 显示直方图
plt.figure(figsize=(6, 4))
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram, color='black')
plt.xlim([0, 256])  # 显示范围
plt.grid()
plt.show()

# 根据直方图选择全局阈值T，并且进行全局阈值化
T = 127
threshold_image = np.zeros(image.shape, np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] < T:
            threshold_image[i, j] = 0
        else:
            threshold_image[i, j] = 255

# 显示处理后的图像
cv2.imshow('Before Global Thresholding', image)
cv2.imshow('After Global Thresholding', threshold_image)
cv2.waitKey(0)
