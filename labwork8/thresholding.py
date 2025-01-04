import cv2
import numpy as np
# 读取灰度图像
image = cv2.imread('resources_labwork8/2fingerprint.png', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊

# 初始化阈值T，使用图像的平均值
T = int(np.mean(image))

# 基本全局阈值处理算法，迭代式地求T
while True:
    # 用T分割图像
    foreground = image[image >= T]
    background = image[image < T]
    # 分别计算两部分地均值
    mu1 = foreground.mean()
    mu2 = background.mean()

    # 更新阈值
    T_new = int((mu1 + mu2) / 2)
    # 判断阈值是否收敛
    if abs(T_new - T) < 1:  # 设定收敛误差为1
        break

    T = T_new

print('Final threshold value:', T)
# 根据最终的阈值进行全局阈值化
threshold_image = np.zeros(image.shape, np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] >= T:
            threshold_image[i, j] = 255
        else:
            threshold_image[i, j] = 0

# 显示处理后的图像
cv2.imshow('Before Global Thresholding', image)
cv2.imshow('After Global Thresholding', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
