import cv2
import numpy as np

# 读取图像
input_image = cv2.imread('./resources_labwork7/vessel.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否为彩色，如果是则转换为灰度
if input_image.ndim == 3:
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = input_image

# 二值化
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 定义核
kernel = np.ones((3, 3), np.uint8)
skeleton = np.zeros_like(binary_image)
# 骨架提取 反复的腐蚀和膨胀
while True:
    # 腐蚀
    eroded = cv2.erode(binary_image, kernel)
    # 膨胀
    dilated = cv2.dilate(eroded, kernel)
    # 不断将检测到的骨架部分添加到已有的骨架上
    skeleton = cv2.bitwise_or(skeleton, cv2.subtract(binary_image, dilated))
    if cv2.countNonZero(eroded) == 0: # 如果腐蚀后的图像不再包括前景像素
        break

    binary_image = eroded.copy()

# 叠加到原图
overlay_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
rows, cols = np.where(skeleton == 255)
for r, c in zip(rows, cols):
    overlay_image[r, c] = [0, 0, 255]  # 用红色表示骨架

# 显示结果
cv2.imshow('原始图像', input_image)
cv2.imshow('骨架图像', overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite('./resources_labwork7/result_vessel_skeleton.png', overlay_image)


