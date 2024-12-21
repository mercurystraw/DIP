import cv2
import numpy as np
from skimage.morphology import skeletonize
# 读取图像
input_image = cv2.imread('./resources_labwork7/vessel.png')
# 检查图像是否为彩色，如果是则转换为灰度
if len(input_image.shape) == 3:
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = input_image

# 二值化
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 图像形态学细化
skeleton_image = skeletonize(binary_image // 255) # 要求输入0和1之间的值

# 叠加到原图
overlay_image = input_image.copy()
rows, cols = np.where(skeleton_image)
for r, c in zip(rows, cols):
    overlay_image[r, c] = [0, 0, 255]  # 用红色中心线

# 显示结果
cv2.imshow('原始图像', input_image)
cv2.imshow('叠加中心线的图像', overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite('./resources_labwork7/result_vessel.png', overlay_image)
