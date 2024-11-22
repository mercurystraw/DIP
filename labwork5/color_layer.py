import cv2
import numpy as np


def color_layering(image, target_color, threshold, neutral_color):
    # 获取图像尺寸
    rows, cols, channels = image.shape

    # 创建结果图像
    result = np.copy(image)

    # 遍历每个像素
    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            color_diff = np.sum((pixel - target_color) ** 2)

            if color_diff > threshold:
                # 替换为中性色
                result[i, j] = neutral_color
            else:
                # 保持原色
                result[i, j] = pixel

    return result


# 读取图像
image = cv2.imread('./resources_labwork5/strawberry.png')

# 定义目标颜色和阈值
target_color = np.array([0, 0, 255])  # 范例目标颜色（红色） OpenCV BGR 格式
threshold = 21000              # 范例阈值

netural_color = np.array([128, 128, 128])  # 范例中性色（灰色）
# 应用彩色分层法
layered_image = color_layering(image, target_color, threshold, netural_color)

# 显示结果
cv2.imwrite('./resources_labwork5/layered_image.png', layered_image)
