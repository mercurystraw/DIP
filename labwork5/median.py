import cv2
import numpy as np


def median_filter(image, kernel_size=3):
    height,width = image.shape[:2]
    pad = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    filtered_image = np.zeros_like(image)
    # 对每个像素进行滤波
    for i in range(height):
        for j in range(width):
            # 获取邻域区域
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            # 计取值
            filtered_image[i, j] = np.median(neighborhood)

    return filtered_image


# 自适应中值滤波函数
def adaptive_median_filter(image, max_size):
    temp = np.zeros_like(image)
    h, w = image.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            size = 3
            while size <= max_size:
                k = size // 2
                if i - k < 0 or i+k >= h or j-k < 0 or j+k >= w:
                    break
                window = image[i - k:i + k + 1, j - k:j + k + 1]
                z_min = window.min()
                z_max = window.max()
                z_med = np.median(window)
                z_xy = image[i, j]
                if z_min < z_med < z_max:
                    if z_min < z_xy < z_max:
                        temp[i, j] = z_xy
                    else:
                        temp[i, j] = z_med
                    break
                else:
                    size += 2
            else:
                print("这是else里的zmed",z_med)
                temp[i, j] = z_med

    return temp


# 读取图像
original_image = cv2.imread('./resources_labwork5/core.png', cv2.IMREAD_GRAYSCALE)
# 中值滤波
median_filtered = cv2.medianBlur(original_image, 3)
# 自适应中值滤波
adaptive_median_filtered = adaptive_median_filter(original_image, 7)
# 计算差值
difference = cv2.absdiff(median_filtered, adaptive_median_filtered)

# 保存差值结果图像
cv2.imwrite('./resources_labwork5/difference_image.png', difference)
# 保存结果图像
cv2.imwrite('./resources_labwork5/core_median_filtered.png', median_filtered)
cv2.imwrite('./resources_labwork5/core_adaptive_median_filtered.png', adaptive_median_filtered)
