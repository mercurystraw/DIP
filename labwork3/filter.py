import cv2
import numpy as np


def padding_and_filter(image, kernel):
    height,width = image.shape[:2]
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    # 对四周进行padding填充
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_DEFAULT)

    # 创建输出图像
    filtered_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(neighborhood * kernel)

    return filtered_image


def gaussian_filter(image, kernel_size=3, sigma=1.0):
    # 生成3*3大小高斯核
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    # 归一化
    kernel = kernel / np.sum(kernel)

    filtered_image = padding_and_filter(image, kernel)
    return filtered_image

def median_filter(image, kernel_size=3):
    height,width = image.shape[:2]
    pad = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_DEFAULT)


    filtered_image = np.zeros_like(image)
    # 对每个像素进行滤波
    for i in range(height):
        for j in range(width):
            # 获取邻域区域
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            # 计取值
            filtered_image[i, j] = np.median(neighborhood)

    return filtered_image


# 读取原始图像
image1_1 = cv2.imread('./resources_labwork3/img1_1.png', cv2.IMREAD_GRAYSCALE)
image1_2 = cv2.imread('./resources_labwork3/img1_2.png', cv2.IMREAD_GRAYSCALE)
image2_1 = cv2.imread('./resources_labwork3/img2_1.png', cv2.IMREAD_GRAYSCALE)
image2_2 = cv2.imread('./resources_labwork3/img2_2.png', cv2.IMREAD_GRAYSCALE)

# 应用高斯滤波
gaussian_filtered_image1_1 = gaussian_filter(image1_1)
gaussian_filtered_image1_2 = gaussian_filter(image1_2)
gaussian_filtered_image2_1 = gaussian_filter(image2_1)
gaussian_filtered_image2_2 = gaussian_filter(image2_2)
#均值滤波
median_filtered_image1_1 = median_filter(image1_1)
median_filtered_image1_2 = median_filter(image1_2)
median_filtered_image2_1 = median_filter(image2_1)
median_filtered_image2_2 = median_filter(image2_2)


# 保存滤波结果
cv2.imwrite('./resources_labwork3/gaussian_filtered_image1_1.png', gaussian_filtered_image1_1)
cv2.imwrite('./resources_labwork3/gaussian_filtered_image1_2.png', gaussian_filtered_image1_2)
cv2.imwrite('./resources_labwork3/gaussian_filtered_image2_1.png', gaussian_filtered_image2_1)
cv2.imwrite('./resources_labwork3/gaussian_filtered_image2_2.png', gaussian_filtered_image2_2)
cv2.imwrite('./resources_labwork3/median_filtered_image1_1.png', median_filtered_image1_1)
cv2.imwrite('./resources_labwork3/median_filtered_image1_2.png', median_filtered_image1_2)
cv2.imwrite('./resources_labwork3/median_filtered_image2_1.png', median_filtered_image2_1)
cv2.imwrite('./resources_labwork3/median_filtered_image2_2.png', median_filtered_image2_2)

print("过滤后的图像已保存。")
