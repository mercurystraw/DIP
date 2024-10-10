import cv2
import numpy as np
import os
# 定义四张图像的路径
image_paths = [
    './resources_labwork2/pollen1.png',
    './resources_labwork2/pollen2.png',
    './resources_labwork2/pollen3.png',
    './resources_labwork2/pollen4.png'
]


def histogram_equalization(img):
    # 计算图像的直方图
    # 调用flatten函数，将图像展平为一维数组用于灰度累计
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])

    # 计算累积分布函数（CDF）
    cdf = hist.cumsum()

    # 将CDF归一化到[0, 255]范围

    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    # 重新变为原图像的形状
    img_equalized = cdf_normalized[img.flatten()].reshape(img.shape).astype(np.uint8)

    return img_equalized


# 遍历图像路径
for image_path in image_paths:
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 进行直方图均衡化
    equalized_img = histogram_equalization(img)

    # 设置直方均衡后的图像名字并且存储
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_equalized.png"  # 新文件名
    output_path = os.path.join(os.path.dirname(image_path), output_filename)
    cv2.imwrite(output_path, equalized_img)

print("处理完成！")

