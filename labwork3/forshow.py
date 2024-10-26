import matplotlib.pyplot as plt
import cv2

def show_filtered_images(image_name):
    original_path = f'./resources_labwork3/{image_name}.png'
    gaussian_filtered_path = f'./resources_labwork3/gaussian_filtered_{image_name.replace("img", "image")}.png'
    median_filtered_path = f'./resources_labwork3/median_filtered_{image_name.replace("img", "image")}.png'

    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    gaussian_img = cv2.imread(gaussian_filtered_path, cv2.IMREAD_GRAYSCALE)
    median_img = cv2.imread(median_filtered_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 高斯滤波图像
    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_img, cmap='gray')
    plt.title('Gaussian Filtered Image')
    plt.axis('off')

    # 中值滤波图像
    plt.subplot(1, 3, 3)
    plt.imshow(median_img, cmap='gray')
    plt.title('Median Filtered Image')
    plt.axis('off')

    plt.show()


# 直接填充函数调用
show_filtered_images('img1_1')
show_filtered_images('img1_2')
show_filtered_images('img2_1')
show_filtered_images('img2_2')


