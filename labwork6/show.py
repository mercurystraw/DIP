import os
from PIL import Image
import matplotlib.pyplot as plt

# 文件路径
base_path = 'resources_labwork6'

# 读取图像文件
noisy_image_path = os.path.join(base_path, 'noisy_image.tif')
denoised_images = {
    'coif1': os.path.join(base_path, 'denoised_image_coif1.tif'),
    'db1': os.path.join(base_path, 'denoised_image_db1.tif'),
    'db2': os.path.join(base_path, 'denoised_image_db2.tif'),
    'haar': os.path.join(base_path, 'denoised_image_haar.tif'),
    'sym2': os.path.join(base_path, 'denoised_image_sym2.tif')
}


def show_images():
    # 显示噪声图像和五种去噪图像（共6张）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示噪声图像
    noisy_image = Image.open(noisy_image_path)
    axes[0, 0].imshow(noisy_image, cmap='gray')
    axes[0, 0].set_title('Noisy Image')
    axes[0, 0].axis('off')

    # 显示去噪图像
    for ax, (key, path) in zip(axes.flatten()[1:], denoised_images.items()):
        image = Image.open(path)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Denoised {key}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 显示边缘检测图像
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for ax, (key, path) in zip(axes, denoised_images.items()):
        edge_path = os.path.join(base_path, f'edges_{key}.tif')
        image = Image.open(edge_path)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Edges {key}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_images()
