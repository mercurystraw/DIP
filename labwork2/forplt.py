import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_air_and_spine():
    # 显示 air 图像
    air_images = [
        ('resources_labwork2/air.png', 'Original Air'),
        ('resources_labwork2/air_enhanced_gamma2.png', 'gamma 2'),
        ('resources_labwork2/air_enhanced_gamma4.png', 'gamma 4'),
        ('resources_labwork2/air_enhanced_gamma6.png', 'gamma 6')
    ]

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))

    # 显示 air 图像
    for i, (image_path, title) in enumerate(air_images):
        image = mpimg.imread(image_path)
        axs[0, i].imshow(image)
        axs[0, i].set_title(title)
        axs[0, i].axis('off')

    # 显示 spine 图像
    spine_images = [
        ('resources_labwork2/spine.png', 'Original Spine'),
        ('resources_labwork2/spine_enhanced_gamma0.2.png', 'gamma 0.2'),
        ('resources_labwork2/spine_enhanced_gamma0.4.png', 'gamma 0.4'),
        ('resources_labwork2/spine_enhanced_gamma0.6.png', 'gamma 0.6')
    ]

    for i, (image_path, title) in enumerate(spine_images):
        image = mpimg.imread(image_path)
        axs[1, i].imshow(image)
        axs[1, i].set_title(title)
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def show_pollen_images():
    # 显示 pollen 图像
    pollen_pairs = [
        ('resources_labwork2/pollen1.png', 'resources_labwork2/pollen1_equalized.png', 'Pollen 1'),
        ('resources_labwork2/pollen2.png', 'resources_labwork2/pollen2_equalized.png', 'Pollen 2'),
        ('resources_labwork2/pollen3.png', 'resources_labwork2/pollen3_equalized.png', 'Pollen 3'),
        ('resources_labwork2/pollen4.png', 'resources_labwork2/pollen4_equalized.png', 'Pollen 4')
    ]

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))

    for i, (original_path, enhanced_path, title) in enumerate(pollen_pairs):
        # 原图显示（灰度）
        original_image = mpimg.imread(original_path)
        axs[i // 2, (i % 2) * 2].imshow(original_image, cmap='gray')
        axs[i // 2, (i % 2) * 2].set_title(title + ' Original')
        axs[i // 2, (i % 2) * 2].axis('off')

        # 增强图显示（灰度）
        enhanced_image = mpimg.imread(enhanced_path)
        axs[i // 2, (i % 2) * 2 + 1].imshow(enhanced_image, cmap='gray')
        axs[i // 2, (i % 2) * 2 + 1].set_title(title + ' equalized')
        axs[i // 2, (i % 2) * 2 + 1].axis('off')

    plt.tight_layout()
    plt.show()

# 调用函数显示图像
show_air_and_spine()
show_pollen_images()
