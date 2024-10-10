import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 定义函数来展示图像
def show_images(images, titles, rows=1, cols=5):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(mpimg.imread(images[i]))
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 图片和标题的列表
images_list_rotated = [
    "./resources_labwork1/seu_initial.png",
    "./resources_labwork1/seu_rotated.png",
    "./resources_labwork1/seu_rotated_inverted.png",
    "./resources_labwork1/seu_rotated_registration_image.png",
    "./resources_labwork1/seu_rotated_difference_image.png"
]

titles_rotated = ["Original", "Rotated", "Inverted", "Registered", "Difference"]

images_list_scaled = [
    "./resources_labwork1/seu_initial.png",
    "./resources_labwork1/seu_scaled.png",
    "./resources_labwork1/seu_scaled_inverted.png",
    "./resources_labwork1/seu_scaled_registration_image.png",
    "./resources_labwork1/seu_scaled_difference_image.png"
]

titles_scaled = ["Original", "Scaled", "Inverted", "Registered", "Difference"]

images_list_translated = [
    "./resources_labwork1/seu_initial.png",
    "./resources_labwork1/seu_translated.png",
    "./resources_labwork1/seu_translated_inverted.png",
    "./resources_labwork1/seu_translated_registration_image.png",
    "./resources_labwork1/seu_translated_difference_image.png"
]

titles_translated = ["Original", "Translated", "Inverted", "Registered", "Difference"]

# 显示图片
show_images(images_list_rotated, titles_rotated)
show_images(images_list_scaled, titles_scaled)
show_images(images_list_translated, titles_translated)
