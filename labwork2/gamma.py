import cv2
import numpy as np

def gamma_transformation(image, gamma):
    c = 1  # 常数 c 取1
    image = image / 255.0  # 将图像归一化到 [0, 1]
    enhanced_image = c * (image ** gamma)
    enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)  # 还原到 [0, 255]
    return enhanced_image


# 读取图像
spine_image = cv2.imread('./resources_labwork2/spine.png')
air_image = cv2.imread('./resources_labwork2/air.png')

# 定义不同的 gamma 值
gammas_spine = [0.2,0.4, 0.6 ]
gammas_air = [2, 4, 6 ]

# 对每幅图像进行变换并保存结果
for i, gamma in enumerate(gammas_spine):
    enhanced_spine = gamma_transformation(spine_image, gamma)
    # 保存增强后的图像
    cv2.imwrite(f'./resources_labwork2/spine_enhanced_gamma{gamma}.png', enhanced_spine)


for i, gamma in enumerate(gammas_air):
    enhanced_air = gamma_transformation(air_image, gamma)
    # 保存增强后的图像
    cv2.imwrite(f'./resources_labwork2/air_enhanced_gamma{gamma}.png', enhanced_air)

print("增强后的图像已保存。")
