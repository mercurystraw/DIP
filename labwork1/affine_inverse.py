from PIL import Image
import numpy as np

# 读取图像
image = Image.open('../resources/seu.png')
init_image_array = np.array(image)
original_height, original_width, channels = init_image_array.shape

# 存储校徽在大平面上的图片 用于后续比较
transformed_height = original_height * 4  # 增加高度
transformed_width = original_width * 4  # 增加宽度
transformed_image = np.zeros((transformed_height, transformed_width, channels),dtype=init_image_array.dtype)
transformed_image[0:original_height, 0:original_width] = init_image_array
init_image = Image.fromarray(transformed_image)
init_image.save('../resources/seu_initial.png')

def affine_transform(image_array, transform_matrix):
    # 获取图像尺寸
    height, width, channels = image_array.shape
    transformed_height = original_height * 4  # 增加高度
    transformed_width = original_width * 4  # 增加宽度
    transformed_image = np.zeros((transformed_height, transformed_width, channels), dtype=image_array.dtype)

    for y in range(height):
        for x in range(width):
            # 原始坐标
            original_coordinate = np.array([x, y, 1])  # 增加齐次坐标
            # 计算新坐标 矩阵乘法
            new_coordinate = transform_matrix @ original_coordinate

            new_x, new_y = int(np.round(new_coordinate[0])), int(np.round(new_coordinate[1]))

            # 确保新坐标在图像范围内
            if 0 <= new_x < transformed_width and 0 <= new_y < transformed_height:
                transformed_image[new_y, new_x] = image_array[y, x]

    return transformed_image


# 平移变换
tx, ty = 200, 200
translation_matrix = np.array([[1, 0, tx],
                               [0, 1, ty],
                               [0, 0, 1]])
# 应用平移变换
translated_image = affine_transform(init_image_array, translation_matrix)
# 确保数据类型为 uint8
translated_image = np.clip(translated_image, 0, 255).astype(np.uint8)
Image.fromarray(translated_image).save('../resources/seu_translated.png')

# 平移的逆变换 调用Numpy的linalg.inv()函数求逆矩阵
inverted_translation_matrix = np.linalg.inv(translation_matrix)
translated_image_array = np.array(translated_image)
# 对逆变换的结果进行处理
inverted_translated_image = affine_transform(translated_image_array, inverted_translation_matrix)
inverted_translated_image = np.clip(inverted_translated_image, 0, 255).astype(np.uint8)
Image.fromarray(inverted_translated_image).save('../resources/seu_translated_inverted.png')


# 旋转变换
angle = 90  # 旋转角度（度）
theta = np.radians(angle)  # 转换为弧度
rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                             [-np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]])
# 计算图像中心坐标
center_x, center_y = original_width / 2, original_height / 2
# 创建平移矩阵使图像中心对齐到原点
translate_to_origin = np.array([[1, 0, -center_x],
                                 [0, 1, -center_y],
                                 [0, 0, 1]])
# 创建平移矩阵使图像中心回到原来的位置
translate_back = np.array([[1, 0, center_x],
                            [0, 1, center_y],
                            [0, 0, 1]])
# 组合变换矩阵
combined_matrix = translate_back @ (rotation_matrix @ translate_to_origin)

# 应用旋转变换
rotated_image = affine_transform(init_image_array, combined_matrix)
# 确保数据类型为 uint8
rotated_image = np.clip(rotated_image, 0, 255).astype(np.uint8)
Image.fromarray(rotated_image).save('../resources/seu_rotated.png')

# 旋转的逆变换矩阵
inverted_rotation_matrix = np.array([[np.cos(-theta), np.sin(-theta), 0],
                                      [-np.sin(-theta), np.cos(-theta), 0],
                                      [0, 0, 1]])
# 计算逆变换的组合矩阵
inverted_combined_matrix = translate_back @ (inverted_rotation_matrix @ translate_to_origin)
# 应用逆旋转变换
inverted_rotated_image = affine_transform(rotated_image, inverted_combined_matrix)
inverted_rotated_image = np.clip(inverted_rotated_image, 0, 255).astype(np.uint8)
Image.fromarray(inverted_rotated_image).save('../resources/seu_rotated_inverted.png')

# 放缩变换矩阵
cx, cy = 1.5, 1.5  # x 和 y 方向的缩放因子
scaling_matrix = np.array([[cx, 0, 0],
                            [0, cy, 0],
                            [0, 0, 1]])

# 放缩变换
scaled_image = affine_transform(init_image_array, scaling_matrix)
scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)
Image.fromarray(scaled_image).save('../resources/seu_scaled.png')

# 放缩的逆变换 可以直接写出来
inverted_scaling_matrix = np.array([[1/cx, 0, 0],
                                     [0, 1/cy, 0],
                                     [0, 0, 1]])
inverted_scaled_image = affine_transform(scaled_image, inverted_scaling_matrix)
inverted_scaled_image = np.clip(inverted_scaled_image, 0, 255).astype(np.uint8)
Image.fromarray(inverted_scaled_image).save('../resources/seu_scaled_inverted.png')