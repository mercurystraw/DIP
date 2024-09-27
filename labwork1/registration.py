import numpy as np
from PIL import Image



# 读取图像
image = Image.open('../resources/seu.png')
init_image_array = np.array(image)
original_height, original_width, channels = init_image_array.shape

translated_image = Image.open('../resources/seu_translated.png')
translated_image_array = np.array(translated_image)

rotated_image = Image.open('../resources/seu_rotated.png')
rotated_image_array = np.array(rotated_image)

scaled_image = Image.open('../resources/seu_scaled.png')
scaled_image_array = np.array(scaled_image)


def affine_transform_plus(image_array, transform_matrix):
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
            new_coordinate = original_coordinate @ transform_matrix

            new_x, new_y = int(np.round(new_coordinate[0])), int(np.round(new_coordinate[1]))

            # 确保新坐标在图像范围内
            if 0 <= new_x < transformed_width and 0 <= new_y < transformed_height:
                transformed_image[new_y, new_x] = image_array[y, x]

    return transformed_image
def compute_registration_transform(points_a, points_b):
    # 输入的点集，points_a 和 points_b 都是形状为 (n, 2) 的数组
    assert points_a.shape == points_b.shape, "Point sets must have the same shape"

    # 将点转换为齐次坐标
    n = points_a.shape[0]
    P = np.hstack((points_a, np.ones((n, 1))))  # 扩展成三维坐标 Shape: (n, 3)
    Q = np.hstack((points_b, np.ones((n, 1))))  # Shape: (n, 3)

    # 使用最小二乘法求解 T
    T = np.linalg.inv(P.T @ P) @ P.T @ Q # 最小二乘法闭合解公式

    return T


points_initial = np.array([[200,52], [201,204], [3,127],[226,129],[88,46],[89,207],[164,62],[164,192]]) # 左点右点上点 下点左点右点 1 京
points_translated = np.array([[402,253] ,[402,404], [203,329],[426,329],[289,248],[290,412], [364,262],[363,391]])
points_rotated = np.array([[53, 56],[202,55],[125,252],[129,28],[45,167],[208,166], [61,91],[190,93]])
points_scaled = np.array([[302,79],[302,305],[6,188],[342,194],[133,69],[133,314],[245, 92],[245,287]])
# 平移配准
translated_T = compute_registration_transform(points_initial, points_translated)
inversed_translated_T = np.linalg.inv(translated_T)

translated_registration_image = affine_transform_plus(translated_image_array, inversed_translated_T)

translated_registration_image = Image.fromarray(translated_registration_image)
translated_registration_image.save('../resources/seu_translated_registration_image.png')

# 旋转配准
rotated_T = compute_registration_transform(points_initial, points_rotated)
inversed_rotated_T = np.linalg.inv(rotated_T)

rotated_registration_image = affine_transform_plus(rotated_image_array, rotated_T)
rotated_registration_image = Image.fromarray(rotated_registration_image)
rotated_registration_image.save('../resources/seu_rotated_registration_image.png')

# 缩放配准
scaled_T = compute_registration_transform(points_initial, points_scaled)
inversed_scaled_T = np.linalg.inv(scaled_T)

scaled_registration_image = affine_transform_plus(scaled_image_array, inversed_scaled_T)
scaled_registration_image = Image.fromarray(scaled_registration_image)
scaled_registration_image.save('../resources/seu_scaled_registration_image.png')