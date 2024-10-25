import cv2
import numpy as np



def compute_laplacian(image):
    # 转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    # 创建一个输出图像，初始化为零
    laplacian_image = np.zeros((height, width), dtype=np.float32)

    # 遍历每个像素，忽略边界
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            # 根据拉普拉斯算子的定义计算值
            laplacian_value = (
                    image[x + 1, y] + image[x - 1, y] +
                    image[x, y + 1] + image[x, y - 1] -
                    4 * image[x, y]
            )
            laplacian_image[x, y] = np.clip(laplacian_value, 0, 255)  # 限制结果在0到255之间

    return laplacian_image

def laplacian_sharpen(image):
    # 转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    c= -1.0
    laplacian = compute_laplacian(image)
    cv2.imwrite('./resources_labwork3/moon_laplacian.png', laplacian)

    # g(x, y) = f(x, y) + c * laplacian
    sharpened_image = cv2.addWeighted(image.astype(np.float32), 1, laplacian.astype(np.float32), c, 0)
    # 转换成8位防止溢出
    sharpened_image = cv2.convertScaleAbs(sharpened_image)
    return sharpened_image

img_moon = cv2.imread('./resources_labwork3/moon.png')

img_moon = img_moon.astype(np.float32)
img_moon_sharpened = laplacian_sharpen(img_moon)
moom_sharpened = cv2.imwrite('./resources_labwork3/moon_sharpened.png', img_moon_sharpened)


def mask_sharpen(image):
    # 转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调用之前写的中值滤波
    image_blurred = cv2.GaussianBlur(image, (5, 5), 2,cv2.BORDER_DEFAULT)
    mask = cv2.subtract(image, image_blurred)
    cv2.imwrite('./resources_labwork3/alphabet_mask.png', mask)

    k = 7.0
    # 将原图像与k倍的模板进行相加
    sharpened_image = cv2.addWeighted(image,1, mask, k,0)
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image

img_alphabet = cv2.imread('./resources_labwork3/alphabet.png')
sharpened_alphabet = mask_sharpen(img_alphabet)
cv2.imwrite('./resources_labwork3/alphabet_sharpened.png', sharpened_alphabet)