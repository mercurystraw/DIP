import numpy as np
from PIL import Image
import os

image_pth = "./resources_labwork1/lena.png" # 图片资源存储在resources文件夹下，这里保存使用图片的路径

init_image = np.array(Image.open(image_pth)) # 读取图片


def nearest_neighbor_interpolation(image, new_height, new_width): # 最近邻插值
    old_height, old_width = image.shape[:2] # 获取原图的大小
    channels = image.shape[2] # 获取图片的通道数,灰度图像一般为1, RGB图像为3

    new_image = np.zeros((new_height, new_width, channels),dtype=np.uint8) # (H,W,C)形式 unit8表示无符号的 8 位整数，像素值范围为0-255

    h_rate = old_height/new_height # 计算height方向的缩放比例
    w_rate = old_width/new_width # 计算width方向的缩放比例
    for i in range(new_height):
        for j in range(new_width):
            x = min(round(i*h_rate), old_height-1) # 采用round函数对坐标进行四舍五入，从而找到映射前距离最近的整数(也就是最近邻插值),同时防止越界
            y = min(round(j*w_rate), old_width-1) # 同x
            new_image[i, j] = image[x, y] # 注意是(H,W,C)形式
            # 防止插值后的值越界
            new_image[i, j] = np.clip(new_image[i, j], 0, 255) # 确保新图像中每个像素的值在0到255之间

    return new_image


def linear_interpolation(image, new_height, new_width): # 线性插值
    old_height, old_width = image.shape[:2]
    channels = image.shape[2]
    new_image = np.zeros((new_height, new_width, channels),dtype=np.uint8)
    h_rate = old_height/new_height
    w_rate = old_width/new_width
    for i in range(new_height):
        for j in range(new_width): # 这里都同上
            x = i*h_rate
            y = j*w_rate
            x0 = int(x) # 向下取整得到x0
            y0 = int(y) # 向下取整得到y0
            # 计算小数部分
            x_diff = x - x0
            y_diff = y - y0
            if x0 == old_height - 1 or y0 == old_width - 1:
                new_image[i, j] = image[x0, y0]
                continue
            # 处理映射的不同情况
            if x_diff == 0 and y_diff == 0:
                # 正好映射到原图像的整数坐标
                new_image[i, j] = image[x0, y0]
            elif x_diff == 0:
                # y_diff != 0
                y1 = y0 + (1 if y_diff > 0 else -1)
                # x 为整数，y 不为整数
                new_image[i,j] = (y1-y)*image[x0, y0]/(y1-y0)+y_diff*image[x0, y1]/(y1-y0)
            elif y_diff == 0:
                x1 = x0 + (1 if x_diff > 0 else 0)
                # y 为整数，x 不为整数
                new_image[i,j] = (x1-x)*image[x0, y0]/(x1-x0)+x_diff*image[x1,y0]/(x1-x0)
            else: # x,y都不为整数
                x1 = x0 + (1 if x_diff > 0 else 0)
                y1 = y0 + (1 if y_diff > 0 else -1)
                new_image[i,j] = (x1-x)*image[x0, y0]+x_diff*image[x1, y1]/(x1-x0) # 线性插值公式
            new_image[i, j] = np.clip(new_image[i, j], 0, 255)
    return new_image


def bilinear_interpolation(image, new_height, new_width): # 双线性插值
    old_height, old_width = image.shape[:2]
    channels = image.shape[2]

    new_image = np.zeros((new_height, new_width, channels),dtype=np.uint8)
    h_rate = old_height/new_height
    w_rate = old_width/new_width

    for i in range(new_height):
        for j in range(new_width): # 这里都同上
            x = i*h_rate
            y = j*w_rate
            x0 = int(x)
            y0 = int(y)
            # 计算小数部分
            x_diff = x - x0
            y_diff = y - y0
            if x0 == old_height - 1 or y0 == old_width - 1:
                new_image[i, j] = image[x0, y0]
                continue
            if x_diff == 0 and y_diff == 0:
                # 正好映射到原图像的整数坐标
                new_image[i, j] = image[x0, y0]
            elif x_diff == 0:
                y1 = y0 + (1 if y_diff > 0 else -1)
                new_image[i,j] = (y1-y)*image[x0, y0]/(y1-y0)+y_diff*image[x0, y1]/(y1-y0)
            elif y_diff == 0:
                x1 = x0 + (1 if x_diff > 0 else 0)
                # y 为整数，x 不为整数
                new_image[i, j] = (x1-x)*image[x0, y0]/(x1-x0)+x_diff*image[x1, y0]/(x1-x0)
            else: # x,y都不为整数
                x1 = x0 + (1 if x_diff > 0 else 0)
                y1 = y0 + (1 if y_diff > 0 else -1)
                new_image[i, j] = (x1-x)*(y1-y)*image[x0,y0]/((x1-x0)*(y1-y0))
                + (x-x0)*(y1-y)*image[x1,y0]/((x1-x0)*(y1-y0))
                + (x1-x)*(y-y0)*image[x0,y1]/((x1-x0)*(y1-y0))
                + (x-x0)*(y-y0)*image[x1,y1]/((x1-x0)*(y1-y0))
                # 双线性插值公式
            new_image[i, j] = np.clip(new_image[i, j], 0, 255)
    return new_image


# 双三次插值权重函数
def Bicubic(x,a=-0.5): # 选取了a=-0.5
    x = abs(x)
    # 双三次插值公式
    if x <= 1:
        return (a + 2) * (x ** 3) - (a + 3) * (x ** 2) + 1
    elif x < 2:
        return a * (x ** 3) - 5 * a * (x ** 2) + 8 * a * x - 4 * a
    else:
        return 0


def bicubic_interpolation(image, new_height, new_width): # 三次样条插值
    old_height, old_width = image.shape[:2]
    channels = image.shape[2]

    new_image = np.zeros((new_height, new_width, channels),dtype=np.uint8)
    h_rate = old_height/new_height
    w_rate = old_width/new_width

    for i in range(new_height):
        for j in range(new_width): # 这里都同上
            x = i*h_rate
            y = j*w_rate
            x0 = int(x)
            y0 = int(y)
            # 计算小数部分
            x_diff = x - x0
            y_diff = y - y0

            value = np.zeros(channels)

            for m in range(-1,3):
                for n in range(-1,3):
                    if 0<=x0+m<old_height and 0<=y0+n<old_width:
                        weight = Bicubic(m-x_diff)*Bicubic(n-y_diff)
                        value += weight*image[x0+m,y0+n]
            new_image[i,j] = np.clip(value,0,255)  # 确保新图像中每个像素的值在0到255之间

    return new_image


old_height, old_width = init_image.shape[:2] # 获取原图的大小
new_height, new_width = old_height*2, old_width*2 # 设置新的高度和宽度

new_image_nn = nearest_neighbor_interpolation(init_image, new_height, new_width)  # 最近邻插值
new_image_linear = linear_interpolation(init_image, new_height, new_width)  # 线性插值
new_image_bilinear = bilinear_interpolation(init_image, new_height, new_width)  # 双线性插值
new_image_bicubic = bicubic_interpolation(init_image, new_height, new_width)  # 三次插值

output_dir = 'resources_labwork1'
Image.fromarray(new_image_nn).save(os.path.join(output_dir, 'lena_nearest_neighbor.png'))
Image.fromarray(new_image_linear).save(os.path.join(output_dir, 'lena_linear_interpolation.png'))
Image.fromarray(new_image_bilinear).save(os.path.join(output_dir, 'lena_bilinear_interpolation.png'))
Image.fromarray(new_image_bicubic).save(os.path.join(output_dir, 'lena_bicubic_interpolation.png'))
