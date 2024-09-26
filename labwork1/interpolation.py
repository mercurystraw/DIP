import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image_pth = "../resources/lena.png" # 图片资源存储在resources文件夹下，这里保存使用图片的路径

init_image = np.array(Image.open(image_pth)) # 读取图片
# 显示原图像并保存
plt.imshow(init_image.astype(np.uint8))  # 转换为 uint8 类型
plt.axis('on')  # 显示坐标轴
plt.savefig('../resources/lena_original.png')  # 保存原图像

def nearest_neighbor_interpolation(image, new_height, new_width): # 最近邻插值
    old_height, old_width = image.shape[:2] # 获取原图的大小
    channels = image.shape[2] # 获取图片的通道数,灰度图像一般为1, RGB图像为3

    new_image = np.zeros((new_height, new_width, channels)) # (H,W,C)形式

    h_rate = old_height/new_height # 计算height方向的缩放比例
    w_rate = old_width/new_width # 计算width方向的缩放比例
    for i in range(new_height):
        for j in range(new_width):
            x = min(round(i*h_rate), old_height-1) # 采用round函数对坐标进行四舍五入，从而找到映射前距离最近的整数(也就是最近邻插值),同时防止越界
            y = min(round(j*w_rate), old_width-1) # 同x
            new_image[i, j] = image[x, y] # 注意是(H,W,C)形式
    return new_image

def linear_interpolation(image, new_height, new_width): # 线性插值
    old_height, old_width = image.shape[:2]
    channels = image.shape[2]

    new_image = np.zeros((new_height, new_width, channels))
    h_rate = old_height/new_height
    w_rate = old_width/new_width

    for i in range(new_height):
        for j in range(new_width): # 这里都同上
            x = i*h_rate
            y = j*w_rate
            x0 = int(x) # 向下取整得到x0
            y0 = int(y) # 向下取整得到y0
            # 确保坐标在有效范围内

            # 计算小数部分
            x_diff = x - x0
            y_diff = y - y0

            # 处理映射的不同情况
            if x_diff == 0 and y_diff == 0:
                # 正好映射到原图像的整数坐标
                new_image[i, j] = image[x0, y0]
            elif x_diff == 0:
                # y_diff != 0
                y1 = y0 + (1 if y_diff > 0 else -1)
                # x 为整数，y 不为整数
                if y0== old_width-1:
                    new_image[i, j] = image[x0, y0]
                else:
                    new_image[i,j] = (y1-y)*image[x0, y0]/(y1-y0)+y_diff*image[x0, y1]/(y1-y0)
            elif y_diff == 0:
                x1 = x0 + (1 if x_diff > 0 else 0)
                # y 为整数，x 不为整数
                if x0 == old_height-1:
                    new_image[i, j] = image[x0, y0]
                else:
                    new_image[i,j] = (x1-x)*image[x0, y0]/(x1-x0)+x_diff*image[x1,y0]/(x1-x0)
            else: # x,y都不为整数
                x1 = x0 + (1 if x_diff > 0 else 0)
                y1 = y0 + (1 if y_diff > 0 else -1)
                if y0 == old_width - 1 or x0==old_height-1:
                    new_image[i, j] = image[x0, y0]
                else:
                    new_image[i,j] = (x1-x)*image[x0, y0]+x_diff*image[x1, y1]/(x1-x0) # 线性插值公式
    return new_image


def bilinear_interpolation(image, new_height, new_width): # 双线性插值
    old_height, old_width = image.shape[:2]
    channels = image.shape[2]

    new_image = np.zeros((new_height, new_width, channels))
    h_rate = old_height/new_height
    w_rate = old_width/new_width

    for i in range(new_height):
        for j in range(new_width): # 这里都同上
            x = i*h_rate
            y = j*w_rate
            x0 = round(x)
            y0 = round(y)
            # 计算小数部分
            x_diff = x - x0
            y_diff = y - y0
            if x_diff == 0 and y_diff == 0:
                # 正好映射到原图像的整数坐标
                value = image[x0, y0]
            elif x_diff == 0 and y_diff!= 0:
                if(y_diff > 0):  y1 = min(y0 + 1, old_width - 1) # 边界处理
                else: y1 = max(y0 - 1, 0)
                value = image[x0, y0] + y_diff * (image[x0, y1] - image[x0, y0]) / (y1 - y0)
            elif y_diff == 0 and x_diff!= 0:
                if(x_diff > 0): x1=min(x0 + 1, old_height - 1) # 边界处理
                else: x1= max(x0 - 1, 0)
                # y 为整数，x 不为整数
                value = image[x0, y0]+x_diff*(image[x1,y0]-image[x0,y0])/(x1-x0)
            else: # x,y都不为整数
                if (x_diff > 0):
                    x1 = min(x0 + 1, old_height - 1)  # 边界处理
                else:
                    x1 = max(x0 - 1, 0)
                if (y_diff > 0):
                    y1 = min(y0 + 1, old_width - 1)
                else:
                    y1 = max(y0 - 1, 0)
                value = (image[x0, y0] * (1 - x_diff) * (1 - y_diff) +
                         image[x1, y0] * x_diff * (1 - y_diff) +
                         image[x0, y1] * (1 - x_diff) * y_diff +
                         image[x1, y1] * x_diff * y_diff)
            new_image[i, j] = value
    return new_image


new_height, new_width = 1023,1023 # 设置新的高度和宽度

new_image = nearest_neighbor_interpolation(init_image, new_height, new_width)  # 调用插值函数
plt.imshow(new_image.astype(np.uint8))  # 转换为 uint8 类型
plt.axis('on')  # 显示坐标轴
plt.savefig('../resources/lena_nearest_neighbor_interpolation.png')

new_image = linear_interpolation(init_image, new_height, new_width)  # 调用插值函数
plt.imshow(new_image.astype(np.uint8))  # 转换为 uint8 类型
plt.axis('on')  # 显示坐标轴
plt.savefig('../resources/lena_linear_interpolation.png')

# new_image = bilinear_interpolation(init_image, new_height, new_width)  # 调用插值函数
# plt.imshow(new_image)
# plt.savefig(f'../results/bilinear_interpolation.png')

