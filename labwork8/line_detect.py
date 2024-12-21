import cv2
import numpy as np

# 读取图像
image = cv2.imread('resources_labwork8/1runway.png', cv2.IMREAD_GRAYSCALE);
# 高斯模糊
blur_gray = cv2.GaussianBlur(image, (3, 3), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blur_gray, 50, 150, 3)
# 使用霍夫变换检测直线，阈值为投票数
lines = cv2.HoughLines(edges, 1, np.pi / 180, 225)
# 创建一个彩色图像副本用于绘制直线
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# 在原图上绘制检测到的直线
if lines is not None:
    for rho, theta in lines[:, 0]:
        # 只关注接近垂直的线条
        if (theta < np.pi / 200):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(color_image, (x1, y1), (x2, y2), (255,0,0), 4, cv2.LINE_AA)

# 显示结果
cv2.imshow("Detected Vertical Lines", color_image)  # 显示绘制了
cv2.imshow("Detected Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
