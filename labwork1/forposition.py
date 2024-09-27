import cv2


# 鼠标回调函数
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
        print(f'Clicked coordinates: (x={x}, y={y})')


# 加载图像
image = cv2.imread('../resources/seu_rotated.png')

# 显示窗口
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_coordinates)

while True:
    cv2.imshow('Image', image)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()