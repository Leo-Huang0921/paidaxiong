import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色和深度流 (D435 推荐分辨率 640x480)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
profile = pipeline.start(config)

# 创建对齐对象：这一步极其重要，必须将深度帧对齐到彩色帧，
# 这样你在彩色画面上点击的 (x,y) 才能对应到正确的深度值。
align_to = rs.stream.color
align = rs.align(align_to)

# 用于保存鼠标点击坐标的全局变量
click_point = None


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


# 设置窗口和鼠标回调
cv2.namedWindow('D435 Depth Test - Click on Cylinders')
cv2.setMouseCallback('D435 Depth Test - Click on Cylinders', mouse_callback)

print("测试已启动：请在画面中点击 40mm 圆柱体的顶部...")

try:
    while True:
        # 等待获取新的一帧
        frames = pipeline.wait_for_frames()

        # 执行对齐操作
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 转换为 numpy 数组以便 OpenCV 处理
        color_image = np.asanyarray(color_frame.get_data())

        # 如果有鼠标点击，获取深度并在画面上绘制
        if click_point:
            x, y = click_point

            # get_distance() 直接返回单位为"米"的真实物理距离
            distance = aligned_depth_frame.get_distance(x, y)

            # 将距离转换为毫米并打印
            dist_mm = distance * 1000
            print(f"坐标 ({x}, {y}) 处的距离为: {dist_mm:.1f} mm")

            # 在图像上标注测量的点和距离
            text = f"{dist_mm:.1f} mm"
            cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(color_image, text, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示彩色图像
        cv2.imshow('D435 Depth Test - Click on Cylinders', color_image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止管道，释放资源
    pipeline.stop()
    cv2.destroyAllWindows()