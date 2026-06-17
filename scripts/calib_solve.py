import cv2  # 负责核心视觉算法 (角点提取、PnP求解、Tsai-Lenz标定)
import numpy as np  # 负责高精度矩阵运算，机器人的世界本质上就是一堆多维数组
import os, glob, math  # 负责文件路径读取和基础数学(弧度转换)

# ==============================================================================
# 模块一：全局物理参数锁死区 (零误差的前提)
# ==============================================================================
# 【物理原理】：OpenCV 标定算法寻找的是黑白方块交汇的“十字内角点”，而不是方块本身。
# 你打印的是 8行11列 的方块，因此内角点必须是 (11, 8)。(OpenCV 习惯宽X在前，高Y在后)
CHECKERBOARD = (8, 11)

# 【绝对尺度】：单个方格的物理真实边长，单位：毫米(mm)。
# 视觉本身是没有“大小”概念的，全靠这个数值给系统注入真实的物理世界尺度。
SQUARE_SIZE = 10.0

IMAGE_DIR = "calib_data/images/"
POSE_DIR = "calib_data/poses/"

# 【相机内参矩阵 (Camera Matrix)】：这是相机的“视网膜参数”
# fx, fy (910.0) 是焦距，决定了物体在画面中的缩放比例。
# cx, cy (640.0, 360.0) 是光学中心，通常在 1280x720 画面的正中央。
# ⚠️ 在真实工业现场，此矩阵应通过张正友标定法专门求出，此处使用 D435 标准参考值。
camera_matrix = np.array([[910.0, 0.0, 640.0],
                          [0.0, 910.0, 360.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
dist_coeffs = np.zeros((5, 1))  # 假设已做畸变校正


# ==============================================================================
# 模块二：空间几何转换引擎 (示教器语言 -> 数学矩阵)
# ==============================================================================
def txt_to_matrix(txt_path):
    """
    【算法名称】：Z-Y-X 欧拉角转旋转矩阵 (Euler to Rotation Matrix)

    【为什么要有这个函数？】
    机械臂示教器上显示的是人类看得懂的欧拉角 (比如倾斜了 15 度：Rx=15)。
    但是，计算机底层的 Tsai-Lenz 算法不认识度数，它只认识 3x3 的旋转矩阵。
    此函数就是连接机械臂物理世界和 OpenCV 数学世界的“翻译官”。
    """
    with open(txt_path, 'r') as f:
        data = f.read().replace(',', ' ').split()

    # 1. 提取平移向量 Translation (X, Y, Z)，单位：毫米
    t_vec = np.array([[float(data[0])], [float(data[1])], [float(data[2])]])

    # 2. 将欧拉角从“度(Degree)”转换为“弧度(Radian)” (计算机三角函数只认弧度)
    rx, ry, rz = map(math.radians, [float(data[3]), float(data[4]), float(data[5])])

    # 3. 构建三个基础旋转矩阵
    # 绕 X 轴旋转矩阵 (Roll)
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])

    # 绕 Y 轴旋转矩阵 (Pitch)
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])

    # 绕 Z 轴旋转矩阵 (Yaw)
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])

    # 4. 矩阵相乘 (核心避坑点)
    # 【工业标准】：矩阵乘法不满足交换律！主流机械臂(如Dobot/KUKA)默认采用 Z-Y-X 的内在旋转顺序。
    # 数学表达为：R = R_z * R_y * R_x (代码中用 @ 符号表示矩阵点乘)
    R_matrix = R_z @ R_y @ R_x

    return R_matrix, t_vec


# ==============================================================================
# 模块三：数据装载与 A、B 矩阵构建 (AX = XB 的前置准备)
# ==============================================================================
# 准备空列表，用来装方程 AX=XB 中的 A 和 B
R_gripper2base, t_gripper2base = [], []  # 装载矩阵 A (机械臂末端在基座下的姿态)
R_target2cam, t_target2cam = [], []  # 装载矩阵 B (标定板在相机下的姿态)

img_files = sorted(glob.glob(IMAGE_DIR + "*.jpg"))
print(f"🔄 [系统流] 正在读取 {len(img_files)} 组标定数据，准备构建空间方程...")

for img_path in img_files:
    idx = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(POSE_DIR, f"{idx}.txt")

    # ---------------------------------------------------------
    # 步骤 1：利用视觉计算矩阵 B (标定板相对于相机的空间位姿)
    # ---------------------------------------------------------
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找标定板特征点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret: continue

    # 【亚像素优化】：将角点精度从 1 个像素，强行逼近到 0.001 个像素，直接决定了抓取的毫米级精度！
    corners_subpix = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # 构建标定板在真实世界的三维坐标系 (假设标定板左上角第一个点为 X:0, Y:0, Z:0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # 【PnP 求解算法 (Perspective-n-Point)】
    # 原理：已知 3D 物理点(objp)和对应的 2D 像素点(corners_subpix)，结合相机内参，
    # 反推出相机拍摄这张照片时，相机相对于标定板在空间中的 旋转向量(rvec) 和 平移向量(tvec)。
    _, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs)

    # 将旋转向量(3x1)转换为计算所需的旋转矩阵(3x3)
    R_cam, _ = cv2.Rodrigues(rvec)

    R_target2cam.append(R_cam)
    t_target2cam.append(tvec)

    # ---------------------------------------------------------
    # 步骤 2：利用示教器数据计算矩阵 A (法兰盘相对于基座的空间位姿)
    # ---------------------------------------------------------
    R_grip, t_grip = txt_to_matrix(txt_path)
    R_gripper2base.append(R_grip)
    t_gripper2base.append(t_grip)

# ==============================================================================
# 模块四：终极矩阵解算 (寻找那个被螺丝拧死的神秘 X 矩阵)
# ==============================================================================
print(f"⚙️ [数学引擎] 数据就绪。正在启动 Tsai-Lenz 算法破解 AX=XB 方程组...")

# 【Tsai-Lenz 算法原理】：
# 目标：解方程 A * X = X * B
# 痛点：X 是一个包含旋转和平移的 4x4 矩阵，直接解极其困难。
# Tsai 的智慧：分而治之！
# 1. 先利用机械臂的旋转轴和相机相对旋转轴在空间中平行的原理，建立线性方程，用最小二乘法解出 3x3 旋转矩阵 R_X。
# 2. 将 R_X 代入原方程，未知数只剩平移，瞬间变成一元一次方程组，再次用最小二乘法解出平移向量 t_X。
# 优势：无须非线性迭代猜测，运算速度极快，且能平均抵消掉你手动采数据时的微小误差。

R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI  # 指定使用 Tsai-Lenz 算法
)

# ==============================================================================
# 模块五：组装与下发 4x4 齐次变换矩阵
# ==============================================================================
# 【齐次变换矩阵 (Homogeneous Transformation Matrix) 结构科普】：
# 它是一个 4x4 的方阵，工业机器人坐标系转换的通用语言。
# [ R11 R12 R13  Tx ]  <-- 左上 3x3 是旋转矩阵 (Rotation)
# [ R21 R22 R23  Ty ]  <-- 右上 3x1 是平移向量 (Translation)
# [ R31 R32 R33  Tz ]
# [  0   0   0   1  ]  <-- 底部是为了能做矩阵乘法而补充的数学占位符

T_matrix = np.eye(4)  # 先生成一个 4x4 的对角线为1的单位矩阵
T_matrix[:3, :3] = R_cam2grip  # 把求出来的 3x3 旋转矩阵塞进左上角
T_matrix[:3, 3] = t_cam2grip.flatten()  # 把求出来的平移塞进右上角

print("\n🎯 标定圆满成功！相机的“灵魂”已完美映射到机械臂的躯干上。")
print("👇 终极 4x4 手眼变换矩阵 (T_cam_to_gripper) 如下：")
print(np.round(T_matrix, 4))

# 导出供 GUI 控制总线直接调用
output_file = 'hand_eye_result.yaml'
cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE).write("Transformation_Matrix", T_matrix).release()
print(f"📦 已将矩阵打包导出至：{output_file}，部署完成！")