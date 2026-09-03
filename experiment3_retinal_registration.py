import os
import sys
import glob
import time
import numpy as np

# 关键: 切换到脚本所在目录，避免中文路径编码问题导致cv2.imread读取失败
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === 中文字体设置 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import cv2
from skimage import exposure
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 全局配置（使用相对路径，避免中文路径编码问题）
BASE_DIR = os.path.join('FIRE', 'FIRE')
OUTPUT_DIR = os.path.join('exp3_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图像缩放目标尺寸（原图2912×2912，缩放加速处理）
# 注：缩小会丢失细微血管细节，但大幅加速配准。如有GPU可尝试1024或原尺寸
IMG_SIZE = 512

# SIFT参数
SIFT_NFEATURES = 500           # 最大关键点数
FLANN_LOWE_RATIO = 0.75        # Lowe's ratio test阈值

# ECC参数
ECC_MAX_ITER = 200             # ECC最大迭代次数
ECC_EPSILON = 1e-5             # ECC收敛阈值

# 数据划分
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2               # 测试集 = 1 - 0.6 - 0.2 = 0.2
RANDOM_SEED = 42

print(f'[INFO] 数据目录: {BASE_DIR}')
print(f'[INFO] 输出目录: {OUTPUT_DIR}')
print(f'[INFO] 图像缩放至: {IMG_SIZE}×{IMG_SIZE}')
print(f'[INFO] 数据划分: 训练{TRAIN_RATIO*100:.0f}% / 验证{VAL_RATIO*100:.0f}% / 测试{(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%')


# 1. 数据加载

def load_fire_pairs(base_dir):
    #加载FIRE数据集的所有配准图像对。
    print('\n' + '=' * 60)
    print('[数据加载] 加载FIRE数据集...')

    gt_dir = os.path.join(base_dir, 'Ground Truth')
    img_dir = os.path.join(base_dir, 'Images')
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
    print(f'  找到 {len(gt_files)} 个控制点文件')

    pairs = []
    for gt_path in gt_files:
        basename = os.path.basename(gt_path)
        # control_points_A01_1_2.txt → A01_1_2
        pair_id = basename.replace('control_points_', '').replace('.txt', '')
        parts = pair_id.split('_')
        patient = parts[0]
        img1_idx = parts[1]
        img2_idx = parts[2]

        # 构建图像路径
        img1_path = os.path.join(img_dir, f'{patient}_{img1_idx}.jpg')
        img2_path = os.path.join(img_dir, f'{patient}_{img2_idx}.jpg')

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f'  警告: 图像缺失 - {patient}_{img1_idx} 或 {patient}_{img2_idx}')
            continue

        # 读取GT控制点（每行: x1 y1 x2 y2）
        gt_points = np.loadtxt(gt_path)
        if gt_points.shape[0] != 10:
            print(f'  警告: {basename} 控制点数={gt_points.shape[0]}, 期望10')
            continue

        pts1 = gt_points[:, :2]  # 图像1上的点
        pts2 = gt_points[:, 2:]  # 图像2上的点

        pairs.append({
            'patient': patient,
            'pair_id': pair_id,
            'img1_path': img1_path,
            'img2_path': img2_path,
            'pts1': pts1.astype(np.float32),
            'pts2': pts2.astype(np.float32),
        })

    print(f'[数据加载] 成功加载 {len(pairs)} 对图像')
    return pairs


# 2. 图像预处理

def preprocess_image(img, target_size=IMG_SIZE):
    #视网膜图像预处理流程:
    #1. 缩放至目标尺寸
    #2. CLAHE自适应直方图均衡化（增强血管对比度）
    #3. 高斯滤波去噪（轻度，保留血管边缘）
    #4. 归一化到[0, 1]

    # 步骤1: 缩放
    img_resized = cv2.resize(img, (target_size, target_size))

    # 步骤2: CLAHE均衡化 —— 在局部区域增强对比度，突出血管
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)

    # 步骤3: 轻度高斯去噪 —— 去除传感器噪声，σ=1.0保守设置
    img_denoised = cv2.GaussianBlur(img_clahe, (3, 3), sigmaX=1.0)

    # 步骤4: 归一化
    img_norm = img_denoised.astype(np.float32) / 255.0

    return img_norm


# 3. 配准方法实现
def register_sift_ransac(img1, img2):
    #基于SIFT特征匹配 + RANSAC的配准。
    #1. 在两幅图像上提取SIFT关键点和描述子
    #2. 使用FLANN进行描述子匹配
    #3. Lowe's ratio test过滤低质量匹配
    #4. RANSAC估计单应性变换矩阵
    #5. 对图像1应用变换进行配准

    # 转为uint8供SIFT
    img1_u8 = (img1 * 255).astype(np.uint8)
    img2_u8 = (img2 * 255).astype(np.uint8)

    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

    # 提取关键点和描述子
    kp1, desc1 = sift.detectAndCompute(img1_u8, None)
    kp2, desc2 = sift.detectAndCompute(img2_u8, None)

    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        print('    [SIFT] 关键点不足，配准失败')
        return np.eye(3, dtype=np.float32), 0, False

    # FLANN匹配器：对图1的每个关键点，在图2里找最相似的2个关键点
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),   # KD-tree
        dict(checks=50)               # 搜索检查次数
    )
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test: 最佳匹配的距离应显著小于次佳匹配
    good_matches = []
    for m, n in matches:
        if m.distance < FLANN_LOWE_RATIO * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print(f'    [SIFT] 有效匹配不足 ({len(good_matches)} < 4)，配准失败')
        return np.eye(3, dtype=np.float32), len(good_matches), False

    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC鲁棒估计单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None:
        print('    [SIFT] RANSAC估计失败')
        return np.eye(3, dtype=np.float32), len(good_matches), False

    n_inliers = np.sum(mask) if mask is not None else 0
    print(f'    [SIFT] 关键点: {len(kp1)}/{len(kp2)}, '
          f'匹配: {len(good_matches)}, 内点: {n_inliers}')

    return H.astype(np.float32), n_inliers, True


def register_ecc(img1, img2):
    #基于ECC的配准。
    # ECC需要float32 [0, 1]或uint8
    img1_f32 = img1.astype(np.float32)
    img2_f32 = img2.astype(np.float32)
    # 初始变换矩阵（identity + 微小偏移的仿射）
    warp_init = np.eye(2, 3, dtype=np.float32)
    #设置运动模式
    motion_type = cv2.MOTION_AFFINE
    # 定义终止条件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ECC_MAX_ITER, ECC_EPSILON)
    try:
        # 执行ECC配准，将两图重叠计算相似度，重复调参直到满足条件
        cc, warp_matrix = cv2.findTransformECC(
            img1_f32, img2_f32, warp_init, motion_type, criteria
        )
        print(f'    [ECC] 相关系数: {cc:.4f}, 收敛成功')
        # 将2×3仿射转为3×3齐次矩阵，便于统一处理
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = warp_matrix
        return H, True
    except cv2.error as e:
        print(f'    [ECC] 优化失败: {str(e)[:80]}')
        return np.eye(3, dtype=np.float32), False


def warp_image(img, H, output_shape=None):

    #使用单应性矩阵对图像进行变换。


    if output_shape is None:
        h, w = img.shape
        output_shape = (w, h)
    else:
        output_shape = (output_shape[1], output_shape[0])  # cv2: (width, height)

    warped = cv2.warpPerspective(img, H, output_shape,
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    return warped

# 4. 评估指标

def compute_msd(pts1_transformed, pts2_gt):

    distances = np.linalg.norm(pts1_transformed - pts2_gt, axis=1)
    return np.mean(distances)


def compute_mse(img_warped, img_target, mask=None):

    #计算均方误差 (Mean Square Error)
    if mask is None:
        # 自动计算重叠区域：两幅图像都非零的区域
        mask = (img_warped > 0.01) & (img_target > 0.01)

    if mask.sum() < 100:
        # 重叠区域太小，返回NaN
        return np.nan

    diff = (img_warped[mask] - img_target[mask]) ** 2
    return np.mean(diff)


def apply_transform_to_points(pts, H):

    #将3×3单应性矩阵应用于2D点集。
    N = pts.shape[0]
    pts_homo = np.hstack([pts, np.ones((N, 1), dtype=np.float32)])  # (N, 3)
    pts_t = (H @ pts_homo.T).T  # (N, 3)
    pts_t = pts_t[:, :2] / pts_t[:, 2:3]  # 齐次坐标归一化
    return pts_t

# 5. 主实验流程
def evaluate_pair(img1, img2, pts1, pts2, pair_info=''):
    #对一对图像使用两种方法进行配准评估。
    H, W = img1.shape

    #配准前（baseline: identity变换）
    msd_before = compute_msd(pts1, pts2)  # identity, 点不变换
    mse_before = compute_mse(img1, img2)

    results = {
        'msd_before': msd_before,
        'mse_before': mse_before,
        'SIFT': {'success': False, 'msd_after': None, 'mse_after': None,
                 'n_matches': 0, 'transform': None},
        'ECC': {'success': False, 'msd_after': None, 'mse_after': None,
                'transform': None},
    }

    #方法1: SIFT + RANSAC
    H_sift, n_matches, ok = register_sift_ransac(img1, img2)
    results['SIFT']['n_matches'] = n_matches
    results['SIFT']['transform'] = H_sift

    if ok:
        # 变换控制点评估MSD
        pts1_t = apply_transform_to_points(pts1, H_sift)
        results['SIFT']['msd_after'] = compute_msd(pts1_t, pts2)
        # 变换图像评估MSE
        img1_warped = warp_image(img1, H_sift, (H, W))
        results['SIFT']['mse_after'] = compute_mse(img1_warped, img2)
        results['SIFT']['success'] = True

    #方法2: ECC
    H_ecc, ok = register_ecc(img1, img2)
    results['ECC']['transform'] = H_ecc

    if ok:
        pts1_t = apply_transform_to_points(pts1, H_ecc)
        results['ECC']['msd_after'] = compute_msd(pts1_t, pts2)
        img1_warped = warp_image(img1, H_ecc, (H, W))
        results['ECC']['mse_after'] = compute_mse(img1_warped, img2)
        results['ECC']['success'] = True

    return results


def main():
    start_time = time.time()
    print('#' * 60)
    print('#  实验三：基于计算机视觉的视网膜图像配准')
    print('#  数据集: FIRE (134对视网膜图像)')
    print(f'#  开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('#' * 60)

    # 5.1 数据加载与划分
    pairs = load_fire_pairs(BASE_DIR)

    # 划分: train 60% / val 20% / test 20%
    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=(1 - TRAIN_RATIO - VAL_RATIO) / (1 - TRAIN_RATIO),
        random_state=RANDOM_SEED
    )
    print(f'[数据划分] 训练集: {len(train_pairs)}对 | '
          f'验证集: {len(val_pairs)}对 | 测试集: {len(test_pairs)}对')

    # 5.2 对所有数据进行预处理
    print('\n' + '=' * 60)
    print('[预处理] 对图像进行归一化和去噪...')
    print(f'  步骤: 缩放({IMG_SIZE}×{IMG_SIZE}) → CLAHE均衡化 → 高斯去噪 → 归一化')

    # 为了效率，预处理所有图像并缓存
    all_processed = {}  # key: path → processed image

    t0 = time.time()
    for pair in pairs:
        for img_key, path_key in [('img1', 'img1_path'), ('img2', 'img2_path')]:
            path = pair[path_key]
            if path not in all_processed:
                img_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img_raw is None:
                    print(f'  错误: 无法读取 {path}')
                    continue
                all_processed[path] = preprocess_image(img_raw)

        # 将预处理后的图像存入pair
        pair['img1_proc'] = all_processed[pair['img1_path']]
        pair['img2_proc'] = all_processed[pair['img2_path']]

    print(f'[预处理] 完成 {len(all_processed)} 张图像, 耗时 {time.time()-t0:.1f}s')

    # 5.3 在测试集上评估（仅测试集，因为配准不需要"训练"）
    print('\n' + '=' * 60)
    print(f'[配准评估] 在测试集 ({len(test_pairs)}对) 上评估两种方法...')
    print('=' * 60)

    test_results = []
    for i, pair in enumerate(test_pairs):
        img1 = pair['img1_proc']
        img2 = pair['img2_proc']
        pts1 = pair['pts1'] * (IMG_SIZE / 2912.0)  # 缩放控制点到新尺寸
        pts2 = pair['pts2'] * (IMG_SIZE / 2912.0)

        print(f'\n--- 测试对 {i+1}/{len(test_pairs)}: {pair["pair_id"]} ---')
        res = evaluate_pair(img1, img2, pts1, pts2, pair['pair_id'])
        res['pair_id'] = pair['pair_id']
        test_results.append(res)

    # 5.4 汇总结果
    print('\n' + '=' * 60)
    print('[结果汇总] 表3-1 配准前后性能对比（测试集）')
    print('=' * 60)

    # 收集有效结果
    sift_msd_before, sift_msd_after = [], []
    sift_mse_before, sift_mse_after = [], []
    ecc_msd_before, ecc_msd_after = [], []
    ecc_mse_before, ecc_mse_after = [], []
    sift_success, ecc_success = 0, 0

    for r in test_results:
        if r['SIFT']['success']:
            sift_success += 1
            sift_msd_before.append(r['msd_before'])
            sift_msd_after.append(r['SIFT']['msd_after'])
            sift_mse_before.append(r['mse_before'])
            sift_mse_after.append(r['SIFT']['mse_after'])
        if r['ECC']['success']:
            ecc_success += 1
            ecc_msd_before.append(r['msd_before'])
            ecc_msd_after.append(r['ECC']['msd_after'])
            ecc_mse_before.append(r['mse_before'])
            ecc_mse_after.append(r['ECC']['mse_after'])

    print(f'\nSIFT+RANSAC: 成功 {sift_success}/{len(test_results)} 对')
    print(f'ECC:          成功 {ecc_success}/{len(test_results)} 对')

    # 构建结果表格
    import csv
    csv_path = os.path.join(OUTPUT_DIR, '表3-1_配准性能对比.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['方法', '成功对数', 'MSD前(px)', 'MSD后(px)',
                         'MSD改善(%)', 'MSE前', 'MSE后', 'MSE改善(%)'])
        for method_name, msd_b, msd_a, mse_b, mse_a, n_ok in [
            ('SIFT+RANSAC', sift_msd_before, sift_msd_after,
             sift_mse_before, sift_mse_after, sift_success),
            ('ECC', ecc_msd_before, ecc_msd_after,
             ecc_mse_before, ecc_mse_after, ecc_success),
        ]:
            if n_ok > 0:
                msd_b_mean = np.mean(msd_b)
                msd_a_mean = np.mean(msd_a)
                mse_b_mean = np.mean(mse_b)
                mse_a_mean = np.mean(mse_a)
                msd_improve = (1 - msd_a_mean / msd_b_mean) * 100 if msd_b_mean > 0 else 0
                mse_improve = (1 - mse_a_mean / mse_b_mean) * 100 if mse_b_mean > 0 else 0

                print(f'\n{method_name}:')
                print(f'  MSD: {msd_b_mean:.2f} → {msd_a_mean:.2f} px (改善 {msd_improve:.1f}%)')
                print(f'  MSE: {mse_b_mean:.4f} → {mse_a_mean:.4f} (改善 {mse_improve:.1f}%)')

                writer.writerow([method_name, n_ok,
                                 f'{msd_b_mean:.2f}', f'{msd_a_mean:.2f}',
                                 f'{msd_improve:.1f}',
                                 f'{mse_b_mean:.4f}', f'{mse_a_mean:.4f}',
                                 f'{mse_improve:.1f}'])

    print(f'\n  -> 表格已保存至: {csv_path}')
    # 5.5 可视化
    visualize_results(test_results, test_pairs, all_processed)

    # 5.6 参数与挑战讨论
    print('\n' + '=' * 60)
    print('[参数分析与挑战讨论]')
    print('=' * 60)
    print(f'''
    1. 预处理参数:
       - CLAHE (clipLimit=2.0, tile=8×8): 增强局部血管对比度，对视网膜图像效果显著
       - 高斯去噪 (σ=1.0, kernel=3): 轻度去噪，保留血管边缘信息
       - 缩放至{IMG_SIZE}×{IMG_SIZE}: 平衡精度与速度。原图2912×2912细节更多但处理慢

    2. SIFT+RANSAC参数:
       - SIFT关键点上限: {SIFT_NFEATURES}
         说明: 视网膜血管分叉处是关键特征点，500个足够覆盖主要血管
       - Lowe's ratio: {FLANN_LOWE_RATIO}
         说明: 较低的ratio过滤更严格，减少误匹配但可能丢失正确匹配
       - RANSAC阈值: 3.0px
         说明: 允许±3像素的重投影误差，对视网膜图像的大致刚性变形合适

    3. ECC参数:
       - 运动模型: 仿射(MOTION_AFFINE, 6参数)
         说明: 视网膜配准主要是平移+旋转+缩放，仿射模型足够描述
       - 最大迭代: {ECC_MAX_ITER}
         说明: ECC收敛较慢，需要较多迭代；如果配准失败可尝试增加
       - 初始化: identity
         说明: 从恒等变换开始优化。若两图差异大，ECC可能陷入局部最优

    4. 遇到的挑战及解决方案:
       - SIFT在低纹理区域关键点少: 使用Lowe's ratio test严格过滤误匹配
       - ECC对大位移收敛差: 采用多尺度策略或先用SIFT粗对齐，再用ECC精调
       - 缩放损失细节: {IMG_SIZE}px是小尺寸，可在报告中讨论全分辨率结果差异
       - 部分图像配准失败: 视网膜图像质量差异大(模糊、曝光不均)，需针对处理

    ''')

    elapsed = (time.time() - start_time)
    print(f'\n[INFO] 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)')
    print(f'[INFO] 所有输出已保存至: {OUTPUT_DIR}')
    print('[INFO] 实验三代码执行完毕！')


# 6. 可视化函数

def visualize_results(test_results, test_pairs, all_processed):
    """生成配准结果可视化图"""
    print('\n[可视化] 生成结果图...')

    # ---- 图3-1: 配准前后MSD/MSE对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 准备数据
    methods = ['配准前\n(Identity)', 'SIFT+\nRANSAC', 'ECC']
    colors = ['#999999', '#5B9BD5', '#FF6B6B']

    sift_msd = [np.mean([r['msd_before'] for r in test_results if r['SIFT']['success']]),
                np.mean([r['SIFT']['msd_after'] for r in test_results if r['SIFT']['success']]),
                np.mean([r['ECC']['msd_after'] for r in test_results if r['ECC']['success']])]
    sift_mse = [np.mean([r['mse_before'] for r in test_results if r['SIFT']['success']]),
                np.mean([r['SIFT']['mse_after'] for r in test_results if r['SIFT']['success']]),
                np.mean([r['ECC']['mse_after'] for r in test_results if r['ECC']['success']])]

    # MSD
    ax = axes[0]
    bars = ax.bar(methods, sift_msd, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, sift_msd):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('MSD (像素)')
    ax.set_title('图3-1(a) 平均表面距离(MSD)对比\n(越小越好)')
    ax.grid(axis='y', alpha=0.3)

    # MSE
    ax = axes[1]
    bars = ax.bar(methods, sift_mse, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, sift_mse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('MSE')
    ax.set_title('图3-1(b) 均方误差(MSE)对比\n(越小越好)')
    ax.set_ylim(0, max(sift_mse) * 1.35)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3-1_msd_mse_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> 图3-1 已保存')

    # ---- 图3-2: 配准成功率饼图 ----
    sift_ok = sum(1 for r in test_results if r['SIFT']['success'])
    ecc_ok = sum(1 for r in test_results if r['ECC']['success'])
    n_total = len(test_results)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, ok, method in zip(axes, [sift_ok, ecc_ok], ['SIFT+RANSAC', 'ECC']):
        sizes = [ok, n_total - ok]
        labels = [f'成功 ({ok})', f'失败 ({n_total-ok})']
        colors_pie = ['#70AD47', '#FF6B6B']
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0))
        ax.set_title(f'图3-2 {method}\n配准成功率')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3-2_success_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> 图3-2 已保存')

    # ---- 图3-3: 配准前后图像对比（选取一个成功案例） ----
    # 找一个SIFT和ECC都成功的案例
    demo_pair = None
    demo_res = None
    for pair, res in zip(test_pairs, test_results):
        if res['SIFT']['success'] and res['ECC']['success']:
            demo_pair = pair
            demo_res = res
            break

    if demo_pair is not None:
        img1 = demo_pair['img1_proc']
        img2 = demo_pair['img2_proc']
        H, W = img1.shape

        # 生成配准后的图像
        img1_sift_warped = warp_image(img1, demo_res['SIFT']['transform'], (H, W))
        img1_ecc_warped = warp_image(img1, demo_res['ECC']['transform'], (H, W))

        # 棋盘格融合图（棋盘格对比配准效果）
        def make_checkerboard(img_a, img_b, grid=16):
            h, w = img_a.shape
            cb = np.zeros((h, w), dtype=np.float32)
            for i in range(0, h, grid):
                for j in range(0, w, grid):
                    if (i//grid + j//grid) % 2 == 0:
                        cb[i:i+grid, j:j+grid] = img_a[i:i+grid, j:j+grid]
                    else:
                        cb[i:i+grid, j:j+grid] = img_b[i:i+grid, j:j+grid]
            return cb

        checker_sift = make_checkerboard(img1_sift_warped, img2)
        checker_ecc = make_checkerboard(img1_ecc_warped, img2)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行: 输入图像和配准前
        axes[0, 0].imshow(img1, cmap='gray')
        axes[0, 0].set_title('(a) 待配准图像 (img1)', fontsize=10)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img2, cmap='gray')
        axes[0, 1].set_title('(b) 参考图像 (img2)', fontsize=10)
        axes[0, 1].axis('off')

        # 配准前叠加（直接叠加显示错位）
        overlay_before = cv2.addWeighted(
            (img1*255).astype(np.uint8), 0.5,
            (img2*255).astype(np.uint8), 0.5, 0)
        axes[0, 2].imshow(overlay_before, cmap='gray')
        axes[0, 2].set_title('(c) 配准前叠加\n(50%透明度)', fontsize=10)
        axes[0, 2].axis('off')

        # 第二行: 两种方法的结果
        axes[1, 0].imshow(checker_sift, cmap='gray')
        axes[1, 0].set_title(f'(d) SIFT+RANSAC 棋盘格融合\n'
                             f'MSD={demo_res["SIFT"]["msd_after"]:.1f}px',
                             fontsize=10)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(checker_ecc, cmap='gray')
        axes[1, 1].set_title(f'(e) ECC 棋盘格融合\n'
                             f'MSD={demo_res["ECC"]["msd_after"]:.1f}px',
                             fontsize=10)
        axes[1, 1].axis('off')

        # 配准前后的控制点位移
        pts1 = demo_pair['pts1'] * (512 / 2912.0)
        pts2 = demo_pair['pts2'] * (512 / 2912.0)
        pts1_sift = apply_transform_to_points(pts1, demo_res['SIFT']['transform'])

        axes[1, 2].imshow(img2, cmap='gray')
        axes[1, 2].scatter(pts2[:, 0], pts2[:, 1], c='green', s=30,
                          marker='o', label='GT (img2)')
        axes[1, 2].scatter(pts1_sift[:, 0], pts1_sift[:, 1], c='red', s=20,
                          marker='x', label='SIFT变换后')
        # 连线显示位移
        for j in range(len(pts2)):
            axes[1, 2].plot([pts1_sift[j, 0], pts2[j, 0]],
                           [pts1_sift[j, 1], pts2[j, 1]],
                           'y-', linewidth=0.5, alpha=0.7)
        axes[1, 2].set_title('(f) 控制点对齐验证\n(绿=GT, 红=SIFT变换后, 黄=残差)',
                            fontsize=10)
        axes[1, 2].legend(fontsize=7, loc='upper right')
        axes[1, 2].axis('off')

        plt.suptitle(f'图3-3 配准结果可视化 ({demo_pair["pair_id"]})', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig3-3_registration_demo.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print('  -> 图3-3 已保存')
    else:
        print('  [警告] 无完全成功的案例，跳过图3-3')

    #  图3-4: 各测试对的MSD改善分布
    fig, ax = plt.subplots(figsize=(12, 5))
    pair_ids = []
    sift_improvements = []
    ecc_improvements = []

    for pair, res in zip(test_pairs, test_results):
        pair_ids.append(pair['pair_id'])
        if res['SIFT']['success'] and res['msd_before'] > 0:
            sift_improvements.append(
                (1 - res['SIFT']['msd_after'] / res['msd_before']) * 100)
        else:
            sift_improvements.append(np.nan)
        if res['ECC']['success'] and res['msd_before'] > 0:
            ecc_improvements.append(
                (1 - res['ECC']['msd_after'] / res['msd_before']) * 100)
        else:
            ecc_improvements.append(np.nan)

    x = np.arange(len(pair_ids))
    width = 0.35
    bars1 = ax.bar(x - width/2, sift_improvements, width, label='SIFT+RANSAC',
                   color='#5B9BD5', edgecolor='black', linewidth=0.3)
    bars2 = ax.bar(x + width/2, ecc_improvements, width, label='ECC',
                   color='#FF6B6B', edgecolor='black', linewidth=0.3)

    ax.set_xlabel('测试图像对')
    ax.set_ylabel('MSD 改善率 (%)')
    ax.set_title('图3-4 各测试对的MSD改善率分布\n(正值=改善, 负值=恶化, 空白=失败)')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, rotation=60, fontsize=6, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3-4_per_pair_improvement.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> 图3-4 已保存')


if __name__ == '__main__':
    main()
