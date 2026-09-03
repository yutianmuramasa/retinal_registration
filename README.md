# Retinal Image Registration (FIRE Dataset)

计算机视觉实习项目：在 **FIRE（Fundus Image Registration）眼底图像数据集**上实现并对比两种配准方法：

1. **SIFT + FLANN + RANSAC（基于特征）**：SIFT 检测血管分叉/视盘边缘等稳定关键点 → FLANN 最近邻匹配 + Lowe's ratio 筛选 → RANSAC 鲁棒估计单应性矩阵
2. **ECC（基于灰度）**：以增强相关系数为目标函数，梯度下降优化仿射变换参数

并对两类方法在**空间对齐精度（MSD）**与**灰度一致性（MSE）**两个维度上的表现做了系统对比分析。

## 主要结果

- 测试集 27 对图像上，**SIFT+RANSAC 将平均点距误差 MSD 由 56.6px 降至 20.5px（改善 63.9%）**，解剖结构空间对齐显著优于 ECC
- ECC 几乎无空间对齐效果（MSD 56.6 → 56.3px）：分析表明其优化的是"两幅图像看起来更相似"的灰度相关性，而非解剖点的重合，易陷入"灰度匹配但空间错位"的局部最优
- 实验同时揭示：**空间对齐与灰度一致性是两个不同（甚至冲突）的优化目标**，配准评估必须多维（MSD/MSE），不能只看单一指标
- 提出了改进方向："SIFT 粗配准（单应性）+ ECC 精调"两阶段级联方案

## 结果图表

`exp3_output/` 目录包含实验产出的全部图表：

| 文件 | 内容 |
|---|---|
| `fig3-1_msd_mse_comparison.png` | 配准前后 MSD / MSE 对比 |
| `fig3-2_success_rate.png` | 两方法"配准成功率"对比 |
| `fig3-3_registration_demo.png` | 棋盘格融合 + 控制点验证的可视化演示 |
| `fig3-4_per_pair_improvement.png` | 27 对测试图像各自的 MSD 改善率分布 |
| `表3-1_配准性能对比.csv` | 性能指标明细 |

## 运行

```bash
# 依赖
pip install opencv-python scikit-image scikit-learn numpy matplotlib

# 需先放置 FIRE 数据集（结构与脚本约定的 FIRE/FIRE 目录一致）：
#   FIRE/FIRE/Images/   眼底图像（2912×2912）
#   FIRE/FIRE/Ground Truth/  人工标注控制点（每对 10 对）

python experiment3_retinal_registration.py
```

说明：
- 图像默认缩放至 512×512 以平衡速度与精度（全分辨率下 SIFT+RANSAC 的 MSD 可进一步降至约 12px，但耗时约为 10 倍）
- FIRE 数据集（134 对眼底图像）需通过官方渠道申请下载，**不随本仓库分发**
- 脚本内的 `FIRE/` 已在 `.gitignore` 中排除

## 目录结构

```
retinal_registration/
├── experiment3_retinal_registration.py   # 实验主脚本（两种配准方法 + 评估可视化）
├── exp3_output/                          # 结果图表与指标 CSV
└── .gitignore
```
