"""
可视化工具 — 扩散过程 GIF、路径对比、训练曲线

需要实现:
1. save_diffusion_gif(model, coords, output_path, n_frames=20):
   - 每 50 步截一帧 (t=1000 → t=0)
   - t=1000: 全灰噪声 → t=0: 清晰城市连线
   - 用 imageio 保存为 GIF (fps=5)

2. plot_tour_comparison(coords, model_tour, opt_tour, save_path):
   - 左图: LKH 最优解 (蓝色)
   - 右图: 模型解 (橙色)
   - 下方标注: LKH Cost / Model Cost / Gap%

3. plot_heatmap(coords, heatmap, title, save_path):
   - N×N 边概率热力图 (imshow)
   - 叠加城市坐标点

4. plot_training_curve(losses, save_path):
   - 横轴=epoch, 纵轴=BCE Loss

5. plot_ablation_bar(results_dict, save_path):
   - 消融实验柱状图 (不同架构的 Gap)

6. plot_generalization_curve(sizes, gaps, save_path):
   - 横轴=TSP 规模, 纵轴=Gap%

依赖: matplotlib, imageio[ffmpeg], numpy
"""

import matplotlib.pyplot as plt
import numpy as np

# TODO: 实现可视化函数
