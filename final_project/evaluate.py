"""
评估脚本 — 模型推理 + Optimality Gap 计算

需要实现:
1. 参数解析:
   --checkpoint: 模型 checkpoint 路径
   --data_file: 测试数据路径
   --inference_steps: 推理步数 (默认 50)
   --decode_method: 解码方法 (greedy / beam_search / greedy_2opt)
   --beam_width: Beam Search 宽度 (默认 5)
   --batch_size: 推理批大小

2. 评估流程:
   - 加载训练好的模型 (使用 EMA 权重)
   - 对每个测试实例:
     * 运行扩散去噪 → 概率热力图
     * 解码为合法路径
     * 计算路径长度
   - 计算 Optimality Gap: (model_cost - opt_cost) / opt_cost * 100%

3. 输出格式:
   TSP-50 Test Results (1000 instances):
   Avg Optimality Gap: X.XX%
   Avg Inference Time: X.XXs
   Valid Tour Rate: XX.X%
   Best Gap: X.XX%
   Worst Gap: X.XX%

用法:
    python evaluate.py --checkpoint checkpoints/best.pt --data_file data/tsp50_train.txt
"""

import argparse
import torch

# TODO: 实现评估逻辑

if __name__ == '__main__':
    pass
