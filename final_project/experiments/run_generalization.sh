#!/bin/bash
# 泛化实验：TSP-50 训练模型在不同规模上的表现
# 测试: TSP-20, TSP-50, TSP-100

set -e

CHECKPOINT="checkpoints/gated_gcn/best.pt"

echo "=== 实验2: 跨规模泛化测试 ==="

echo "[1/3] Testing on TSP-20..."
python evaluate.py --checkpoint $CHECKPOINT --data_file data/tsp20_train.txt

echo "[2/3] Testing on TSP-50..."
python evaluate.py --checkpoint $CHECKPOINT --data_file data/tsp50_train.txt

echo "[3/3] Testing on TSP-100..."
python evaluate.py --checkpoint $CHECKPOINT --data_file data/tsp100_test.txt

# 可选: 混合规模训练
# echo "=== 混合规模训练 ==="
# cat data/tsp20_train.txt data/tsp50_train.txt > data/mixed_train.txt
# python train.py --data_file data/mixed_train.txt --save_dir checkpoints/mixed
