#!/bin/bash
# 解码策略对比实验 (可选)
# 同一模型，不同解码方法的 Gap vs 时间 Pareto 分析

set -e

CHECKPOINT="checkpoints/gated_gcn/best.pt"
DATA="data/tsp50_train.txt"

echo "=== 实验3: 解码策略对比 ==="

echo "[1/5] Greedy decode..."
python evaluate.py --checkpoint $CHECKPOINT --data_file $DATA --decode_method greedy

echo "[2/5] Beam Search k=3..."
python evaluate.py --checkpoint $CHECKPOINT --data_file $DATA --decode_method beam_search --beam_width 3

echo "[3/5] Beam Search k=5..."
python evaluate.py --checkpoint $CHECKPOINT --data_file $DATA --decode_method beam_search --beam_width 5

echo "[4/5] Beam Search k=10..."
python evaluate.py --checkpoint $CHECKPOINT --data_file $DATA --decode_method beam_search --beam_width 10

echo "[5/5] Greedy + 2-opt..."
python evaluate.py --checkpoint $CHECKPOINT --data_file $DATA --decode_method greedy_2opt
