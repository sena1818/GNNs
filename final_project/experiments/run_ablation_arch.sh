#!/bin/bash
# 消融实验：GNN 架构对比 (Gated GCN vs GAT vs Simple GCN)
# 在 TSP-50 上各训练 50 epoch，对比 Optimality Gap

set -e

DATA="data/tsp50_train.txt"
EPOCHS=50
BATCH=32

echo "=== 实验1: 架构消融 ==="

# 变体A: Gated GCN (DIFUSCO baseline)
echo "[1/3] Training Gated GCN..."
python train.py --data_file $DATA --encoder_type gated_gcn --epochs $EPOCHS --batch_size $BATCH --save_dir checkpoints/gated_gcn

# 变体B: GAT (Graph Attention Network)
echo "[2/3] Training GAT..."
python train.py --data_file $DATA --encoder_type gat --epochs $EPOCHS --batch_size $BATCH --save_dir checkpoints/gat

# 变体C: Simple GCN
echo "[3/3] Training Simple GCN..."
python train.py --data_file $DATA --encoder_type gcn --epochs $EPOCHS --batch_size $BATCH --save_dir checkpoints/gcn

# 评估
echo "=== 评估 ==="
for model in gated_gcn gat gcn; do
    echo "--- $model ---"
    python evaluate.py --checkpoint checkpoints/$model/best.pt --data_file $DATA
done
