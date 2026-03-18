#!/bin/bash
# 消融实验 1：GNN 架构对比 (Gated GCN vs GAT vs Simple GCN)
# 三种架构在相同 FM 框架 + TSP-50 数据上训练，对比 Gap / 速度 / 参数量
#
# 用法（在 final_project/ 目录下运行）:
#   bash experiments/run_ablation_arch.sh
#
# 结果保存到 experiments/results/ablation_arch_*.json

set -e
cd "$(dirname "$0")/.."   # 切到 final_project/ 目录

DATA="data/tsp50_train.txt"
TEST="data/tsp50_test.txt"    # 独立测试集（不同于训练集）
EPOCHS=50
BATCH=32

# 若测试集不存在，自动生成
if [ ! -f "$TEST" ]; then
    echo "Generating TSP-50 test set (1000 instances)..."
    python data/generate_tsp_data.py --num_nodes 50 --num_samples 1000 --output_file "$TEST"
fi
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " Architecture Ablation (Flow Matching, TSP-50)"
echo "============================================"

# 变体 A: Gated GCN（最强，参数最多）
echo "[1/3] Training Gated GCN..."
python train.py \
    --data_file "$DATA" \
    --encoder_type gated_gcn \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --save_dir checkpoints/fm_gated_gcn

# 变体 B: GAT
echo "[2/3] Training GAT..."
python train.py \
    --data_file "$DATA" \
    --encoder_type gat \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --save_dir checkpoints/fm_gat

# 变体 C: Simple GCN（最轻量）
echo "[3/3] Training Simple GCN..."
python train.py \
    --data_file "$DATA" \
    --encoder_type gcn \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --save_dir checkpoints/fm_gcn

# 评估（greedy + 2opt 两种解码）
echo ""
echo "============================================"
echo " Evaluation"
echo "============================================"
for model in fm_gated_gcn fm_gat fm_gcn; do
    echo "--- $model (greedy) ---"
    python evaluate.py \
        --checkpoint "checkpoints/$model/best.pt" \
        --data_file "$TEST" \
        --decode greedy \
        --save_result "$RESULTS_DIR/${model}_greedy.json"

    echo "--- $model (greedy+2opt) ---"
    python evaluate.py \
        --checkpoint "checkpoints/$model/best.pt" \
        --data_file "$TEST" \
        --decode greedy --use_2opt \
        --save_result "$RESULTS_DIR/${model}_greedy2opt.json"
done

echo ""
echo "Done. Results in $RESULTS_DIR/"
