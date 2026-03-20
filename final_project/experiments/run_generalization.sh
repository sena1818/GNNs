#!/bin/bash
# 实验 2：跨规模泛化测试
# 用 TSP-50 训练的最佳模型，直接在 TSP-20 / TSP-50 / TSP-100 上测试（不重新训练）
# 验证 Flow Matching + GNN 的泛化能力
#
# 用法（在 final_project/ 目录下运行）:
#   bash experiments/run_generalization.sh
#   bash experiments/run_generalization.sh checkpoints/fm_gat/best.pt  # 指定其他模型

set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-checkpoints/fm_gated_gcn/best.pt}"
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " Cross-Scale Generalization Test"
echo " Checkpoint: $CHECKPOINT"
echo "============================================"

for scale in 20 50 100; do
    # 所有规模都用独立测试集（不是训练集）
    DATA="data/tsp${scale}_test.txt"

    # 若测试集不存在，自动生成
    if [ ! -f "$DATA" ]; then
        echo "Generating TSP-${scale} test set (500 instances)..."
        python data/generate_tsp_data.py \
            --num_nodes "$scale" --num_samples 500 --output_file "$DATA"
    fi

    if [ ! -f "$DATA" ]; then
        echo "Skipping TSP-$scale: $DATA not found"
        continue
    fi

    echo "--- TSP-$scale (greedy) ---"
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data_file "$DATA" \
        --decode greedy \
        --save_result "$RESULTS_DIR/gen_tsp${scale}_greedy.json"

    echo "--- TSP-$scale (greedy+2opt) ---"
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data_file "$DATA" \
        --decode greedy --use_2opt \
        --save_result "$RESULTS_DIR/gen_tsp${scale}_greedy2opt.json"
done

echo ""
echo "Done. Results in $RESULTS_DIR/"
echo ""
echo "Tip: for mixed-scale training to improve generalization:"
echo "  cat data/tsp20_train.txt data/tsp50_train.txt > data/mixed_train.txt"
echo "  python train.py --data_file data/mixed_train.txt --save_dir checkpoints/fm_mixed"
