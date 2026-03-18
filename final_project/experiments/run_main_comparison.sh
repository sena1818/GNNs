#!/bin/bash
# 主实验：三种扩散框架对比
# Flow Matching vs Discrete DDPM (D3PM) vs Continuous DDPM
# 相同 GNN 架构 (Gated GCN)，相同数据 (TSP-50)，相同 epoch 数
#
# 用法（在 final_project/ 目录下运行）:
#   bash experiments/run_main_comparison.sh           # TSP-50，50 epoch
#   bash experiments/run_main_comparison.sh 20 10     # TSP-20，10 epoch（快速调试）
#
# 结果保存到 experiments/results/main_*.json
# 训练日志保存到 checkpoints/<mode>_gated_gcn/history.json

set -e
cd "$(dirname "$0")/.."

SCALE="${1:-50}"          # TSP 规模：20 / 50 / 100
EPOCHS="${2:-50}"         # 训练轮数
BATCH=32
ENCODER="gated_gcn"
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

# 数据文件
if [ "$SCALE" -eq 100 ]; then
    TRAIN_DATA="data/tsp100_train.txt"
    TEST_DATA="data/tsp100_test.txt"
else
    TRAIN_DATA="data/tsp${SCALE}_train.txt"
    TEST_DATA="data/tsp${SCALE}_test.txt"   # 独立测试集，与训练集不同
fi

# 若测试集不存在，自动生成（1000 个独立实例）
if [ ! -f "$TEST_DATA" ] && [ "$SCALE" -ne 100 ]; then
    echo "Generating TSP-${SCALE} test set (1000 instances)..."
    python data/generate_tsp_data.py \
        --num_nodes "$SCALE" --num_samples 1000 --output_file "$TEST_DATA"
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA not found. Run data/generate_tsp_data.py first."
    exit 1
fi

echo "============================================"
echo " Main Comparison: FM vs D3PM vs Gaussian"
echo " Scale   : TSP-$SCALE"
echo " Epochs  : $EPOCHS"
echo " Encoder : $ENCODER"
echo " Train   : $TRAIN_DATA"
echo "============================================"
echo ""

# ── 训练 ────────────────────────────────────────

echo "[1/3] Training Flow Matching..."
python train.py \
    --mode flow_matching \
    --data_file "$TRAIN_DATA" \
    --encoder_type "$ENCODER" \
    --epochs "$EPOCHS" \
    --batch_size $BATCH \
    --save_dir "checkpoints/flow_matching_${ENCODER}"

echo ""
echo "[2/3] Training Discrete DDPM (D3PM)..."
python train.py \
    --mode discrete_ddpm \
    --data_file "$TRAIN_DATA" \
    --encoder_type "$ENCODER" \
    --epochs "$EPOCHS" \
    --batch_size $BATCH \
    --save_dir "checkpoints/discrete_ddpm_${ENCODER}"

echo ""
echo "[3/3] Training Continuous DDPM (Gaussian)..."
python train.py \
    --mode continuous_ddpm \
    --data_file "$TRAIN_DATA" \
    --encoder_type "$ENCODER" \
    --epochs "$EPOCHS" \
    --batch_size $BATCH \
    --save_dir "checkpoints/continuous_ddpm_${ENCODER}"

# ── 评估 ────────────────────────────────────────

echo ""
echo "============================================"
echo " Evaluation (greedy + greedy+2opt)"
echo "============================================"

for mode in flow_matching discrete_ddpm continuous_ddpm; do
    CKPT="checkpoints/${mode}_${ENCODER}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found, skipping."
        continue
    fi

    echo ""
    echo "--- $mode (greedy) ---"
    python evaluate.py \
        --checkpoint "$CKPT" \
        --data_file "$TEST_DATA" \
        --decode greedy \
        --save_result "$RESULTS_DIR/main_${mode}_greedy.json"

    echo "--- $mode (greedy+2opt) ---"
    python evaluate.py \
        --checkpoint "$CKPT" \
        --data_file "$TEST_DATA" \
        --decode greedy --use_2opt \
        --save_result "$RESULTS_DIR/main_${mode}_greedy2opt.json"
done

# ── 汇总表格 ─────────────────────────────────────

echo ""
echo "============================================"
echo " Results Summary (TSP-$SCALE, $EPOCHS epochs)"
echo "============================================"
printf "%-22s  %-10s  %-10s  %-12s\n" "Model" "Gap(greedy)" "Gap(+2opt)" "Time(ms)"
printf "%-22s  %-10s  %-10s  %-12s\n" "----------------------" "----------" "----------" "------------"

for mode in flow_matching discrete_ddpm continuous_ddpm; do
    f_greedy="$RESULTS_DIR/main_${mode}_greedy.json"
    f_2opt="$RESULTS_DIR/main_${mode}_greedy2opt.json"

    if [ -f "$f_greedy" ] && [ -f "$f_2opt" ]; then
        python - "$mode" "$f_greedy" "$f_2opt" <<'PYEOF'
import sys, json
mode = sys.argv[1]
d1   = json.load(open(sys.argv[2]))
d2   = json.load(open(sys.argv[3]))
print(f"{mode:<22}  {d1['avg_gap']:.2f}%{'':<6}  {d2['avg_gap']:.2f}%{'':<6}  {d1['avg_infer_ms']:.1f}ms")
PYEOF
    fi
done

echo ""
echo "Done. Full results in $RESULTS_DIR/"
echo "Training curves: checkpoints/*/history.json"
