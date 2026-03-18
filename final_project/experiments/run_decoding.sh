#!/bin/bash
# 实验 3：解码策略对比
# 同一模型（FM + Gated GCN + TSP-50），对比 5 种解码方式的 Gap vs 推理时间
#
# 用法（在 final_project/ 目录下运行）:
#   bash experiments/run_decoding.sh
#   bash experiments/run_decoding.sh checkpoints/discrete_ddpm_gated_gcn/best.pt

set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-checkpoints/flow_matching_gated_gcn/best.pt}"
DATA="data/tsp50_train.txt"
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

# 从 checkpoint 路径提取模型名（用于结果文件命名）
MODEL_TAG=$(basename "$(dirname "$CHECKPOINT")")

echo "============================================"
echo " Decoding Strategy Comparison"
echo " Checkpoint : $CHECKPOINT"
echo " Data       : $DATA"
echo "============================================"

echo "[1/5] Greedy..."
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_file "$DATA" \
    --decode greedy \
    --save_result "$RESULTS_DIR/${MODEL_TAG}_decode_greedy.json"

echo "[2/5] Beam Search k=3..."
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_file "$DATA" \
    --decode beam_search --beam_k 3 \
    --save_result "$RESULTS_DIR/${MODEL_TAG}_decode_beam3.json"

echo "[3/5] Beam Search k=5..."
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_file "$DATA" \
    --decode beam_search --beam_k 5 \
    --save_result "$RESULTS_DIR/${MODEL_TAG}_decode_beam5.json"

echo "[4/5] Beam Search k=10..."
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_file "$DATA" \
    --decode beam_search --beam_k 10 \
    --save_result "$RESULTS_DIR/${MODEL_TAG}_decode_beam10.json"

echo "[5/5] Greedy + 2-opt..."
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_file "$DATA" \
    --decode greedy --use_2opt \
    --save_result "$RESULTS_DIR/${MODEL_TAG}_decode_greedy2opt.json"

echo ""
echo "============================================"
echo " Summary (check JSON files for full stats)"
echo "============================================"
for f in "$RESULTS_DIR/${MODEL_TAG}_decode_"*.json; do
    tag=$(basename "$f" .json | sed "s/${MODEL_TAG}_decode_//")
    gap=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"avg_gap\"]:.2f}%')" 2>/dev/null || echo "N/A")
    ms=$(python  -c "import json; d=json.load(open('$f')); print(f'{d[\"avg_infer_ms\"]:.1f}ms')" 2>/dev/null || echo "N/A")
    echo "  $tag : gap=$gap  time=$ms"
done

echo ""
echo "Done. Results in $RESULTS_DIR/"
