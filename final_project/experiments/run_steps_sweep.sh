#!/bin/bash
# 实验二：推理步数扫描 — 核心实验
# 固定训练好的模型，用不同推理步数评估，生成 steps-vs-gap 曲线
#
# 用法:
#   bash experiments/run_steps_sweep.sh              # 默认 TSP-50
#   bash experiments/run_steps_sweep.sh 20           # TSP-20

set -e
cd "$(dirname "$0")/.."

SCALE="${1:-50}"
ENCODER="gated_gcn"
TEST_DATA="data/tsp${SCALE}_test.txt"
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: $TEST_DATA not found."
    exit 1
fi

STEPS_LIST="5 10 20 50 100"

echo "============================================"
echo " Steps Sweep: inference steps vs gap"
echo " Scale: TSP-$SCALE"
echo " Steps: $STEPS_LIST"
echo "============================================"
echo ""

for mode in flow_matching discrete_ddpm continuous_ddpm; do
    CKPT="checkpoints/${mode}_${ENCODER}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found, skipping $mode."
        continue
    fi

    echo "--- $mode ---"
    for steps in $STEPS_LIST; do
        echo "  steps=$steps ..."
        python evaluate.py \
            --checkpoint "$CKPT" \
            --data_file "$TEST_DATA" \
            --inference_steps "$steps" \
            --decode greedy \
            --save_result "$RESULTS_DIR/steps_${mode}_s${steps}.json" \
            2>/dev/null
    done
    echo ""
done

# 汇总表格
echo "============================================"
echo " Steps Sweep Results (TSP-$SCALE, greedy)"
echo "============================================"
printf "%-6s" "Steps"
for mode in FM D3PM DDPM; do
    printf "  %-12s" "$mode"
done
echo ""
printf "%-6s" "------"
for _ in FM D3PM DDPM; do
    printf "  %-12s" "------------"
done
echo ""

for steps in $STEPS_LIST; do
    printf "%-6s" "$steps"
    for mode in flow_matching discrete_ddpm continuous_ddpm; do
        f="$RESULTS_DIR/steps_${mode}_s${steps}.json"
        if [ -f "$f" ]; then
            gap=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['avg_gap']:.2f}%\")")
            printf "  %-12s" "$gap"
        else
            printf "  %-12s" "N/A"
        fi
    done
    echo ""
done

echo ""
echo "Done. Plot with: python experiments/plot_results.py --plot steps_sweep"
