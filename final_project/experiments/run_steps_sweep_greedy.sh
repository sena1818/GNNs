#!/bin/bash
# 实验：推理步数扫描（纯 greedy 解码，不带 2opt）
# 目的：展示真实的热力图质量随步数的变化，不被 2opt 后处理掩盖
# 这是论文中最关键的 figure 之一：证明 FM 5步 vs DDPM 需要更多步
#
# 用法:
#   bash experiments/run_steps_sweep_greedy.sh

set -e
cd "$(dirname "$0")/.."

SCALE="50"
ENCODER="gated_gcn"
TEST_DATA="data/tsp${SCALE}_test.txt"
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

STEPS_LIST="5 10 20 50 100"

echo "============================================"
echo " Steps Sweep (GREEDY ONLY — no 2opt)"
echo " Purpose: reveal true heatmap quality"
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
        echo "  steps=$steps (greedy) ..."
        python evaluate.py \
            --checkpoint "$CKPT" \
            --data_file "$TEST_DATA" \
            --inference_steps "$steps" \
            --decode greedy \
            --save_result "$RESULTS_DIR/steps_greedy_${mode}_s${steps}.json" \
            2>/dev/null
    done
    echo ""
done

# 汇总表格
echo "============================================"
echo " Steps Sweep Results (TSP-$SCALE, GREEDY)"
echo "============================================"
printf "%-6s" "Steps"
for mode in FM D3PM DDPM; do
    printf "  %-14s" "$mode"
done
echo ""
printf "%-6s" "------"
for _ in FM D3PM DDPM; do
    printf "  %-14s" "--------------"
done
echo ""

for steps in $STEPS_LIST; do
    printf "%-6s" "$steps"
    for mode in flow_matching discrete_ddpm continuous_ddpm; do
        f="$RESULTS_DIR/steps_greedy_${mode}_s${steps}.json"
        if [ -f "$f" ]; then
            info=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['avg_gap']:.2f}% v={d['valid_rate']*100:.0f}%\")")
            printf "  %-14s" "$info"
        else
            printf "  %-14s" "N/A"
        fi
    done
    echo ""
done

echo ""
echo "Compare with merge+2opt sweep in steps_*.json to see decoder contribution."
