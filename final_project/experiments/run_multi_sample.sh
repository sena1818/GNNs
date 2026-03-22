#!/bin/bash
# 实验：多次采样取最优（对标 DIFUSCO 的 "10 steps × 16 samples"）
# 对每个实例生成 N 个独立热力图，各自解码，取最短路径
#
# 注意：FM 是确定性 ODE，多次采样结果相同，因此跳过 FM
# D3PM 和 DDPM 有随机性，多次采样有意义
#
# 用法:
#   bash experiments/run_multi_sample.sh

set -e
cd "$(dirname "$0")/.."

ENCODER="gated_gcn"
TEST_DATA="data/tsp50_test.txt"
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

SAMPLE_COUNTS="1 4 8 16"

echo "============================================"
echo " Multi-Sample Experiment"
echo " Take best tour from N independent samples"
echo " DIFUSCO uses: 10 steps × 16 samples"
echo "============================================"
echo ""

# D3PM: 使用 10 步（对标 DIFUSCO 的 sampling 设置）
echo "=== Discrete DDPM (D3PM) — 10 steps ==="
CKPT="checkpoints/discrete_ddpm_${ENCODER}/best.pt"
if [ -f "$CKPT" ]; then
    for ns in $SAMPLE_COUNTS; do
        echo "  n_samples=$ns ..."
        python evaluate.py \
            --checkpoint "$CKPT" \
            --data_file "$TEST_DATA" \
            --inference_steps 10 \
            --decode merge --use_2opt \
            --n_samples "$ns" \
            --save_result "$RESULTS_DIR/multisample_d3pm_n${ns}.json" \
            2>/dev/null
    done
else
    echo "WARNING: $CKPT not found."
fi
echo ""

# Continuous DDPM: 使用 50 步
echo "=== Continuous DDPM — 50 steps ==="
CKPT="checkpoints/continuous_ddpm_${ENCODER}/best.pt"
if [ -f "$CKPT" ]; then
    for ns in $SAMPLE_COUNTS; do
        echo "  n_samples=$ns ..."
        python evaluate.py \
            --checkpoint "$CKPT" \
            --data_file "$TEST_DATA" \
            --inference_steps 50 \
            --decode merge --use_2opt \
            --n_samples "$ns" \
            --save_result "$RESULTS_DIR/multisample_ddpm_n${ns}.json" \
            2>/dev/null
    done
else
    echo "WARNING: $CKPT not found."
fi
echo ""

# FM: 确定性 ODE，但还是测一下（验证多次采样无效果）
echo "=== Flow Matching — 20 steps (deterministic, as control) ==="
CKPT="checkpoints/flow_matching_${ENCODER}/best.pt"
if [ -f "$CKPT" ]; then
    for ns in 1 8; do
        echo "  n_samples=$ns ..."
        python evaluate.py \
            --checkpoint "$CKPT" \
            --data_file "$TEST_DATA" \
            --inference_steps 20 \
            --decode merge --use_2opt \
            --n_samples "$ns" \
            --save_result "$RESULTS_DIR/multisample_fm_n${ns}.json" \
            2>/dev/null
    done
else
    echo "WARNING: $CKPT not found."
fi

# 汇总
echo ""
echo "============================================"
echo " Multi-Sample Results Summary"
echo "============================================"
printf "%-10s  %-12s  %-12s  %-12s\n" "Samples" "D3PM(10s)" "DDPM(50s)" "FM(20s)"
printf "%-10s  %-12s  %-12s  %-12s\n" "--------" "----------" "----------" "----------"
for ns in $SAMPLE_COUNTS; do
    printf "%-10s" "$ns"
    for prefix in d3pm ddpm fm; do
        f="$RESULTS_DIR/multisample_${prefix}_n${ns}.json"
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
echo "Done."
