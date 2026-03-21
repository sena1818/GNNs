#!/bin/bash
# ============================================================================
# 一键运行所有消融实验 — 在 GPU 服务器上执行
#
# 用法:
#   cd final_project
#   nohup bash run_all_experiments.sh > experiments_log.txt 2>&1 &
#
# 预计耗时 (RTX 4090):
#   Step 1: Steps Sweep       ~15 min (纯推理)
#   Step 2: Generalization     ~10 min (纯推理)
#   Step 3: Decoding Ablation  ~10 min (纯推理)
#   Step 4: Visualization      ~5 min  (纯推理)
#   Step 5: GNN Architecture   ~4-6 h  (需训练 GAT + GCN)
#   Step 6: Generate Plots     ~10 sec
#   总计: ~5-6 h (大部分是架构消融训练)
# ============================================================================

set -e
cd "$(dirname "$0")"

RESULTS_DIR="experiments/results"
FIGS_DIR="report/figs"
mkdir -p "$RESULTS_DIR" "$FIGS_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting all experiments..."
echo ""

# ── Step 1: 推理步数扫描 (核心实验) ─────────────────────────────────────
echo "============================================"
echo " STEP 1: Inference Steps Sweep"
echo "============================================"
bash experiments/run_steps_sweep.sh

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | Step 1 done."

# ── Step 2: 跨规模泛化 ─────────────────────────────────────────────────
echo ""
echo "============================================"
echo " STEP 2: Cross-Scale Generalization"
echo "============================================"
for model in flow_matching_gated_gcn discrete_ddpm_gated_gcn continuous_ddpm_gated_gcn; do
    CKPT="checkpoints/${model}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "--- $model ---"
        bash experiments/run_generalization.sh "$CKPT"
    fi
done

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | Step 2 done."

# ── Step 3: 解码策略消融 ───────────────────────────────────────────────
echo ""
echo "============================================"
echo " STEP 3: Decoding Strategy Ablation"
echo "============================================"
for model in flow_matching_gated_gcn discrete_ddpm_gated_gcn; do
    CKPT="checkpoints/${model}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "--- $model ---"
        bash experiments/run_decoding.sh "$CKPT"
    fi
done

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | Step 3 done."

# ── Step 4: 扩散过程可视化 ─────────────────────────────────────────────
echo ""
echo "============================================"
echo " STEP 4: Diffusion Visualization"
echo "============================================"
python visualize_diffusion.py --out_dir "$FIGS_DIR"

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | Step 4 done."

# ── Step 5: GNN 架构消融 (可选, 耗时长) ────────────────────────────────
echo ""
echo "============================================"
echo " STEP 5: GNN Architecture Ablation (Training)"
echo "============================================"
echo "This step trains 2 new models (~2h each on RTX 4090)."
echo "If you want to skip, press Ctrl+C within 5 seconds."
sleep 5

bash experiments/run_ablation_arch.sh

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | Step 5 done."

# ── Step 6: 生成所有图表 ───────────────────────────────────────────────
echo ""
echo "============================================"
echo " STEP 6: Generate All Plots"
echo "============================================"
python experiments/plot_results.py

echo ""
echo "============================================"
echo " ALL EXPERIMENTS COMPLETE!"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Results:     $RESULTS_DIR/"
echo "Figures:     $FIGS_DIR/"
echo "Next step:   Write the LaTeX report!"
