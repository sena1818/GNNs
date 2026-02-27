
# Final Project Proposal

**Team Members**: [YanGang & PengXinyao]
**Course**: Generative Neural Networks (GNNs), Winter Semester 2025/26
**Submission Date**: January 22, 2026

---

# Proposal 1: Probabilistic Power Load Forecasting via Diffusion Models

## 1. Scientific Problem and Research Value

**Core Problem**: How to achieve accurate probabilistic forecasting of grid load to address uncertainties introduced by renewable energy integration?

**Research Value**: With European wind and solar penetration exceeding 30%, severe load fluctuations are introduced. Probabilistic forecasting can reduce reserve capacity requirements by 15-20% and improve grid stability. This study fills the gap in validating Diffusion-TS (ICLR 2024) in the energy domain and advances interpretable applications of diffusion models in time series forecasting.

## 2. Literature Review

**Core Papers**:
- **Diffusion-TS** (Yuan & Qiao, ICLR 2024): First interpretable time series diffusion model achieving long-term forecasting through trend/seasonal decomposition, but only validated on financial/traffic data
- **TimeGrad** (Rasul et al., ICML 2021): Autoregressive diffusion model with slow sampling (1000 steps) and limited interpretability
- **DeepAR** (Salinas et al., IJF 2020): Industry baseline with Gaussian assumption limiting distributional flexibility

**Our Extensions**: (1) Validate Diffusion-TS generalization on ETT energy dataset; (2) Ablation analysis quantifying contributions of trend extraction, seasonal modeling, and frequency-domain loss; (3) Uncertainty quantification comparing confidence interval quality via CRPS metrics; (4) Interpretability through visualizing decomposition components' correlation with real grid events (weekly patterns, temperature fluctuations).

## 3. Methodology and Feasibility

**Technical Roadmap**: The study proceeds in four phases: 
(1) **Data preprocessing**: Z-score normalization of ETTh1 dataset, identifying dominant periodic patterns through autocorrelation and Fourier spectrum analysis; 
(2) **Model implementation**: Reproduce Diffusion-TS core architecture including forward diffusion , seasonal modeling through Fourier synthesis layers, denoising network using 4-layer Transformer with direct x₀ prediction, and loss design L = MSE(x₀, x̂₀) + λ·‖FFT(x₀) - FFT(x̂₀)‖²; 
(3) **Experiments**: Train full model , reproduce baselines (TimeGrad, DeepAR), conduct systematic ablation studies; 
(4) **Analysis**: Quantitative evaluation using MAE/RMSE/CRPS metrics, visualization of prediction curves and uncertainty intervals.

**Feasibility**: (1) Diffusion-TS published at ICLR 2024 with solid theoretical foundation and peer-reviewed methodology; 
(2) ETT dataset widely used by 100+ time series forecasting papers with verified data quality; 
(3) Computational tasks feasible on Google Colab free tier with moderate model parameters; 
(4) Detailed hyperparameters and training strategies provided in paper.

## 4. Data Source

**ETT Dataset** (Electricity Transformer Temperature): ETTh1 (17,420 hourly samples, 7 variables including oil temperature and load, 1-hour sampling rate) and ETTm1 (69,680 samples, 15-minute sampling rate). Source: [GitHub - zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset). Quality: Clear daily/weekly cycles, missing rate < 0.5% (interpolated), temperature-load correlation 0.72.

## 5. Computational Resources

**Hardware**: Google Colab free tier (T4 GPU, 16GB VRAM)
**Estimated Time**: Data preprocessing 1h (CPU), Diffusion-TS training 30-40h (GPU, across sessions), baseline training 20h (TimeGrad 15h + DeepAR 5h), evaluation 3h. 
**Total GPU time**: ~60h.

## 6. Potential Risks and Mitigation

**Main Risks**: 
(1) **Memory limitations**: Transformer model may exceed GPU memory. Mitigation: Progressive hyperparameter tuning, reduce batch size (32→16) or hidden dim (256→128), use gradient accumulation; 
(2) **Training instability**: Risk of gradient explosion or mode collapse. Mitigation: Strictly follow paper hyperparameters, cosine annealing scheduler, gradient clipping (max_norm=1.0); 
(3) **Insufficient decomposition quality**: Trend/seasonal decomposition may underperform on energy data. Mitigation: Adjust Fourier layer frequencies (k=5/10/20), compare with classical STL/Wavelet methods.

---

# Proposal 2: Solving TSP via Diffusion Models

## 1. Scientific Problem and Research Value

**Core Problem**: Can discrete combinatorial optimization (NP-hard TSP) be reformulated as continuous generative modeling to achieve gradient-based learning with cross-scale generalization?

**Research Value**: TSP is a classical problem in combinatorial optimization with applications in logistics, chip design, and robotic navigation. Recent integration of diffusion models and GNNs provides a new paradigm, transforming discrete search space (N! permutations) into continuous probability modeling to generate near-optimal solutions (< 2% gap) in < 1 second. This study aims to: (1) Explore whether lightweight GNN architectures (GAT, GCN) maintain solution quality while significantly reducing computational cost; (2) Validate cross-scale generalization—whether training on small instances (TSP-50) enables solving larger problems (TSP-100/200); (3) Explore diverse solution strategies through probabilistic generation for emergency planning (e.g., road closures in logistics), combined with traditional 2-opt refinement.

## 2. Literature Review

**Core Papers**:
- **T2T-CO** (Chen et al., NeurIPS 2024): Anisotropic GNN + diffusion achieving 0.3% optimality gap on TSP-100 (SOTA), but high computational cost with Graph Transformer
- **DIFUSCO** (Sun et al., NeurIPS 2023): First TSP diffusion framework with complex discrete decoding
- **LKH Algorithm** (Helsgaun 2017): Near-optimal but slow (hours for TSP-10000), no learning capability
- **Attention Model** (Kool et al., ICLR 2019): RL end-to-end with poor generalization (TSP-50→100 gap > 10%)

**Our Extensions**: 
(1) Lightweight architecture ablation comparing Graph Transformer vs. GAT vs. GCN for efficiency-performance tradeoffs; 
(2) Cross-scale generalization testing (TSP-50 training → TSP-20/100/200 testing, target: TSP-100 gap < 2%); 
(3) Multi-modal diffusion generating 10 diverse high-quality paths; 
(4) Hybrid solver combining diffusion initialization with 2-opt local refinement.

## 3. Methodology and Feasibility

**Technical Roadmap**: 
(1) **Infrastructure setup**: Configure PyTorch Geometric and PyConcorde, generate multi-scale TSP instances (TSP-20: 1000, TSP-50: 5000) with LKH-computed optimal solutions; 
(2) **Core model implementation**: Anisotropic graph convolution encoder (4 layers) with separate weight matrices for in/out edges, U-Net diffusion decoder predicting denoising trajectory for adjacency matrix A ∈ {0,1}^(N×N), forward noising q(A_t|A_0) and reverse denoising with direct x₀ prediction; 
(3) **Extension experiments**: Architecture ablation (Graph Transformer/GAT/GCN), cross-scale generalization (TSP-50 training → TSP-20/100/200 testing), decoding strategy optimization (beam search k=10, greedy, temperature sampling), multi-modal generation (10-choose-1 strategy), hybrid solver (diffusion + 2-opt); 
(4) **Evaluation**: Quantitative metrics (optimality gap, inference time, success rate) and visualization (diffusion process GIF, edge probability heatmap, path comparison).

**Feasibility**: 
(1) T2T-CO provides complete implementation framework; 
(2) Self-generated data eliminates external dependencies, LKH solver produces deterministic labels quickly; 
(3) Memory requirements (~10GB) within Google Colab T4 capacity; 
(4) Fast inference (TSP-100 < 1 second) enables rapid experimentation; 
(5) NeurIPS 2024 paper provides detailed implementation and hyperparameters.

## 4. Data Source

**Self-Generated TSP Instances**: Sample N city coordinates uniformly from [0,1]², compute optimal paths via LKH solver , convert to adjacency matrix A[i,j]=1 (if edge in path), augment via rotation/reflection (×8 per instance).
**Scale and Quantity**: TSP-20: 1000 instances, TSP-50: 5000 instances (main training set), TSP-100/200: 1000/500 instances.

## 5. Computational Resources

**Hardware**: Google Colab free tier (T4 GPU)
**Estimated Time**: Data generation 5h (LKH-dominated), TSP-20/50 training 20h, architecture ablation 20h (3 variants × 5h), hybrid solver experiments 5h, evaluation and visualization 5h. 
**Total GPU time**: ~50h.

## 6. Potential Risks and Mitigation

**Main Risks**: 
(1) **Discrete decoding difficulty**: Decoding from continuous probability matrix to valid TSP path may fail. Mitigation: Use T2T-CO validated beam search, fallback to greedy decoding if needed; 
(2) **Cross-scale generalization degradation**: Model may underperform on large instances . Mitigation: Mixed-scale training; even if suboptimal, analyze failure modes as research findings; 
(3) **GNN training instability**: Deep GNNs risk over-smoothing or gradient vanishing. Mitigation: Exponential moving average (EMA) for weights, learning rate warmup; 
(4) **Resource constraints**: GPU memory/time limitations. Mitigation: Reduce batch size (32→16) with gradient accumulation, prioritize TSP-50 core experiments, simplify ablation if necessary.