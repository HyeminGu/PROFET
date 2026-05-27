# PROFET — Particle-based Reconstruction Of generative Force-matched Expression Trajectories

**PROFET** reconstructs continuous gene expression dynamics from static, time-stamped single-cell RNA sequencing (scRNA-seq) snapshots. Unlike conventional methods that rely on discrete timepoints or assume linear transitions, PROFET models cell state evolution as a principled generative process. It has been validated on both synthetic and experimental datasets and applied to uncover treatment-induced heterogeneity in breast cancer. By recovering dynamic expression trajectories from static scRNA-seq data, PROFET provides a scalable and principled tool for modeling cell state transitions in development, disease, and therapeutic response.

## Method

1. **Step 1 — Particle transport (GPA):** constructs optimal transport plans between empirical distributions at consecutive timepoints using a Lipschitz-regularised KL divergence minimisation, producing temporally smooth and distribution-consistent particle trajectories (`run_GPA.py`, TensorFlow).

2. **Step 2 — Force matching:** fits a time-dependent neural ODE velocity field to the particle flows from Step 1, yielding a continuous global vector field (`run_ForceMatching.py`, TensorFlow). At inference, the fitted field is loaded via `models/velocityfield.py` (PyTorch) and integrated with a forward-Euler ODE solver.

3. **Step 3 — Downstream analysis:** the inferred continuous trajectory is used for four types of biological analysis, all implemented in `util/downstream.py`:
   - **Trajectory visualisation and subtrajectory classification:** reconstructed cell trajectories are visualised in PCA space and classified into subgroups based on either fate (target time point clustering) or ancestral state (source time point clustering), revealing distinct cell fate decisions.
   - **Gene expression dynamics** *(EMT, mESC):* per-gene expression is reconstructed over continuous time from the trajectory, enabling comparison of dynamic gene programmes across subtrajectories via average dynamics, violin plots, fold change, and KDE distribution comparisons at held-out intermediate timepoints.
   - **Phenotypic shift heterogeneity** *(breast cancer datasets):* cells are classified into Low / Medium / High phenotypic shift groups based on displacement in PCA space before and after treatment, and per-gene expression dynamics are reconstructed within each group to characterise transcriptional diversity in treatment response.
   - **Fate analysis** *(LARRY, Axolotl):* cell fates are predefined from published studies. For LARRY, fate labels are provided at day 6 for three groups (neutrophils, monocytes, and others); for Axolotl, at day 7 for four groups (BE, IE, RCP, and CT cells). Each inferred trajectory is assigned to the nearest fate group centroid at the corresponding reference timepoint (day 6 for LARRY, day 7 for Axolotl), and per-gene expression dynamics are reconstructed within each fate class.

## Installation

```bash
git clone https://github.com/HyeminGu/PROFET.git
cd PROFET
pip install -r requirements.txt
```

Key dependencies: `torch`, `tensorflow`, `geomloss`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.

## Project Structure

```
PROFET/                                  ← project root
│
├── PROFET/                              ← core code (lib_dir in notebooks)
│   ├── run_GPA.py                       ← Step 1: particle transport
│   ├── run_ForceMatching.py             ← Step 2: velocity field training
│   ├── models/
│   │   ├── velocityfield.py             ← PyTorch VelocityField (load + ODE integrate)
│   │   └── discriminator.py             ← GPA discriminator network
│   └── configs/
│       ├── GPA.yaml                     ← default GPA hyperparameters
│       └── GPA-Toy1.yaml                ← toy-data config
│
├── util/                                ← shared Python utilities
│   ├── utils.py                         ← data I/O, PCA, animation, W2 metric
│   └── downstream.py                    ← all downstream analysis functions
│                                           (gene dynamics, trajectory visualisation,
│                                            subtrajectory classification)
│
├── notebooks/                           ← one self-contained notebook per dataset
│   ├── Emt_72.ipynb                     ← EMT 
│   ├── Stem_cell_differentiation.ipynb  ← mESC 
│   ├── MCF7_Cell_Line.ipynb             ← MCF7 breast cancer cell line
│   ├── Patient_PA3.ipynb                ← Patient PA3 (BMC cohort)
│   ├── Patient_862.ipynb                ← Patient 862 (NatMed cohort)
│   ├── Patient_887.ipynb                ← Patient 887 (NatMed cohort)
│   ├── Synthetic.ipynb                  ← synthetic trajectory benchmark
│   ├── LARRY_3000_benchmark.ipynb       ← LARRY dataset benchmark
│   ├── Axolotl_data_2000.ipynb          ← Axolotl limb regeneration
│   └── OU_process-GPA.ipynb             ← Ornstein-Uhlenbeck toy example
│
├── data/                                ← raw data and preprocessed .pkl files
│                                           (not included in the repository)
├── assets/                              ← outputs: GIFs, plots, model weights
├── requirements.txt
├── LICENSE
└── README.md
```

## Examples

PROFET has been applied and benchmarked across nine datasets spanning multiple biological contexts:

| Notebook | Dataset | Context |
|---|---|---|
| `Emt_72.ipynb` | EMT (72 genes, 12,588 cells) | Epithelial-to-mesenchymal transition; 6 timepoints (days 0–8); trains on days 0, 4; holds out days 1, 2, 3, 8 |
| `Stem_cell_differentiation.ipynb` | mESC differentiation (100 genes, 456 cells) | Mouse embryonic stem cell differentiation; 5 timepoints (days 0–4); trains on days 0, 2, 4; holds out days 1, 3 |
| `MCF7_Cell_Line.ipynb` | MCF7 breast cancer cell line (117 genes, 14,160 cells) | Palbociclib treatment response (NDPR cohort); day 0 → day 1 |
| `Patient_PA3.ipynb` | Patient PA3 (116 genes, 4,692 cells) | Palbociclib treatment (BMC cohort); day 0 → day 1 |
| `Patient_862.ipynb` | Patient 862 (115 genes, 17,260 cells) | Palbociclib treatment (NatMed cohort); day 0 → day 1 |
| `Patient_887.ipynb` | Patient 887 (115 genes, 10,174 cells) | Palbociclib treatment (NatMed cohort); day 0 → day 1 |
| `LARRY_3000_benchmark.ipynb` | LARRY (3,000 genes, 49,302 cells) | Lineage-tracing benchmark; 3 timepoints (days 2, 4, 6); trains on days 2, 6; holds out day 4 |
| `Axolotl_data_2000.ipynb` | Axolotl limb regeneration (2,000 genes, 18,648 cells) | 5 timepoints (days 0–4); trains on days 0, 2, 4; holds out days 1, 3 |
| `Synthetic.ipynb` | Synthetic trajectory (26 genes, 1,195 cells) | 5 timepoints (days 0–4); trains on days 0, 2, 4; holds out days 1, 3 |

An Ornstein-Uhlenbeck toy example (`OU_process-GPA.ipynb`) is also provided.

## Usage

Each notebook is self-contained and walks through the full pipeline for one dataset.

### Typical workflow

```
notebooks/<Dataset>.ipynb
│
├── 1. Preprocessing
│      Input:  raw gene expression matrix (.txt) + cell time annotation (.txt)
│      Output: preprocessed dataset saved as data/<name>_preprocessed.pkl
│              PCA variance ratio plot saved to data/
│
├── 2. PROFET
│      Step 1 (GPA)
│        Input:  preprocessed .pkl (projected PCA coordinates)
│        Output: GPA transport plan saved as assets/<name>/KL-Lipschitz_...pickle
│      Step 2 (Force Matching)
│        Input:  GPA .pickle file(s) from Step 1
│        Output: velocity field weights + hyperparameters saved to assets/<name>/<exp_memo>/
│      ODE integration
│        Input:  velocity field from assets/<name>/<exp_memo>/
│        Output: X1_trpts — list of cell positions at each time step
│
├── 3. Trajectory Visualization & Subtrajectory Classification
│      Input:  X1_trpts, pca, mats (per-timepoint expression matrices)
│      Output: static trajectory plots (.png, with/without snapshots)
│              animated subtrajectory GIFs (.gif)
│              cluster label CSV ({exp_memo}_X1_hat_clusters.csv or _X2_hat_clusters.csv)
│
└── 4. Downstream Analysis
       EMT / mESC
         Input:  X1_trpts, cluster label CSV, gene expression matrices
         Output: per-gene average dynamics plots, violin plots by subtrajectory,
                 fold change / p-value CSVs and plots, single-cell trajectory plots,
                 KDE distribution comparisons at intermediate timepoints
       Breast cancer (MCF7 / PA3 / 862 / 887)
         Input:  X1_trpts, gene expression matrices
         Output: displacement distribution plots and CSVs,
                 Low / Medium / High phenotypic shift classification plots,
                 per-gene single-cell dynamics by shift class
       LARRY / Axolotl
         Input:  X1_trpts, predefined fate labels (neutrophils / monocytes / others for LARRY;
                 BE / IE / RCP / CT for Axolotl), gene expression matrices
         Output: fate-classified subtrajectory plots,
                 per-gene dynamics by fate class
```

### Data directory layout

Raw datasets are available for download at:
[https://drive.google.com/drive/folders/1ba-skCOxvosDQTWz1Rq3GlClk-NH8-eV](https://drive.google.com/drive/folders/1ba-skCOxvosDQTWz1Rq3GlClk-NH8-eV)

Preprocessed datasets (`.pkl` files) are available for download at:
[https://drive.google.com/drive/folders/1jrh3L8ZrHaGbSQDNA95ZXK383PaJvl9I?usp=drive_link](https://drive.google.com/drive/folders/1jrh3L8ZrHaGbSQDNA95ZXK383PaJvl9I?usp=drive_link)

Place each dataset under `data/`:

| Dataset | Pickle file | Timepoints | Genes | Total cells | Training tp | Held-out |
|---|---|---|---|---|---|---|
| EMT | `emt_72_preprocessed.pkl` | 0, 1, 2, 3, 4, 8 | 72 | 12,588 | 0, 4 | 1, 2, 3, 8 |
| Stem cell differentiation (mESC) | `stem_cell_differentiation_preprocessed.pkl` | 0, 1, 2, 3, 4 | 100 | 456 | 0, 2, 4 | 1, 3 |
| MCF7 cell line | `MCF7_Cell_Line_preprocessed.pkl` | 0, 1 | 117 | 14,160 | 0, 1 | — |
| Patient PA3 | `Patient_PA3_preprocessed.pkl` | 0, 1 | 116 | 4,692 | 0, 1 | — |
| Patient 862 | `Patient_862_preprocessed.pkl` | 0, 1 | 115 | 17,260 | 0, 1 | — |
| Patient 887 | `Patient_887_preprocessed.pkl` | 0, 1 | 115 | 10,174 | 0, 1 | — |
| LARRY benchmark | `LARRY_3000_benchmark_preprocessed.pkl` | 2, 4, 6 | 3,000 | 49,302 | 2, 6 | 4 |
| Axolotl limb regeneration | `Axolotl_data_2000_preprocessed.pkl` | 0, 1, 2, 3, 4 | 2,000 | 18,648 | 0, 2, 4 | 1, 3 |
| Synthetic | `synthetic_preprocessed.pkl` | 0, 1, 2, 3, 4 | 26 | 1,195 | 0, 2, 4 | 1, 3 |

## Utility modules

### `util/utils.py`

| Function | Description |
|---|---|
| `load_preprocessed_data` | Load a saved `.pkl` dataset |
| `save_preprocessed_data` | Save preprocessed data to `.pkl` |
| `reduce_dimension` | Fit full-rank PCA and save variance plot |
| `visualize_data` | Per-timepoint 2D PCA scatter plots |
| `generate_animation` | Animated GIF of trajectory + optional vector field |
| `generate_W2distance_plot` | W₂ distance between predicted trajectory and data over time |
| `W2` | Sinkhorn W₂ between two sample sets |
| `save_trajectories` | Save a list of trajectory snapshots to a `.pkl` file |
| `ResourceMonitor` | Context manager measuring wall-clock time and peak GPU / CPU memory |

### `util/downstream.py`

Contains all downstream analysis and visualization functions, organized in two sections:

**Gene Expression Dynamics**
- `Average_gene_dynamics_whole_saveonly` — mean trajectory with 95 % CI
- `Average_gene_dynamics_whole_saveonly_with_violin_plot_sample1_EMT` — violin plots by subtrajectory (EMT)
- `Average_gene_dynamics_whole_saveonly_with_violin_plot_sample_3_stem` — violin plots by subtrajectory (mESC)
- `Average_gene_dynamics_whole_saveonly_single_trajectory_EMT/mESC` — single-cell trajectories
- `Average_gene_dynamics_whole_saveonly_single_trajectory_Axolotl` — single-cell trajectories (Axolotl)
- `Average_gene_dynamics_whole_saveonly_with_violin_plot_Axolotl` — violin plots by subtrajectory (Axolotl)
- `Average_gene_dynamics_whole_saveonly_single_trajectory_NDPR_breast_cancer` — single-cell (MCF7)
- `Average_gene_dynamics_whole_saveonly_single_trajectory_clinical` — single-cell (PA3, 862, 887)
- `Compute_and_Plot_FoldChange_MeanDiff_PValues` — fold change, mean difference, p-values
- `difference_of_means_emt / difference_of_means_stem` — between-subgroup statistics
- `Compare_Distribution_Trajectories_Intermediate_EMT/mESC` — KDE comparisons at intermediate times
- `plot_X1_hat_displacement_distribution` — displacement histogram (breast cancer)
- `generate_static_cluster_plot_deviation_colormap_MCF7/PA3/862/887` — phenotypic shift classification

**Trajectory Visualization & Subtrajectory Classification**
- `generate_static_trajectory_plots_three_timepoints` — static plots, 3 training timepoints
- `generate_static_trajectory_plots_two_timepoints` — static plots, 2 training + 1 test
- `generate_static_trajectory_plots_two_timepoints_no_middle` — static plots, 2 training, no test
- `generate_static_cluster_plot_target` — static subtrajectory plot, clustered by fate
- `generate_static_cluster_plot_target_with_dfcluster_selected_clusters` — fate-classified plot using pre-labelled clusters with optional subset selection
- `generate_static_cluster_plot_target_LARRY_benchmark_fate` — fate-classified subtrajectory plot for the LARRY benchmark
- `generate_static_cluster_plot_source` — static subtrajectory plot, clustered by ancestor
- `classify_X1_hat` — animated fate classification
- `classify_X2_hat` — animated ancestral classification

## Citation

If you use PROFET in your research, please cite:

```bibtex
@article{cheng2025profet,
  title={PROFET Predicts Continuous Gene Expression Dynamics
from scRNA-seq Data to Elucidate Resistance to Cancer Therapy},
  author={},
  journal={Preprint},
  year={2025}
}
```
