from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as mcolors
import os
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.lines as mlines
import matplotlib.animation as animation
from IPython.display import Image
import pandas as pd
import torch # for W2 calculation purpose
from geomloss import SamplesLoss
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from math import ceil
from scipy.stats import gaussian_kde
from matplotlib import colormaps

def generate_static_cluster_plot_target_with_dfcluster_selected_clusters(
    pca,
    source_t, target_t, X1_trpts, mats, optimal_k, start_i, index, p,
    df_cluster,
    day_value=None,
    day_col='day', cluster_col='cluster',
    reverse=False, intermediate_t=[1,2,3],
    d_red=2, random_state=42, exp_memo='experiment', output_file=None,
    selected_clusters=None,          # e.g., [1,6,9]; None => use all present
    hide_unselected=False,           # dim or hide unselected trajectories
    bg_alpha=0.15,                   # background transparency
    traj_alpha=0.9,                  # selected traj opacity
    traj_gray_alpha=0.15,            # unselected traj opacity (if not hidden)
    point_size=3, line_alpha=0.5, line_width=0.6,
    cnv_lut=None,                    # pass your global_cnv_lut here
    strict_colors=True,              # error if label color missing
    color_mode='final'               # 'final' (default) or 'frame'
):
    """
    Build PCA-space centroids for target-day cells grouped by predefined clusters
    (from df_cluster[cluster_col], filtered by day only). Classify trajectory
    snapshots by nearest centroid and plot:
      - background (source/intermediate/target) = light gray (bg_alpha),
      - selected clusters in consistent colors from cnv_lut,
      - unselected either dim gray or hidden.

    color_mode:
      'final' -> color every frame by the final-frame label (stable per trajectory)
      'frame' -> color each frame by instantaneous nearest centroid (changes over time)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---------- helpers ----------
    def _reduce_if_needed(A, pca_obj):
        ncomp = getattr(pca_obj, "n_components_", getattr(pca_obj, "n_components", None))
        return (pca_obj.transform(A).astype(np.float32)
                if A.shape[1] != ncomp else A.astype(np.float32))

    def _build_centroids(X_red, y, keep=None):
        cents = {}
        keep_set = set(keep) if keep is not None else None
        for lab in np.unique(y):
            if keep_set is not None and lab not in keep_set:
                continue
            idx = (y == lab)
            if np.any(idx):
                cents[lab] = X_red[idx].mean(axis=0)
        return cents

    def _assign_by_centroid(X_red, cents):
        labs = np.array(sorted(cents.keys()))
        C = np.vstack([cents[k] for k in labs])  # (k, d)
        d2 = ((X_red[:, None, :] - C[None, :, :])**2).sum(axis=2)
        return labs[d2.argmin(axis=1)]

    def _canon_key(x):
        if pd.isna(x): return "NA"
        try: return f"clone {int(x)}"
        except Exception: return str(x)

    # ---------- filtering by day only ----------
    if day_value is None:
        day_value = target_t

    _ = p['numerical_ts'][-1] / 200  # bookkeeping (kept as-is)

    if day_col not in df_cluster.columns:
        raise ValueError(f"'{day_col}' not found in df_cluster columns.")

    mask = (df_cluster[day_col] == day_value)
    if not mask.any():
        raise ValueError(f"No rows match day=={day_value}.")

    df_sub = df_cluster.loc[mask]

    # target data + predefined labels (must align by row)
    X_target = mats[target_t]
    if cluster_col not in df_sub.columns:
        raise ValueError(f"'{cluster_col}' not found in filtered df_cluster.")
    y_target = df_sub[cluster_col].to_numpy()

    X_target_red = _reduce_if_needed(X_target, pca)
    if len(y_target) != X_target_red.shape[0]:
        raise ValueError(
            f"Alignment mismatch: mats[target_t]={X_target_red.shape[0]} vs labels={len(y_target)}"
        )

    # selected clusters (optional)
    if selected_clusters is not None:
        selected_clusters = list(selected_clusters)
        present = set(y_target)
        missing = [c for c in selected_clusters if c not in present]
        if len(missing) == len(selected_clusters):
            raise ValueError(f"None of selected_clusters {selected_clusters} are present for the chosen day.")

    # centroids restricted to selected (if provided)
    centroids = _build_centroids(X_target_red, y_target, keep=selected_clusters)
    if not centroids:
        raise ValueError("No centroids built (after applying selected_clusters).")

    # color policy: strictly from cnv_lut
    if cnv_lut is None:
        raise ValueError("Please pass `cnv_lut=global_cnv_lut` to enforce consistent clone colors.")

    labels_we_might_draw = sorted(np.unique(list(centroids.keys())))
    missing_keys = [lab for lab in labels_we_might_draw if _canon_key(lab) not in cnv_lut]
    if missing_keys:
        msg = f"Missing colors in cnv_lut for labels: {missing_keys} (canonical: {[ _canon_key(l) for l in missing_keys ]})"
        if strict_colors:
            raise KeyError(msg)
        else:
            print("[warn]", msg)

    def _color(lab):
        return cnv_lut.get(_canon_key(lab), "#333333")  # only used if strict_colors=False and label missing

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(8, 6))

    # background (light gray)
    X2_vis = _reduce_if_needed(mats[target_t], pca)
    ax.scatter(X2_vis[:,0], X2_vis[:,1], color='lightgray', alpha=bg_alpha, s=10, zorder=5)

    X1_vis = _reduce_if_needed(mats[source_t], pca)
    ax.scatter(X1_vis[:,0], X1_vis[:,1], color='lightgray', alpha=bg_alpha, s=10, zorder=5)

    for t in intermediate_t:
        X_mid = _reduce_if_needed(mats[t], pca)
        ax.scatter(X_mid[:,0], X_mid[:,1], color='lightgray', alpha=bg_alpha, s=10, zorder=5)

    if index <= 0:
        index = 1

    # Precompute FINAL labels for coloring if requested
    X1_hat_last = np.asarray(X1_trpts[-1], dtype=np.float32)
    X1_hat_last_red = _reduce_if_needed(X1_hat_last, pca)
    labels_final = _assign_by_centroid(X1_hat_last_red, centroids)  # final fates for each cell

    sel_set = None if selected_clusters is None else set(selected_clusters)
    prev_X = None
    iters = range(len(X1_trpts))
    if reverse:
        iters = reversed(iters)

    for i in iters:
        if i < start_i or (i % index) != 0:
            continue

        X_now = np.asarray(X1_trpts[i], dtype=np.float32)
        if np.isnan(X_now).any():
            continue

        X_now_red = _reduce_if_needed(X_now, pca)

        if color_mode == 'final':
            labels_i = labels_final
        elif color_mode == 'frame':
            labels_i = _assign_by_centroid(X_now_red, centroids)
        else:
            raise ValueError("color_mode must be 'final' or 'frame'.")

        if sel_set is None:
            # draw all in color
            for lab in np.unique(labels_i):
                idx = (labels_i == lab)
                ax.scatter(X_now_red[idx,0], X_now_red[idx,1],
                           c=_color(lab), alpha=traj_alpha, s=point_size, zorder=10)
                if prev_X is not None:
                    ax.plot(np.vstack([prev_X[idx,0], X_now_red[idx,0]]),
                            np.vstack([prev_X[idx,1], X_now_red[idx,1]]),
                            color=_color(lab), alpha=line_alpha, linewidth=line_width, zorder=9)
        else:
            sel_mask = np.isin(labels_i, list(sel_set))
            # selected in color
            if np.any(sel_mask):
                for lab in np.unique(labels_i[sel_mask]):
                    idx = (labels_i == lab)
                    ax.scatter(X_now_red[idx,0], X_now_red[idx,1],
                               c=_color(lab), alpha=traj_alpha, s=point_size, zorder=10)
                    if prev_X is not None:
                        ax.plot(np.vstack([prev_X[idx,0], X_now_red[idx,0]]),
                                np.vstack([prev_X[idx,1], X_now_red[idx,1]]),
                                color=_color(lab), alpha=line_alpha, linewidth=line_width, zorder=9)
            # unselected dim gray (unless hidden)
            if not hide_unselected:
                unsel_mask = ~sel_mask
                if np.any(unsel_mask):
                    ax.scatter(X_now_red[unsel_mask,0], X_now_red[unsel_mask,1],
                               color='lightgray', alpha=traj_gray_alpha, s=point_size, zorder=6)

        prev_X = X_now_red

    ax.set_xlabel("PC 1", fontsize=24)
    ax.set_ylabel("PC 2", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_title("")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Static cluster plot (target, selected clones) saved to {output_file}")
    plt.close(fig)

    # Return final-frame labels (useful downstream)
    return labels_final


def generate_static_cluster_plot_target_LARRY_benchmark_fate(
    pca,
    source_t, target_t, X1_trpts, mats, optimal_k, start_i, index, p,
    reverse=False, intermediate_t=[1,2,3], d_red=2, random_state=42,
    exp_memo='experiment', output_file=None,
    meta=None,                      # metadata DataFrame
    background_alpha=0.18,          # lighter background
    traj_alpha=0.90,                # alpha for Monocyte/Neutrophil trajectories
    other_traj_alpha=0.45,          # alpha for "Other" trajectories (more transparent)
    line_alpha_main=0.55,           # connector alpha for Monocyte/Neutrophil
    line_alpha_other=0.35,          # connector alpha for "Other"
    # --- NEW: optional day-2 highlighting (adds marks; no trajectory changes) ---
    day2_df=None,                   # DataFrame aligned BY ORDER to day-2 cells
    day2_fate_col="fate",           # column in day2_df to select who to mark
    highlight_fates=("Monocyte","Neutrophil"),
    cross_size=36, cross_lw=1.6, cross_alpha=0.95,
    cross_colors=None               # e.g. {"Monocyte":"#123b6d","Neutrophil":"#8b2e00"}
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if meta is None:
        raise ValueError("Please pass `meta` (with 'Developmental time point' and 'Cell type annotation').")

    def _col(df, name):
        candidates = {
            "Time point": [
                "Time point"
            ],
            "Cell type annotation": [
                "Cell type annotation","Cell type","cell_type","CellType","celltype","Annotation","annotation"
            ],
        }[name]
        canon = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
        for cand in candidates:
            k = "".join(ch for ch in cand.lower() if ch.isalnum())
            if k in canon:
                return canon[k]
        raise KeyError(f"Could not find column for {name} in meta.")

    dev_col = _col(meta, "Time point")
    ct_col  = _col(meta, "Cell type annotation")

    # (not otherwise used; kept for parity)
    dt = p['numerical_ts'][-1] / 200

    # last-day data -> 2D space
    last_day_gene     = mats[target_t]
    last_day_reduced  = pca.transform(last_day_gene).astype(np.float32)

    # meta mask for last day == 6
    dev_series = meta[dev_col].astype(str)
    dev_num    = pd.to_numeric(dev_series.str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    last_mask  = (dev_num == 6) if not dev_num.isna().all() else (dev_series == "6")

    meta_last = meta.loc[last_mask]
    n_meta_last = len(meta_last)
    n_last_day  = last_day_reduced.shape[0]
    if n_meta_last != n_last_day:
        n_min = min(n_meta_last, n_last_day)
        print(f"Warning: last-day meta rows ({n_meta_last}) != mats[{target_t}] rows ({n_last_day}). "
              f"Proceeding with first {n_min} rows by position.")
        last_day_reduced = last_day_reduced[:n_min, :]
        meta_last = meta_last.iloc[:n_min, :]

    # group mapping: 0=Monocyte, 1=Neutrophil, 2=Other
    ct_lower = meta_last[ct_col].astype(str).str.lower()
    def _map_group(x: str) -> int:
        if "monocyt" in x:   return 0
        if "neutro"  in x:   return 1
        return 2
    last_day_labels = np.array([_map_group(x) for x in ct_lower], dtype=int)

    present = np.unique(last_day_labels)
    centroids = {g: last_day_reduced[last_day_labels == g].mean(axis=0) for g in present}
    present_sorted = sorted(present.tolist())  # e.g., [0,1,2] if all present

    # label FINAL predicted state by nearest centroid
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    C = np.stack([centroids[g] for g in present], axis=0)
    dists = ((X1_hat_last[:, None, :] - C[None, :, :])**2).sum(axis=2)
    nearest_idx = dists.argmin(axis=1)
    map_back = np.array(present)
    X1_hat_labels = map_back[nearest_idx]

    # colors per group
    default_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}  # blue, orange, green
    subgroup_colors = {g: default_colors.get(g, 'gray') for g in present_sorted}

    # alpha and zorder per group
    alpha_point_by_group = {0: traj_alpha, 1: traj_alpha, 2: other_traj_alpha}
    alpha_line_by_group  = {0: line_alpha_main, 1: line_alpha_main, 2: line_alpha_other}
    zorder_by_group      = {0: 5, 1: 5, 2: 3}   # Monocyte/Neutrophil on top of Other

    # draw OTHER first, then Mono/Neutrophil
    group_plot_order = []
    if 2 in present_sorted: group_plot_order.append(2)
    group_plot_order += [g for g in present_sorted if g in (0,1)]

    # =================== plotting ===================
    fig, ax = plt.subplots(figsize=(8, 6))

    # faint gray backgrounds
    X2_vis = pca.transform(mats[target_t])
    ax.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray',
               alpha=background_alpha, s=10, zorder=1, label='Data')

    for i, X1_trpt in enumerate(X1_trpts):
        if i % index == 0 and i >= start_i:
            if np.isnan(X1_trpt).any():
                continue
            X1_hat_vis = X1_trpt  # 2D already

            # draw groups in the chosen order
            for g in group_plot_order:
                idx = (X1_hat_labels == g)
                ax.scatter(
                    X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                    c=subgroup_colors[g],
                    alpha=alpha_point_by_group[g],
                    s=5,
                    zorder=zorder_by_group[g],
                    label=f'Predicted Subtrajectory {g+1}' if i == start_i else None
                )
                if i > start_i:
                    prev_X1_hat_vis = X1_trpts[i - index]
                    prev_idx = (X1_hat_labels == g)
                    ax.plot(
                        [prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]],
                        [prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]],
                        color=subgroup_colors[g],
                        alpha=alpha_line_by_group[g],
                        linewidth=1,
                        zorder=zorder_by_group[g] - 1
                    )

    # source & intermediates in faint gray
    X1_vis = pca.transform(mats[source_t])
    ax.scatter(X1_vis[:, 0], X1_vis[:, 1],
               color='lightgray', alpha=background_alpha, s=10, zorder=1)
    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=background_alpha, s=10, zorder=1)

    # ====== NEW: overlay crosses at day-2 for selected cells, matched by ORDER ======
    if day2_df is not None:
        if day2_fate_col not in day2_df.columns:
            raise ValueError(f"`day2_df` must contain column '{day2_fate_col}'.")
        # align by order to day-2 (source) cells
        n_src = X1_vis.shape[0]
        f = day2_df[day2_fate_col].astype("string").fillna("Unknown")
        m = min(len(f), n_src)
        if len(f) != n_src:
            print(f"Note: day2_df rows ({len(f)}) != #source day cells ({n_src}); using first {m} rows by order.")
        fate_vals = f.iloc[:m].to_numpy()

        # default cross colors if none provided
        if cross_colors is None:
            cross_colors = {"Monocyte":"#123b6d","Neutrophil":"#8b2e00"}

        # draw crosses per fate name requested
        for fate_name in highlight_fates:
            mask = (fate_vals == fate_name)
            if m < n_src:  # pad to match length if needed
                pad = np.zeros(n_src - m, dtype=bool)
                mask = np.concatenate([mask, pad])
            if np.any(mask):
                ax.scatter(
                    X1_vis[mask, 0], X1_vis[mask, 1],
                    marker='x', s=cross_size, linewidths=cross_lw,
                    c=cross_colors.get(fate_name, "black"),
                    alpha=cross_alpha, zorder=8, label=None
                )
    # ====== end NEW ======

    ax.set_xlabel("PC 1", fontsize=24)
    ax.set_ylabel("PC 2", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_title("")

    if output_file is None:
        output_file = f"{exp_memo}_static_target.png"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static cluster plot saved to {output_file}")
    plt.close(fig)

    return X1_hat_labels


def generate_static_trajectory_plots_two_timepoints(pca,physical_dt,days, intermediate_days, X1_trpts, mats, d_red=26, output_file_with_snapshots=None, output_file_without_snapshots=None, output_file_snapshots_only=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """

    
    # Define color gradient for snapshots
    num_snapshots = len(X1_trpts)
    colormap = cm.viridis  # Can change to "plasma", "inferno", etc.
    snapshot_colors = [colormap(i / num_snapshots) for i in range(num_snapshots)]

    # Rescale time values for the color bar
    time_values = np.linspace(0, physical_dt * num_snapshots, num_snapshots)

    # Create a normalization object for the color mapping
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for color bar

   
    source_t = days[0]
  
    target_t = days[1]
    # target_t = 4
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        intermediate_days[0]: '#ff7f0e',  # Orange
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot source, intermediates, and target
    X1_vis = pca.transform(mats[source_t])
    #Xm_vis = pca.transform(mats[middle_t])
    X2_vis = pca.transform(mats[target_t])
    #ax1.scatter(X1_vis[:, 0], X1_vis[:, 1], facecolors='none', edgecolors=color_map[source_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {source_t}')
    #ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], facecolors='none', edgecolors=color_map[target_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {target_t}')


    # Plot intermediate time points
    for t in intermediate_days:
        X_intermediate_vis = pca.transform(mats[t])
        ax1.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], color=color_map[t], facecolors='none', edgecolors=color_map[t], linewidths=1.0, alpha=0.75, s=10, zorder = 20,  label=f'Day {t} (Test Data)')

    # Plot snapshots from X1_trpts with a color gradient
    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            continue
        X1_hat_vis = X1_trpt
        ax1.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], color=snapshot_colors[i], alpha=0.75, s=2, zorder = 1)

    # Add a small color bar inside the plot
    cax = ax1.inset_axes([1.02, 0.2, 0.03, 0.6])  # [x, y, width, height] (relative position)
    
    # Create the colorbar with increased size
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set manual tick positions
    cbar.set_ticks(np.linspace(0, 4, 5))  # Ensures ticks at 0, 1, 2, 3, 4
    
    # Optional: Explicitly set tick labels if needed
    cbar.set_ticklabels([0, 1, 2, 3, 4])  
    
    # Increase colorbar label font size
    cbar.set_label("Time", fontsize=20)  
    
    # Increase colorbar tick font size
    cbar.ax.tick_params(labelsize=20)

    # Adjust colorbar thickness
    #cbar.ax.set_aspect(20)  # Increase aspect ratio to make it thicker
   
    # Set labels and title
    ax1.set_xlabel("PC 1", fontsize = 20)
    ax1.set_ylabel("PC 2", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax1.legend(loc='upper right', fontsize= 24)
    ax1.set_title("")

    # Save or show the plot
    if output_file_with_snapshots:
        plt.savefig(output_file_with_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITH snapshots saved to {output_file_with_snapshots}")
        plt.close(fig1)
    else:
        plt.show()

    # **Plot 2: Without Snapshots**
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot only source, intermediates, and target
    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=8,  zorder = 15, label=f'Time {source_t} (Training Data)')
    #ax2.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=10,  zorder = 10, label=f'Time {middle_t}')
    ax2.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=8,  zorder = 10, label=f'Time {target_t} (Training Data)')

    # Set labels and title
    ax2.set_xlabel("PC 1", fontsize = 20)
    ax2.set_ylabel("PC 2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title("")

    # Save or show the plot
    if output_file_without_snapshots:
        plt.savefig(output_file_without_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITHOUT snapshots saved to {output_file_without_snapshots}")
        plt.close(fig2)
    else:
        plt.show()


## Static plot function for simple GPA (breast cancer cell line data)

def generate_static_trajectory_plots_two_timepoints_no_middle(pca,physical_dt,days, intermediate_days, X1_trpts, mats, d_red=26, output_file_with_snapshots=None, output_file_without_snapshots=None, output_file_snapshots_only=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """
    

    # Define color gradient for snapshots
    num_snapshots = len(X1_trpts)
    colormap = cm.viridis  # Can change to "plasma", "inferno", etc.
    snapshot_colors = [colormap(i / num_snapshots) for i in range(num_snapshots)]

    # Rescale time values for the color bar
    time_values = np.linspace(0, physical_dt * num_snapshots, num_snapshots)

    # Create a normalization object for the color mapping
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for color bar

    source_t, middle_t, target_t = days[0], days[1], days[-1]
    
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        #intermediate_days[0]: '#ff7f0e',  # Orange
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot source, intermediates, and target
    X1_vis = pca.transform(mats[source_t])
    #Xm_vis = pca.transform(mats[middle_t])
    X2_vis = pca.transform(mats[target_t])
    #ax1.scatter(X1_vis[:, 0], X1_vis[:, 1], facecolors='none', edgecolors=color_map[source_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {source_t}')
    #ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], facecolors='none', edgecolors=color_map[target_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {target_t}')



    # Plot snapshots from X1_trpts with a color gradient
    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            continue
        X1_hat_vis = X1_trpt
        ax1.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], color=snapshot_colors[i], alpha=0.75, s=2, zorder = 1)

    # sort & coerce to ints for discrete ticks like 2,3,4
    d0, d1 = sorted(map(int, days))

    # make sure the color scale matches the requested day range
    if hasattr(sm, "set_clim"):
        sm.set_clim(d0, d1)
    else:
        sm.norm.vmin, sm.norm.vmax = d0, d1

    # create the inset colorbar
    cax = ax1.inset_axes([1.02, 0.2, 0.03, 0.6])
    cbar = plt.colorbar(sm, cax=cax)

    # ticks from start to end (inclusive): e.g., [2, 3, 4]
    ticks = np.arange(d0, d1 + 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

    # styling
    cbar.set_label("Time", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    # Adjust colorbar thickness
    #cbar.ax.set_aspect(20)  # Increase aspect ratio to make it thicker
   
    # Set labels and title
    ax1.set_xlabel("PC 1", fontsize = 20)
    ax1.set_ylabel("PC 2", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax1.legend(loc='upper right', fontsize= 24)
    ax1.set_title("")

    # Save or show the plot
    if output_file_with_snapshots:
        plt.savefig(output_file_with_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITH snapshots saved to {output_file_with_snapshots}")
        plt.close(fig1)
    else:
        plt.show()

    # **Plot 2: Without Snapshots**
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot only source, intermediates, and target
    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=8,  zorder = 15, label=f'Time {source_t} (Training Data)')
    #ax2.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=10,  zorder = 10, label=f'Time {middle_t}')
    ax2.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=8,  zorder = 10, label=f'Time {target_t} (Training Data)')

    # Set labels and title
    ax2.set_xlabel("PC 1", fontsize = 20)
    ax2.set_ylabel("PC 2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title("")

    # Save or show the plot
    if output_file_without_snapshots:
        plt.savefig(output_file_without_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITHOUT snapshots saved to {output_file_without_snapshots}")
        plt.close(fig2)
    else:
        plt.show()


def classify_X2_hat(
    full_matrix,pca,source_t, target_t,X1_trpts,mats, optimal_k, start_i, index,p, reverse=True, intermediate_t=[1], 
    d_red=2, random_state=42, exp_memo='2', output_file = None,output_file_2 = None):
    
   

    dt = p['numerical_ts'][-1] / 200
   
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]

    intermediate_t = np.array(intermediate_t)
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)

    day1, day2 = source_t, target_t

    # Perform clustering analysis on the last day's cell states
    last_day = mats[day1]
    last_day_reduced = pca.transform(last_day).astype(np.float32)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    X1_hat_last = X1_trpts[0].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Generate colors for clusters
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    viridis_colors = default_colors[:optimal_k]

    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, viridis_colors)
    subgroup_colors_red = get_subgroup_colors(last_day_labels, viridis_colors)

    # Define filename paths
    direction = 'backward' if reverse else 'forward'
    img_src = output_file
    initial_img_src = output_file_2
    
    # Plot initial state for animation
    fig, ax = plt.subplots()
    ims = []

    reducer = decomposition.PCA(n_components=2, random_state=0)
    reducer.fit(full_matrix)
    vis_all_days = reducer.transform(full_matrix)
    # Prepare Data for Initial State
    X1_vis = reducer.transform(mats[day1])
    X2_vis = reducer.transform(mats[day2])

    # **(1) Save the Initial State Figure with Black Circle Outlines**
    fig_init, ax_init = plt.subplots(figsize=(8, 6))
    
    # Plot background gray cells
    ax_init.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=8.0, zorder=5)
    
    # Plot last day's clusters **with black outline**
    scatter = ax_init.scatter(X1_vis[:, 0], X1_vis[:, 1], 
                              c=[subgroup_colors_red[label] for label in last_day_labels], 
                              alpha=1.0, s=50, edgecolors='black', linewidth=1.5, zorder=8)
    
    # Axis Labels
    ax_init.set_xlabel("PC 1", fontsize=24)
    ax_init.set_ylabel("PC 2", fontsize=24)
    ax_init.tick_params(axis='both', which='major', labelsize=24)
    ax_init.set_title("", fontsize=16)
    
    # Save the static figure
    plt.savefig(initial_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    

    # **(2) Create a Separate Figure for the Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # Get unique labels
    unique_labels = np.unique(last_day_labels)
    
    # Define legend elements:
    legend_elements = []
    
    # (A) **Fate Labels (Bold Dots with Black Outlines)**
    for i, label in enumerate(unique_labels):
        legend_elements.append(
            mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='None', markersize=12, 
                          markeredgecolor='black', markeredgewidth=3.0, label=f"Ancestor {i+1}")
        )
    
    # (B) **Predicted Trajectories (One Dot with a Centered Horizontal Bar)**
    for i, label in enumerate(unique_labels):
        # Single dot with a horizontal bar
        trajectory_dot_bar = mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='-', 
                                           markersize=6, linewidth=2, alpha=1.0, label=f"Trajectory {i+1}")
    
        # Add to legend
        legend_elements.append(trajectory_dot_bar)
    
    # Create horizontal legend **with a frame**
    ax_legend.legend(
        handles=legend_elements,
        loc="center", fontsize=20, title="Cell Ancestors & Predicted Trajectories",
        title_fontsize=20, ncol=4, frameon=True, framealpha=1.0, edgecolor="black", handletextpad=1.0, columnspacing=1.0
    )
    
    # Save the legend figure
    legend_img_src = initial_img_src.replace(".png", "_legend.png")
    plt.savefig(legend_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    


    # Animation: Initial frame
    im = ax.scatter(X1_vis[:, 0], X1_vis[:, 1], 
                    c=[subgroup_colors_red[label] for label in last_day_labels], 
                    alpha=1.0, s=3.0, zorder=8)

    ttl = ax.text(0.5, 1.05, "t = %.3f" % (0), 
                  bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                  transform=ax.transAxes, ha="center")

    ims.append([im, ttl])

    # Animation: Trajectory updates
    indices = range(len(X1_trpts) - start_i)
    if reverse:
        indices = reversed(indices)

    for i in indices:
        if i % index == 0:
            X1_trpt = X1_trpts[i]
            if np.isnan(X1_trpt).any():
                break
            X1_hat = pca.inverse_transform(X1_trpt)
            X1_hat_vis = reducer.transform(X1_hat)

            im = ax.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], 
                            c=[subgroup_colors_blue[label] for label in X1_hat_labels], 
                            alpha=1.0, s=3.0, zorder=10)
            
            ax.scatter(X1_vis[:, 0], X1_vis[:, 1], 
                       c=[subgroup_colors_red[label] for label in last_day_labels], 
                       alpha=1.0, s=3.0, zorder=8)

            # Keep background cells in the animation
            ax.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=0.5, zorder=5)

            ttl = ax.text(0.5, 1.05, "t = %.3f" % (physical_dt * i), 
                          bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                          transform=ax.transAxes, ha="center")

            ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=200)
    writergif = animation.PillowWriter(fps=3)
    ani.save(img_src, writer=writergif)
    plt.clf()
    
    # Display saved animation
    display(Image(filename=img_src))

    print(f"Initial state figure (with background) saved at: {initial_img_src}")
    print(f"Animation saved at: {img_src}")


def classify_X1_hat(full_matrix,pca,source_t, target_t,X1_trpts,mats, optimal_k, start_i, index,p, reverse=True, intermediate_t=[1], 
    d_red=2, random_state=42, exp_memo='2', output_file = None,output_file_2 = None):
    

    dt = p['numerical_ts'][-1] / 200
    

    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]

    intermediate_t = np.array(intermediate_t)
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)

    day1, day2 = source_t, target_t

    # Perform clustering analysis on the last day's cell states
    last_day = mats[day2]
    last_day_reduced = pca.transform(last_day).astype(np.float32)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Generate colors for clusters
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    viridis_colors = default_colors[:optimal_k]

    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, viridis_colors)
    subgroup_colors_red = get_subgroup_colors(last_day_labels, viridis_colors)

    # Define filename paths
    direction = 'backward' if reverse else 'forward'
    img_src = output_file
    initial_img_src = output_file_2
    # img_src = f"{output_dir}{exp_memo}-movie-cluster-{optimal_k}-{direction}-trajectory.gif"
    # initial_img_src = os.path.join(output_dir, f"{exp_memo}_initial_state_with_background.png")  # NEW STATIC FIGURE

    # Plot initial state for animation
    fig, ax = plt.subplots()
    ims = []

    # Prepare Data for Initial State
    reducer = decomposition.PCA(n_components=2, random_state=0)
    reducer.fit(full_matrix)
    vis_all_days = reducer.transform(full_matrix)
    
    X1_vis = reducer.transform(mats[day1])
    X2_vis = reducer.transform(mats[day2])

    # **(1) Save the Initial State Figure with Black Circle Outlines**
    fig_init, ax_init = plt.subplots(figsize=(8, 6))
    
    # Plot background gray cells
    ax_init.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=8.0, zorder=5)
    
    # Plot last day's clusters **with black outline**
    scatter = ax_init.scatter(X2_vis[:, 0], X2_vis[:, 1], 
                              c=[subgroup_colors_red[label] for label in last_day_labels], 
                              alpha=1.0, s=50, edgecolors='black', linewidth=1.5, zorder=8)
    
    # Axis Labels
    ax_init.set_xlabel("PC 1", fontsize=24)
    ax_init.set_ylabel("PC 2", fontsize=24)
    ax_init.tick_params(axis='both', which='major', labelsize=24)
    ax_init.set_title("", fontsize=16)
    
    # Save the static figure
    plt.savefig(initial_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    

    # **(2) Create a Separate Figure for the Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # Get unique labels
    unique_labels = np.unique(last_day_labels)
    
    # Define legend elements:
    legend_elements = []
    
    # (A) **Fate Labels (Bold Dots with Black Outlines)**
    for i, label in enumerate(unique_labels):
        legend_elements.append(
            mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='None', markersize=12, 
                          markeredgecolor='black', markeredgewidth=3.0, label=f"Fate {i+1}")
        )
    
    # (B) **Predicted Trajectories (One Dot with a Centered Horizontal Bar)**
    for i, label in enumerate(unique_labels):
        # Single dot with a horizontal bar
        trajectory_dot_bar = mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='-', 
                                           markersize=6, linewidth=2, alpha=1.0, label=f"Trajectory {i+1}")
    
        # Add to legend
        legend_elements.append(trajectory_dot_bar)
    
    # Create horizontal legend **with a frame**
    ax_legend.legend(
        handles=legend_elements,
        loc="center", fontsize=20, title="Cell Fates & Predicted Trajectories",
        title_fontsize=20, ncol=4, frameon=True, framealpha=1.0, edgecolor="black", handletextpad=1.0, columnspacing=1.0
    )
    
    # Save the legend figure
    legend_img_src = initial_img_src.replace(".png", "_legend.png")
    plt.savefig(legend_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    


    # Animation: Initial frame
    im = ax.scatter(X2_vis[:, 0], X2_vis[:, 1], 
                    c=[subgroup_colors_red[label] for label in last_day_labels], 
                    alpha=1.0, s=3.0, zorder=8)

    ttl = ax.text(0.5, 1.05, "t = %.3f" % (0), 
                  bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                  transform=ax.transAxes, ha="center")

    ims.append([im, ttl])

    # Animation: Trajectory updates
    indices = range(len(X1_trpts) - start_i)
    if reverse:
        indices = reversed(indices)

    for i in indices:
        if i % index == 0:
            X1_trpt = X1_trpts[i]
            if np.isnan(X1_trpt).any():
                break
            X1_hat = pca.inverse_transform(X1_trpt)
            X1_hat_vis = reducer.transform(X1_hat)

            im = ax.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], 
                            c=[subgroup_colors_blue[label] for label in X1_hat_labels], 
                            alpha=1.0, s=3.0, zorder=10)
            
            ax.scatter(X2_vis[:, 0], X2_vis[:, 1], 
                       c=[subgroup_colors_red[label] for label in last_day_labels], 
                       alpha=1.0, s=3.0, zorder=8)

            # Keep background cells in the animation
            ax.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=0.5, zorder=5)

            ttl = ax.text(0.5, 1.05, "t = %.3f" % (physical_dt * i), 
                          bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                          transform=ax.transAxes, ha="center")

            ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=200)
    writergif = animation.PillowWriter(fps=3)
    ani.save(img_src, writer=writergif)
    plt.clf()
    
    # Display saved animation
    display(Image(filename=img_src))

    print(f"Initial state figure (with background) saved at: {initial_img_src}")
    print(f"Animation saved at: {img_src}")


def generate_static_cluster_plot_source(
    pca,
    source_t, target_t, X1_trpts, mats, optimal_k, start_i, index,p, reverse=False, intermediate_t=[1,2,3], 
    d_red=2, random_state=42, exp_memo='experiment', output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories.
    
    Parameters:
    - pca - a two dimensional pca
    - source_t (int): Source time step.
    - target_t (int): Target time step.
    - X1_trpts: list of cell positions over time generated by velocity field
    - Mats:  dictionary that groups gene expression data by timepoint.
    - optimal_k (int): Number of clusters.
    - start_i (int): Starting index for X1_trpts.
    - index (int): Step size for selecting snapshots.
    -p: velocity model parameters
    - reverse (bool): Whether to reverse trajectory direction.
    - intermediate_t (list): List of intermediate time points.
    - d_red (int): PCA dimension reduction.
    - random_state (int): Random seed for clustering.
    - exp_memo (str): Experiment identifier for file naming.
    - output_file: The output filepath for the graph
    """
    

    # Compute trajectory integration
    dt = p['numerical_ts'][-1] / 200
   
    # Perform clustering on last day's cell states
    last_day = mats[source_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    # Classify final predicted states
    X1_hat_last = X1_trpts[0].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Define colors for each cluster
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    subgroup_colors_blue = {label: default_colors[i] for i, label in enumerate(np.unique(X1_hat_labels))}
    subgroup_colors_red = {label: default_colors[i] for i, label in enumerate(np.unique(last_day_labels))}


    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot last day's clusters (target) in red subgroup colors
   # Plot last day's clusters (target) in gray as actual data
    X2_vis = pca.transform(mats[target_t])
    ax.scatter(X2_vis[:, 0], X2_vis[:, 1], facecolors='none', edgecolors='gray', linewidths=0.7, alpha=0.7, s=10, zorder=10, label='Data')


    # Plot transported states (X1_hat) using assigned cluster colors (predicted sub-trajectories)
    for i, X1_trpt in enumerate(X1_trpts):
        if i % index == 0 and i >= start_i:
            if np.isnan(X1_trpt).any():
                continue
            X1_hat_vis = X1_trpt
            for label in np.unique(X1_hat_labels):
                idx = (X1_hat_labels == label)
                ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           c=subgroup_colors_blue[label], alpha=0.75, s=3, zorder=1,
                           label=f'Predicted Subtrajectory {label+1}' if i == start_i else None) 
                

    # Plot source day
    X1_vis = pca.transform(mats[source_t])
    ax.scatter(X1_vis[:, 0], X1_vis[:, 1], facecolors='none', edgecolors='gray', linewidths=0.7,  alpha=0.7, s=10, zorder = 10, label=f'Source: Day {source_t}')

    # Plot intermediate time points
    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], facecolors='none', edgecolors='gray', linewidths=0.7,  alpha=0.7, s=10, zorder = 10, label=f'Intermediate: Day {t}')

    # Set labels, legend, and title
    ax.set_xlabel("PC 1", fontsize = 24)
    ax.set_ylabel("PC 2", fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=24)  # Increases tick font size
    ax.set_title("")

    # Save or show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static cluster plot saved to {output_file}")
    plt.close(fig)

    return X1_hat_labels


def generate_static_cluster_plot_target(pca,
    source_t, target_t, X1_trpts, mats, optimal_k, start_i, index,p, reverse=False, intermediate_t=[1,2,3], 
    d_red=2, random_state=42, exp_memo='experiment', output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories.
    
    Parameters:
    - pca - a two dimensional pca
    - source_t (int): Source time step.
    - target_t (int): Target time step.
    - X1_trpts: list of cell positions over time generated by velocity field
    - Mats:  dictionary that groups gene expression data by timepoint.
    - optimal_k (int): Number of clusters.
    - start_i (int): Starting index for X1_trpts.
    - index (int): Step size for selecting snapshots.
    -p: velocity model parameters
    - reverse (bool): Whether to reverse trajectory direction.
    - intermediate_t (list): List of intermediate time points.
    - d_red (int): PCA dimension reduction.
    - random_state (int): Random seed for clustering.
    - exp_memo (str): Experiment identifier for file naming.
    - output_file: The output filepath for the graph
    """
    
    

    # Compute trajectory integration
    dt = p['numerical_ts'][-1] / 200
    
 
    # Perform clustering on last day's cell states
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    # Classify final predicted states
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Define colors for each cluster
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    subgroup_colors_blue = {label: default_colors[i] for i, label in enumerate(np.unique(X1_hat_labels))}
    subgroup_colors_red = {label: default_colors[i] for i, label in enumerate(np.unique(last_day_labels))}

   
    

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot last day's clusters (target) in gray as actual data
    X2_vis = pca.transform(mats[target_t])
    ax.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=10, label='Data')
        
    # Plot transported states (X1_hat) using assigned cluster colors (predicted sub-trajectories)
    for i, X1_trpt in enumerate(X1_trpts):
        if i % index == 0 and i >= start_i:
            if np.isnan(X1_trpt).any():
                continue
            X1_hat_vis = X1_trpt
    
            for label in np.unique(X1_hat_labels):
                idx = (X1_hat_labels == label)
    
                # Scatter Plot: Individual Points
                ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           c=subgroup_colors_blue[label], alpha=0.75, s=3, zorder=1,
                           label=f'Predicted Subtrajectory {label+1}' if i == start_i else None)
    
                # **Line Plot: Connect Points from Previous Iteration**
                if i > start_i:  # Ensure there's a previous iteration
                    prev_X1_hat_vis = X1_trpts[i - index]  # Get previous iteration
                    prev_idx = (X1_hat_labels == label)
    
                    ax.plot([prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]],
                            [prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]],
                            color=subgroup_colors_blue[label], alpha=0.5, linewidth=1, zorder=0)

    
    # Plot source and intermediate days in gray circles (empty)
    X1_vis = pca.transform(mats[source_t])
    ax.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=10)
    
    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)
    
    # Set labels and title
    ax.set_xlabel("PC 1", fontsize = 24)
    ax.set_ylabel("PC 2", fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=24)  # Increases tick font size
    ax.set_title("")
    
    # Add custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='24')
    
    # Save or show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static cluster plot saved to {output_file}")
    plt.close(fig)


    return X1_hat_labels

def generate_static_trajectory_plots_three_timepoints(pca,physical_dt, days, intermediate_days, X1_trpts, mats, d_red=26, output_file_with_snapshots=None, output_file_without_snapshots=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """
    
    # Define color gradient for snapshots
    num_snapshots = len(X1_trpts)
    colormap = cm.viridis  # Can change to "plasma", "inferno", etc.
    snapshot_colors = [colormap(i / num_snapshots) for i in range(num_snapshots)]

    # Rescale time values for the color bar
    time_values = np.linspace(0, physical_dt * num_snapshots, num_snapshots)

    # Create a normalization object for the color mapping
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for color bar

    source_t, middle_t, target_t = days[0], days[1], days[-1]
    
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        intermediate_days[0]: '#2ca02c',  # Green
        middle_t: '#ff7f0e',  # Orange
        intermediate_days[1]: '#8c564b',  # Brown
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot source, intermediates, and target
    X1_vis = pca.transform(mats[source_t])
    Xm_vis = pca.transform(mats[middle_t])
    X2_vis = pca.transform(mats[target_t])
    ax1.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=12, zorder = 10, label=f'Time {source_t} (Training Data)')
    ax1.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=12, zorder = 10, label=f'Time {middle_t} (Training Data)')
    ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=12, zorder = 10, label=f'Time {target_t} (Training Data)')

    # Plot intermediate time points
    for t in intermediate_days:
        X_intermediate_vis = pca.transform(mats[t])
        ax1.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], color=color_map[t], facecolors='none', edgecolors=color_map[t], linewidths=1.2, alpha=1.0, s=15, zorder = 20,  label=f'Time {t} (Test Data)')

    # Plot snapshots from X1_trpts with a color gradient
    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            continue
        X1_hat_vis = X1_trpt
        ax1.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], color=snapshot_colors[i], alpha=0.75, s=5, zorder = 1)

    
    # Add a small color bar inside the plot
    cax = ax1.inset_axes([1.02, 0.2, 0.03, 0.6])  # [x, y, width, height] (relative position)
    
    # Create the colorbar with increased size
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set manual tick positions
    cbar.set_ticks(np.linspace(0, 4, 5))  # Ensures ticks at 0, 1, 2, 3, 4
    
    # Optional: Explicitly set tick labels if needed
    cbar.set_ticklabels([0, 1, 2, 3, 4])  
    
    # Increase colorbar label font size
    cbar.set_label("Time", fontsize=20)  
    
    # Increase colorbar tick font size
    cbar.ax.tick_params(labelsize=20)

    # Adjust colorbar thickness
    #cbar.ax.set_aspect(20)  # Increase aspect ratio to make it thicker
   
    # Set labels and title
    ax1.set_xlabel("PC 1", fontsize = 20)
    ax1.set_ylabel("PC 2", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax1.legend(loc='upper right', fontsize= 24)
    ax1.set_title("")

    # Save or show the plot
    if output_file_with_snapshots:
        plt.savefig(output_file_with_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITH snapshots saved to {output_file_with_snapshots}")
        plt.close(fig1)
    else:
        plt.show()

    # **Plot 2: Without Snapshots**
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=12, zorder = 10, label=f'Time {source_t} (Training Data)')
    ax2.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=12, zorder = 10, label=f'Time {middle_t} (Training Data)')
    ax2.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=12, zorder = 10, label=f'Time {target_t} (Training Data)')

    # Plot intermediate time points
    for t in intermediate_days:
        X_intermediate_vis = pca.transform(mats[t])
        ax2.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], color=color_map[t], facecolors='none', edgecolors=color_map[t], linewidths=1.2, alpha=1.0, s=15, zorder = 20,  label=f'Time {t} (Test Data)')

    # Set labels and title
    ax2.set_xlabel("PC 1", fontsize = 20)
    ax2.set_ylabel("PC 2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title("")

    # Save or show the plot
    if output_file_without_snapshots:
        plt.savefig(output_file_without_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITHOUT snapshots saved to {output_file_without_snapshots}")
        plt.close(fig2)
    else:
        plt.show()


    
    # Extract legend elements
    handles, labels = ax1.get_legend_handles_labels()
    
    # Extract numeric values from "Time X (Input Data)" and "Time X (Test Data)"
    time_labels = []
    for label in labels:
        try:
            time_value = int(label.split(" ")[1])  # Extract the numerical value after "Time"
            time_labels.append((time_value, label))  # Store (time, label) pairs
        except ValueError:
            time_labels.append((float('inf'), label))  # Place non-time labels at the end
    
    # Sort legend by time values
    time_labels.sort(key=lambda x: x[0])  # Sort by the extracted numeric value
    sorted_labels = [item[1] for item in time_labels]
    sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
    
    # **Increase marker size in legend**
    for handle in sorted_handles:
        if isinstance(handle, plt.Line2D):  # Ensure we're modifying scatter markers
            handle.set_markersize(30)  # Adjust marker size
    
    # Create a separate figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Adjust size as needed
    ax_legend.axis("off")  # Remove axes
        
    # Create legend with larger markers for scatter plots
    legend = ax_legend.legend(
        sorted_handles, sorted_labels, fontsize=24, loc='center',
        ncol=len(sorted_labels), markerscale=4  # Increase scatter marker size
    )
    
    # Save the legend separately
    # legend_path = os.path.join(result_dir, "legend_only.png")
    # fig_legend.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_legend)  # Close the legend figure
    

def gene_dynamics_whole_saveonly(full_matrix, pca,gene_names, source_t, target_t,X1_trpts,mats, optimal_k, gene_of_interest, index,p, max_i,
                              intermediate_t = [1], img_src = None, img_src_2 = None, img_src_3 = None):

    dt = p['numerical_ts'][-1]/200
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t
    X1_trpt = X1_trpts[-1]
    
    
    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day2]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    
    #X1_hat_last_reduced = pca.transform(X1_hat_last)

    X1_hat_last = X1_trpts[-1].astype(np.float32) 
    X1_hat_labels = kmeans.predict(X1_hat_last)
    
    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(last_day_labels)
    print(f"Number of unique labels in last_day_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(last_day_labels, red_colors)

    #mask = last_day_labels == 0
    
    # Visualization in the original space 
    
    fig, ax = plt.subplots()
    ims = []
    
    # Extract the gene index for the gene of interest
    # gene_index = full_matrix.columns.get_loc(gene_of_interest) - 1
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    for i in indices:
        if i % index == 0 and i <= max_i:
            X1_trpt = X1_trpts[i]
            if np.isnan(X1_trpt).any():
                break
            X1_hat = pca.inverse_transform(X1_trpt)
            #X1_hat_vis = reducer.transform(X1_hat)
            X1_hat_vis = pca.transform(X1_hat)
            vis_all_days = pca.transform(full_matrix)
            #mask_2 = X1_hat_labels == 0
            #X1_hat_vis = X1_hat_vis[mask_2]

            # Plot all points in X1_hat_vis with colormap based on precomputed normalized gene expression values
            gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
            all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update the list to exclude the used values
            im = ax.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], c=gene_expression_values, cmap='viridis', alpha=1.0, s=0.5, zorder=10, vmin=vmin, vmax=vmax)
            #ax.scatter(X1_intermediate_vis[:, 0], X1_intermediate_vis[:, 1], color=colors[t], alpha=0.5, s=0.5, zorder=5)
            ax.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=0.3, s=0.5, zorder=1)
    
            ttl = ax.text(0.5, 1.05, "t = %.3f" % (physical_dt*i), bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha="center")
            ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=200)
    writergif = animation.PillowWriter(fps=3)
    ani.save(img_src, writer=writergif)
    # Close the figure to free up memory
    plt.close(fig)

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot



    output_file = img_src_2
    
    # Plot averaged gene expressions with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Plot the averaged gene expressions as a line
    plt.plot(
        extended_indices,
        avg_gene_expressions,
        label='Average Gene Expression',
        color='green',
        linestyle='-',  # Use a line instead of dots
    )
    
    # Fill the confidence intervals
    plt.fill_between(
        extended_indices,
        np.array(avg_gene_expressions) - np.array(ci_gene_expressions),
        np.array(avg_gene_expressions) + np.array(ci_gene_expressions),
        alpha=0.2,
        color='lightgreen',
        label='95% CI'
    )
    
    # Ensure rescaled_indices and all_avg_expressions have the same length
    assert len(rescaled_indices) == len(all_avg_expressions), (
        f"Length mismatch: rescaled_indices ({len(rescaled_indices)}) != all_avg_expressions ({len(all_avg_expressions)})"
    )
    
    # Plot the intermediate and boundary time points
    plt.errorbar(
        rescaled_indices,  # Use rescaled indices for the x-axis
        all_avg_expressions,
        yerr=all_ci_expressions,
        fmt='o',
        color='blue',
        label='Discrete Points'
    )
    
    # Update the x-axis ticks and labels
    plt.xticks(
        ticks=rescaled_indices,  # Tick positions based on rescaled indices
        labels=combined_indices,  # Relabel using combined_indices
        rotation=0,  # Optional: Rotate labels for better visibility
        fontsize=10    # Adjust font size for readability
    )
    
    plt.xlabel('Time Point (Day)')
    plt.ylabel('Gene Expression')
    plt.title(f'Average {gene_of_interest} Expression Over Time')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save with high resolution
    
    # Close the figure to free up memory
    plt.close()



    # (2) Plot averaged gene expression and confidence intervals for subgroups at each time point based on X1_hat_labels
    # Perform KMeans clustering with the optimal number of clusters
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)
    
    # Initialize dictionaries to store subgroup averages and confidence intervals
    subgroup_avg_gene_expressions = {label: [] for label in np.unique(X1_hat_labels)}
    subgroup_ci_gene_expressions = {label: [] for label in np.unique(X1_hat_labels)}
    
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Compute averages and confidence intervals for subgroups
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]
    
        # Compute subgroup-specific averages and confidence intervals
        for label in np.unique(X1_hat_labels):
            subgroup_values = gene_expression_values[X1_hat_labels == label]
            subgroup_avg_gene_expressions[label].append(np.mean(subgroup_values))
            ci = stats.sem(subgroup_values) * stats.t.ppf((1 + 0.95) / 2., len(subgroup_values) - 1)
            subgroup_ci_gene_expressions[label].append(ci)

    
    # Define the filename for saving the subgroup plot
    subgroup_output_file = img_src_3


    # Plot subgroup averages and confidence intervals
    plt.figure(figsize=(10, 6))
    for label in np.unique(X1_hat_labels):
        plt.plot(
            extended_indices,
            subgroup_avg_gene_expressions[label],
            label=f'Subgroup {label} Average',
            linestyle='-'
        )
        plt.fill_between(
            extended_indices,
            np.array(subgroup_avg_gene_expressions[label]) - np.array(subgroup_ci_gene_expressions[label]),
            np.array(subgroup_avg_gene_expressions[label]) + np.array(subgroup_ci_gene_expressions[label]),
            alpha=0.2,
            label=f'Subgroup {label} 95% CI'
        )
    
    
    # Update the x-axis ticks and labels
    plt.xticks(
        ticks=rescaled_indices,  # Tick positions based on rescaled indices
        labels=combined_indices,  # Relabel using combined_indices
        rotation=0,  # Optional: Rotate labels for better visibility
        fontsize=10  # Adjust font size for readability
    )
    
    plt.xlabel('Time Point (Day)')
    plt.ylabel('Gene Expression')
    plt.title(f'Average {gene_of_interest} Expression Over Time by Subgroup')
    plt.legend()
    
    # Save the subgroup plot
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close()
    
    print(f"Subgroup plot saved at: {subgroup_output_file}")

def Average_gene_dynamics_whole_saveonly(pca, gene_names, source_t, target_t,X1_trpts,mats, gene_of_interest, index, p, max_i,
                              intermediate_t = [1], img_src = None):

    
    
    dt = p['numerical_ts'][-1]/200
    
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t

    X1_trpt = X1_trpts[-1]
    

    # Create a color mapping for the specific indices
   

    
    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day1]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    

    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    

    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot



    output_file = img_src
    
    # Plot averaged gene expressions with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Plot the averaged gene expressions as a line
    plt.plot(
        extended_indices,
        avg_gene_expressions,
        label='Average Gene Expression',
        color='orange',
        linestyle='-',  # Use a line instead of dots
    )
    
    # Fill the confidence intervals
    plt.fill_between(
        extended_indices,
        np.array(avg_gene_expressions) - np.array(ci_gene_expressions),
        np.array(avg_gene_expressions) + np.array(ci_gene_expressions),
        alpha=0.2,
        color='lightsalmon',
        label='95% CI'
    )
    
    # Ensure rescaled_indices and all_avg_expressions have the same length
    assert len(rescaled_indices) == len(all_avg_expressions), (
        f"Length mismatch: rescaled_indices ({len(rescaled_indices)}) != all_avg_expressions ({len(all_avg_expressions)})"
    )
    
    # Plot the intermediate and boundary time points
    plt.errorbar(
        rescaled_indices,  # Use rescaled indices for the x-axis
        all_avg_expressions,
        yerr=all_ci_expressions,
        fmt='o',
        color='blue',
        label='Discrete Points'
    )
    
    # Update the x-axis ticks and labels
    plt.xticks(
        ticks=rescaled_indices,  # Tick positions based on rescaled indices
        labels=combined_indices,  # Relabel using combined_indices
        rotation=0,  # Optional: Rotate labels for better visibility
        fontsize=32    # Adjust font size for readability
    )

    plt.yticks(fontsize=32)  # Increase y-axis tick font size

    
    plt.xlabel('Time', fontsize=32)
    plt.ylabel('Gene Expression', fontsize=32)
    plt.title(f'Average {gene_of_interest} Dynamics', fontsize=32)
    #plt.legend(fontsize=16)  # Adjust font size as needed
    
    # Save the plot to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save with high resolution
    
    # Close the figure to free up memory
    plt.close()




    # **Save Separate Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))
    ax_legend.axis("off")

    legend_elements = [
        mlines.Line2D([], [], color='orange', linestyle='-', linewidth=3, label='Average Gene Dynamics'),
        mpatches.Patch(color='lightsalmon', alpha=0.7, label='95% Confidence Interval'),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='-', markersize=8, linewidth=2, label='Data Points')
    ]

    ax_legend.legend(handles=legend_elements, loc="center", fontsize=32, title="", title_fontsize=32, 
                     frameon=True, ncol=len(legend_elements), handletextpad=2, columnspacing=2)

    legend_output_file = output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Subgroup trajectory plot saved at: {output_file}")
    print(f"Legend plot saved separately at: {legend_output_file}")    

## This is for EMT data (Three time points: Time [0, 2, 4])
## Subtrajectroies with_violin_plot

def Average_gene_dynamics_whole_saveonly_with_violin_plot_sample1_EMT(pca, gene_names, source_t, target_t,X1_trpts,mats, gene_of_interest, index, p, max_i,
                              intermediate_t = [1], img_src = None, cluster_save_path = "X2_hat_clusters.csv"):

    dt = p['numerical_ts'][-1]/200
   
    
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t
        

    X1_trpt = X1_trpts[-1]
    
    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day1]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    

    
    # Load previously saved cluster labels
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)

    #mask = last_day_labels == 0
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    


    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot


    # (1) Perform clustering on the last day's cell states from `mats`
    last_day = mats[day1]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering

    
    # Define colors for subgroups
    subgroup_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subgroup_colors[i % len(subgroup_colors)] for i, label in enumerate(unique_labels)}
    
    # Define filename
    subgroup_output_file = img_src
    
    
    # (2) Initialize Storage for Mean and CI
    subgroup_avg_gene_expressions = {label: [] for label in unique_labels}
    subgroup_ci_gene_expressions = {label: [] for label in unique_labels}
    
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized.copy()
    
    # (3) Compute Mean & Confidence Intervals
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:  # Apply truncation
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract gene expression values
        X1_hat = pca.inverse_transform(X1_trpt)
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]
    
        # Compute subgroup averages & CI
        for label in unique_labels:
            mask = (X1_hat_labels == label)  # Use labels **only from step 1**
            subgroup_values = np.array(gene_expression_values)[mask]
    
            if len(subgroup_values) > 0:
                subgroup_avg_gene_expressions[label].append(np.mean(subgroup_values))
                ci = stats.sem(subgroup_values) * stats.t.ppf((1 + 0.95) / 2., len(subgroup_values) - 1)
                subgroup_ci_gene_expressions[label].append(ci)
            else:
                subgroup_avg_gene_expressions[label].append(np.nan)
                subgroup_ci_gene_expressions[label].append(np.nan)
    

    
    # (4) **Plot**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # **Get x-axis positions for the line plot (scale to [0, 4])**
    num_points = len(next(iter(subgroup_avg_gene_expressions.values())))  # Number of time points
    x_positions = np.linspace(0, 4, num_points)  # Ensure correct x-spacing for trajectories
    
    # **Plot Predicted Trajectories & Confidence Intervals**
    subgroup_legend_handles = []  # Store for separate legend


    # **Plot Subgroup Averages & Confidence Intervals**
    for i, label in enumerate(unique_labels):
        # **Plot the Mean Trajectory Line**
        line, = ax1.plot(
            x_positions, subgroup_avg_gene_expressions[label], zorder=10,
            linestyle='-', color=subgroup_color_map[label], linewidth=2,
            label=f'Predicted Trajectory {i+1}'
        )
    
        # **Plot the Confidence Interval (Shaded Region)**
        ax1.fill_between(
            x_positions,
            np.array(subgroup_avg_gene_expressions[label]) - np.array(subgroup_ci_gene_expressions[label]),
            np.array(subgroup_avg_gene_expressions[label]) + np.array(subgroup_ci_gene_expressions[label]),
            alpha=0.2, zorder=5, color=subgroup_color_map[label],
            label=f'95% CI of Trajectory {i+1}'
        )
    
        # **Legend entry for Mean + Confidence Interval**
        ci_patch = mpatches.Patch(
            color=subgroup_color_map[label], alpha=0.2, label=f'95% CI of Trajectory {i+1}'
        )
    
        # **Store in Legend Handles**
        subgroup_legend_handles.append(line)
        subgroup_legend_handles.append(ci_patch)

        
    # (5) **Ensure Violin Plots are at `[0, 2, 4]`**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    # **Manually set violin plot positions to `[0, 2, 4]`**
    violin_x_positions = np.array([0, 2, 4])  # Explicitly define positions
    violin_colors = ["black", "gray", "black", "gray", "black"]  # Set distinct colors
    
    # 🎻 **Plot Violin Plots One-by-One to Force Correct Positioning**
    for i, (x_pos, data, color) in enumerate(zip(violin_x_positions, violin_data, violin_colors)):
        violin_parts = sns.violinplot(
            data=[data],  # Must be wrapped in a list to avoid merging violins
            ax=ax1,
            inner=None,
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=color,  # ✅ Assign distinct colors
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  # Move to correct x-location
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  # ✅ Extend range
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 2, 4])  # ✅ Force labels at `[0, 2, 4]`
    ax1.set_xticklabels([0, 2, 4], fontsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    
    ax1.set_xlabel('Day', fontsize=35)
    ax1.set_ylabel('Gene Expression', fontsize=35)
    ax1.set_title(f'Subtrajectory {gene_of_interest} Expression', fontsize=35)
    
    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data"),
        mpatches.Patch(color="gray", label="Test Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (VERTICAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(8, 2))  # Tall aspect ratio for vertical layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = subgroup_legend_handles + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=18, title="",
        title_fontsize=18, ncol=6, frameon=True, handletextpad=1.5, columnspacing=1.5
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Subgroup trajectory plot saved at: {subgroup_output_file}")
    print(f"Legend plot saved separately at: {legend_output_file}")

## save the gene expression dynamics png as pdf for EMT data



def create_pdf_from_gene_images(png_files, gene_list, pdf_path, images_per_page=25, grid_size=(5, 5)):
    """
    Create a PDF with gene expression PNG images arranged in a grid layout while preserving original resolution.

    Parameters:
        output_dir (str): Directory containing the PNG files.
        exp_memo (str): Base name used in the PNG filenames.
        gene_list (list): List of genes corresponding to the PNG files.
        pdf_path (str): Path to save the output PDF file.
        images_per_page (int): Number of images per page (default: 25).
        grid_size (tuple): Grid size (rows, cols) for each page (default: 5x5).
    """

    # Generate list of PNG file paths
    # png_files = [
    #     f"{output_dir}{exp_memo}_subtrajectories_violin_plots_{gene}.png" for gene in gene_list
    # ]

    # Check if all PNG files exist
    missing_files = [file for file in png_files if not os.path.exists(file)]
    if missing_files:
        print(f"Warning: The following files are missing and will be skipped:\n{missing_files}")

    # Filter out missing files
    png_files = [file for file in png_files if os.path.exists(file)]

    # Calculate the total number of pages
    total_pages = ceil(len(png_files) / images_per_page)

    # Create the PDF
    with PdfPages(pdf_path) as pdf:
        for page in range(total_pages):
            # Create a figure with dynamically sized subplots
            fig, axes = plt.subplots(*grid_size, figsize=(15, 15))  # Increased size for better resolution
            axes = axes.flatten()

            # Plot images for the current page
            start_idx = page * images_per_page
            end_idx = start_idx + images_per_page

            for i, ax in enumerate(axes):
                img_idx = start_idx + i
                if img_idx < len(png_files):
                    img = plt.imread(png_files[img_idx])
                    ax.imshow(img, aspect='auto')  # Preserve aspect ratio
                    ax.axis('off')  # Remove axes
                    # Add filename as the title
                    gene_name = gene_list[img_idx]
                    ax.set_title('', fontsize=8)
                else:
                    ax.axis('off')  # Hide empty axes

            # Save the page to the PDF with high resolution
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory

    print(f"✅ PDF saved to {pdf_path} with original image resolution.")

## Dynamics of p-values and fold change arross subtrajectories (with csv files and visulization results) 

def Compute_and_Plot_FoldChange_MeanDiff_PValues(pca,gene_names,X1_trpts, gene_of_interest, index,p, max_i, subtraj_dir = "Transport_genes/assets", cluster_save_path = None):
 
   
    dt = p['numerical_ts'][-1] / 200

    # Load previously saved cluster labels
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")

    # Extract gene expression values
    gene_index = list(gene_names).index(gene_of_interest)


    all_gene_expression_values_normalized_X1 = np.concatenate(
        [pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i]
    )

    # Initialize lists for storing results
    fold_change_results = []
    mean_diff_results = []
    p_value_results = []

    indices = range(0, len(X1_trpts), index)
    eps = 1e-6  # Small constant to prevent division by zero

    # Compute mean difference, fold-change, and p-values over time
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            continue

        X1_hat = pca.inverse_transform(X1_trpt)
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]

        for label1 in unique_labels:
            for label2 in unique_labels:
                if label1 >= label2:
                    continue

                mask1 = (X1_hat_labels == label1)
                mask2 = (X1_hat_labels == label2)

                expr_values1 = np.array(gene_expression_values)[mask1]
                expr_values2 = np.array(gene_expression_values)[mask2]

                if len(expr_values1) > 1 and len(expr_values2) > 1:
                    mean1, mean2 = np.mean(expr_values1), np.mean(expr_values2)
                    
                    # Compute Mean Difference
                    mean_diff = mean1 - mean2
                    
                    # Compute Fold Change
                    if mean1 <= 0 or mean2 <= 0 or np.isnan(mean1) or np.isnan(mean2):
                        log2_fc = np.nan
                    else:
                        log2_fc = np.log2(mean1 / mean2)

                    # Compute p-value
                    t_stat, p_val = stats.ttest_ind(expr_values1, expr_values2, equal_var=False, nan_policy='omit')
                else:
                    mean_diff = np.nan
                    log2_fc = np.nan
                    p_val = np.nan

                fold_change_results.append({
                    "Time": time_idx,
                    "Cluster 1": label1,
                    "Cluster 2": label2,
                    "Log2 Fold Change": log2_fc
                })

                mean_diff_results.append({
                    "Time": time_idx,
                    "Cluster 1": label1,
                    "Cluster 2": label2,
                    "Mean Difference": mean_diff
                })

                p_value_results.append({
                    "Time": time_idx,
                    "Cluster 1": label1,
                    "Cluster 2": label2,
                    "p-value": p_val
                })

    # Convert results to DataFrame
    df_fc = pd.DataFrame(fold_change_results)
    df_md = pd.DataFrame(mean_diff_results)
    df_pval = pd.DataFrame(p_value_results)

    # Save as CSV files
    # output_dir = os.path.join(result_dir, "output", exp_memo)
    # os.makedirs(output_dir, exist_ok=True)

    # Function to propagate NaNs forward **only if the last finite value was negative**
    def propagate_nans_for_negative_fc(df, column="Log2 Fold Change"):
        """Once NaN appears after a negative `column` value, all subsequent values become NaN."""
        df = df.copy()
        log2_fc_values = df[column].values  # Extract column values as an array
    
        # Identify first NaN index
        nan_mask = np.isnan(log2_fc_values)
        if nan_mask.any():
            first_nan_idx = np.where(nan_mask)[0][0]  # Find first NaN index
    
            # Check the last valid value before NaN
            last_valid_idx = first_nan_idx - 1 if first_nan_idx > 0 else None
            if last_valid_idx is not None and log2_fc_values[last_valid_idx] < 0:
                # If last valid log2 fold change was negative, set all subsequent values to NaN
                log2_fc_values[first_nan_idx:] = np.nan
    
        df[column] = log2_fc_values  # Update the DataFrame
        return df
    
    
    # **Create and Save the Fold Change CSV with NaN Propagation for Negative Values**
    df_fc_nan_propagated = df_fc.groupby(["Cluster 1", "Cluster 2"]).apply(propagate_nans_for_negative_fc)
    df_fc_nan_propagated.to_csv(os.path.join(subtraj_dir, f"fold_change_nan_propagated_{gene_of_interest}.csv"), index=False)
    
    print(f"Saved fold change CSV with NaN propagation for negative values: fold_change_nan_propagated_{gene_of_interest}.csv")


    df_fc.to_csv(os.path.join(subtraj_dir, f"fold_change_{gene_of_interest}.csv"), index=False)
    df_md.to_csv(os.path.join(subtraj_dir, f"mean_difference_{gene_of_interest}.csv"), index=False)
    df_pval.to_csv(os.path.join(subtraj_dir, f"p_values_{gene_of_interest}.csv"), index=False)

 

    # -------- Visualization --------

    ## **(1) Line Plot for Mean Difference (Each Cluster Pair)**
    for (cluster1, cluster2), group in df_md.groupby(["Cluster 1", "Cluster 2"]):
        plt.figure(figsize=(8, 5))
        plt.plot(group["Time"], group["Mean Difference"], label=f"Clusters {cluster1} vs {cluster2}", marker='o', linestyle='-')
        plt.axhline(y=0, color="black", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Mean Difference")
        plt.title(f"Mean Difference Over Time: {gene_of_interest}\nCluster {cluster1} vs {cluster2}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(subtraj_dir, f"mean_difference_cluster_{cluster1}_vs_{cluster2}.png"), dpi=300)
        plt.close()

    ## **(2) Line Plot for Fold-Change (All Cluster Pairs)**
    plt.figure(figsize=(10, 6))
    for (cluster1, cluster2), group in df_fc.groupby(["Cluster 1", "Cluster 2"]):
        plt.plot(group["Time"], group["Log2 Fold Change"], label=f"Clusters {cluster1} vs {cluster2}", marker='o')

    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Log2 Fold Change")
    plt.title(f"Log2 Fold Change Over Time - {gene_of_interest}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(subtraj_dir, f"fold_change_all_clusters_{gene_of_interest}.png"), dpi=300)
    plt.close()

    ## **(3) Line Plot of p-values (All Cluster Pairs)**
    plt.figure(figsize=(10, 6))
    for (cluster1, cluster2), group in df_pval.groupby(["Cluster 1", "Cluster 2"]):
        plt.plot(group["Time"], group["p-value"], label=f"Clusters {cluster1} vs {cluster2}", marker='o')

    plt.axhline(y=0.05, color="r", linestyle="--", label="Significance Threshold (p=0.05)")
    plt.yscale("log")
    plt.ylim(1e-6, 1)
    plt.xlabel("Time")
    plt.ylabel("p-value (log scale)")
    plt.title(f"P-values for Differential Expression - {gene_of_interest}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(subtraj_dir, f"p_values_all_clusters_{gene_of_interest}.png"), dpi=300)
    plt.close()


def difference_of_means_emt(gene_names, subtraj_dir):
# Define paths
    genes_of_interest = gene_names  # List of all genes

    # Load all CSVs for fold change, mean difference, and p-values
    df_fc_list, df_md_list, df_pval_list = [], [], []

    for gene in genes_of_interest:
        fc_file_path = os.path.join(subtraj_dir, f"fold_change_nan_propagated_{gene}.csv")
        md_file_path = os.path.join(subtraj_dir, f"mean_difference_{gene}.csv")
        pval_file_path = os.path.join(subtraj_dir, f"p_values_{gene}.csv")

        if os.path.exists(fc_file_path):
            df_fc = pd.read_csv(fc_file_path)
            df_fc["Gene"] = gene  
            df_fc_list.append(df_fc)

        if os.path.exists(md_file_path):
            df_md = pd.read_csv(md_file_path)
            df_md["Gene"] = gene  
            df_md_list.append(df_md)

        if os.path.exists(pval_file_path):
            df_pval = pd.read_csv(pval_file_path)
            df_pval["Gene"] = gene  
            df_pval_list.append(df_pval)

    # Merge all DataFrames
    df_fc_all = pd.concat(df_fc_list, ignore_index=True)
    df_md_all = pd.concat(df_md_list, ignore_index=True)
    df_pval_all = pd.concat(df_pval_list, ignore_index=True)

    # Merge p-values into fold change and mean difference DataFrames
    df_fc_all = df_fc_all.merge(df_pval_all, on=["Time", "Cluster 1", "Cluster 2", "Gene"], how="left")
    df_md_all = df_md_all.merge(df_pval_all, on=["Time", "Cluster 1", "Cluster 2", "Gene"], how="left")

    # **Rescale Time by dividing by 50**
    df_fc_all["Time"] = df_fc_all["Time"] / 50
    df_md_all["Time"] = df_md_all["Time"] / 50

    # Define thresholds
    pos_threshold_fc = 5.0  
    neg_threshold_fc = -5.0  
    pos_threshold_md = 0.11  
    neg_threshold_md = -0.15  
    p_value_threshold = 1e-3  # Significance threshold

    # **Identify genes passing p-value threshold**
    significant_genes = df_pval_all.groupby("Gene")["p-value"].apply(lambda x: x.min(skipna=True) < p_value_threshold)
    significant_genes = significant_genes[significant_genes].index  # Only genes passing p-value

    # **Filter fold-change and mean-difference genes, but retain all passing p-value**
    def filter_significant_genes(df, metric_col, threshold_high, threshold_low):
        """Find genes that exceed thresholds (colored) and others that stay gray (but pass p-value)."""
        gene_criteria = df.groupby("Gene")[metric_col].apply(lambda x: ((x > threshold_high) | (x < threshold_low)).any())
        
        highlighted_genes = gene_criteria[gene_criteria].index  # Genes exceeding threshold
        retained_genes = significant_genes.intersection(gene_criteria.index)  # Only keep genes passing p-value
        
        return retained_genes, highlighted_genes  # Return both full set and colored ones

    # **Apply filtering**
    all_genes_fc, highlighted_genes_fc = filter_significant_genes(df_fc_all, "Log2 Fold Change", pos_threshold_fc, neg_threshold_fc)
    all_genes_md, highlighted_genes_md = filter_significant_genes(df_md_all, "Mean Difference", pos_threshold_md, neg_threshold_md)

    # Compute per-gene normalization
    df_md_all["Normalized Mean Difference"] = df_md_all.groupby("Gene")["Mean Difference"].transform(lambda x: x / x.abs().max())

    # Identify genes with significant normalized mean difference changes
    gene_change_norm = df_md_all.groupby("Gene")["Normalized Mean Difference"].apply(lambda x: x.max() - x.min())
    all_genes_norm = significant_genes.intersection(gene_change_norm.index)
    highlighted_genes_norm = gene_change_norm[gene_change_norm > 0.45].index

    # Define colormaps
    cmap1 = plt.colormaps["tab20b"]
    cmap2 = plt.colormaps["tab20c"]
    cmap3 = plt.colormaps["Set1"]
    cmap4 = plt.colormaps["Set3"]
    cmap5 = plt.colormaps["Paired"]

    # Generate color lists
    color_list = (
        [cmap3(i) for i in range(min(9, len(cmap3.colors)))] +  
        [cmap4(i) for i in range(min(12, len(cmap4.colors)))] +  
        [cmap1(i) for i in range(20)] +
        [cmap2(i) for i in range(20)] +
        [cmap5(i) for i in range(min(12, len(cmap5.colors)))] +  
        list(plt.cm.hsv(np.linspace(0, 1, 15)))  
    )

    # Assign colors
    all_highlighted_genes = sorted(set(highlighted_genes_fc).union(set(highlighted_genes_md)).union(set(highlighted_genes_norm)))
    gene_colors = {gene: color_list[i % len(color_list)] for i, gene in enumerate(all_highlighted_genes)}

    # Group by cluster pairs and plot
    for (cluster1, cluster2), group_fc in df_fc_all.groupby(["Cluster 1", "Cluster 2"]):

        # **(1) Fold Change Plot**
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_handles = []

        for gene in all_genes_fc:
            sub_group = group_fc[group_fc["Gene"] == gene]
            if gene in highlighted_genes_fc:
                color = gene_colors.get(gene)
                alpha_value, linestyle = 1.0, "-"
            else:
                color = "gray"
                alpha_value, linestyle = 0.2, "--"

            line, = ax.plot(sub_group["Time"], sub_group["Log2 Fold Change"], marker="o", markersize=4, linestyle=linestyle, color=color, alpha=alpha_value, label=gene)
            if gene in highlighted_genes_fc:
                legend_handles.append(line)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Log2 Fold Change")
        ax.set_title(f"Fold Change Over Time (Cluster {cluster1} vs {cluster2})")

        if legend_handles:
            ax.legend(handles=legend_handles, title="Significant Genes", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(subtraj_dir, f"fold_change_comparison_cluster_{cluster1}_vs_{cluster2}.png"), dpi=300)
        plt.close()

        # **(2) Mean Difference Plot**
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_handles = []
        group_md = df_md_all[(df_md_all["Cluster 1"] == cluster1) & (df_md_all["Cluster 2"] == cluster2)]

        for gene in all_genes_md:
            sub_group = group_md[group_md["Gene"] == gene]
            if gene in highlighted_genes_md:
                color = gene_colors.get(gene)
                alpha_value, linestyle = 1.0, "-"
            else:
                color = "gray"
                alpha_value, linestyle = 0.2, "--"

            line, = ax.plot(sub_group["Time"], sub_group["Mean Difference"], marker="o", markersize=4, linestyle=linestyle, color=color, alpha=alpha_value, label=gene)
            if gene in highlighted_genes_md:
                legend_handles.append(line)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Difference")
        ax.set_title(f"Mean Difference Over Time (Cluster {cluster1} vs {cluster2})")

        if legend_handles:
            ax.legend(handles=legend_handles, title="Significant Genes", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(subtraj_dir, f"mean_difference_comparison_cluster_{cluster1}_vs_{cluster2}.png"), dpi=300)
        plt.close()




        
        # **(3) Normalized Mean Difference Plot**
        fig, ax = plt.subplots(figsize=(10, 8))
        legend_handles = []
        
        # **Filter data for the given cluster pair**
        group_md = df_md_all[(df_md_all["Cluster 1"] == cluster1) & (df_md_all["Cluster 2"] == cluster2)]
        
        # **Only keep genes passing the p-value threshold**
        genes_to_plot = significant_genes.intersection(group_md["Gene"].unique())
        
        # **Get Last Time Point Values**
        last_time_values = group_md[group_md["Time"] == group_md["Time"].max()].set_index("Gene")["Normalized Mean Difference"]
        
        # **Categorize Highlighted Genes into Four Groups Based on Their Last Time Point Values**
        group_green = [gene for gene in highlighted_genes_norm if gene in last_time_values and 1.0 >= last_time_values[gene] > 0.8]
        group_red = [gene for gene in highlighted_genes_norm if gene in last_time_values and 0.8 >= last_time_values[gene] > -0.3]
        group_blue = [gene for gene in highlighted_genes_norm if gene in last_time_values and -0.3 >= last_time_values[gene] > -0.8]
        group_orange = [gene for gene in highlighted_genes_norm if gene in last_time_values and -0.8 >= last_time_values[gene] >= -1.0]
        
        # **Define function to get only bright colors from a colormap (skip dark colors)**
        def get_bright_colormap(cmap_name, num_colors):
            cmap = plt.get_cmap(cmap_name)
            if num_colors == 1:
                return [cmap(0.75)]  # Single color case, select a bright shade
            return [cmap(0.5 + (i / (2 * (num_colors - 1)))) for i in range(num_colors)]  # Use only upper 50% of colormap
        
        # **Generate Bright Color Maps for Each Group**
        colors_green = get_bright_colormap("Greens", max(len(group_green), 1))
        colors_blue = get_bright_colormap("Purples", max(len(group_blue), 1))
        colors_red = get_bright_colormap("Reds", max(len(group_red), 1))
        colors_orange = get_bright_colormap("Oranges", max(len(group_orange), 1))
        
        # **Assign Colors to Highlighted Genes Based on Group**
        color_mapped_genes = {}
        
        for i, gene in enumerate(group_green):
            color_mapped_genes[gene] = colors_green[i]
        for i, gene in enumerate(group_blue):
            color_mapped_genes[gene] = colors_blue[i]
        for i, gene in enumerate(group_red):
            color_mapped_genes[gene] = colors_red[i]
        for i, gene in enumerate(group_orange):
            color_mapped_genes[gene] = colors_orange[i]
        
        # **Plot Data (Exclude group_green and group_red)**
        for gene in genes_to_plot:  # **Only plot genes that satisfy p-value threshold**
            if gene in group_green or gene in group_orange:  # Skip plotting genes in group_green and group_red
                continue  
        
            sub_group = group_md[group_md["Gene"] == gene]
        
            if gene in highlighted_genes_norm:  # **Highlight genes that satisfy both p-value & change criteria**
                color = color_mapped_genes.get(gene, "black")
                alpha_value, linestyle = 1.0, "-"
                line, = ax.plot(sub_group["Time"], sub_group["Normalized Mean Difference"], 
                                marker="o", markersize=4, linestyle=linestyle, 
                                color=color, alpha=alpha_value, label=gene)
                legend_handles.append((line, gene))  # Save for grouped legend
            else:  # **Gray for genes that satisfy p-value but not change threshold**
                color = "gray"
                alpha_value, linestyle = 0.2, "--"
                ax.plot(sub_group["Time"], sub_group["Normalized Mean Difference"], 
                        marker="o", markersize=4, linestyle=linestyle, 
                        color=color, alpha=alpha_value, label=gene)

        # **Labeling and Formatting**
        ax.set_xlabel("Day", fontsize=30)
        ax.set_ylabel("Normalized Mean Difference", fontsize=30)
        ax.set_title(f"Trajectory {cluster1+1} vs {cluster2+1}", fontsize=30)
        ax.tick_params(axis="both", labelsize=30)
        
        # **Save Main Plot Without Legend**
        plot_output_path = os.path.join(subtraj_dir, f"normalized_mean_difference_cluster_{cluster1}_vs_{cluster2}.png")
        plt.tight_layout()
        plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✅ Normalized Mean Difference plot saved: {plot_output_path}")
        
        # **Create and Save Separate Legend**
        if legend_handles:
            fig_legend, ax_legend = plt.subplots(figsize=(10, 10))  # Wider figure to fit grouped layout
            ax_legend.axis("off")
        
            # **Group Legend by Color**
            grouped_legend_handles = []
            
            for gene_list, title in zip(
                #[group_green, group_blue, group_red, group_orange],
                [group_red, group_blue],
                ["", "", "", ""]  # No explicit labels, but groups remain visually separate
            ):
                if gene_list:
                    handles = [mpatches.Patch(color=color_mapped_genes[gene], label=gene) for gene in gene_list]
                    grouped_legend_handles.append(handles)
        
            # **Flatten List of Legends**
            flattened_handles = [item for sublist in grouped_legend_handles for item in sublist]
        
            # **Plot the Grouped Legend**
            ax_legend.legend(
                handles=flattened_handles, 
                title="Significant Genes", 
                loc="center", 
                fontsize=26, 
                title_fontsize=26, 
                frameon=True, 
                ncol=min(len(flattened_handles), 2)  # Adjust to avoid overflow
            )
        
            # **Save Legend as Separate PNG**
            legend_output_path = os.path.join(subtraj_dir, f"legend_normalized_mean_difference_cluster_{cluster1}_vs_{cluster2}.png")
            plt.savefig(legend_output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"✅ Legend saved separately: {legend_output_path}")


## Subtrajectroies defined by source
def Average_gene_dynamics_whole_saveonly_single_trajectory_EMT(pca,gene_names,source_t, target_t,X1_trpts,mats, gene_of_interest, index,p, max_i,
                              intermediate_t = [1], subgroup_output_file = "assets/Transport_genes", cluster_save_path = None):

   
    
    dt = p['numerical_ts'][-1]/200
    
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t


    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]

    # Step 1: Perform clustering analysis on the last day's cell states from mats
    
    # Load previously saved cluster labels
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot


 
    # (1) **Assign Labels for Subgroups Based on Step 1**

    
    # Define **subtrajectory colors** (for cell trajectories)
    #subtrajectory_colors = ['red', 'blue', 'brown']
    subtrajectory_colors = ['green']
    
    # Define **violin plot colors** for the three time points
    violin_colors = ["black", "gray", "black"]  # Green, Orange, Purple
    
    # Map each subgroup label to a **trajectory color** and shift labels from 0,1 → 1,2
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subtrajectory_colors[i % len(subtrajectory_colors)] for i, label in enumerate(unique_labels)}
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Define filename for saving
    
    # (2) **Create Figure**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # (3) **Ensure Proper x-axis Scaling**
    num_points = len(indices)
    x_positions = np.linspace(0, 4, num_points)  # Scale to match `[0, 2, 4]`
    
    # (4) **Extract Cell Trajectories for Each Gene**
    cell_trajectories = {cell_idx: [] for cell_idx in range(X1_trpts[0].shape[0])}
    
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract **expression values of the gene of interest** from each cell at this time point
        gene_expression_values = pca.inverse_transform(X1_trpt)[:, gene_index]
    
        # Append the expression value at this time to each cell’s trajectory
        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)
    
    # (5) **Plot Individual Trajectories per Subgroup**
    legend_patches = []  # Store legend handles
    for label in unique_labels:
        first_plotted = False  # Track if we added a legend entry for this subgroup
        
        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue  # Ensure trajectories align with time points
    
            if X1_hat_labels[cell_idx] == label:  # Match subgroup label from step 1
                ax1.plot(
                    x_positions, traj,  
                    color=subgroup_color_map[label],  # ✅ Use the **subtrajectory colors**
                    alpha=0.1, linewidth=0.8  
                )
                
                # Add a single legend entry for each subgroup (renaming from 0,1 → 1,2)
                if not first_plotted:
                    legend_patches.append(mpatches.Patch(color=subgroup_color_map[label], label=f'Trajectory {label_mapping[label]}'))
                    first_plotted = True
    
    # (6) **Ensure Violin Plots are at `[0, 2, 4]` & Appear in Front**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    violin_x_positions = np.array([0, 2, 4])  # Ensure correct positions
    
    # 🎻 **Plot Violin Plots with Correct Colors and Transparency**
    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        violin_parts = sns.violinplot(
            data=[data],  
            ax=ax1,
            inner=None,  # ✅ REMOVE QUARTILE LINES
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=violin_colors[i],  # ✅ Assign correct color
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0,2, 4])  
    ax1.set_xticklabels([0, 2, 4], fontsize=32)
    ax1.tick_params(axis='y', labelsize=32)
    
    ax1.set_xlabel('Day', fontsize=32)
    ax1.set_ylabel('Gene Expression', fontsize=32)
    ax1.set_title(f'Single Cell {gene_of_interest} Expression Dynamics', fontsize=32)


    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

    # 🎨 **Redefine `legend_patches` to Include a Green Bar**
    legend_patches = [
        mlines.Line2D([], [], color="green", linestyle="-", linewidth=3, 
                      label="Gene dynamics of each single cell")
    ]

    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data"),
        mpatches.Patch(color="gray", label="Test Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
## Distribution of single genes comparions (Intermediate time points only) - EMT data

def Compare_Distribution_Trajectories_Intermediate_EMT(pca, gene_names, source_t, target_t,X1_trpts,mats, gene_of_interest, p, intermediate_t = [1], output_file = None):

    dt = p['numerical_ts'][-1] / 200
   
    # Extract gene index
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Test data distributions
    kde_test_data = [
        pca.inverse_transform(pca.transform(mats[t]))[:, gene_index] for t in intermediate_t
    ]

    # Correct snapshot extraction based on scaling
    snapshots_per_day = len(X1_trpts) / (target_t - source_t)
    scaled_intermediate_indices = [int(day * snapshots_per_day) for day in intermediate_t]

    kde_predicted_data = []
    for idx in scaled_intermediate_indices:
        if idx >= len(X1_trpts):
            idx = len(X1_trpts) - 1
        gene_expr_predicted = pca.inverse_transform(X1_trpts[idx])[:, gene_index]
        kde_predicted_data.append(gene_expr_predicted)

    # Visualization setup
    num_plots = len(intermediate_t)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)

    if num_plots == 1:
        axes = [axes]

    #test_data_colors = ["#2ca02c", "#8c564b"] #Sample 3
    #predicted_colors = ["#1b6420", "#5c3930"] #Sample 3

    test_data_colors = ["#f58231", "#911eb4", "#3cb44b"]  #Sample 1
    predicted_colors = ["#D2691E", "#800080", "#228B22"]  #Sample 1

    

        

    # Initialize list to store legend handles per intermediate time
    legend_patches_list = []
    
    # Generate KDE plots
    for i, (ax, t, test_vals, pred_vals) in enumerate(zip(axes, intermediate_t, kde_test_data, kde_predicted_data)):
    
        all_vals = np.concatenate([test_vals, pred_vals])
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        x_margin = (x_max - x_min) * 0.2
        x_range = np.linspace(x_min - x_margin, x_max + x_margin, 300)
    
        kde_test = gaussian_kde(test_vals)
        kde_pred = gaussian_kde(pred_vals)
    
        test_density = kde_test(x_range)
        pred_density = kde_pred(x_range)
    
        y_max = max(test_density.max(), pred_density.max()) * 2
    
        ax.fill_between(x_range, test_density, color=test_data_colors[i % len(test_data_colors)], alpha=0.5)
        ax.plot(x_range, test_density, color=test_data_colors[i % len(test_data_colors)], linewidth=2)
    
        ax.fill_between(x_range, pred_density, color=predicted_colors[i % len(predicted_colors)], alpha=0.5)
        ax.plot(x_range, pred_density, color=predicted_colors[i % len(predicted_colors)], linestyle="dashed", linewidth=2)
    
        ax.set_title(f"Day {t}", fontsize=26)
        ax.set_xlabel("Gene Expression", fontsize=26)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Density", fontsize=26)
        ax.tick_params(axis='both', which='major', labelsize=26)
    
        plt.suptitle(f"KDE for {gene_of_interest}", fontsize=26)
    

        # **Legend Entry for This Time Point**
        legend_patches_list.append([
            # **Test Data: Dashed Line**
            mlines.Line2D([], [], color=test_data_colors[i % len(test_data_colors)], linestyle="solid", linewidth=3,
                          label=f"Test Data day {t}"),
            
            # **Predicted Data: Solid Line**
            mlines.Line2D([], [], color=predicted_colors[i % len(predicted_colors)], linestyle="dashed", linewidth=3,
                          label=f"Predicted day {t}")
        ])
        
    # **Save KDE plot (without legend)**
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"KDE plot saved: {output_file}")
    
    # **(2) Create a Separate Figure for the Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(len(intermediate_t) * 3, 2))  # Adjust width dynamically
    ax_legend.axis("off")  # Hide axes
    
    # **Flatten legend handles into a single row-style list**
    flattened_legend_patches = []
    for group in legend_patches_list:
        for entry in group:
            flattened_legend_patches.append(entry)
    
    # **Create a Row-Style Legend with Box + Line for Each Entry**
    ax_legend.legend(
        handles=flattened_legend_patches,
        loc="center",
        fontsize=22,
        ncol=2,  # Ensures (Test, Predicted) pairs stay together
        frameon=True,
        handletextpad=1.5,
        columnspacing=2
    )
    
    # **Save the separate legend**
    legend_output_file = output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Legend plot saved separately at: {legend_output_file}")

#### the start of the breast cancer functions
#####

## For clinical data to identify cells with most change


def plot_X1_hat_displacement_distribution(X1_trpts, csv_output,plot_output, hist_output_path, exp_memo='2'):
   

    # Extract first and last time point
    X1_hat_first = X1_trpts[0].astype(np.float32)
    X1_hat_last = X1_trpts[-1].astype(np.float32)

    # Compute Euclidean distances for each cell
    displacements = np.linalg.norm(X1_hat_last - X1_hat_first, axis=1)

    # Compute summary statistics
    mean_disp = np.mean(displacements)
    std_disp = np.std(displacements)
    iqr_disp = np.percentile(displacements, 75) - np.percentile(displacements, 25)
    cv_disp = std_disp / mean_disp if mean_disp > 0 else np.nan

    hist_counts, _ = np.histogram(displacements, bins=40)
    hist_probs = hist_counts / hist_counts.sum()
    entropy_disp = entropy(hist_probs)

    # Save stats to CSV
    stats_df = pd.DataFrame([{
        "exp_memo": exp_memo,
        "mean_displacement": mean_disp,
        "std_displacement": std_disp,
        "iqr_displacement": iqr_disp,
        "cv_displacement": cv_disp,
        "entropy": entropy_disp
    }])

    #stats_output_path = f"{output_dir}{exp_memo}_X1_hat_displacement_stats.csv"
    stats_output_path = csv_output
    stats_df.to_csv(stats_output_path, index=False)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(displacements, bins=40, color="skyblue", edgecolor="black")
    plt.xlabel("Displacement (Euclidean Distance)", fontsize=16)
    plt.ylabel("Number of Cells", fontsize=16)
    plt.title("Distribution of Cell Displacements\nX1_hat First vs Last Time Point", fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    #output_path = f"{output_dir}{exp_memo}_X1_hat_displacement_distribution.png"
    output_path = plot_output
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"📊 Displacement plot saved at: {output_path}")
    print(f"📄 Stats CSV saved at: {stats_output_path}")

    # Save histogram data
    hist_counts, bin_edges = np.histogram(displacements, bins=40)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    hist_df = pd.DataFrame({
        "bin_center": bin_centers,
        "count": hist_counts
    })
    
    #hist_output_path = f"{output_dir}{exp_memo}_X1_hat_displacement_histogram.csv"
    hist_df.to_csv(hist_output_path, index=False)

def generate_static_cluster_plot_deviation_colormap_Robin_PT2(pca,
    source_t, target_t, start_i,X1_trpts,mats, index,intermediate_t=[1, 2, 3],output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories with gradient coloring.
    Also generates a separate legend figure and individual plots per subgroup.
    """


    # Step 1: Reduce real data and predictions to PCA space
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_first = X1_trpts[0].astype(np.float32)
    displacements = np.linalg.norm(X1_hat_last - X1_hat_first, axis=1)

    # 862, 3 and 5 (ER)
    # 887 0.8 and 1.3 (ER)
    # BMC 1.0 and 3.0 (ER)
    # Rinath 2.89 and 2.90 (ER)

    # 862, 1.3 and 2 (R)
    # 887 0.8 and 1.2 (R)
    # BMC 1.2 and 2.4 (R)
    # Rinath 3.2 and 3.8 (R)

    ## R genes
    #862 local minima x-values: [1.288 2.365]
    #887 local minima x-values: [0.699 1.149]
    #BMC local minima x-values: [1.145 1.937]
    #In vitro local minima x-values: [3.017 3.853]


    # Assign labels based on displacement
    X1_hat_labels = np.full(displacements.shape, 'low', dtype=object)
    X1_hat_labels[(displacements > 6) & (displacements <= 8)] = 'medium'
    X1_hat_labels[displacements > 8] = 'high'

    # Define colormaps per subgroup (avoid lightest tones by clipping range)
    label_to_cmap = {
        'low': colormaps['Oranges'],
        'medium': colormaps['Purples'],
        'high': colormaps['Greens']
    }
    cmap_clip = slice(75, 256)
    color_range = np.linspace(0, 1, 256)[cmap_clip]  # consistent use

    # Set file path for main plot
    #output_file = f"{result_dir}{exp_memo}_static_celltypes_plot_deviation_colormap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    X2_vis = pca.transform(mats[target_t])
    X1_vis = pca.transform(mats[source_t])

    total_steps = len([i for i in range(len(X1_trpts)) if i % index == 0 and i >= start_i])

    for label in np.unique(X1_hat_labels):
        cmap = label_to_cmap[label]
        idx = (X1_hat_labels == label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1], color=color, alpha=0.9, s=3, zorder=1)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2, zorder=0)

    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)

    ax.set_xlabel("PC 1", fontsize=20)
    ax.set_ylabel("PC 2", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title("")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Static cluster plot saved to {output_file}")

    # --- Save colormap legend with thinner bars side by side ---
    fig_legend, axs = plt.subplots(1, len(label_to_cmap), figsize=(20, 1.5))
    if len(label_to_cmap) == 1:
        axs = [axs]

    for ax, (label, base_cmap) in zip(axs, label_to_cmap.items()):
        # Create a new clipped colormap
        clipped_cmap = base_cmap(np.linspace(0, 1, 256)[cmap_clip])
        custom_cmap = plt.matplotlib.colors.ListedColormap(clipped_cmap)
    
        # Use that in the legend
        gradient = np.linspace(0, 1, cmap_clip.stop - cmap_clip.start).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 0, 0.03])
        ax.set_title(f"{label.capitalize()} phenotypic shift \nPre-treatment → Post-treatment", fontsize=20)
        ax.axis('off')


    legend_path = output_file.replace('.png', '_legend.png')
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to {legend_path}")

    # --- Additional per-label plots ---
    for current_label in np.unique(X1_hat_labels):
        fig_lbl, ax_lbl = plt.subplots(figsize=(8, 6))
        cmap = label_to_cmap[current_label]
        idx = (X1_hat_labels == current_label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax_lbl.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           color=color, alpha=0.8, s=3, zorder=2)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax_lbl.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2)

        ax_lbl.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)
        ax_lbl.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)

        ax_lbl.set_xlabel("PC 1", fontsize=20)
        ax_lbl.set_ylabel("PC 2", fontsize=20)
        ax_lbl.tick_params(axis='both', labelsize=20)
        ax_lbl.set_title("", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output_file.replace('.png', f'_label_{current_label}.png')}", dpi=300, bbox_inches='tight')
        plt.close(fig_lbl)

    return X1_hat_labels



def generate_static_cluster_plot_deviation_colormap_MCF7(pca,
    source_t, target_t, start_i,X1_trpts,mats, index,intermediate_t=[1, 2, 3],output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories with gradient coloring.
    Also generates a separate legend figure and individual plots per subgroup.
    """


    # Step 1: Reduce real data and predictions to PCA space
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_first = X1_trpts[0].astype(np.float32)
    displacements = np.linalg.norm(X1_hat_last - X1_hat_first, axis=1)

    # 862, 3 and 5 (ER)
    # 887 0.8 and 1.3 (ER)
    # BMC 1.0 and 3.0 (ER)
    # Rinath 2.89 and 2.90 (ER)

    # 862, 1.3 and 2 (R)
    # 887 0.8 and 1.2 (R)
    # BMC 1.2 and 2.4 (R)
    # Rinath 3.2 and 3.8 (R)

    ## R genes
    #862 local minima x-values: [1.288 2.365]
    #887 local minima x-values: [0.699 1.149]
    #BMC local minima x-values: [1.145 1.937]
    #In vitro local minima x-values: [3.017 3.853]


    # Assign labels based on displacement
    X1_hat_labels = np.full(displacements.shape, 'low', dtype=object)
    X1_hat_labels[(displacements > 3.017) & (displacements <= 3.853)] = 'medium'
    X1_hat_labels[displacements > 3.853] = 'high'

    # Define colormaps per subgroup (avoid lightest tones by clipping range)
    label_to_cmap = {
        'low': colormaps['Oranges'],
        'medium': colormaps['Purples'],
        'high': colormaps['Greens']
    }
    cmap_clip = slice(75, 256)
    color_range = np.linspace(0, 1, 256)[cmap_clip]  # consistent use

    # Set file path for main plot
    #output_file = f"{result_dir}{exp_memo}_static_celltypes_plot_deviation_colormap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    X2_vis = pca.transform(mats[target_t])
    X1_vis = pca.transform(mats[source_t])

    total_steps = len([i for i in range(len(X1_trpts)) if i % index == 0 and i >= start_i])

    for label in np.unique(X1_hat_labels):
        cmap = label_to_cmap[label]
        idx = (X1_hat_labels == label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1], color=color, alpha=0.9, s=3, zorder=1)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2, zorder=0)

    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)

    ax.set_xlabel("PC 1", fontsize=32)
    ax.set_ylabel("PC 2", fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_title("")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Static cluster plot saved to {output_file}")

    # --- Save colormap legend with thinner bars side by side ---
    fig_legend, axs = plt.subplots(1, len(label_to_cmap), figsize=(20, 1.5))
    if len(label_to_cmap) == 1:
        axs = [axs]

    for ax, (label, base_cmap) in zip(axs, label_to_cmap.items()):
        # Create a new clipped colormap
        clipped_cmap = base_cmap(np.linspace(0, 1, 256)[cmap_clip])
        custom_cmap = plt.matplotlib.colors.ListedColormap(clipped_cmap)
    
        # Use that in the legend
        gradient = np.linspace(0, 1, cmap_clip.stop - cmap_clip.start).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 0, 0.03])
        ax.set_title(f"{label.capitalize()} phenotypic shift \nPre-treatment → Post-treatment", fontsize=20)
        ax.axis('off')


    legend_path = output_file.replace('.png', '_legend.png')
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to {legend_path}")

    # --- Additional per-label plots ---
    for current_label in np.unique(X1_hat_labels):
        fig_lbl, ax_lbl = plt.subplots(figsize=(8, 6))
        cmap = label_to_cmap[current_label]
        idx = (X1_hat_labels == current_label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax_lbl.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           color=color, alpha=0.8, s=3, zorder=2)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax_lbl.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2)

        ax_lbl.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)
        ax_lbl.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)

        ax_lbl.set_xlabel("PC 1", fontsize=32)
        ax_lbl.set_ylabel("PC 2", fontsize=32)
        ax_lbl.tick_params(axis='both', labelsize=32)
        ax_lbl.set_title("", fontsize=32)
        plt.tight_layout()
        plt.savefig(f"{output_file.replace('.png', f'_label_{current_label}.png')}", dpi=300, bbox_inches='tight')
        plt.close(fig_lbl)

    return X1_hat_labels


def generate_static_cluster_plot_deviation_colormap_PA3(pca,
    source_t, target_t, start_i,X1_trpts,mats, index,intermediate_t=[1, 2, 3],output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories with gradient coloring.
    Also generates a separate legend figure and individual plots per subgroup.
    """


    # Step 1: Reduce real data and predictions to PCA space
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_first = X1_trpts[0].astype(np.float32)
    displacements = np.linalg.norm(X1_hat_last - X1_hat_first, axis=1)

    # 862, 3 and 5 (ER)
    # 887 0.8 and 1.3 (ER)
    # BMC 1.0 and 3.0 (ER)
    # Rinath 2.89 and 2.90 (ER)

    # 862, 1.3 and 2 (R)
    # 887 0.8 and 1.2 (R)
    # BMC 1.2 and 2.4 (R)
    # Rinath 3.2 and 3.8 (R)

    ## R genes
    #862 local minima x-values: [1.288 2.365]
    #887 local minima x-values: [0.699 1.149]
    #BMC local minima x-values: [1.145 1.937]
    #In vitro local minima x-values: [3.017 3.853]


    # Assign labels based on displacement
    X1_hat_labels = np.full(displacements.shape, 'low', dtype=object)
    X1_hat_labels[(displacements > 1.145) & (displacements <= 1.937)] = 'medium'
    X1_hat_labels[displacements > 1.937] = 'high'

    # Define colormaps per subgroup (avoid lightest tones by clipping range)
    label_to_cmap = {
        'low': colormaps['Oranges'],
        'medium': colormaps['Purples'],
        'high': colormaps['Greens']
    }
    cmap_clip = slice(75, 256)
    color_range = np.linspace(0, 1, 256)[cmap_clip]  # consistent use

    # Set file path for main plot
    #output_file = f"{result_dir}{exp_memo}_static_celltypes_plot_deviation_colormap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    X2_vis = pca.transform(mats[target_t])
    X1_vis = pca.transform(mats[source_t])

    total_steps = len([i for i in range(len(X1_trpts)) if i % index == 0 and i >= start_i])

    for label in np.unique(X1_hat_labels):
        cmap = label_to_cmap[label]
        idx = (X1_hat_labels == label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1], color=color, alpha=0.9, s=3, zorder=1)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2, zorder=0)

    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)

    ax.set_xlabel("PC 1", fontsize=32)
    ax.set_ylabel("PC 2", fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_title("")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Static cluster plot saved to {output_file}")

    # --- Save colormap legend with thinner bars side by side ---
    fig_legend, axs = plt.subplots(1, len(label_to_cmap), figsize=(20, 1.5))
    if len(label_to_cmap) == 1:
        axs = [axs]

    for ax, (label, base_cmap) in zip(axs, label_to_cmap.items()):
        # Create a new clipped colormap
        clipped_cmap = base_cmap(np.linspace(0, 1, 256)[cmap_clip])
        custom_cmap = plt.matplotlib.colors.ListedColormap(clipped_cmap)
    
        # Use that in the legend
        gradient = np.linspace(0, 1, cmap_clip.stop - cmap_clip.start).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 0, 0.03])
        ax.set_title(f"{label.capitalize()} phenotypic shift \nPre-treatment → Post-treatment", fontsize=20)
        ax.axis('off')


    legend_path = output_file.replace('.png', '_legend.png')
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to {legend_path}")

    # --- Additional per-label plots ---
    for current_label in np.unique(X1_hat_labels):
        fig_lbl, ax_lbl = plt.subplots(figsize=(8, 6))
        cmap = label_to_cmap[current_label]
        idx = (X1_hat_labels == current_label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax_lbl.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           color=color, alpha=0.8, s=3, zorder=2)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax_lbl.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2)

        ax_lbl.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)
        ax_lbl.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)

        ax_lbl.set_xlabel("PC 1", fontsize=32)
        ax_lbl.set_ylabel("PC 2", fontsize=32)
        ax_lbl.tick_params(axis='both', labelsize=32)
        ax_lbl.set_title("", fontsize=32)
        plt.tight_layout()
        plt.savefig(f"{output_file.replace('.png', f'_label_{current_label}.png')}", dpi=300, bbox_inches='tight')
        plt.close(fig_lbl)

    return X1_hat_labels


def generate_static_cluster_plot_deviation_colormap_862(pca,
    source_t, target_t, start_i,X1_trpts,mats, index,intermediate_t=[1, 2, 3],output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories with gradient coloring.
    Also generates a separate legend figure and individual plots per subgroup.
    """


    # Step 1: Reduce real data and predictions to PCA space
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_first = X1_trpts[0].astype(np.float32)
    displacements = np.linalg.norm(X1_hat_last - X1_hat_first, axis=1)

    # 862, 3 and 5 (ER)
    # 887 0.8 and 1.3 (ER)
    # BMC 1.0 and 3.0 (ER)
    # Rinath 2.89 and 2.90 (ER)

    # 862, 1.3 and 2 (R)
    # 887 0.8 and 1.2 (R)
    # BMC 1.2 and 2.4 (R)
    # Rinath 3.2 and 3.8 (R)

    ## R genes
    #862 local minima x-values: [1.288 2.365]
    #887 local minima x-values: [0.699 1.149]
    #BMC local minima x-values: [1.145 1.937]
    #In vitro local minima x-values: [3.017 3.853]


    # Assign labels based on displacement
    X1_hat_labels = np.full(displacements.shape, 'low', dtype=object)
    X1_hat_labels[(displacements > 1.288) & (displacements <= 2.365)] = 'medium'
    X1_hat_labels[displacements > 2.365] = 'high'

    # Define colormaps per subgroup (avoid lightest tones by clipping range)
    label_to_cmap = {
        'low': colormaps['Oranges'],
        'medium': colormaps['Purples'],
        'high': colormaps['Greens']
    }
    cmap_clip = slice(75, 256)
    color_range = np.linspace(0, 1, 256)[cmap_clip]  # consistent use

    # Set file path for main plot
    #output_file = f"{result_dir}{exp_memo}_static_celltypes_plot_deviation_colormap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    X2_vis = pca.transform(mats[target_t])
    X1_vis = pca.transform(mats[source_t])

    total_steps = len([i for i in range(len(X1_trpts)) if i % index == 0 and i >= start_i])

    for label in np.unique(X1_hat_labels):
        cmap = label_to_cmap[label]
        idx = (X1_hat_labels == label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1], color=color, alpha=0.9, s=3, zorder=1)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2, zorder=0)

    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)

    ax.set_xlabel("PC 1", fontsize=32)
    ax.set_ylabel("PC 2", fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_title("")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Static cluster plot saved to {output_file}")

    # --- Save colormap legend with thinner bars side by side ---
    fig_legend, axs = plt.subplots(1, len(label_to_cmap), figsize=(20, 1.5))
    if len(label_to_cmap) == 1:
        axs = [axs]

    for ax, (label, base_cmap) in zip(axs, label_to_cmap.items()):
        # Create a new clipped colormap
        clipped_cmap = base_cmap(np.linspace(0, 1, 256)[cmap_clip])
        custom_cmap = plt.matplotlib.colors.ListedColormap(clipped_cmap)
    
        # Use that in the legend
        gradient = np.linspace(0, 1, cmap_clip.stop - cmap_clip.start).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 0, 0.03])
        ax.set_title(f"{label.capitalize()} phenotypic shift \nPre-treatment → Post-treatment", fontsize=20)
        ax.axis('off')


    legend_path = output_file.replace('.png', '_legend.png')
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to {legend_path}")

    # --- Additional per-label plots ---
    for current_label in np.unique(X1_hat_labels):
        fig_lbl, ax_lbl = plt.subplots(figsize=(8, 6))
        cmap = label_to_cmap[current_label]
        idx = (X1_hat_labels == current_label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax_lbl.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           color=color, alpha=0.8, s=3, zorder=2)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax_lbl.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2)

        ax_lbl.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)
        ax_lbl.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)

        ax_lbl.set_xlabel("PC 1", fontsize=32)
        ax_lbl.set_ylabel("PC 2", fontsize=32)
        ax_lbl.tick_params(axis='both', labelsize=32)
        ax_lbl.set_title("", fontsize=32)
        plt.tight_layout()
        plt.savefig(f"{output_file.replace('.png', f'_label_{current_label}.png')}", dpi=300, bbox_inches='tight')
        plt.close(fig_lbl)

    return X1_hat_labels


def generate_static_cluster_plot_deviation_colormap_887(pca,
    source_t, target_t, start_i,X1_trpts,mats, index,intermediate_t=[1, 2, 3],output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories with gradient coloring.
    Also generates a separate legend figure and individual plots per subgroup.
    """


    # Step 1: Reduce real data and predictions to PCA space
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_first = X1_trpts[0].astype(np.float32)
    displacements = np.linalg.norm(X1_hat_last - X1_hat_first, axis=1)

    # 862, 3 and 5 (ER)
    # 887 0.8 and 1.3 (ER)
    # BMC 1.0 and 3.0 (ER)
    # Rinath 2.89 and 2.90 (ER)

    # 862, 1.3 and 2 (R)
    # 887 0.8 and 1.2 (R)
    # BMC 1.2 and 2.4 (R)
    # Rinath 3.2 and 3.8 (R)

    ## R genes
    #862 local minima x-values: [1.288 2.365]
    #887 local minima x-values: [0.699 1.149]
    #BMC local minima x-values: [1.145 1.937]
    #In vitro local minima x-values: [3.017 3.853]


    # Assign labels based on displacement
    X1_hat_labels = np.full(displacements.shape, 'low', dtype=object)
    X1_hat_labels[(displacements > 0.699) & (displacements <= 1.149)] = 'medium'
    X1_hat_labels[displacements > 1.149] = 'high'

    # Define colormaps per subgroup (avoid lightest tones by clipping range)
    label_to_cmap = {
        'low': colormaps['Oranges'],
        'medium': colormaps['Purples'],
        'high': colormaps['Greens']
    }
    cmap_clip = slice(75, 256)
    color_range = np.linspace(0, 1, 256)[cmap_clip]  # consistent use

    # Set file path for main plot
    #output_file = f"{result_dir}{exp_memo}_static_celltypes_plot_deviation_colormap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    X2_vis = pca.transform(mats[target_t])
    X1_vis = pca.transform(mats[source_t])

    total_steps = len([i for i in range(len(X1_trpts)) if i % index == 0 and i >= start_i])

    for label in np.unique(X1_hat_labels):
        cmap = label_to_cmap[label]
        idx = (X1_hat_labels == label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1], color=color, alpha=0.9, s=3, zorder=1)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2, zorder=0)

    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)

    ax.set_xlabel("PC 1", fontsize=32)
    ax.set_ylabel("PC 2", fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_title("")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Static cluster plot saved to {output_file}")

    # --- Save colormap legend with thinner bars side by side ---
    fig_legend, axs = plt.subplots(1, len(label_to_cmap), figsize=(20, 1.5))
    if len(label_to_cmap) == 1:
        axs = [axs]

    for ax, (label, base_cmap) in zip(axs, label_to_cmap.items()):
        # Create a new clipped colormap
        clipped_cmap = base_cmap(np.linspace(0, 1, 256)[cmap_clip])
        custom_cmap = plt.matplotlib.colors.ListedColormap(clipped_cmap)
    
        # Use that in the legend
        gradient = np.linspace(0, 1, cmap_clip.stop - cmap_clip.start).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 0, 0.03])
        ax.set_title(f"{label.capitalize()} phenotypic shift \nPre-treatment → Post-treatment", fontsize=20)
        ax.axis('off')


    legend_path = output_file.replace('.png', '_legend.png')
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to {legend_path}")

    # --- Additional per-label plots ---
    for current_label in np.unique(X1_hat_labels):
        fig_lbl, ax_lbl = plt.subplots(figsize=(8, 6))
        cmap = label_to_cmap[current_label]
        idx = (X1_hat_labels == current_label)

        for step_idx, i in enumerate(range(start_i, len(X1_trpts), index)):
            if np.isnan(X1_trpts[i]).any():
                continue
            X1_hat_vis = X1_trpts[i]
            norm_val = step_idx / max(total_steps - 1, 1)
            color_idx = int(norm_val * (len(color_range) - 1))
            color = cmap(color_range[color_idx])
            ax_lbl.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           color=color, alpha=0.8, s=3, zorder=2)

            if i > start_i:
                prev_X1_hat_vis = X1_trpts[i - index]
                prev_idx = idx
                ax_lbl.plot([
                    prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]
                ], [
                    prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]
                ], color=color, alpha=0.6, linewidth=1.2)

        ax_lbl.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)
        ax_lbl.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=1)

        ax_lbl.set_xlabel("PC 1", fontsize=32)
        ax_lbl.set_ylabel("PC 2", fontsize=32)
        ax_lbl.tick_params(axis='both', labelsize=32)
        ax_lbl.set_title("", fontsize=32)
        plt.tight_layout()
        plt.savefig(f"{output_file.replace('.png', f'_label_{current_label}.png')}", dpi=300, bbox_inches='tight')
        plt.close(fig_lbl)

    return X1_hat_labels


## ## This is for breast cancer cell line's data, Time [0 , 4]
# Plot gene dynamis for each trajectory
## Subtrajectories defined by source

def Average_gene_dynamics_whole_saveonly_single_trajectory_Robin(
    pca, gene_names, source_t, target_t, X1_trpts, mats,
    gene_of_interest, index, p, max_i,
    intermediate_t=[1],
    subgroup_output_file=None,
    cluster_save_path=None
):

    dt = p['numerical_ts'][-1] / 200
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]

    intermediate_t = np.array(intermediate_t)

    if len(intermediate_t) == 0:
        intermediate_t = range(source_t + 1, target_t)

    day1, day2 = source_t, target_t
    X1_trpt = X1_trpts[-1]

    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")

    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values

    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")

    gene_index = list(gene_names).index(gene_of_interest)

    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)

    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(
            X1_intermediate_vis_i_pca[:, gene_index]
        )

    gene_expression_X1_trpts = np.concatenate([
        pca.inverse_transform(X1_trpt)[:, gene_index]
        for i, X1_trpt in enumerate(X1_trpts)
        if i % index == 0 and i <= max_i
    ])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts

    indices = range(0, len(X1_trpts), index)

    subtrajectory_colors = ['green', 'orange', 'purple', 'blue', 'red', 'brown']
    violin_colors = ["black", "black"]

    subgroup_color_map = {
        label: subtrajectory_colors[i % len(subtrajectory_colors)]
        for i, label in enumerate(unique_labels)
    }

    label_mapping = {
        old_label: new_label + 1
        for new_label, old_label in enumerate(unique_labels)
    }

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # ============================================
    # CHANGED: rescale trajectory from 0 → 1
    # ============================================
    num_points = len(indices)
    x_positions = np.linspace(0, 1, num_points)

    cell_trajectories = {
        cell_idx: []
        for cell_idx in range(X1_trpts[0].shape[0])
    }

    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break

        X1_trpt = X1_trpts[time_idx]

        if np.isnan(X1_trpt).any():
            break

        gene_expression_values = pca.inverse_transform(
            X1_trpt
        )[:, gene_index]

        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)

    legend_patches = []

    for label in unique_labels:
        first_plotted = False

        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue

            if X1_hat_labels[cell_idx] == label:
                ax1.plot(
                    x_positions,
                    traj,
                    color=subgroup_color_map[label],
                    alpha=0.3,
                    linewidth=1.5
                )

                if not first_plotted:
                    legend_patches.append(
                        mpatches.Patch(
                            color=subgroup_color_map[label],
                            label=f'Trajectory of {label_mapping[label]} phenotypic shift'
                        )
                    )
                    first_plotted = True

    # Only plot the two real input time points:
    # source_t = 0, target_t = 1
    violin_data = [
        gene_expression_X1_normalized,  # mats[source_t]
        gene_expression_X2_normalized   # mats[target_t]
    ]
    
    violin_x_positions = np.array([0, 1])

    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        sns.violinplot(
            data=[data],
            ax=ax1,
            inner=None,
            linewidth=1.2,
            width=0.15,
            cut=0,
            scale="width",
            color=violin_colors[i],
            alpha=0.8,
            zorder=3
        )

        for violin in ax1.collections[-1:]:
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()

    # ============================================
    # CHANGED: x-axis limits
    # ============================================
    ax1.set_xlim(-0.15, 1.15)

    # ============================================
    # CHANGED: ticks now 0 and 1
    # ============================================
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(
        ["Primary", "Recurrence"],
        fontsize=24
    )

    ax1.tick_params(axis='y', labelsize=24)

    ax1.set_xlabel('Time', fontsize=24)
    ax1.set_ylabel('Gene Expression', fontsize=24)
    ax1.set_title(f'{gene_of_interest}', fontsize=24)

    plt.savefig(
        subgroup_output_file,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    label_descriptions = {
    "low": "Trajectory of low phenotypic shift",
    "medium": "Trajectory of medium phenotypic shift",
    "high": "Trajectory of high phenotypic shift"}


    # Thicker lines using `linewidth`
    legend_patches = [
        mlines.Line2D(
            [], [], color=color, linestyle='-', linewidth=3,  # ← thicker line here
            markersize=10,
            label=f"{label_descriptions.get(ctype, '')}"
        )
        for ctype, color in zip(unique_labels, subtrajectory_colors)
    ]


    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()


def Average_gene_dynamics_whole_saveonly_single_trajectory_NDPR_breast_cancer(pca, gene_names, source_t, target_t, X1_trpts,mats, gene_of_interest, index,p, max_i, intermediate_t = [1], subgroup_output_file = None, cluster_save_path = None):

    dt = p['numerical_ts'][-1]/200
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t

    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]

    # Create a color mapping for the specific indices
    
    # Step 1: Perform clustering analysis on the last day's cell states from mats
    
    # Load previously saved cluster labels
    # cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_deviation.csv"
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot




 
    # (1) **Assign Labels for Subgroups Based on Step 1**

    
    # Define **subtrajectory colors** (for cell trajectories)
    # real_cell_types = np.array(cell_ids_by_day[day2])
    #unique_cell_types = np.unique(real_cell_types)
    # unique_cell_types = unique_labels
    #subtrajectory_colors = list(sns.color_palette("tab20", len(unique_cell_types)))
    #subtrajectory_colors = ['green', 'orange', 'purple', 'blue', 'red', 'brown']
    subtrajectory_colors = ['violet']
    
    # Define **violin plot colors** for the three time points
    violin_colors = ["black", "black"]  # Green, Orange, Purple
    
    # Map each subgroup label to a **trajectory color** and shift labels from 0,1 → 1,2
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subtrajectory_colors[i % len(subtrajectory_colors)] for i, label in enumerate(unique_labels)}
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Define filename for saving
    #subgroup_output_file = f"{output_dir}/Celltypes_deviated_trajectories_violin_plot_{gene_of_interest}.png"
    
    # (2) **Create Figure**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # (3) **Ensure Proper x-axis Scaling**
    num_points = len(indices)
    x_positions = np.linspace(0, 4, num_points)  # Scale to match `[0, 2, 4]`
    
    # (4) **Extract Cell Trajectories for Each Gene**
    cell_trajectories = {cell_idx: [] for cell_idx in range(X1_trpts[0].shape[0])}
    
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract **expression values of the gene of interest** from each cell at this time point
        gene_expression_values = pca.inverse_transform(X1_trpt)[:, gene_index]
    
        # Append the expression value at this time to each cell’s trajectory
        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)
    
    # (5) **Plot Individual Trajectories per Subgroup**
    legend_patches = []  # Store legend handles
    for label in unique_labels:
        first_plotted = False  # Track if we added a legend entry for this subgroup
        
        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue  # Ensure trajectories align with time points
    
            if X1_hat_labels[cell_idx] == label:  # Match subgroup label from step 1
                ax1.plot(
                    x_positions, traj,  
                    color=subgroup_color_map[label],  # ✅ Use the **subtrajectory colors**
                    alpha=0.1, linewidth=0.8  
                )
                
                # Add a single legend entry for each subgroup (renaming from 0,1 → 1,2)
                if not first_plotted:
                    legend_patches.append(mpatches.Patch(color=subgroup_color_map[label], label=f'Trajectory of {label_mapping[label]} phenotypic shift'))
                    first_plotted = True
    
    # (6) **Ensure Violin Plots are at `[0, 2, 4]` & Appear in Front**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    violin_x_positions = np.array([0, 4])  # Ensure correct positions
    
    # 🎻 **Plot Violin Plots with Correct Colors and Transparency**
    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        violin_parts = sns.violinplot(
            data=[data],  
            ax=ax1,
            inner=None,  # ✅ REMOVE QUARTILE LINES
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=violin_colors[i],  # ✅ Assign correct color
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 4])  
    #ax1.set_xticklabels([0, 4], fontsize=32)
    ax1.set_xticklabels(["Pre-treatment", "Post-treatment"], fontsize=46)
    ax1.tick_params(axis='y', labelsize=46)
    
    ax1.set_xlabel('Time', fontsize=46)
    ax1.set_ylabel('Gene Expression', fontsize=46)
    ax1.set_title(f'{gene_of_interest}', fontsize=46)


    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

    # 🎨 **Redefine `legend_patches` to Include a Green Bar**
    
    legend_patches = [
        mlines.Line2D([], [], color="violet", linestyle="-", linewidth=3, 
                      label="Hallmark dynamics of each single cell")
    ]




    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()


def Average_gene_dynamics_whole_saveonly_single_trajectory_clinical(pca, gene_names, source_t, target_t, X1_trpts,mats, gene_of_interest, index,p, max_i, intermediate_t = [1], subgroup_output_file = None, cluster_save_path = None):

    dt = p['numerical_ts'][-1]/200
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t

    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]

    # Create a color mapping for the specific indices
    
    # Step 1: Perform clustering analysis on the last day's cell states from mats
    
    # Load previously saved cluster labels
    # cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_deviation.csv"
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot




 
    # (1) **Assign Labels for Subgroups Based on Step 1**

    
    # Define **subtrajectory colors** (for cell trajectories)
    # real_cell_types = np.array(cell_ids_by_day[day2])
    #unique_cell_types = np.unique(real_cell_types)
    unique_cell_types = unique_labels
    #subtrajectory_colors = list(sns.color_palette("tab20", len(unique_cell_types)))
    subtrajectory_colors = ['green', 'orange', 'purple', 'blue', 'red', 'brown']
    #subtrajectory_colors = ['violet']
    
    # Define **violin plot colors** for the three time points
    violin_colors = ["black", "black"]  # Green, Orange, Purple
    
    # Map each subgroup label to a **trajectory color** and shift labels from 0,1 → 1,2
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subtrajectory_colors[i % len(subtrajectory_colors)] for i, label in enumerate(unique_labels)}
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Define filename for saving
    #subgroup_output_file = f"{output_dir}/Celltypes_deviated_trajectories_violin_plot_{gene_of_interest}.png"
    
    # (2) **Create Figure**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # (3) **Ensure Proper x-axis Scaling**
    num_points = len(indices)
    x_positions = np.linspace(0, 4, num_points)  # Scale to match `[0, 2, 4]`
    
    # (4) **Extract Cell Trajectories for Each Gene**
    cell_trajectories = {cell_idx: [] for cell_idx in range(X1_trpts[0].shape[0])}
    
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract **expression values of the gene of interest** from each cell at this time point
        gene_expression_values = pca.inverse_transform(X1_trpt)[:, gene_index]
    
        # Append the expression value at this time to each cell’s trajectory
        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)
    
    # (5) **Plot Individual Trajectories per Subgroup**
    legend_patches = []  # Store legend handles
    for label in unique_labels:
        first_plotted = False  # Track if we added a legend entry for this subgroup
        
        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue  # Ensure trajectories align with time points
    
            if X1_hat_labels[cell_idx] == label:  # Match subgroup label from step 1
                ax1.plot(
                    x_positions, traj,  
                    color=subgroup_color_map[label],  # ✅ Use the **subtrajectory colors**
                    alpha=0.1, linewidth=0.8  
                )
                
                # Add a single legend entry for each subgroup (renaming from 0,1 → 1,2)
                if not first_plotted:
                    legend_patches.append(mpatches.Patch(color=subgroup_color_map[label], label=f'Trajectory of {label_mapping[label]} phenotypic shift'))
                    first_plotted = True
    
    # (6) **Ensure Violin Plots are at `[0, 2, 4]` & Appear in Front**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    violin_x_positions = np.array([0, 4])  # Ensure correct positions
    
    # 🎻 **Plot Violin Plots with Correct Colors and Transparency**
    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        violin_parts = sns.violinplot(
            data=[data],  
            ax=ax1,
            inner=None,  # ✅ REMOVE QUARTILE LINES
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=violin_colors[i],  # ✅ Assign correct color
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 4])  
    #ax1.set_xticklabels([0, 4], fontsize=32)
    ax1.set_xticklabels(["Pre-treatment", "Post-treatment"], fontsize=46)
    ax1.tick_params(axis='y', labelsize=46)
    
    ax1.set_xlabel('Time', fontsize=46)
    ax1.set_ylabel('Gene Expression', fontsize=46)
    ax1.set_title(f'{gene_of_interest}', fontsize=46)


    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

    # 🎨 **Redefine `legend_patches` to Include a Green Bar**
    
    #legend_patches = [
    #    mlines.Line2D([], [], color="violet", linestyle="-", linewidth=3, 
    #                  label="Hallmark dynamics of each single cell")
    #]

    label_descriptions = {
    "low": "Trajectory of low phenotypic shift",
    "medium": "Trajectory of medium phenotypic shift",
    "high": "Trajectory of high phenotypic shift"}


    # Thicker lines using `linewidth`
    legend_patches = [
        mlines.Line2D(
            [], [], color=color, linestyle='-', linewidth=3,  # ← thicker line here
            markersize=10,
            label=f"{label_descriptions.get(ctype, '')}"
        )
        for ctype, color in zip(unique_cell_types, subtrajectory_colors)
    ]


    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()

## ## This is for clinical data, Time [0 , 4]
## Plot gene dynamis for each trajectory (deviation cell types)

## Subtrajectroies defined by source
def Average_gene_dynamics_whole_saveonly_single_trajectory_clinical_old(pca,gene_names, source_t, target_t, X1_trpts,mats, gene_of_interest,p, index, max_i,
                              intermediate_t = [1], 
                              d_red=2, random_state=42, exp_memo = '2', cluster_save_path = None, subgroup_output_file = None):

   
    dt = p['numerical_ts'][-1]/200
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t

    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]

    # Create a color mapping for the specific indices
    # Step 1: Perform clustering analysis on the last day's cell states from mats
    
    # Load previously saved cluster labels
    #cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_deviation.csv"

    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    # Define the filename for saving the plot




 
    # (1) **Assign Labels for Subgroups Based on Step 1**

    
    # Define **subtrajectory colors** (for cell trajectories)
    # real_cell_types = np.array(cell_ids_by_day[day2])
    #unique_cell_types = np.unique(real_cell_types)
    unique_cell_types = unique_labels
    #subtrajectory_colors = list(sns.color_palette("tab20", len(unique_cell_types)))
    subtrajectory_colors = ['green', 'orange', 'purple', 'blue', 'red', 'brown']
    #subtrajectory_colors = ['violet']
    
    # Define **violin plot colors** for the three time points
    violin_colors = ["black", "black"]  # Green, Orange, Purple
    
    # Map each subgroup label to a **trajectory color** and shift labels from 0,1 → 1,2
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subtrajectory_colors[i % len(subtrajectory_colors)] for i, label in enumerate(unique_labels)}
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Define filename for saving
    #subgroup_output_file = f"{output_dir}/Celltypes_deviated_trajectories_violin_plot_{gene_of_interest}.png"
    
    # (2) **Create Figure**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # (3) **Ensure Proper x-axis Scaling**
    num_points = len(indices)
    x_positions = np.linspace(0, 4, num_points)  # Scale to match `[0, 2, 4]`
    
    # (4) **Extract Cell Trajectories for Each Gene**
    cell_trajectories = {cell_idx: [] for cell_idx in range(X1_trpts[0].shape[0])}
    
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract **expression values of the gene of interest** from each cell at this time point
        gene_expression_values = pca.inverse_transform(X1_trpt)[:, gene_index]
    
        # Append the expression value at this time to each cell’s trajectory
        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)
    
    # (5) **Plot Individual Trajectories per Subgroup**
    legend_patches = []  # Store legend handles
    for label in unique_labels:
        first_plotted = False  # Track if we added a legend entry for this subgroup
        
        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue  # Ensure trajectories align with time points
    
            if X1_hat_labels[cell_idx] == label:  # Match subgroup label from step 1
                ax1.plot(
                    x_positions, traj,  
                    color=subgroup_color_map[label],  # ✅ Use the **subtrajectory colors**
                    alpha=0.1, linewidth=0.8  
                )
                
                # Add a single legend entry for each subgroup (renaming from 0,1 → 1,2)
                if not first_plotted:
                    legend_patches.append(mpatches.Patch(color=subgroup_color_map[label], label=f'Trajectory of {label_mapping[label]} phenotypic shift'))
                    first_plotted = True
    
    # (6) **Ensure Violin Plots are at `[0, 2, 4]` & Appear in Front**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    violin_x_positions = np.array([0, 4])  # Ensure correct positions
    
    # 🎻 **Plot Violin Plots with Correct Colors and Transparency**
    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        violin_parts = sns.violinplot(
            data=[data],  
            ax=ax1,
            inner=None,  # ✅ REMOVE QUARTILE LINES
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=violin_colors[i],  # ✅ Assign correct color
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 4])  
    #ax1.set_xticklabels([0, 4], fontsize=32)
    ax1.set_xticklabels(["Pre-treatment", "Post-treatment"], fontsize=46)
    ax1.tick_params(axis='y', labelsize=46)
    
    ax1.set_xlabel('Time', fontsize=46)
    ax1.set_ylabel('Gene Expression', fontsize=46)
    ax1.set_title(f'{gene_of_interest}', fontsize=46)


    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

    # 🎨 **Redefine `legend_patches` to Include a Green Bar**
    
    #legend_patches = [
    #    mlines.Line2D([], [], color="violet", linestyle="-", linewidth=3, 
    #                  label="Hallmark dynamics of each single cell")
    #]

    label_descriptions = {
    "low": "Trajectory of low phenotypic shift",
    "medium": "Trajectory of medium phenotypic shift",
    "high": "Trajectory of high phenotypic shift"}


    # Thicker lines using `linewidth`
    legend_patches = [
        mlines.Line2D(
            [], [], color=color, linestyle='-', linewidth=3,  # ← thicker line here
            markersize=10,
            label=f"{label_descriptions.get(ctype, '')}"
        )
        for ctype, color in zip(unique_cell_types, subtrajectory_colors)
    ]


    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()

###start of the stem cell functions
###
## ## This is for Stem cell data (Five time points: Time [0, 1, 2, 3, 4])

## Subtrajectroies with_violin_plot

def Average_gene_dynamics_whole_saveonly_with_violin_plot_sample_3_stem(pca, gene_names, source_t, target_t,X1_trpts,mats,optimal_k, gene_of_interest, index, p, max_i, intermediate_t = [1], img_src = None, cluster_save_path = "X1_hat_clusters.csv"):

    dt = p['numerical_ts'][-1]/200
   
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t

    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]

    # Create a color mapping for the specific indices

    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day2]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    
    # Load previously saved cluster labels
    #cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_clusters.csv"
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)

    #mask = last_day_labels == 0
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )




    
    # (1) Perform clustering on the last day's cell states from `mats`
    last_day = mats[day2]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    

    
    # Define colors for subgroups
    subgroup_colors = ['red', 'blue', '#ffe119', '#f58231', '#3cb44b']
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subgroup_colors[i % len(subgroup_colors)] for i, label in enumerate(unique_labels)}
    
    # Define filename
    subgroup_output_file = img_src
    
    # (2) Initialize Storage for Mean and CI
    subgroup_avg_gene_expressions = {label: [] for label in unique_labels}
    subgroup_ci_gene_expressions = {label: [] for label in unique_labels}
    
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized.copy()
    
    # (3) Compute Mean & Confidence Intervals
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:  # Apply truncation
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract gene expression values
        X1_hat = pca.inverse_transform(X1_trpt)
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]
    
        # Compute subgroup averages & CI
        for label in unique_labels:
            mask = (X1_hat_labels == label)  # Use labels **only from step 1**
            subgroup_values = np.array(gene_expression_values)[mask]
    
            if len(subgroup_values) > 0:
                subgroup_avg_gene_expressions[label].append(np.mean(subgroup_values))
                ci = stats.sem(subgroup_values) * stats.t.ppf((1 + 0.95) / 2., len(subgroup_values) - 1)
                subgroup_ci_gene_expressions[label].append(ci)
            else:
                subgroup_avg_gene_expressions[label].append(np.nan)
                subgroup_ci_gene_expressions[label].append(np.nan)
    

            
    
    # (4) **Plot**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # **Get x-axis positions for the line plot (scale to [0, 4])**
    num_points = len(next(iter(subgroup_avg_gene_expressions.values())))  # Number of time points
    x_positions = np.linspace(0, 4, num_points)  # Ensure correct x-spacing for trajectories
    
    # **Plot Predicted Trajectories & Confidence Intervals**
    subgroup_legend_handles = []  # Store for separate legend


    # **Plot Subgroup Averages & Confidence Intervals**
    for i, label in enumerate(unique_labels):
        # **Plot the Mean Trajectory Line**
        line, = ax1.plot(
            x_positions, subgroup_avg_gene_expressions[label], zorder=10,
            linestyle='-', color=subgroup_color_map[label], linewidth=2,
            label=f'Predicted Trajectory {i+1}'
        )
    
        # **Plot the Confidence Interval (Shaded Region)**
        ax1.fill_between(
            x_positions,
            np.array(subgroup_avg_gene_expressions[label]) - np.array(subgroup_ci_gene_expressions[label]),
            np.array(subgroup_avg_gene_expressions[label]) + np.array(subgroup_ci_gene_expressions[label]),
            alpha=0.2, zorder=5, color=subgroup_color_map[label],
            label=f'95% CI of Trajectory {i+1}'
        )
    
        # **Legend entry for Mean + Confidence Interval**
        ci_patch = mpatches.Patch(
            color=subgroup_color_map[label], alpha=0.2, label=f'95% CI of Trajectory {i+1}'
        )
    
        # **Store in Legend Handles**
        subgroup_legend_handles.append(ci_patch)
        subgroup_legend_handles.append(line)
        
    # (5) **Ensure Violin Plots are at `[0, 2, 4]`**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    # **Manually set violin plot positions to `[0, 2, 4]`**
    violin_x_positions = np.array([0, 1, 2, 3, 4])  # Explicitly define positions
    violin_colors = ["black", "gray", "black", "gray", "black"]  # Set distinct colors
    
    # 🎻 **Plot Violin Plots One-by-One to Force Correct Positioning**
    for i, (x_pos, data, color) in enumerate(zip(violin_x_positions, violin_data, violin_colors)):
        violin_parts = sns.violinplot(
            data=[data],  # Must be wrapped in a list to avoid merging violins
            ax=ax1,
            inner=None,
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=color,  # ✅ Assign distinct colors
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  # Move to correct x-location
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  # ✅ Extend range
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 1, 2, 3, 4])  # ✅ Force labels at `[0, 2, 4]`
    ax1.set_xticklabels([0, 1, 2, 3, 4], fontsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    
    ax1.set_xlabel('Time', fontsize=35)
    ax1.set_ylabel('Gene Expression', fontsize=35)
    ax1.set_title(f'Subtrajectory {gene_of_interest} Expression', fontsize=34)
    
    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data"),
        mpatches.Patch(color="gray", label="Test Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (VERTICAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(4, 8))  # Tall aspect ratio for vertical layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = subgroup_legend_handles + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=18, title="Trajectories & Violin Plots",
        title_fontsize=18, ncol=1, frameon=True, handletextpad=1.5, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Subgroup trajectory plot saved at: {subgroup_output_file}")
    print(f"Legend plot saved separately at: {legend_output_file}")

## Plot the difference of means across subtrajectories (Stem cell data)

def difference_of_means_stem(gene_names, subtraj_dir):
    # Define function to get evenly spaced colors from a colormap
    def get_colormap_colors(cmap_name, num_colors):
        if num_colors == 1:
            return [plt.get_cmap(cmap_name)(0.5)]  # Select the middle of the colormap
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    # Define paths
    genes_of_interest = gene_names  
    #subtraj_dir = os.path.join(result_dir, "output", exp_memo, "subtraj")

    # Load all CSVs for fold change, mean difference, and p-values
    df_fc_list, df_md_list, df_pval_list = [], [], []

    for gene in genes_of_interest:
        fc_file_path = os.path.join(subtraj_dir, f"fold_change_nan_propagated_{gene}.csv")
        md_file_path = os.path.join(subtraj_dir, f"mean_difference_{gene}.csv")
        pval_file_path = os.path.join(subtraj_dir, f"p_values_{gene}.csv")

        if os.path.exists(fc_file_path):
            df_fc = pd.read_csv(fc_file_path)
            df_fc["Gene"] = gene  
            df_fc_list.append(df_fc)

        if os.path.exists(md_file_path):
            df_md = pd.read_csv(md_file_path)
            df_md["Gene"] = gene  
            df_md_list.append(df_md)

        if os.path.exists(pval_file_path):
            df_pval = pd.read_csv(pval_file_path)
            df_pval["Gene"] = gene  
            df_pval_list.append(df_pval)

    # Merge all DataFrames
    df_fc_all = pd.concat(df_fc_list, ignore_index=True)
    df_md_all = pd.concat(df_md_list, ignore_index=True)
    df_pval_all = pd.concat(df_pval_list, ignore_index=True)

    # Merge p-values into fold change and mean difference DataFrames
    df_fc_all = df_fc_all.merge(df_pval_all, on=["Time", "Cluster 1", "Cluster 2", "Gene"], how="left")
    df_md_all = df_md_all.merge(df_pval_all, on=["Time", "Cluster 1", "Cluster 2", "Gene"], how="left")

    # **Rescale Time**
    df_fc_all["Time"] = df_fc_all["Time"] / 50
    df_md_all["Time"] = df_md_all["Time"] / 50

    # Define thresholds
    pos_threshold_fc = 5.0  
    neg_threshold_fc = -5.0  
    pos_threshold_md = 0.11  
    neg_threshold_md = -0.15  
    p_value_threshold = 1e-4  

    # **Identify significant genes based on p-value threshold**
    significant_genes = df_pval_all.groupby("Gene")["p-value"].apply(lambda x: x.min(skipna=True) < p_value_threshold)
    significant_genes = significant_genes[significant_genes].index  

    # Compute per-gene normalization
    df_md_all["Normalized Mean Difference"] = df_md_all.groupby("Gene")["Mean Difference"].transform(lambda x: x / x.abs().max())

    # **Filter fold-change and mean-difference genes, but retain all passing p-value**
    def filter_significant_genes(df, metric_col, threshold_high, threshold_low):
        """Find genes that exceed thresholds (colored) and others that stay gray (but pass p-value)."""
        gene_criteria = df.groupby("Gene")[metric_col].apply(lambda x: ((x > threshold_high) | (x < threshold_low)).any())
        
        highlighted_genes = gene_criteria[gene_criteria].index  # Genes exceeding threshold
        retained_genes = significant_genes.intersection(gene_criteria.index)  # Only keep genes passing p-value
        
        return retained_genes, highlighted_genes  # Return both full set and colored ones

    # **Apply filtering**
    all_genes_fc, highlighted_genes_fc = filter_significant_genes(df_fc_all, "Log2 Fold Change", pos_threshold_fc, neg_threshold_fc)
    all_genes_md, highlighted_genes_md = filter_significant_genes(df_md_all, "Mean Difference", pos_threshold_md, neg_threshold_md)

    # Identify genes with significant normalized mean difference changes
    gene_max_diff = df_md_all.groupby("Gene")["Normalized Mean Difference"].apply(lambda x: x.max() - x.min())

    # **Determine which genes to highlight**
    highlighted_genes_norm = significant_genes.intersection(gene_max_diff[gene_max_diff > 0.62].index)
    gray_genes = significant_genes.difference(highlighted_genes_norm)  # Genes passing p-value but not gene_max_diff

    # **Split Highlighted Genes into Positive and Negative Groups**
    positive_genes = []
    negative_genes = []

    for gene in highlighted_genes_norm:
        initial_value = df_md_all[df_md_all["Gene"] == gene]["Normalized Mean Difference"].iloc[0]  
        if initial_value >= 0:
            positive_genes.append(gene)
        else:
            negative_genes.append(gene)



    # **Assign Colors Using Custom Colormaps (Avoiding White)**
    num_positive = max(len(positive_genes), 1)  
    num_negative = max(len(negative_genes), 1)  

    # Load colormaps
    cmap_reds = plt.get_cmap("Reds")  # Red to White
    cmap_blues = plt.get_cmap("Blues_r")  # Blue to White (Reversed)

    # Clip the colormap range to avoid very light colors (too close to white)
    colors_positive = [cmap_reds(0.3 + 0.7 * (i / (num_positive - 1))) for i in range(num_positive)]  # Avoid very light red
    colors_negative = [cmap_blues(0.1 + 0.6 * (i / max(1, num_negative - 1))) for i in range(num_negative)]

    gene_colors_scaled = {}

    for i, gene in enumerate(positive_genes):
        gene_colors_scaled[gene] = colors_positive[i % len(colors_positive)]

    for i, gene in enumerate(negative_genes):
        gene_colors_scaled[gene] = colors_negative[i % len(colors_negative)]




    # **Separate Legends for Positive & Negative Genes**
    legend_handles_pos = [mpatches.Patch(color=gene_colors_scaled[gene], label=gene) for gene in positive_genes]
    legend_handles_neg = [mpatches.Patch(color=gene_colors_scaled[gene], label=gene) for gene in negative_genes]

    # **Save Separate Legend for Positive Genes**
    if legend_handles_pos:
        fig_legend_pos, ax_legend_pos = plt.subplots(figsize=(6, max(1, len(legend_handles_pos) // 2)))
        ax_legend_pos.axis("off")
        ax_legend_pos.legend(handles=legend_handles_pos, loc="center", title="Positive  Difference",
                            title_fontsize=36, fontsize=36, frameon=True, ncol=1)
        plt.savefig(os.path.join(subtraj_dir, "legend_positive_genes.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # **Save Separate Legend for Negative Genes**
    if legend_handles_neg:
        fig_legend_neg, ax_legend_neg = plt.subplots(figsize=(6, max(1, len(legend_handles_neg) // 2)))
        ax_legend_neg.axis("off")
        ax_legend_neg.legend(handles=legend_handles_neg, loc="center", title="Negative Difference",
                            title_fontsize=36, fontsize=36, frameon=True, ncol=1)
        plt.savefig(os.path.join(subtraj_dir, "legend_negative_genes.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # **Generate Main Plot (Without Legend)**
    for (cluster1, cluster2), sub_group in df_md_all.groupby(["Cluster 1", "Cluster 2"]):

        fig, ax = plt.subplots(figsize=(10, 9))

        for gene in significant_genes:  # Only genes passing p-value threshold are plotted
            gene_data = sub_group[sub_group["Gene"] == gene]

            if gene in highlighted_genes_norm:
                color = gene_colors_scaled.get(gene, "gray")  
                alpha_value, linestyle = 1.0, "-"
            else:  # Genes passing p-value but NOT max diff
                color = "gray"
                alpha_value, linestyle = 0.2, "--"

            ax.plot(
                gene_data["Time"], gene_data["Normalized Mean Difference"], 
                marker="o", markersize=3, linestyle=linestyle, color=color, alpha=alpha_value
            )

        # Labels and Title
        ax.set_xlabel("Time", fontsize=30)
        ax.set_ylabel("Normalized Mean Difference", fontsize=30)
        ax.set_title(f"Trajectory {cluster1 + 1} vs {cluster2 + 1}", fontsize=30)
        ax.tick_params(axis="both", labelsize=30)

        # Save Plot
        plt.tight_layout()
        plt.savefig(os.path.join(subtraj_dir, f"normalized_mean_difference_cluster_{cluster1}_vs_{cluster2}.png"), dpi=300)
        plt.close()

        cmap1 = plt.colormaps["tab20b"]
        cmap2 = plt.colormaps["tab20c"]
        cmap3 = plt.colormaps["Set1"]
        cmap4 = plt.colormaps["Set3"]
        cmap5 = plt.colormaps["Paired"]

        # Generate color lists
        color_list = (
            [cmap3(i) for i in range(min(9, len(cmap3.colors)))] +  
            [cmap4(i) for i in range(min(12, len(cmap4.colors)))] +  
            [cmap1(i) for i in range(20)] +
            [cmap2(i) for i in range(20)] +
            [cmap5(i) for i in range(min(12, len(cmap5.colors)))] +  
            list(plt.cm.hsv(np.linspace(0, 1, 15)))  
        )

        # Assign colors
        all_highlighted_genes = sorted(set(highlighted_genes_fc).union(set(highlighted_genes_md)).union(set(highlighted_genes_norm)))
        gene_colors = {gene: color_list[i % len(color_list)] for i, gene in enumerate(all_highlighted_genes)}

        # **(1) Fold Change Plot**
        group_fc = df_fc_all[(df_fc_all["Cluster 1"] == cluster1) & (df_fc_all["Cluster 2"] == cluster2)]
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_handles = []

        for gene in all_genes_fc:
            sub_group = group_fc[group_fc["Gene"] == gene]
            if gene in highlighted_genes_fc:
                color = gene_colors.get(gene)
                alpha_value, linestyle = 1.0, "-"
            else:
                color = "gray"
                alpha_value, linestyle = 0.2, "--"

            line, = ax.plot(sub_group["Time"], sub_group["Log2 Fold Change"], marker="o", markersize=4, linestyle=linestyle, color=color, alpha=alpha_value, label=gene)
            if gene in highlighted_genes_fc:
                legend_handles.append(line)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("Time", fontsize = 30)
        ax.set_ylabel("Log2 Fold Change", fontsize = 30)
        ax.set_title(f"Fold Change Over Time (Cluster {cluster1} vs {cluster2})", fontsize = 30)

        if legend_handles:
            ax.legend(handles=legend_handles, title="Significant Genes", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(subtraj_dir, f"fold_change_comparison_cluster_{cluster1}_vs_{cluster2}.png"), dpi=300)
        plt.close()

        # **(2) Mean Difference Plot**
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_handles = []
        group_md = df_md_all[(df_md_all["Cluster 1"] == cluster1) & (df_md_all["Cluster 2"] == cluster2)]

        for gene in all_genes_md:
            sub_group = group_md[group_md["Gene"] == gene]
            if gene in highlighted_genes_md:
                color = gene_colors.get(gene)
                alpha_value, linestyle = 1.0, "-"
            else:
                color = "gray"
                alpha_value, linestyle = 0.2, "--"

            line, = ax.plot(sub_group["Time"], sub_group["Mean Difference"], marker="o", markersize=4, linestyle=linestyle, color=color, alpha=alpha_value, label=gene)
            if gene in highlighted_genes_md:
                legend_handles.append(line)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("Time", fontsize = 30)
        ax.set_ylabel("Mean Difference", fontsize = 30)
        ax.set_title(f"Mean Difference Over Time (Cluster {cluster1} vs {cluster2})", fontsize = 30)

        if legend_handles:
            ax.legend(handles=legend_handles, title="Significant Genes", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(subtraj_dir, f"mean_difference_comparison_cluster_{cluster1}_vs_{cluster2}.png"), dpi=300)
        plt.close()

    print("✅ Main plots saved.")
    print("✅ Separate legend for positive genes saved as 'legend_positive_genes.png'.")
    print("✅ Separate legend for negative genes saved as 'legend_negative_genes.png'.")

## ## This is for Stem Cell data, Time [0 , 1,  2, 3,  4]
## Plot gene dynamis for each trajectory




def Average_gene_dynamics_whole_saveonly_single_trajectory_Axolotl(pca, gene_names, source_t, target_t,X1_trpts,mats,optimal_k, gene_of_interest, index, p, max_i,
                              intermediate_t = [1], img_src = None, cluster_save_path = "X1_hat_clusters.csv",subgroup_output_file = None):


    
    dt = p['numerical_ts'][-1]/200
    
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t
    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]


    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day2]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    
    # Load previously saved cluster labels
    #cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_clusters.csv"
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")

    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)

    #mask = last_day_labels == 0
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    
    
    # Define **subtrajectory colors** (for cell trajectories)
    subtrajectory_colors = ["teal","magenta","orange","navy","gold"]
    #subtrajectory_colors = ['violet']
    
    # Define **violin plot colors** for the three time points
    #violin_colors = ["#3cb44b", "#f58231", "#3cb44b", "#f58231", "#3cb44b"]  # Green, Orange, Purple
    violin_colors = ["black", "gray", "black", "gray", "black"] 
    
    # Map each subgroup label to a **trajectory color** and shift labels from 0,1 → 1,2
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subtrajectory_colors[i % len(subtrajectory_colors)] for i, label in enumerate(unique_labels)}
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Define filename for saving
    #subgroup_output_file = f"{output_dir}/Individual_trajectories_violin_plot_{gene_of_interest}.png"
    
    # (2) **Create Figure**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # (3) **Ensure Proper x-axis Scaling**
    num_points = len(indices)
    x_positions = np.linspace(0, 4, num_points)  # Scale to match `[0, 2, 4]`
    
    # (4) **Extract Cell Trajectories for Each Gene**
    cell_trajectories = {cell_idx: [] for cell_idx in range(X1_trpts[0].shape[0])}
    
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract **expression values of the gene of interest** from each cell at this time point
        gene_expression_values = pca.inverse_transform(X1_trpt)[:, gene_index]
    
        # Append the expression value at this time to each cell’s trajectory
        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)
    
    # (5) **Plot Individual Trajectories per Subgroup**
    legend_patches = []  # Store legend handles
    for label in unique_labels:
        first_plotted = False  # Track if we added a legend entry for this subgroup
        
        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue  # Ensure trajectories align with time points
    
            if X1_hat_labels[cell_idx] == label:  # Match subgroup label from step 1
                ax1.plot(
                    x_positions, traj,  
                    color=subgroup_color_map[label],  # ✅ Use the **subtrajectory colors**
                    alpha=0.7, linewidth=1.0 
                )
                
                # Add a single legend entry for each subgroup (renaming from 0,1 → 1,2)
                if not first_plotted:
                    legend_patches.append(mpatches.Patch(color=subgroup_color_map[label], label=f'Trajectory {label_mapping[label]}'))
                    first_plotted = True
    
    # (6) **Ensure Violin Plots are at `[0, 2, 4]` & Appear in Front**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    violin_x_positions = np.array([0, 1, 2, 3, 4])  # Ensure correct positions
    
    # 🎻 **Plot Violin Plots with Correct Colors and Transparency**
    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        violin_parts = sns.violinplot(
            data=[data],  
            ax=ax1,
            inner=None,  # ✅ REMOVE QUARTILE LINES
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=violin_colors[i],  # ✅ Assign correct color
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 1, 2, 3, 4])  
    ax1.set_xticklabels([0, 1, 2, 3, 4], fontsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    
    ax1.set_xlabel('Time', fontsize=35)
    ax1.set_ylabel('Gene Expression', fontsize=35)
    ax1.set_title(f'Single Cell {gene_of_interest} Expression Dynamics', fontsize=35)


    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

    # 🎨 **Redefine `legend_patches` to Include a Green Bar**
    legend_patches = [
        mlines.Line2D([], [], color="violet", linestyle="-", linewidth=3, 
                      label="Gene dynamics of each single cell")
    ]

    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data"),
        mpatches.Patch(color="gray", label="Test Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()

## Subtrajectroies defined by source
def Average_gene_dynamics_whole_saveonly_single_trajectory_mESC(pca, gene_names, source_t, target_t,X1_trpts,mats,optimal_k, gene_of_interest, index, p, max_i,
                              intermediate_t = [1], img_src = None, cluster_save_path = "X1_hat_clusters.csv",subgroup_output_file = None):


    
    dt = p['numerical_ts'][-1]/200
    
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t
    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]


    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day2]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    
    # Load previously saved cluster labels
    #cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_clusters.csv"
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")

    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)

    #mask = last_day_labels == 0
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )

    
    
    # Define **subtrajectory colors** (for cell trajectories)
    #subtrajectory_colors = ['red', 'blue']
    subtrajectory_colors = ['violet']
    
    # Define **violin plot colors** for the three time points
    #violin_colors = ["#3cb44b", "#f58231", "#3cb44b", "#f58231", "#3cb44b"]  # Green, Orange, Purple
    violin_colors = ["black", "gray", "black", "gray", "black"] 
    
    # Map each subgroup label to a **trajectory color** and shift labels from 0,1 → 1,2
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subtrajectory_colors[i % len(subtrajectory_colors)] for i, label in enumerate(unique_labels)}
    label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)}
    
    # Define filename for saving
    #subgroup_output_file = f"{output_dir}/Individual_trajectories_violin_plot_{gene_of_interest}.png"
    
    # (2) **Create Figure**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # (3) **Ensure Proper x-axis Scaling**
    num_points = len(indices)
    x_positions = np.linspace(0, 4, num_points)  # Scale to match `[0, 2, 4]`
    
    # (4) **Extract Cell Trajectories for Each Gene**
    cell_trajectories = {cell_idx: [] for cell_idx in range(X1_trpts[0].shape[0])}
    
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract **expression values of the gene of interest** from each cell at this time point
        gene_expression_values = pca.inverse_transform(X1_trpt)[:, gene_index]
    
        # Append the expression value at this time to each cell’s trajectory
        for cell_idx, expr_value in enumerate(gene_expression_values):
            cell_trajectories[cell_idx].append(expr_value)
    
    # (5) **Plot Individual Trajectories per Subgroup**
    legend_patches = []  # Store legend handles
    for label in unique_labels:
        first_plotted = False  # Track if we added a legend entry for this subgroup
        
        for cell_idx, traj in cell_trajectories.items():
            if len(traj) != len(x_positions):
                continue  # Ensure trajectories align with time points
    
            if X1_hat_labels[cell_idx] == label:  # Match subgroup label from step 1
                ax1.plot(
                    x_positions, traj,  
                    color=subgroup_color_map[label],  # ✅ Use the **subtrajectory colors**
                    alpha=0.7, linewidth=1.0 
                )
                
                # Add a single legend entry for each subgroup (renaming from 0,1 → 1,2)
                if not first_plotted:
                    legend_patches.append(mpatches.Patch(color=subgroup_color_map[label], label=f'Trajectory {label_mapping[label]}'))
                    first_plotted = True
    
    # (6) **Ensure Violin Plots are at `[0, 2, 4]` & Appear in Front**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    violin_x_positions = np.array([0, 1, 2, 3, 4])  # Ensure correct positions
    
    # 🎻 **Plot Violin Plots with Correct Colors and Transparency**
    for i, (x_pos, data) in enumerate(zip(violin_x_positions, violin_data)):
        violin_parts = sns.violinplot(
            data=[data],  
            ax=ax1,
            inner=None,  # ✅ REMOVE QUARTILE LINES
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=violin_colors[i],  # ✅ Assign correct color
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 1, 2, 3, 4])  
    ax1.set_xticklabels([0, 1, 2, 3, 4], fontsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    
    ax1.set_xlabel('Time', fontsize=35)
    ax1.set_ylabel('Gene Expression', fontsize=35)
    ax1.set_title(f'Single Cell {gene_of_interest} Expression Dynamics', fontsize=35)


    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

    # 🎨 **Redefine `legend_patches` to Include a Green Bar**
    legend_patches = [
        mlines.Line2D([], [], color="violet", linestyle="-", linewidth=3, 
                      label="Gene dynamics of each single cell")
    ]

    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data"),
        mpatches.Patch(color="gray", label="Test Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (HORIZONTAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = legend_patches + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=24, title="",
        title_fontsize=24, ncol=len(combined_legend),  # Horizontal layout
        frameon=True, handletextpad=2, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
## Distribution of single genes comparions (Intermediate time points only) - Stem Cell data
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Compare_Distribution_Trajectories_Intermediate_mESC(pca, gene_names, source_t, target_t,X1_trpts,mats, gene_of_interest, p, intermediate_t = [1], output_file = None):

    if intermediate_t is None:
        intermediate_t = [1]

    
    dt = p['numerical_ts'][-1] / 200

    # Extract gene index
    gene_index = list(gene_names).index(gene_of_interest)

    # Test data distributions
    kde_test_data = [
        pca.inverse_transform(pca.transform(mats[t]))[:, gene_index] for t in intermediate_t
    ]

    # Correct snapshot extraction based on scaling
    snapshots_per_day = len(X1_trpts) / (target_t - source_t)
    scaled_intermediate_indices = [int(day * snapshots_per_day) for day in intermediate_t]

    kde_predicted_data = []
    for idx in scaled_intermediate_indices:
        if idx >= len(X1_trpts):
            idx = len(X1_trpts) - 1
        gene_expr_predicted = pca.inverse_transform(X1_trpts[idx])[:, gene_index]
        kde_predicted_data.append(gene_expr_predicted)

    # Visualization setup
    num_plots = len(intermediate_t)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)

    if num_plots == 1:
        axes = [axes]

    #test_data_colors = ["#2ca02c", "#8c564b"] #Sample 3
    #predicted_colors = ["#1b6420", "#5c3930"] #Sample 3

    test_data_colors = ["#2ca02c", "#8c564b", "#3cb44b"]  #Sample 1
    predicted_colors = ["#1b6420", "#5c3930", "#228B22"]  #Sample 1


    

        

    # Initialize list to store legend handles per intermediate time
    legend_patches_list = []
    
    # Generate KDE plots
    for i, (ax, t, test_vals, pred_vals) in enumerate(zip(axes, intermediate_t, kde_test_data, kde_predicted_data)):
    
        all_vals = np.concatenate([test_vals, pred_vals])
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        x_margin = (x_max - x_min) * 0.2
        x_range = np.linspace(x_min - x_margin, x_max + x_margin, 300)
    
        kde_test = gaussian_kde(test_vals)
        kde_pred = gaussian_kde(pred_vals)
    
        test_density = kde_test(x_range)
        pred_density = kde_pred(x_range)
    
        y_max = max(test_density.max(), pred_density.max()) * 2
    
        ax.fill_between(x_range, test_density, color=test_data_colors[i % len(test_data_colors)], alpha=0.5)
        ax.plot(x_range, test_density, color=test_data_colors[i % len(test_data_colors)], linewidth=2)
    
        ax.fill_between(x_range, pred_density, color=predicted_colors[i % len(predicted_colors)], alpha=0.5)
        ax.plot(x_range, pred_density, color=predicted_colors[i % len(predicted_colors)], linestyle="dashed", linewidth=2)
    
        ax.set_title(f"Time {t}", fontsize=26)
        ax.set_xlabel("Gene Expression", fontsize=26)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Density", fontsize=26)
        ax.tick_params(axis='both', which='major', labelsize=26)
    
        plt.suptitle(f"KDE for {gene_of_interest}", fontsize=26)
    

        # **Legend Entry for This Time Point**
        legend_patches_list.append([
            # **Test Data: Dashed Line**
            mlines.Line2D([], [], color=test_data_colors[i % len(test_data_colors)], linestyle="solid", linewidth=3,
                          label=f"Test Data time {t}"),
            
            # **Predicted Data: Solid Line**
            mlines.Line2D([], [], color=predicted_colors[i % len(predicted_colors)], linestyle="dashed", linewidth=3,
                          label=f"Predicted time {t}")
        ])
        
    # **Save KDE plot (without legend)**
    #output_file = f"{output_dir}/KDE_Intermediate_Only_updated_{gene_of_interest}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"KDE plot saved: {output_file}")
    
    # **(2) Create a Separate Figure for the Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(len(intermediate_t) * 3, 2))  # Adjust width dynamically
    ax_legend.axis("off")  # Hide axes
    
    # **Flatten legend handles into a single row-style list**
    flattened_legend_patches = []
    for group in legend_patches_list:
        for entry in group:
            flattened_legend_patches.append(entry)
    
    # **Create a Row-Style Legend with Box + Line for Each Entry**
    ax_legend.legend(
        handles=flattened_legend_patches,
        loc="center",
        fontsize=22,
        ncol=2,  # Ensures (Test, Predicted) pairs stay together
        frameon=True,
        handletextpad=1.5,
        columnspacing=2
    )
    
    # **Save the separate legend**
    legend_output_file = output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Legend plot saved separately at: {legend_output_file}")


def Average_gene_dynamics_whole_saveonly_with_violin_plot_Axolotl(pca, gene_names, source_t, target_t,X1_trpts,mats,optimal_k, gene_of_interest, index, p, max_i, intermediate_t = [1], img_src = None, cluster_save_path = "X1_hat_clusters.csv"):

    dt = p['numerical_ts'][-1]/200
   
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]
    
    intermediate_t = np.array(intermediate_t)
    
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)
        
    # data parameters
    day1, day2 = source_t, target_t

    X1_trpt = X1_trpts[-1]
    
    
    contrast_colors = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#d62728',  # red 
    '#9467bd'  # purple (to be used for index 8)
    ]

    # Create a color mapping for the specific indices

    # Step 1: Perform clustering analysis on the last day's cell states from mats
    last_day = mats[day2]

    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    
    # Load previously saved cluster labels
    #cluster_save_path = f"{result_dir}{exp_memo}_X1_hat_clusters.csv"
    if not os.path.exists(cluster_save_path):
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_save_path}")
    
    df_clusters = pd.read_csv(cluster_save_path)
    X1_hat_labels = df_clusters["Cluster_Label"].values  # Load saved labels

    # Print the number of unique labels in last_day_labels
    unique_labels = np.unique(X1_hat_labels)
    print(f"Number of unique labels in X1_hat_labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")
    
    
    # Define a function to create colors for the subgroups using a predefined set of colors
    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        if len(colors) < len(unique_labels):
            raise ValueError("Not enough colors for the number of unique labels.")
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    # Define specific sets of colors for the blue and red subgroups
    blue_colors = ['#1f77b4', '#878ceb', '#104E8B', '#87CEEB', '#4682B4', '#6495ED', '#5F9EA0']  # Add more shades of blue as needed
    red_colors = ['#d62728',  '#eb8787', '#FF4500', '#DC143C', '#FF6347', '#B22222', '#8B0000']  # Add more shades of red as needed
    light_red_colors = ['#f99fa1', '#ffb1b1', '#ffaf86', '#f48585', '#ffb5a5', '#ff9c9c', '#ff5f5f']
    
    # Get the subgroup colors based on the labels
    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, blue_colors)
    subgroup_colors_red = get_subgroup_colors(X1_hat_labels, red_colors)

    #mask = last_day_labels == 0
    
    
    # Extract the gene index for the gene of interest
    gene_index = list(gene_names).index(gene_of_interest)
    
    # Extract gene expression values from mats[day1], intermediate time points, and mats[day2]
    X1_vis_pca = pca.transform(mats[source_t])
    X1_vis_i_pca = pca.inverse_transform(X1_vis_pca)
    X2_vis_pca = pca.transform(mats[target_t])
    X2_vis_i_pca = pca.inverse_transform(X2_vis_pca)

    gene_expression_X1 = X1_vis_i_pca[:, gene_index]
    gene_expression_X2 = X2_vis_i_pca[:, gene_index]

    gene_expression_intermediates = []
    for t in intermediate_t:
        X1_intermediate_vis_pca = pca.transform(mats[t])
        X1_intermediate_vis_i_pca = pca.inverse_transform(X1_intermediate_vis_pca)
        gene_expression_intermediates.append(X1_intermediate_vis_i_pca[:, gene_index])

    # Extract gene expression values from X1_trpts based on the given condition
    
    gene_expression_X1_trpts = np.concatenate([pca.inverse_transform(X1_trpt)[:, gene_index] for i, X1_trpt in enumerate(X1_trpts) if i % index == 0 and i <= max_i])
    
    # Combine all gene expression values
    all_gene_expression_values = np.concatenate([gene_expression_X1, *gene_expression_intermediates, gene_expression_X2, gene_expression_X1_trpts])

    gene_expression_X1_normalized = gene_expression_X1
    gene_expression_intermediates_normalized = gene_expression_intermediates
    gene_expression_X2_normalized = gene_expression_X2
    gene_expression_X1_trpts_normalized = gene_expression_X1_trpts
    
    vmin = all_gene_expression_values.min()
    vmax = all_gene_expression_values.max()
    
    # Plot dynamics for X1_trpts with subgroup colors
    indices = range(len(X1_trpts))

    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    

    
    # (1) Plot the averaged gene expressions across X1_trpt at each time point with confidence intervals
    
    # Compute the average gene expression and confidence intervals
    avg_gene_expressions = []
    ci_gene_expressions = []
    
    # Reset normalized gene expression values for X1_trpts
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized
    
    # Use indices with the specified step size defined by `index`
    indices = range(0, len(X1_trpts), index)

    
    # Iterate through indices to compute averages and confidence intervals
    for i in indices:
        if i > max_i:  # Apply truncation based on max_i
            break
        X1_trpt = X1_trpts[i]
        if np.isnan(X1_trpt).any():
            break
    
        # Inverse transform the current trajectory
        X1_hat = pca.inverse_transform(X1_trpt)
    
        # Extract gene expression values for the current step
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]  # Update to exclude used values
    
        # Compute average and confidence interval
        avg_gene_expressions.append(np.mean(gene_expression_values))
        ci = stats.sem(gene_expression_values) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_values) - 1)
        ci_gene_expressions.append(ci)
    
    # Process intermediate time points
    intermediate_avg_expressions = []
    intermediate_ci_expressions = []
    intermediate_indices = []


    for idx, t in enumerate(intermediate_t):
        gene_expression_intermediate = gene_expression_intermediates_normalized[idx]
        intermediate_avg_expressions.append(np.mean(gene_expression_intermediate))
        ci = stats.sem(gene_expression_intermediate) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_intermediate) - 1)
        intermediate_ci_expressions.append(ci)
    
        # Rescale the intermediate time points to align with `index`
        shifted_value_1 = intermediate_t - 1
        shifted_value_2 = intermediate_t[0] - 1
        shifted_t_1 = t - shifted_value_1
        shifted_t_2 = t - shifted_value_2
        time_index = int((float(shifted_t_2) / (float(max(shifted_t_1)) + 1)) * len(indices))
        intermediate_indices.append(time_index)

    
    # Include first and last time points
    all_avg_expressions = [np.mean(gene_expression_X1_normalized)] + intermediate_avg_expressions + [np.mean(gene_expression_X2_normalized)]
    all_ci_expressions = [
        stats.sem(gene_expression_X1_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X1_normalized) - 1)
    ] + intermediate_ci_expressions + [
        stats.sem(gene_expression_X2_normalized) * stats.t.ppf((1 + 0.95) / 2., len(gene_expression_X2_normalized) - 1)
    ]

        
    all_indices = [0] + intermediate_indices + [len(indices)]
    combined_indices = sorted([day1] + intermediate_t.tolist() + [day2])

    print(combined_indices)

    
    # Ensure extended_indices align with avg_gene_expressions
    extended_indices = np.array([x * index for x in range(len(avg_gene_expressions))])
    
    # Ensure all_indices and extended_indices are NumPy arrays
    combined_indices = np.array(combined_indices)
    extended_indices = np.array(extended_indices)
    
    # Linearly rescale all_indices to be equally distributed in extended_indices
    rescaled_indices = np.interp(
        combined_indices,  # Original indices
        [combined_indices[0], combined_indices[-1]],  # Range of all_indices
        [extended_indices[0], extended_indices[-1]]  # Range of extended_indices
    )




    
    # (1) Perform clustering on the last day's cell states from `mats`
    last_day = mats[day2]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=40)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_
    

    
    # Define colors for subgroups
    subgroup_colors = ["teal","magenta","orange","navy","gold"]
    unique_labels = np.unique(X1_hat_labels)
    subgroup_color_map = {label: subgroup_colors[i % len(subgroup_colors)] for i, label in enumerate(unique_labels)}
    
    # Define filename
    subgroup_output_file = img_src
    
    # (2) Initialize Storage for Mean and CI
    subgroup_avg_gene_expressions = {label: [] for label in unique_labels}
    subgroup_ci_gene_expressions = {label: [] for label in unique_labels}
    
    all_gene_expression_values_normalized_X1 = gene_expression_X1_trpts_normalized.copy()
    
    # (3) Compute Mean & Confidence Intervals
    for i, time_idx in enumerate(indices):
        if time_idx > max_i:  # Apply truncation
            break
        X1_trpt = X1_trpts[time_idx]
        if np.isnan(X1_trpt).any():
            break
    
        # Extract gene expression values
        X1_hat = pca.inverse_transform(X1_trpt)
        gene_expression_values = all_gene_expression_values_normalized_X1[:len(X1_hat)]
        all_gene_expression_values_normalized_X1 = all_gene_expression_values_normalized_X1[len(X1_hat):]
    
        # Compute subgroup averages & CI
        for label in unique_labels:
            mask = (X1_hat_labels == label)  # Use labels **only from step 1**
            subgroup_values = np.array(gene_expression_values)[mask]
    
            if len(subgroup_values) > 0:
                subgroup_avg_gene_expressions[label].append(np.mean(subgroup_values))
                ci = stats.sem(subgroup_values) * stats.t.ppf((1 + 0.95) / 2., len(subgroup_values) - 1)
                subgroup_ci_gene_expressions[label].append(ci)
            else:
                subgroup_avg_gene_expressions[label].append(np.nan)
                subgroup_ci_gene_expressions[label].append(np.nan)
    

            
    
    # (4) **Plot**
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # **Get x-axis positions for the line plot (scale to [0, 4])**
    num_points = len(next(iter(subgroup_avg_gene_expressions.values())))  # Number of time points
    x_positions = np.linspace(0, 4, num_points)  # Ensure correct x-spacing for trajectories
    
    # **Plot Predicted Trajectories & Confidence Intervals**
    subgroup_legend_handles = []  # Store for separate legend


    # **Plot Subgroup Averages & Confidence Intervals**
    for i, label in enumerate(unique_labels):
        # **Plot the Mean Trajectory Line**
        line, = ax1.plot(
            x_positions, subgroup_avg_gene_expressions[label], zorder=10,
            linestyle='-', color=subgroup_color_map[label], linewidth=2,
            label=f'Predicted Trajectory {i+1}'
        )
    
        # **Plot the Confidence Interval (Shaded Region)**
        ax1.fill_between(
            x_positions,
            np.array(subgroup_avg_gene_expressions[label]) - np.array(subgroup_ci_gene_expressions[label]),
            np.array(subgroup_avg_gene_expressions[label]) + np.array(subgroup_ci_gene_expressions[label]),
            alpha=0.2, zorder=5, color=subgroup_color_map[label],
            label=f'95% CI of Trajectory {i+1}'
        )
    
        # **Legend entry for Mean + Confidence Interval**
        ci_patch = mpatches.Patch(
            color=subgroup_color_map[label], alpha=0.2, label=f'95% CI of Trajectory {i+1}'
        )
    
        # **Store in Legend Handles**
        subgroup_legend_handles.append(ci_patch)
        subgroup_legend_handles.append(line)
        
    # (5) **Ensure Violin Plots are at `[0, 2, 4]`**
    violin_data = [
        gene_expression_X1_normalized,
        *gene_expression_intermediates_normalized,
        gene_expression_X2_normalized
    ]
    
    # **Manually set violin plot positions to `[0, 2, 4]`**
    violin_x_positions = np.array([0, 1, 2, 3, 4])  # Explicitly define positions
    violin_colors = ["black", "gray", "black", "gray", "black"]  # Set distinct colors
    
    # 🎻 **Plot Violin Plots One-by-One to Force Correct Positioning**
    for i, (x_pos, data, color) in enumerate(zip(violin_x_positions, violin_data, violin_colors)):
        violin_parts = sns.violinplot(
            data=[data],  # Must be wrapped in a list to avoid merging violins
            ax=ax1,
            inner=None,
            linewidth=1.2,
            width=0.7,
            cut=0,
            scale="width",
            color=color,  # ✅ Assign distinct colors
            alpha=0.8,  # ✅ MAKE TRANSPARENT
            zorder=3  # ✅ BRINGS VIOLINS TO THE FRONT
        )
        
        # **Manually Adjust X-Position of Each Violin**
        for violin in ax1.collections[-1:]:  # Only adjust the last added violin
            for path in violin.get_paths():
                path.vertices[:, 0] += x_pos - path.vertices[:, 0].mean()  # Move to correct x-location
    
    # **Expand x-axis limits to prevent cutting off last violin plot**
    ax1.set_xlim(-0.5, 4.5)  # ✅ Extend range
    
    # 🛠 **Fix x-axis labels and ensure proper alignment**
    ax1.set_xticks([0, 1, 2, 3, 4])  # ✅ Force labels at `[0, 2, 4]`
    ax1.set_xticklabels([0, 1, 2, 3, 4], fontsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    
    ax1.set_xlabel('Time', fontsize=35)
    ax1.set_ylabel('Gene Expression', fontsize=35)
    ax1.set_title(f'Subtrajectory {gene_of_interest} Expression', fontsize=34)
    
    # 🎨 **Violin Plot Legend**
    violin_legend_patches = [
        mpatches.Patch(color="black", label="Input Data"),
        mpatches.Patch(color="gray", label="Test Data")
    ]
    
    # 🎨 **Create Separate Legend Figure (VERTICAL LAYOUT)**
    fig_legend, ax_legend = plt.subplots(figsize=(4, 8))  # Tall aspect ratio for vertical layout
    ax_legend.axis("off")  # Hide axes
    
    # **Combine both legends**
    combined_legend = subgroup_legend_handles + violin_legend_patches
    
    ax_legend.legend(
        handles=combined_legend,
        loc="center", fontsize=18, title="Trajectories & Violin Plots",
        title_fontsize=18, ncol=1, frameon=True, handletextpad=1.5, columnspacing=2
    )
    
    # Save the separate legend
    legend_output_file = subgroup_output_file.replace(".png", "_legend.png")
    plt.savefig(legend_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🎨 **Save the main figure without a legend**
    plt.savefig(subgroup_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Subgroup trajectory plot saved at: {subgroup_output_file}")
    print(f"Legend plot saved separately at: {legend_output_file}")




