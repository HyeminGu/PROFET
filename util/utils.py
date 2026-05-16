import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import decomposition
import argparse
import pickle
import torch
import os
import threading
import time as _time

try:
    from geomloss import SamplesLoss
except ImportError:
    import subprocess
    subprocess.run(["pip3", "install", "geomloss"])
    from geomloss import SamplesLoss


# ── Constants ──────────────────────────────────────────────────────────────────

# Canonical path to the shared data directory (benchmarks/data/)
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Default color palette indexed by time point value
contrast_colors = {
    0: "#1f77b4",  # blue
    1: "#2ca02c",  # green
    2: "#ff7f0e",  # orange
    3: "#8c564b",  # brown
    4: "#d62728",  # red
    6: "#17becf",  # cyan / teal
    8: "#9467bd",  # purple
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse common benchmark command-line arguments.

    Returns an argparse.Namespace with:
      --dataset           : str   dataset name
      --d_red             : int   PCA dimension (optional)
      --days              : list  observed time points (comma-separated)
      --intermediate_days : list  held-out time points (comma-separated)
    """
    parser = argparse.ArgumentParser(description="Genetic simulation argument parser")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or path."
    )
    parser.add_argument(
        "--d_red",
        type=int,
        default=None,
        help="Dimension reduction size (optional)."
    )
    parser.add_argument(
        "--days",
        type=lambda s: [int(x) for x in s.split(",")],
        required=True,
        help="Comma-separated list of days, e.g. --days 1,2,3"
    )
    parser.add_argument(
        "--intermediate_days",
        type=lambda s: [int(x) for x in s.split(",")],
        required=True,
        help="Comma-separated list of intermediate days, e.g. --intermediate_days 5,10,15"
    )

    return parser.parse_args()


# ── Data I/O ──────────────────────────────────────────────────────────────────

def load_preprocessed_data(example_name):
    """Load a preprocessed dataset pickle from data/.

    Parameters
    ----------
    example_name : str
        Dataset identifier (e.g. 'EMT_72', 'stem_cell_differentiation').

    Returns
    -------
    (time_label, full_matrix, projected_matrix, pca) if PCA data was stored,
    (time_label, full_matrix) otherwise.
    """
    path = os.path.join(_DATA_DIR, f"{example_name}_preprocessed.pkl")
    with open(path, "rb") as fr:
        data = pickle.load(fr)
    if len(data) == 4:
        return data  # time_label, full_matrix, projected_matrix, pca
    else:
        return data  # time_label, full_matrix


def save_preprocessed_data(data, example_name):
    """Save preprocessed data as <example_name>_preprocessed.pkl in data/.

    Parameters
    ----------
    data : dict
        Must contain 'time_label' and 'full_matrix'. Optionally 'projected_matrix'
        and 'pca' for PCA-reduced datasets.
    example_name : str
        Dataset identifier used as the filename stem.
    """
    time_label = data['time_label']
    full_matrix = data['full_matrix']
    out_path = os.path.join(_DATA_DIR, f"{example_name}_preprocessed.pkl")
    try:
        projected_matrix = data['projected_matrix']
        pca = data['pca']
        pickle.dump([time_label, full_matrix, projected_matrix, pca], open(out_path, "wb"))
    except KeyError:
        pickle.dump([time_label, full_matrix], open(out_path, "wb"))


def reduce_dimension(full_matrix, d_reds, example_name):
    """Fit PCA on full_matrix and save a cumulative variance ratio plot to data/.

    Parameters
    ----------
    full_matrix : np.ndarray, shape (n_samples, n_features)
    d_reds : list of int
        Candidate reduced dimensions to annotate on the plot. Pass [] to skip.
    example_name : str
        Dataset identifier; used for the output plot filename.

    Returns
    -------
    pca : sklearn PCA fitted on full_matrix
    projected_matrix : np.ndarray — full_matrix projected by pca
    """
    if d_reds == []:
        print("No dimension reduction")
        return

    dimension = full_matrix.shape[1]

    pca = decomposition.PCA(n_components=dimension, random_state=0)
    pca.fit(full_matrix)
    projected_matrix = pca.transform(full_matrix)

    variances = pca.explained_variance_ratio_ * 100

    plt.plot(range(1, len(variances) + 1), np.cumsum(variances), '-')
    plt.xlabel("number of PCA basis", fontsize=16)
    plt.ylabel("Explained variance (%)", fontsize=16)

    for d_red in d_reds:
        plt.vlines(x=d_red, ymin=variances[0], ymax=np.sum(variances), color='r', linestyles='dashed')
        plt.text(x=d_red - 0.01 * dimension, y=np.sum(variances[:d_red]) - 3,
                 s="$d'=%d$\n%.2f %%" % (d_red, np.sum(variances[:d_red])), fontsize=12)

    plt.ylim([variances[0], np.sum(variances)])
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(os.path.join(_DATA_DIR, f"{example_name}_pca_variance_ratios.png"))
    plt.show()

    return pca, projected_matrix


def save_trajectories(X1_trpts, save_path):
    """Save a list of trajectory snapshots to a pickle file.

    Parameters
    ----------
    X1_trpts : list of np.ndarray
        Trajectory snapshots, one array per time step.
    save_path : str
        Destination .pkl path (parent directory is created if needed).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(X1_trpts, f)
    print(f"Trajectories saved to {save_path}")


# ── Metrics ────────────────────────────────────────────────────────────────────

def W2(X, Y):
    """Compute the Wasserstein-2 distance between sample sets X and Y via Sinkhorn."""
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(Y, dtype=np.float32))
    return SamplesLoss(loss="sinkhorn", p=2)(X, Y).numpy()


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_data(full_matrix, time_label, colors={}):
    """Scatter-plot each time point in PCA axes 1–2, one figure per time point.

    All time points are shown as a light-gray background; the highlighted time
    point is drawn in its assigned color.

    Parameters
    ----------
    full_matrix : np.ndarray, shape (n_samples, n_features)
    time_label : np.ndarray
        Time label for each row of full_matrix.
    colors : dict, optional
        Map from time point value to matplotlib color. Randomly assigned if empty.
    """
    pca = decomposition.PCA(n_components=2, random_state=0)
    pca.fit(full_matrix)
    projected_matrix_2D = pca.transform(full_matrix)

    time_points = sorted(set(time_label))
    if colors == {}:
        np.random.seed(127)
        for time in time_points:
            colors[time] = np.random.random(3)

    for time in time_points:
        plt.figure(figsize=(4, 3))
        plt.scatter(projected_matrix_2D[:, 0], projected_matrix_2D[:, 1],
                    color='lightgray', alpha=0.3, s=0.7)  # background
        vis_time = projected_matrix_2D[time_label == time, :]
        plt.scatter(vis_time[:, 0], vis_time[:, 1],
                    alpha=1.0, color=colors[time], s=0.7, label=f'time {time}')  # highlighted
        plt.legend()
        plt.show()


def _compute_time_grid(days, intermediate_days, X1_trpts):
    """Derive a uniform time grid spanning all observed and intermediate days.

    Parameters
    ----------
    days : list of int/float           observed time points
    intermediate_days : list of int/float  held-out time points
    X1_trpts : list of np.ndarray      trajectory frames (at least 2)

    Returns
    -------
    all_days, dt, physical_dt, t_start, t_end
    """
    all_days = sorted(set(days + intermediate_days))
    if len(all_days) == 0:
        raise ValueError("days and intermediate_days cannot both be empty.")
    if len(X1_trpts) < 2:
        raise ValueError("X1_trpts must contain at least 2 trajectory frames.")

    dt = 1.0 / (len(X1_trpts) - 1)
    t_start = min(all_days)
    t_end = max(all_days)
    physical_dt = (t_end - t_start) * dt

    return all_days, dt, physical_dt, t_start, t_end


def generate_animation(
    dataset,
    days,
    intermediate_days,
    X1_trpts,
    img_src,
    d_red,
    dimension_reduction,
    colors=contrast_colors,
    plot_vectorfield=False,
):
    """Save a GIF animating particle trajectories against reference snapshots.

    Each frame shows the transported particles at one trajectory step overlaid
    with reference day snapshots. When plot_vectorfield=True, velocity quivers
    are drawn instead of particle scatter.

    Parameters
    ----------
    dataset : str             dataset name (used to load the preprocessed pkl)
    days : list               observed time points to overlay
    intermediate_days : list  held-out time points to overlay in gray
    X1_trpts : list of np.ndarray  trajectory frames
    img_src : str             output GIF path
    d_red : int               PCA dimension (unused directly, kept for API parity)
    dimension_reduction : bool  if False, apply pca.transform before plotting
    colors : dict             color map per time point
    plot_vectorfield : bool   if True, draw quivers instead of scatter
    """
    path = os.path.join(_DATA_DIR, f"{dataset}_preprocessed.pkl")
    with open(path, "rb") as fr:
        try:
            time_label, full_matrix, projected_matrix, pca = pickle.load(fr)
        except Exception:
            time_label, full_matrix = pickle.load(fr)
            projected_matrix, pca = None, None

    all_days, dt, physical_dt, t_start, _ = _compute_time_grid(
        days, intermediate_days, X1_trpts
    )

    time_points = sorted(set(time_label))
    if colors == {}:
        np.random.seed(127)
        for day in time_points:
            colors[day] = np.random.random(3)

    fig, ax = plt.subplots()
    ims = []

    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            break

        X1_trpt_vis = pca.transform(X1_trpt) if dimension_reduction is False else X1_trpt

        if plot_vectorfield and i < len(X1_trpts) - 1:
            X1_trpt_vis_next = (
                pca.transform(X1_trpts[i + 1]) if dimension_reduction is False else X1_trpts[i + 1]
            )
            vs_vis = (X1_trpt_vis_next - X1_trpt_vis) / dt
            im = ax.quiver(
                X1_trpt_vis[:, 0], X1_trpt_vis[:, 1],
                vs_vis[:, 0], vs_vis[:, 1],
            )
        else:
            im = ax.scatter(
                X1_trpt_vis[:, 0], X1_trpt_vis[:, 1],
                color=colors[days[0]], alpha=1.0, s=0.7, zorder=20, label=f"day {days[0]}",
            )
            for t in days[1:]:
                X2_vis = projected_matrix[time_label == t, :2]
                ax.scatter(
                    X2_vis[:, 0], X2_vis[:, 1],
                    color=colors[t], alpha=1.0, s=0.7, zorder=10, label=f"day {t}",
                )
            for t in intermediate_days:
                X1_intermediate_vis = projected_matrix[time_label == t, :2]
                ax.scatter(
                    X1_intermediate_vis[:, 0], X1_intermediate_vis[:, 1],
                    color="lightgray", alpha=0.3, s=0.7, zorder=1, label=f"day {t}",
                )

        ttl = ax.text(
            0.5, 1.05,
            "t = %.3f" % (t_start + physical_dt * i),
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=ax.transAxes, ha="center",
        )
        ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=200)
    ani.save(img_src, writer=animation.PillowWriter(fps=3))
    plt.clf()


def generate_W2distance_plot(
    dataset,
    days,
    intermediate_days,
    X1_trpts,
    img_src,
    d_red,
    dimension_reduction,
    colors=contrast_colors,
    max_points_for_w2=2000,
):
    """Plot W2 distance between each trajectory frame and each reference day snapshot.

    Skips computation if any sample count exceeds max_points_for_w2, since
    Sinkhorn is expensive at large scale.

    Parameters
    ----------
    dataset : str             dataset name (used to load the preprocessed pkl)
    days : list               observed time points
    intermediate_days : list  held-out time points
    X1_trpts : list of np.ndarray  trajectory frames
    img_src : str             output plot path
    d_red : int               number of PCA dims to use when dimension_reduction is True
    dimension_reduction : bool  if False, use full_matrix; if True, use projected_matrix[:d_red]
    colors : dict             color map per time point
    max_points_for_w2 : int   sample size threshold above which W2 is skipped

    Returns
    -------
    list of float : trajectory time at which W2 is minimised for each day,
    or None if computation was skipped.
    """
    path = os.path.join(_DATA_DIR, f"{dataset}_preprocessed.pkl")
    with open(path, "rb") as fr:
        try:
            time_label, full_matrix, projected_matrix, pca = pickle.load(fr)
        except Exception:
            time_label, full_matrix = pickle.load(fr)
            projected_matrix, pca = None, None

    all_days, dt, physical_dt, t_start, _ = _compute_time_grid(
        days, intermediate_days, X1_trpts
    )

    time_points = sorted(set(time_label))
    if colors == {}:
        np.random.seed(127)
        for day in time_points:
            colors[day] = np.random.random(3)

    rng = np.random.default_rng(0)

    def _maybe_subsample(arr, n):
        if arr.shape[0] > n:
            idx = rng.choice(arr.shape[0], n, replace=False)
            return arr[idx]
        return arr

    max_traj_size = max(X.shape[0] for X in X1_trpts)
    ref_sizes = {
        day: (full_matrix if dimension_reduction is False else projected_matrix)[time_label == day].shape[0]
        for day in all_days
    }
    max_size = max(max_traj_size, max(ref_sizes.values()) if ref_sizes else 0)

    if max_size > max_points_for_w2:
        print(
            f"Sample size ({max_size}) exceeds threshold ({max_points_for_w2}). "
            f"Subsampling to {max_points_for_w2} points for W2 computation."
        )

    argmin_w2_s = []
    for day in all_days:
        reference_mat = full_matrix[time_label == day, :] if dimension_reduction is False \
            else projected_matrix[time_label == day, :d_red]
        traj_d = X1_trpts[0].shape[1] if X1_trpts else reference_mat.shape[1]
        ref_d = min(reference_mat.shape[1], traj_d)
        reference_sub = _maybe_subsample(reference_mat[:, :ref_d], max_points_for_w2)
        w2_s = [W2(_maybe_subsample(X[:, :ref_d], max_points_for_w2), reference_sub).item() for X in X1_trpts]
        plt.plot(
            t_start + np.arange(len(X1_trpts)) * physical_dt,
            w2_s, color=colors[day], label="Day" + str(day),
        )
        argmin_w2_s.append(t_start + np.argmin(w2_s) * physical_dt)

    plt.legend()
    print("minimum values are obtained at x=", argmin_w2_s)
    plt.title(r"$W_2(P_n, Q_{day})$")
    plt.savefig(img_src)
    plt.close()

    return argmin_w2_s


# ── Resource Monitoring ───────────────────────────────────────────────────────

class ResourceMonitor:
    """Context manager that measures wall-clock training time, peak GPU memory,
    and peak CPU RAM.

    Usage::

        with ResourceMonitor() as monitor:
            train(...)
        monitor.report("MyModel training")

    Attributes set after the ``with`` block exits
    -----------------------------------------------
    elapsed_s       : float  – wall-clock seconds
    peak_gpu_bytes  : int    – peak GPU memory allocated (bytes); 0 if no CUDA
    peak_cpu_bytes  : int    – peak process RSS (bytes); 0 if psutil unavailable
    """

    def __init__(self, poll_interval: float = 0.5):
        self._poll_interval = poll_interval

    def _poll(self):
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            while not self._stop.is_set():
                try:
                    rss = proc.memory_info().rss
                    if rss > self._peak_cpu_bytes:
                        self._peak_cpu_bytes = rss
                except Exception:
                    pass
                self._stop.wait(self._poll_interval)
        except ImportError:
            pass  # psutil not installed; peak_cpu_bytes stays 0

    def __enter__(self):
        self._start = _time.time()
        self._peak_cpu_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.elapsed_s = _time.time() - self._start
        self.peak_gpu_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        self.peak_cpu_bytes = self._peak_cpu_bytes

    def report(self, label: str = "Training", dataset: str = None, d_red: int = None, filename: str = None):
        """Print (and optionally append to file) a resource usage summary."""
        gpu_str = (
            f"{self.peak_gpu_bytes / 1e9:.3f} GB" if self.peak_gpu_bytes > 0
            else "N/A (no CUDA device)"
        )
        cpu_str = (
            f"{self.peak_cpu_bytes / 1e9:.3f} GB" if self.peak_cpu_bytes > 0
            else "N/A (psutil not installed)"
        )
        lines = [
            f"[{label}] Wall-clock time : {self.elapsed_s:.2f} s",
            f"[{label}] Peak GPU memory : {gpu_str}",
            f"[{label}] Peak CPU RAM    : {cpu_str}",
        ]
        if dataset is not None:
            lines.append(f"[{label}] Dataset         : {dataset}")
        if d_red is not None:
            lines.append(f"[{label}] Dimensionality  : {d_red}")
        for line in lines:
            print(line)
        if filename is not None:
            with open(filename, "a") as f:
                f.write("\n".join(lines) + "\n")

    def as_dict(self) -> dict:
        return {
            "elapsed_s": self.elapsed_s,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "peak_cpu_bytes": self.peak_cpu_bytes,
        }
