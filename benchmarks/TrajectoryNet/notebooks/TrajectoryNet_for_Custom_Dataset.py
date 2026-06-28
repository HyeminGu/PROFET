#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import subprocess
from ...util.utils import * 
import time

# Load the preprocessed dataset from our Preprocess_datasets notebook

args = parse_args()

preprocessed_filename = f"../data/{args.dataset}_preprocessed.pkl"
with open(preprocessed_filename, "rb") as fr:
    try:
        time_label, full_matrix, projected_matrix, pca = pickle.load(fr)
    except:
        time_label, full_matrix = pickle.load(fr)

# we primarily use (PCA-) projected_matrix unless we apply TrajectoryNet on the original datasets
projected_matrix = np.array(projected_matrix, dtype=np.float32)
time_label = np.array(time_label, dtype=np.float32)



# Input variables for TrajectoryNet: embedding_matrix, sample_labels -> npz
npzname = f'../data/{args.dataset}.npz'

embedding_matrix = []
sample_labels = []
for day in args.days:
    embedding_matrix.append(projected_matrix[time_label==day, :args.d_red])
    sample_labels.append((1+day)*np.ones(embedding_matrix[-1].shape[0]))

embedding_matrix = np.concatenate(embedding_matrix, axis=0)
sample_labels = np.concatenate(sample_labels, axis=0)
print(embedding_matrix.shape, sample_labels.shape)

np.savez(npzname, pca=embedding_matrix, sample_labels=sample_labels)



# Train TrajectoryNet
savedir = f"../results/{args.dataset}_dim{args.d_red}"
cmd = [
    "python3", "-m", "TrajectoryNet.main",
    "--dataset", npzname,
    "--save", savedir,
    "--embedding_name", "pca",
    "--niter", "10000",
    "--dims", "128-128-128-128",
    "--max_dim", str(args.d_red)
]
start_time = time.time()
end_time = time.time()
elapsed = end_time - start_time
print(f"Total wall clock runtime: {elapsed:.2f} seconds")

subprocess.run(cmd)



# Evaluate TrajectoryNet
cmd = [
    "python3", "-m", "TrajectoryNet.eval",
    "--dataset", npzname,
    "--save", savedir,
    "--embedding_name", "pca",
    "--dims", "128-128-128-128",
    "--max_dim", str(args.d_red)
]

subprocess.run(cmd)



# Load trajectory data
# trajectory is stored backward in time
savefile = f"{savedir}/backward_trajectories.npy"
traj = np.load(savefile) # shape: (T, N, D)

X1_trpts = [traj[-i-1] for i in range(traj.shape[0])]
dt = 1/(traj.shape[0]-1)
physical_dt = max(args.days) * dt



img_src1 = f"{savedir}/particle_trajectories.gif"
generate_animation(args.dataset, args.days, args.intermediate_days, X1_trpts, dt, physical_dt,
                       img_src1, args.d_red, dimension_reduction, plot_vectorfield = False)

img_src2 = f"{savedir}/velocity_trajectories.gif"
generate_animation(args.dataset, args.days, args.intermediate_days, X1_trpts, dt, physical_dt,
                       img_src2, args.d_red, dimension_reduction, plot_vectorfield = True)

img_src3 = f"{savedir}/w2distances.png"
generate_W2distance_plot(args.dataset, args.days, args.intermediate_days, X1_trpts, physical_dt, img_src3, args.d_red, dimension_reduction)




