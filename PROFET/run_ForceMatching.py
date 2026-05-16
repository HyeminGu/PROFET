#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import argparse
import time
import tensorflow as tf

t_init = time.time()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--ts', type=float, nargs='+')
parser.add_argument('--files', type=str, nargs='+')
parser.add_argument('--hidden_units', type=int, nargs='+', default=[64, 64, 64, 64])
parser.add_argument('--mini_batch_size_t', type=int, help='mini batch size for time for each time interval', default=25)
parser.add_argument('--mini_batch_size_x', type=int, default=1000)
parser.add_argument('--limsup_window', type=int, help='number of future iterations to monitor for loss convergence', default=500)
parser.add_argument('--exp_memo', type=str, default='test')
parser.add_argument('--lr_NN', type=float, default=0.00005)
parser.add_argument('--iterations', type=int, default=20000)
parser.add_argument('--seed', type=int, default=42)
p, _ = parser.parse_known_args()
p = vars(p)

np.random.seed(p['seed'])
tf.random.set_seed(p['seed'])



# Load data
current_filename = __file__
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_filename)))
assets_dir = os.path.join(_root_dir, 'assets', '')

vectorfields = []
trajectories = []
gpa_Ts = [0.0]


# calculate stopping time
def rolling_stopping_time(records, mode='liminf', window=p['limsup_window']):
    mask = np.argmax(records)
    max_val = np.max(records)
    mask = np.where(records >= max_val-1e-1)[0][-1]
    valid_records = records[mask:]
    N = len(valid_records)
    if mode == 'liminf':
        roll_vals = [min(valid_records[t:min(t+window, N)]) for t in range(N)]
        compare = lambda r, roll: r <= roll
    elif mode == 'limsup':
        roll_vals = [max(valid_records[t:min(t+window, N)]) for t in range(N)]
        compare = lambda r, roll: r >= roll
    else:
        raise ValueError("Mode must be 'liminf' or 'limsup'.")

    for t, (r, roll) in enumerate(zip(valid_records, roll_vals)):
        if compare(r, roll):
            return t + mask
    return N-1 + mask  # fallback

# read GPA results
for i, file in enumerate(p['files']):
    filename = assets_dir + p['dataset'] + '/' + file
    
    with open(filename, "rb") as fr:
        param, result = pickle.load(fr)
        
    error_val = np.array(result['KE_Ps'])
        
    converged_index = rolling_stopping_time(np.abs(error_val))
    vectorfields.append(result['vectorfields'][:int(converged_index/param['save_iter'])])
    trajectories.append([param['Y_']] + result['trajectories'][:int(converged_index/param['save_iter'])-1])
    gpa_Ts.append(gpa_Ts[-1] + converged_index * param['lr_P'])
    # NOTE: unlike KE_Ps, vectorfields and trajectories are saved less frequently, once per save_iter

# --------
# time conversion
if len(p['files']) > 1:  # multiple GPA results to be combined
    max_ratio = 0.0
    ratios = []
    for i in range(len(p['ts'])-1):
        r = (gpa_Ts[i+1] - gpa_Ts[i])/(p['ts'][i+1] - p['ts'][i])
        ratios.append(r)
        max_ratio = np.maximum(max_ratio, r)
        
    numerical_ts = [0.0]
    for i in range(len(p['ts'])-1):
        numerical_ts.append(max_ratio*p['ts'][i+1])
        ratios[i] /= max_ratio
                
    print(f"GPA time = {gpa_Ts}, Ratios between GPA times = {ratios}")
    print(f"Physical time = {p['ts']}, Converted time = {numerical_ts}")
else:
    ratios = [1.0]
    numerical_ts = gpa_Ts
    
    print(f"GPA time = {gpa_Ts}, Physical time = {p['ts']}, Converted time = {numerical_ts}")
    
space_dim = vectorfields[0][0].shape[1]

p['numerical_ts'] = numerical_ts
p['space_dim'] = space_dim

# Velocityfield Network v(x,t)
from models.velocityfield import VelocityField
velocity_net = VelocityField(space_dim, p['hidden_units'], p['lr_NN'], p['seed'])

# Mini-batch function: return true_v, x, t
def mini_batch(vf_ts, x_ts, ts = numerical_ts, ratios = ratios, n_t = p['mini_batch_size_t'], n_x = p['mini_batch_size_x']):
    # vf_ts / x_ts: nested list of velocity / position evaluations at different time points
    #               primary level) different GPA runs in time order
    #               secondary level) velocities / positions at different time steps for each GPA run
    # ts: list of physical time points for v
    # c: real constant that is a multiplier of velocity
    # n_t: number of different time t to be sampled
    # n_x = all samples (= number of samples alloted to the same time t)
    true_vs, true_xs, true_ts= [], [], []

    for i, (vf_t, x_t) in enumerate(zip(vf_ts, x_ts)):
        dt =  (ts[i+1] - ts[i])/ len(vf_t)  # time step size for adapted to time horizon of v(x,t)
        if len(vf_t) > n_t:
            t_ids = np.random.choice(len(vf_t), n_t, replace=False) # sample n_t time points
        else:
            t_ids = np.arange(len(vf_t))
        
        if vf_t[i].shape[0] > n_x:
            x_ids = np.random.choice(vf_t[i].shape[0], n_x, replace=False) # sample n_x spatial points
        else:
            x_ids = np.arange(vf_t[i].shape[0])
        
        t = dt * t_ids + ts[i]   #  sampled time for v(x,t)
        # return their corresponding velocity, x(t), t values
        for j, t_id in enumerate(t_ids):
            true_vs.append(ratios[i] * vf_t[t_id][x_ids])
            true_xs.append(x_t[t_id][x_ids])
            true_ts.append(np.ones([true_xs[-1].shape[0],1]) * t[j])
            
        
    return np.vstack(true_vs), np.vstack(true_xs), np.vstack(true_ts)


# Training loop
for it in range(1,p['iterations']+1):
    true_v, x, t = mini_batch(vectorfields, trajectories)

    # Convert to tensors
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    true_v = tf.convert_to_tensor(true_v, dtype=tf.float32)

    loss = velocity_net.train(x, t, true_v)
    
    if it % int(p['iterations']/10) == 0:
        print(f"Iteration {it}, Loss: {loss.numpy():.6f}")#, Smoothness: {smoothness_loss.numpy():.6f}")
    

# Save model for evaluation
save_dir = f"{assets_dir}{p['dataset']}/{p['exp_memo']}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save model weights
velocity_net.save(save_dir)

with open(f"{save_dir}hyperparameters.pickle", "wb") as f:
    pickle.dump(p, f)

print("Results saved at:", save_dir)


print(f"Total time: {time.time()-t_init} sec")
