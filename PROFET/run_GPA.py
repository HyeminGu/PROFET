#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow as tf
import numpy as np
import re
import sys
import pickle
import argparse
import yaml
from yaml.loader import SafeLoader

# ----------------------------------------
# Argument parsing
# ----------------------------------------
parser = argparse.ArgumentParser()

# A fixed set of configurations
parser.add_argument('--config', type=str, default='GPA')

# Dataset property
parser.add_argument('-N_Q', '--N_samples_Q', type=int, help='total number of target samples')
parser.add_argument('-N_P', '--N_samples_P', type=int, help='total number of prior samples')
parser.add_argument('--N_dim', type=int, help='dimension of input data')
parser.add_argument('--dimension_reduction', type=str, choices=['Y', 'N', 'yes', 'no'], help='true: load projected data; false: load original data')
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='Transport_genes')
parser.add_argument('--label', type=int, nargs='+', help='list of the indices of source and target days')
parser.add_argument('--rescale_factor', type=float, default=1.0, help='rescale data X by X*rescale_factor')

# (f, Gamma)-divergence
parser.add_argument('--f', type=str, choices=['KL', 'alpha', 'reverse_KL', 'reverse_alpha'])
parser.add_argument('-alpha', type=float, help='parameter value for alpha divergence')
parser.add_argument('--formulation', type=str, choices=['LT', 'DV'], help='LT or DV in case of f=KL, otherwise keep LT')
parser.add_argument('--Gamma', type=str, choices=['Lipshitz'], default='Lipshitz')
parser.add_argument('-L', type=float, help='Lipschitz constant: default=inf w/o constraint')
parser.add_argument('--reverse', type=bool, default=False, help='True -> D(Q|P), False -> D(P|Q)')

# Neural Network <phi>
parser.add_argument('-N_fnn_layers', type=int, nargs='+', help='list of the number of FNN hidden layer units')
parser.add_argument('--activation_ftn', type=str, nargs='+', choices=['relu', 'mollified_relu_cos3', 'mollified_relu_poly3', 'softplus', 'leaky_relu', 'elu', 'bounded_relu', 'bounded_elu'], help='[0]: hidden layers, [1]: last hidden layer')
parser.add_argument('-eps', type=float, default=0.5, help='Mollifier shape parameter for mollified relu activations')
parser.add_argument('--N_conditions', type=int, default=1, help='number of classes for the conditional setting')

# Discriminator training parameters
parser.add_argument('--lr_phi', type=float, help='lr for phi')
parser.add_argument('-ep_phi', '--epochs_phi', type=int, help='# updates for phi to find phi*')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], help='optimizer for NN')

# Particles transportation parameters
parser.add_argument('--ode_solver', type=str, choices=['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5', 'ABM1', 'Heun', 'ABM2', 'ABM3', 'ABM4', 'ABM5', 'RK4'], help='ode solver for particle ode')
parser.add_argument('-ep', '--epochs', type=int, default=5000, help='maximum # updates for P')
parser.add_argument('-lr_P', type=float, help='lr for P')
parser.add_argument('--adjust_lr_P_epochs', type=str, choices=['Y', 'N', 'yes', 'no'], default='Y', help='adjust epochs and lr_P w.r.t. N_dim and support size')
parser.add_argument('--f_Lip_threshold', type=float, default=0.05, help='threshold f_Lip value for termination')
parser.add_argument('--exp_no', type=str, default='test', help='short experiment name under the same data')

# Save settings
parser.add_argument('--save_iter', type=int, help='save results per each save_iter')

p, _ = parser.parse_known_args()
p = vars(p)

# ----------------------------------------
# Load YAML config (overrides defaults)
# ----------------------------------------
yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{p['config']}.yaml")
with open(yaml_file, 'r') as f:
    yaml_params = yaml.load(f, Loader=SafeLoader)

param = p.copy()
for key, yaml_val in yaml_params.items():
    param[key] = yaml_val

if param['alpha']:
    par = [param['alpha']]
    param['exptype'] = '%s=%05.2f-%s' % (param['f'], param['alpha'], param['Gamma'])
else:
    par = []
    param['exptype'] = '%s-%s' % (param['f'], param['Gamma'])
if param['L'] == None:
    param['expname'] = '%s_%s' % (param['exptype'], 'inf')
else:
    param['expname'] = '%s_%.4f' % (param['exptype'], param['L'])
if param['reverse'] == True:
    param['expname'] += '_reverse'

# set random seed for reproducibility
tf.random.set_seed(param['random_seed'])
np.random.seed(param['random_seed'])

# Data loading ----------------------------------------
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(_root_dir, 'data', '')

def load_dataset(example_name, N_dim, dimension_reduction, time_points):
    filename = data_dir + example_name + '_preprocessed.pkl'
    with open(filename, "rb") as fr:
        data = pickle.load(fr)
    if dimension_reduction in ['Y', 'yes']:
        if len(data) == 2:
            print("Check parameter --dimension_reduction. Projected matrix cannot be found.")
        else:
            time_labels, _, samples, _ = data
    else:
        if len(data) == 2:
            time_labels, samples = data
        else:
            time_labels, samples, _, _ = data
    time_points = sorted(time_points)
    Y_ = samples[time_labels==time_points[0], :N_dim]
    X_ = samples[time_labels==time_points[1], :N_dim]
    return Y_, X_

X_label, Y_label = None, None

if param['dataset'] == 'Toy1':
    if "true_target" in param['exp_no']:
        X_ = np.random.normal(loc=4.0, scale=1.0, size=(param['N_samples_Q'], param['N_dim']))
    else:
        load_filename = data_dir + "ou_result_%s.pickle" % str(param['exp_no'])
        with open(load_filename, "rb") as fr:
            X_ = pickle.load(fr)[0]
    Y_means, Y_stds = [0.0, 4.0], [0.5, 1.0]
    idx_0 = np.random.random((param['N_samples_P'], 1)) > 2/3
    Y_ = idx_0 * np.random.normal(loc=Y_means[0], scale=Y_stds[0], size=(param['N_samples_P'], param['N_dim'])) + \
        (1-idx_0) * np.random.normal(loc=Y_means[1], scale=Y_stds[1], size=(param['N_samples_P'], param['N_dim']))
else:
    labels = "_".join(str(l) for l in param['label'])
    if param['dimension_reduction'] in ['Y', 'yes']:
        param['expname'] = "%s-%stimes-dim%d" % (param['expname'], labels, param['N_dim'])
    else:
        param['expname'] = "%s-%stimes" % (param['expname'], labels)
    Y_, X_ = load_dataset(param['dataset'], param['N_dim'], param['dimension_reduction'], param['label'])

param['N_samples_Q'], param['N_samples_P'] = len(X_), len(Y_)

Q = tf.constant(X_ * param['rescale_factor'], dtype=tf.float32) # constant
P = tf.Variable(Y_ * param['rescale_factor'], dtype=tf.float32) # variable

if param['N_conditions'] > 1:
    Q_label = tf.constant(X_label, dtype=tf.float32)
    P_label = tf.constant(Y_label, dtype=tf.float32)
else:
    Q_label, P_label = None, None

data_par = {'P_label': P_label, 'Q_label': Q_label}

print("Data prepared.")


# Discriminator + particle transport ---------------------------------
from models.discriminator import GPA, compute_learning_rate_quantile_based, calc_ke

NN_par = {'activation_ftn': param['activation_ftn'], 'N_dim': param['N_dim'],
          'N_fnn_layers': [param['N_dim']] + param['N_fnn_layers'],
          'N_conditions': param['N_conditions'], 'L': param['L'], 'eps': param['eps']}

loss_par = {'f': param['f'], 'formulation': param['formulation'], 'par': par, 'reverse': param['reverse']}

lr_phi = tf.Variable(param['lr_phi'], trainable=False)

# Training setting: nondimensionalized learning rate ---------------
if param['adjust_lr_P_epochs'] in ['Y', 'yes']:
    param['epochs'] = int(param['epochs'] * np.sqrt(param['N_dim']))
    lr_P_init, _, _ = compute_learning_rate_quantile_based([X_, Y_], alpha=2/param['epochs'])
    param['lr_P'] = lr_P_init

lr_P = tf.Variable(param['lr_P'], trainable=False, dtype=tf.float32)
print(f"Example specific step size = {param['lr_P']}")

gpa = GPA(NN_par, loss_par, data_par, Q, lr_phi, param['epochs_phi'], param['optimizer'])

# Save settings -----------------------------------------------
trajectories = []
vectorfields = []
divergences = []
KE_Ps = []

if param['save_iter'] >= param['epochs']:
    param['save_iter'] = 1

assets_dir = os.path.join(_root_dir, 'assets', '')
if not os.path.exists(assets_dir + param['dataset']):
    os.makedirs(assets_dir + param['dataset'])

param['expname'] = param['expname'] + '-%04d_%04d-%02d-%s' % (param['N_samples_Q'], param['N_samples_P'], param['random_seed'], param['exp_no'])
filename = assets_dir + param['dataset'] + '/%s.pickle' % param['expname']


# Train ---------------------------------------------------------------
import time
t0 = time.time()

dPs = []
for it in range(1, param['epochs']+1):
    current_loss, dW_norm = gpa.train(P)
    dPs.append(gpa.calc_vectorfield(P))

    P, dPs, dP = gpa.step(P, lr_P, dPs, param['ode_solver'])

    divergences.append(current_loss)
    KE_P = calc_ke(dP, param['N_samples_P'])
    KE_Ps.append(KE_P)

    if it % param['save_iter'] == 0:
        trajectories.append(P.numpy() / param['rescale_factor'])
        vectorfields.append(-dP.numpy())

    if it % (param['epochs']/10) == 0:
        print('iter %6d: loss = %.10f, norm of dW = %.2f, kinetic energy of P = %.10f, average learning rate for P = %.6f' % (it, current_loss, dW_norm, KE_P, tf.math.reduce_mean(lr_P).numpy()))

    if it > param['save_iter'] and current_loss < param['f_Lip_threshold']:
        param['epochs'] = it + 1
        break

print(f'total time {time.time() - t0:.3f}s')


# Save result ------------------------------------------------------
if param['N_dim'] == 1:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)
    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]

if param['L'] == None:
    param['L'] = 'inf'
param.update({'X_': X_, 'Y_': Y_})
result = {'trajectories': trajectories, 'vectorfields': vectorfields, 'divergences': divergences, 'KE_Ps': KE_Ps}

with open(filename, "wb") as fw:
    pickle.dump([param, result], fw)
print("Results saved at:", filename)
