import tensorflow as tf
import numpy as np
from numpy import pi
from functools import partial


# ── Utilities ──────────────────────────────────────────────────────────────────

def compute_learning_rate_quantile_based(arrays, alpha=0.05, q_low=0.01, q_high=0.99):
    X_combined = np.concatenate(arrays, axis=0)
    entries = X_combined.flatten()
    S = np.quantile(entries, q_high) - np.quantile(entries, q_low)
    d = X_combined.shape[1]
    sqrt_d = np.sqrt(d)
    return alpha * S * sqrt_d, S, sqrt_d

def calc_ke(dP_dt, N_samples_P):
    return np.linalg.norm(dP_dt)**2 / N_samples_P / 2


# ── Activation functions ───────────────────────────────────────────────────────

def _bounded_relu(x):
    return tf.math.subtract(tf.nn.relu(x), tf.nn.relu(x - 1))

def _bounded_elu(x):
    return tf.math.subtract(tf.nn.elu(x), tf.nn.elu(x - 1))

def _mollified_relu_poly3(x):
    return (tf.cast(x >= 2, tf.float32) * (x - 1) +
            tf.cast(tf.logical_and(x >= 0, x < 2), tf.float32) *
            tf.math.multiply(tf.math.pow(x, 4), tf.math.pow(x, 2) - 6*x + 10) / 32)

def _mollified_relu_cos3(x, eps):
    return (tf.cast(x >= 2*eps, tf.float32) * (x - eps) +
            tf.cast(tf.logical_and(x >= 0, x < 2*eps), tf.float32) *
            (tf.math.pow(x, 2) / (4*eps) + eps * (tf.math.cos(pi*x/eps) - 1) / (2*pi**2)))

def _activation(name, eps):
    if name == 'relu':                  return tf.nn.relu
    elif name == 'leaky_relu':          return tf.nn.leaky_relu
    elif name == 'softplus':            return tf.math.softplus
    elif name == 'elu':                 return tf.nn.elu
    elif name == 'bounded_relu':        return _bounded_relu
    elif name == 'bounded_elu':         return _bounded_elu
    elif name == 'mollified_relu_poly3': return _mollified_relu_poly3
    elif name == 'mollified_relu_cos3': return partial(_mollified_relu_cos3, eps=eps)


# ── GPA class ─────────────────────────────────────────────────────────────────

class GPA:
    """Generative Particle Algorithm: discriminator training + particle transport."""

    def __init__(self, NN_par, loss_par, data_par, Q, lr_phi, epochs_phi, optimizer):
        self.NN_par = NN_par
        self.loss_par = loss_par
        self.data_par = data_par
        self.Q = Q
        self.lr_phi = lr_phi
        self.epochs_phi = epochs_phi
        self.optimizer_name = optimizer
        self._adam_state = None

        self.W, self.b = self._initialize_NN()
        self.nu = tf.Variable(0.0, dtype=tf.float32)

    # ── NN construction ────────────────────────────────────────────────────────

    def _initialize_NN(self):
        layers = self.NN_par['N_fnn_layers']
        N_cond = self.NN_par['N_conditions']
        W, b = [], []
        for l in range(len(layers) - 1):
            std = 1.0 / tf.sqrt(tf.cast(layers[l], tf.float32) / 2.)
            W.append(tf.Variable(tf.random.normal([layers[l], layers[l+1]], stddev=std), dtype=tf.float32))
            b.append(tf.Variable(tf.zeros([1, layers[l+1]]), dtype=tf.float32))
        std = 1.0 / tf.sqrt(tf.cast(layers[-1], tf.float32) / 2.)
        W.append(tf.Variable(tf.random.normal([layers[-1], N_cond], stddev=std), dtype=tf.float32))
        b.append(tf.Variable(tf.zeros([1, N_cond]), dtype=tf.float32))
        if self.NN_par['L'] is not None:
            for w in W:
                self._spectral_normalize(w, len(W))
        return W, b

    def _spectral_normalize(self, w, n):
        norm = tf.norm(w, 2)
        if norm < 1e-6:
            print('WARNING: Norm of W is too small')
        w.assign(tf.math.scalar_mul(self.NN_par['L']**(1/n) / norm, w))

    def _fnn(self, x, x_label):
        W, b, NN_par = self.W, self.b, self.NN_par
        act = _activation(NN_par['activation_ftn'][0], NN_par['eps'])
        try:
            act2 = _activation(NN_par['activation_ftn'][1], NN_par['eps'])
        except IndexError:
            act2 = act
        h = x
        for l in range(len(W) - 2):
            h = act(tf.add(tf.matmul(h, W[l]), b[l]))
        h = act2(tf.add(tf.matmul(h, W[-2]), b[-2]))
        out = tf.add(tf.matmul(h, W[-1]), b[-1])
        if NN_par['N_conditions'] > 1:
            out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
        return out

    # ── Loss / divergence ──────────────────────────────────────────────────────

    def _f_star(self, g):
        f = self.loss_par['f']
        if f == 'KL':
            return tf.math.exp(g - 1)
        elif f == 'alpha':
            alpha = self.loss_par['par'][0]
            if alpha > 1:
                return 1/alpha * (1/(alpha-1) + tf.math.pow((alpha-1)*tf.nn.relu(g), alpha/(alpha-1)))
            else:
                return (1/alpha * tf.math.pow((1-alpha)*(tf.nn.relu(-g)+1e-6), -alpha/(1-alpha))
                        - 1/(alpha*(1-alpha)))
        elif f == 'reverse_alpha':
            alpha = self.loss_par['par'][0]
            return -1/alpha - 1/(alpha*(alpha-1)) * (tf.math.pow(alpha*tf.nn.relu(-g), alpha-1) - 1)
        elif f == 'reverse_KL':
            return -1 - tf.math.log(tf.nn.relu(-g) + 1e-6)
        elif f == 'JS':
            max_exp_g = tf.math.reduce_max(tf.math.exp(g))
            threshold = 0.01
            if max_exp_g > 2.0 - threshold:
                return -tf.math.log(2 - tf.math.exp(g) + threshold)
            else:
                return -tf.math.log(2 - tf.math.exp(g))

    def _E_phi(self, x, x_label):
        return tf.math.reduce_mean(self._fnn(x, x_label))

    def _E_fstar_phi(self, x, x_label):
        phi_vals = self._fnn(x, x_label)
        if self.loss_par['formulation'] == 'DV':
            max_phi = tf.math.reduce_max(phi_vals)
            return max_phi + tf.math.log(tf.math.reduce_mean(tf.math.exp(phi_vals - max_phi)))
        else:
            return tf.math.reduce_mean(self._f_star(phi_vals - self.nu)) + self.nu

    def divergence(self, P, Q):
        P_label, Q_label = self.data_par['P_label'], self.data_par['Q_label']
        if not self.loss_par['reverse']:
            return self._E_phi(P, P_label) - self._E_fstar_phi(Q, Q_label)
        else:
            return self._E_phi(Q, Q_label) - self._E_fstar_phi(P, P_label)

    # ── Backward (optimizer step) ──────────────────────────────────────────────

    def _adam_update(self, grad, m, v, step, beta1=0.9, beta2=0.999, eps=1e-8):
        grad = grad.numpy()
        if step == 0:
            m, v = np.zeros_like(grad), np.zeros_like(grad)
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        m_hat = m / (1 - beta1**(step+1))
        v_hat = v / (1 - beta2**(step+1))
        return m_hat / (np.sqrt(v_hat) + eps), m, v

    def _backward(self, dW, db, dnu, step, calc_dW_norm=False):
        """Gradient ascent step on discriminator parameters (maximizes divergence)."""
        lr = -self.lr_phi  # negative → ascent
        dW_norm = 0

        if self.optimizer_name == 'sgd':
            for l in range(len(self.W)):
                if calc_dW_norm:
                    dW_norm = max(dW_norm, tf.norm(dW[l]))
                self.W[l].assign(self.W[l] - lr * dW[l])
                if self.NN_par['L'] is not None:
                    self._spectral_normalize(self.W[l], len(self.W))
                if db[l] is not None:
                    self.b[l].assign(self.b[l] - lr * db[l])
            if dnu is not None:
                self.nu.assign(self.nu - self.lr_phi * dnu)

        elif self.optimizer_name == 'adam':
            if self._adam_state is None:
                n = len(self.W)
                self._adam_state = {
                    'm_W': [0]*n, 'v_W': [0]*n,
                    'm_b': [0]*n, 'v_b': [0]*n,
                    'm_nu': 0,    'v_nu': 0,
                }
            s = self._adam_state
            for l in range(len(self.W)):
                if calc_dW_norm:
                    dW_norm = max(dW_norm, tf.norm(dW[l]))
                dW_hat, s['m_W'][l], s['v_W'][l] = self._adam_update(dW[l], s['m_W'][l], s['v_W'][l], step)
                self.W[l].assign(self.W[l] - lr * dW_hat)
                if self.NN_par['L'] is not None:
                    self._spectral_normalize(self.W[l], len(self.W))
                if db[l] is not None:
                    db_hat, s['m_b'][l], s['v_b'][l] = self._adam_update(db[l], s['m_b'][l], s['v_b'][l], step)
                    self.b[l].assign(self.b[l] - lr * db_hat)
            if dnu is not None:
                dnu_hat, s['m_nu'], s['v_nu'] = self._adam_update(dnu, s['m_nu'], s['v_nu'], step)
                self.nu.assign(self.nu - lr * dnu_hat)

        return dW_norm

    # ── Forward (train discriminator) ──────────────────────────────────────────

    def train(self, P):
        """Train discriminator for self.epochs_phi steps. Returns (loss, dW_norm)."""
        self._adam_state = None
        for step in range(self.epochs_phi):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch([self.W, self.b, self.nu])
                loss = self.divergence(P, self.Q)
            dW, db, dnu = tape.gradient(loss, [self.W, self.b, self.nu])
            dW_norm = self._backward(dW, db, dnu, step, calc_dW_norm=(step == self.epochs_phi - 1))
        return loss.numpy(), dW_norm

    # ── Vectorfield ────────────────────────────────────────────────────────────

    def _first_variation(self, P):
        P_label = self.data_par['P_label']
        g = self._fnn(P, P_label)
        if not self.loss_par['reverse']:
            return g
        else:
            if self.loss_par['formulation'] == 'DV':
                g_max = tf.reduce_max(g)
                logsumexp = tf.math.log(tf.math.reduce_mean(tf.math.exp(g - g_max))) + g_max
                return -(logsumexp + tf.math.exp(g - logsumexp) - 1)
            else:
                return -(self._f_star(g - self.nu) + self.nu)

    @tf.function
    def calc_vectorfield(self, P):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(P)
            fv = self._first_variation(P)
        return tape.gradient(fv, P)

    # ── ODE solvers ────────────────────────────────────────────────────────────

    def _retrain_and_vf(self, P_pred):
        """Retrain discriminator at predicted position and return vectorfield."""
        self.train(P_pred)
        return self.calc_vectorfield(P_pred)

    def step(self, P, lr_P, dPs, ode_solver):
        """Advance particles one ODE step. Returns (P, dPs, dP)."""

        if ode_solver == 'forward_euler':
            vf = dPs[-1]
            P.assign(P - lr_P * vf)
            return P, [], vf

        elif ode_solver == 'AB2':
            if len(dPs) < 2:
                vf = dPs[-1]
            else:
                vf = (-dPs[0] + 3*dPs[1]) / 2
                dPs.pop(0)
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'AB3':
            if len(dPs) < 3:
                vf = dPs[-1]
            else:
                vf = (5*dPs[0] - 16*dPs[1] + 23*dPs[2]) / 12
                dPs.pop(0)
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'AB4':
            if len(dPs) < 4:
                vf = dPs[-1]
            else:
                vf = (-9*dPs[0] + 37*dPs[1] - 59*dPs[2] + 55*dPs[3]) / 24
                dPs.pop(0)
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'AB5':
            if len(dPs) < 5:
                vf = dPs[-1]
            else:
                vf = (251*dPs[0] - 1274*dPs[1] + 2616*dPs[2] - 2774*dPs[3] + 1901*dPs[4]) / 720
                dPs.pop(0)
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'ABM1':
            P_pred = P - lr_P * dPs[0]
            dPs.pop(0)
            vf = self._retrain_and_vf(P_pred)
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'Heun':
            P_pred = P - lr_P * dPs[-1]
            f_pred = self._retrain_and_vf(P_pred)
            vf = (dPs[0] + f_pred) / 2
            dPs.pop(0)
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'ABM2':
            if len(dPs) < 2:
                vf = dPs[-1]
            else:
                P_pred = P - lr_P * (-dPs[0] + 3*dPs[1]) / 2
                dPs.pop(0)
                f_pred = self._retrain_and_vf(P_pred)
                vf = (dPs[0] + f_pred) / 2
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'ABM3':
            if len(dPs) < 3:
                vf = dPs[-1]
            else:
                P_pred = P - lr_P * (5*dPs[0] - 16*dPs[1] + 23*dPs[2]) / 12
                dPs.pop(0)
                f_pred = self._retrain_and_vf(P_pred)
                vf = (-dPs[0] + 8*dPs[1] + 5*f_pred) / 12
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'ABM4':
            if len(dPs) < 4:
                vf = dPs[-1]
            else:
                P_pred = P - lr_P * (-9*dPs[0] + 37*dPs[1] - 59*dPs[2] + 55*dPs[3]) / 24
                dPs.pop(0)
                f_pred = self._retrain_and_vf(P_pred)
                vf = (dPs[0] - 5/24*dPs[1] + 19*dPs[2] + 9*f_pred) / 24
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'ABM5':
            if len(dPs) < 5:
                vf = dPs[-1]
            else:
                P_pred = P - lr_P * (251*dPs[0] - 1274*dPs[1] + 2616*dPs[2] - 2774*dPs[3] + 1901*dPs[4]) / 720
                dPs.pop(0)
                f_pred = self._retrain_and_vf(P_pred)
                vf = (-19*dPs[0] + 106*dPs[1] - 264*dPs[2] + 646*dPs[3] + 251*f_pred) / 720
            P.assign(P - lr_P * vf)
            return P, dPs, vf

        elif ode_solver == 'RK4':
            f1 = dPs[-1]
            f2 = self._retrain_and_vf(P - lr_P/2 * f1)
            f3 = self._retrain_and_vf(P - lr_P/2 * f2)
            f4 = self._retrain_and_vf(P - lr_P * f3)
            vf = (f1 + 2*f2 + 2*f3 + f4) / 6
            P.assign(P - lr_P * vf)
            return P, [], vf
