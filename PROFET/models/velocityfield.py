import pickle
import tensorflow as tf


class VelocityField:
    """Time-dependent velocity field v(x, t) learned by force matching."""

    def __init__(self, space_dim, hidden_units, lr, seed):
        self.space_dim = space_dim
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.net = tf.keras.Sequential()
        for hu in hidden_units:
            self.net.add(tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Dense(hu, kernel_initializer=initializer, activation='tanh')))
        self.net.add(tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Dense(space_dim, kernel_initializer=initializer)))
        self.net.build(input_shape=(None, space_dim + 1))
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    def __call__(self, xt):
        return self.net(xt)

    @tf.function
    def train(self, x, t, v_target):
        with tf.GradientTape() as tape:
            v_pred = self.net(tf.concat([x, t], axis=1))
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(v_target - v_pred), axis=1))
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
        return loss

    def integrate(self, x0, T, dt):
        x = tf.constant(x0, dtype=tf.float32)
        xs = [x0]
        for i in range(int(T/dt)):
            xt = tf.concat([x, dt*i*tf.ones([x.shape[0], 1], dtype=tf.float32)], axis=-1)
            x += dt * self(xt)
            xs.append(x.numpy())
        return xs

    def save(self, save_dir):
        self.net.save_weights(f"{save_dir}neural_network_weights.h5")

    @classmethod
    def load(cls, save_dir):
        with open(f"{save_dir}hyperparameters.pickle", "rb") as f:
            p = pickle.load(f)
        vf = cls(p['space_dim'], p['hidden_units'], lr=0.0, seed=0)
        vf.net.load_weights(f"{save_dir}/neural_network_weights.h5")
        return vf, p
