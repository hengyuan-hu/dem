import tensorflow as tf
import numpy as np
import utils


def kinetic_energy(vel):
    """Returns the kinetic energy computed in the simplest way.

    vel: 2D vector, num_chains * dimension
    """
    return 0.5 * tf.reduce_sum(tf.square(vel), 1)


def hamiltonian(pos, vel, potential_fn):
    return potential_fn(pos) + kinetic_energy(vel)


def metropolis_hastings_accept(prev_energy, next_energy):
    """Return whether to accept the new state using MH and canonical dist."""
    energy_diff = prev_energy - next_energy
    return tf.exp(energy_diff) >= tf.random_uniform(tf.shape(prev_energy), 0, 1)


def simulate_dynamics(init_pos, init_vel, stepsize, num_steps, potential_fn):
    """Return final (pos, vel) obtained after num_steps leapfrog updates."""
    # first vel is half step; pos is full step
    potential = tf.reduce_sum(potential_fn(init_pos), 0)
    new_vel = init_vel - 0.5 * stepsize * tf.gradients(potential, init_pos)[0]
    new_pos = init_pos + stepsize * new_vel

    for _ in xrange(num_steps - 1):
        potential = tf.reduce_sum(potential_fn(new_pos), 0)
        new_vel = new_vel - stepsize * tf.gradients(potential, new_pos)[0]
        new_pos = new_pos + stepsize * new_vel
    potential = tf.reduce_sum(potential_fn(new_pos), 0)
    new_vel = new_vel - 0.5 * stepsize * tf.gradients(potential, new_pos)[0]
    return new_pos, new_vel


def hmc_sample(pos, stepsize, num_steps, potential_fn):
    """Produce next hmc samples with a num_steps trajectory."""
    vel = tf.random_normal(tf.shape(pos))
    final_pos, final_vel = simulate_dynamics(
        pos, vel, stepsize, num_steps, potential_fn
    )
    accept = metropolis_hastings_accept(
        hamiltonian(pos, vel, potential_fn),
        hamiltonian(final_pos, final_vel, potential_fn)
    )
    new_pos = tf.select(accept, final_pos, pos)
    accept_rate = tf.reduce_mean(tf.cast(accept, tf.float32), 0)
    return accept_rate, new_pos


class HamiltonianSampler(object):
    def __init__(self, init_pos,
                 potential_fn,
                 init_stepsize=0.01,
                 target_accept_rate=0.9,
                 num_steps=20,
                 stepsize_min=0.001,
                 stepsize_max=0.25,
                 stepsize_dec=0.98,
                 stepsize_inc=1.02,
                 avg_accept_slowness=0.9):
        """
        init_pos: initial value of the variable to be sampled
        avg_accept_slowness: used in geometric avg. 0.0 means no avg is used
        """
        # variables
        self.pos = tf.Variable(init_pos, dtype=tf.float32)
        self.stepsize = tf.Variable(init_stepsize, dtype=tf.float32)
        self.avg_accept_rate = tf.Variable(target_accept_rate, dtype=tf.float32)
        # constants
        self.potential_fn = potential_fn
        self.target_accept_rate = tf.constant(target_accept_rate, dtype=tf.float32)
        self.num_steps = num_steps
        self.stepsize_min = tf.constant(stepsize_min, dtype=tf.float32)
        self.stepsize_max = tf.constant(stepsize_max, dtype=tf.float32)
        self.stepsize_dec = tf.constant(stepsize_dec, dtype=tf.float32)
        self.stepsize_inc = tf.constant(stepsize_inc, dtype=tf.float32)
        self.avg_accept_slowness = tf.constant(avg_accept_slowness, dtype=tf.float32)

    def sample(self):
        """Define the computation graph for one hmc sampling."""
        accept_rate, new_pos = hmc_sample(
            self.pos, self.stepsize, self.num_steps, self.potential_fn
        )
        new_avg_accept_rate = tf.add(
            self.avg_accept_slowness * self.avg_accept_rate,
            (1.0 - self.avg_accept_slowness) * accept_rate
        )
        new_stepsize = tf.select(new_avg_accept_rate > self.target_accept_rate,
                                 self.stepsize * self.stepsize_inc,
                                 self.stepsize * self.stepsize_dec)
        new_stepsize = tf.clip_by_value(
            new_stepsize, self.stepsize_min, self.stepsize_max
        )
        updates = [self.pos.assign(new_pos),
                   self.stepsize.assign(new_stepsize),
                   self.avg_accept_rate.assign(new_avg_accept_rate)]
        return new_pos, updates


# test =================
def sampler_on_nd_gaussian(burnin, num_chains, num_samples, dim):
    # define the gaussian
    np.random.seed(666)
    mu = np.random.uniform(0, 20, dim).astype(np.float32)
    cov = np.random.uniform(0, 10, (dim, dim)).astype(np.float32)
    cov = np.dot(cov, cov.T)
    cov = cov / cov.max()
    # cov = np.identity(dim).astype(np.float32)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = np.linalg.inv(cov)

    def gaussian_energy(x):
        return 0.5 * tf.reduce_sum(tf.multiply(tf.matmul(x-mu, cov_inv), x-mu), 1)

    init_pos = np.random.normal(size=(num_chains, dim))
    hmc_sampler = HamiltonianSampler(
        init_pos, gaussian_energy, init_stepsize=1, stepsize_max=5,
        avg_accept_slowness=0.9
    )
    sample_op, updates = hmc_sampler.sample()

    samples = []
    sess = utils.get_session()
    with sess.as_default():
        tf.set_random_seed(666)
        tf.global_variables_initializer().run()

        for _ in range(burnin):
            _, _ = sess.run([sample_op, updates])

        for i in range(num_samples):
            new_sample, _ = sess.run([sample_op, updates])
            samples.append(new_sample)
        final_stepsize = sess.run(hmc_sampler.stepsize)
        final_accept_rate = sess.run(hmc_sampler.avg_accept_rate)

    samples = np.array(samples)
    print samples.shape
    samples = samples.T.reshape(dim, -1).T
    print samples.shape

    print '****** TARGET VALUES ******'
    print 'target mean:', mu
    print 'target cov:\n', cov

    print '****** EMPIRICAL MEAN/COV USING HMC ******'
    print 'empirical mean: ', samples.mean(axis=0)
    print 'empirical_cov:\n', np.cov(samples.T)

    print '****** HMC INTERNALS ******'
    print 'final stepsize:', final_stepsize
    print 'final acceptance_rate', final_accept_rate

    print 'DIFF'
    print np.abs(cov - np.cov(samples.T)).sum()
    print cov.sum() - np.cov(samples.T).sum()


if __name__ == '__main__':
    sampler_on_nd_gaussian(1000, 3, 1000, dim=50)
