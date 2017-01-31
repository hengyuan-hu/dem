# rbm trained with hmc sampler
# OR test hmc sampler on a trained rbm.


if __name__ == '__main__':
    from dem_trainer import DEMTrainer
    from dataset_wrapper import MnistWrapper
    import keras.backend as K
    import hmc
    from rbm import RBM, GibbsSampler
    import utils
    import numpy as np

    sess = utils.create_session()
    K.set_session(sess)
    dataset = MnistWrapper.load_default()
    num_vis = 784
    num_hid = 500
    dataset.reshape((num_vis,))

    params_init_file = 'test/mnist_rbm_chain20_lr0.01_save_weights/epoch_20_rbm.h5'
    rbm = RBM(num_vis, num_hid, params_init_file)

    def sampler_generator(num_chains=100):
        init_sample = np.random.uniform(0, 1.0, (num_chains,) + dataset.x_shape)
        hmc_sampler = hmc.HamiltonianSampler(init_sample,
                                             rbm.free_energy,
                                             init_stepsize=0.01,
                                             target_accept_rate=0.8,
                                             num_steps=100,
                                             stepsize_min=0.0001,
                                             stepsize_max=1.0,
                                             stepsize_dec=0.98,
                                             stepsize_inc=1.02)
        return hmc_sampler

    # def sampler_generator(cd_k=1, num_chains=100):
    #     init_vals = np.random.normal(0.0, 1.0, (num_chains,) + dataset.x_shape)
    #     return GibbsSampler(init_vals, rbm, cd_k)

    batch_size = 100
    hmc_sampler = sampler_generator()
    trainer = DEMTrainer(
        sess, dataset, rbm, hmc_sampler, sampler_generator, utils.vis_mnist)
    # trainer.train(0.001, 50, batch_size, 'test/mnist_hmc_rbm')
    # trainer.dump_log('test/mnist_hmc_rbm')
    samples = trainer._draw_samples(200)
    trainer._save_samples(samples, 'test/mnist_hmc_rbm/hmc_sample.png')
