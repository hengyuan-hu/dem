import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from autograd_rbm import *

class Stacked_RBM(object):
    def __init__(self, dim_list, name='srbm'):
        self.num_rbm = len(dim_list)-1
        self.rbm_list = []
        for i in xrange(self.num_rbm):
            self.rbm_list.append(RBM(dim_list[i],dim_list[i+1],'rbm_%d'%i))
    
    # Returns last hidden sample and samples list
    def deep_compute_up(self, vis_samples):
        samples_list = []
        k = self.num_rbm
        # compute up
        for i in xrange(k):
            samples_list.append(vis_samples)
            rbm = self.rbm_list[i]
            hid_p = rbm.compute_up(vis_samples)
            hid_samples = rbm.sample(hid_p)
            hid_samples = tf.stop_gradient(hid_samples)
            vis_samples = hid_samples
        return hid_samples, hid_p, samples_list
    
    # Returns 
    def deep_compute_down(self, hid_samples):
        samples_list = []
        k = self.num_rbm
        for i in xrange(k):
            rbm = self.rbm_list[k-1-i]
            vis_p = rbm.compute_down(hid_samples)
            vis_samples = rbm.sample(vis_p)
            vis_samples = tf.stop_gradient(vis_samples)
            hid_samplse = vis_samples
            samples_list.append(vis_samples)
        
        return vis_samples, vis_p, samples_list[::-1]
        
    # Upward & downward process of a deep rbm
    def deep_vhv(self, vis_samples):
        # compute up
        last_hid_samples,last_hid_p,vis_list = self.deep_compute_up(vis_samples)
        # compute down
        last_vis_samples,last_vis_p,vis_samples_list = self.deep_compute_down(last_hid_samples)
        return last_vis_samples, last_vis_p, vis_list, vis_samples_list
    
    def deep_cd(self, vis, k):
        """Contrastive Divergence.
        params: vis is treated as vis_samples.
        """
        self.vis_list = []
        self.vis_samples_list = []
        vis_samples = vis
        vis_p, vis_samples, vis_list, vis_samples_list = self.deep_vhv(vis_samples)
        return vis_p, vis_samples, vis_list, vis_samples_list

    def deep_get_loss_updates(self, lr, vis, persistent_vis, cd_k):
        if persistent_vis is not None:
            _, recon_vis_samples, _, vis_samples_list = self.deep_cd(persistent_vis, cd_k)
            _, _, vis_list = self.deep_compute_up(vis)
            
        else:
            _, recon_vis_samples, vis_list, vis_samples_list = self.deep_cd(vis, cd_k)
            
        self.vis_list = vis_list
        self.vis_samples_list = vis_samples_list
        
        energy_vis = tf.zeros([], tf.float32)
        energy_sample = tf.zeros([], tf.float32)
        
        for i in xrange(self.num_rbm):
            each_vis = self.vis_list[i]
            rbm = self.rbm_list[i]
            energy_vis += tf.reduce_mean(rbm.free_energy(each_vis))
            
        for i in xrange(self.num_rbm):
            each_sample = self.vis_samples_list[i]
            rbm = self.rbm_list[i]
            energy_sample += tf.reduce_mean(rbm.free_energy(each_sample))
            
        cost = energy_vis - energy_sample
        
        loss = self.l2_loss()
        return loss, cost, recon_vis_samples
        
    def l2_loss(self):
        vis = self.vis_list[0]
        _, recon_vis, _, _ = self.deep_vhv(vis)
        num_dims = len(vis.get_shape().as_list())
        dims = range(num_dims)
        total_loss = tf.reduce_sum(tf.square(vis - recon_vis), dims[1:])
        return tf.reduce_mean(total_loss)
    
    def sample_from_rbm(self, num_steps, num_examples, vis):
        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples, _, _ = self.deep_vhv(vis_samples)
            return x+1, vis_p, vis_samples

        _, prob_imgs, sampled_imgs = tf.while_loop(cond, body, [0, vis, vis], back_prop=False)
        return prob_imgs, sampled_imgs