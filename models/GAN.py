# This script is inspired by https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
import time

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Add, Reshape, Flatten, Dropout, BatchNormalization, Conv2D, ReLU, add, \
    ZeroPadding2D, GaussianNoise
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from constant.model_constant import CHANNEL
from processing import create_safe_directory
from utils.load_dataset import load_data, save_images
import numpy as np
from models.losses import modified_discriminator_loss, modified_generator_loss, total_generator_loss, \
    discriminator_loss, generator_loss, calc_cycle_loss, noisy_discriminator_loss, discriminator_loss2, load_loss
from ruamel import yaml
import random
import os


class GAN():
    def __init__(self, train_yaml, model_yaml, sess):
        """:param train_yaml,model_yaml two dictionnaries"""
        self.k_step = train_yaml["k_step"]
        print(train_yaml)
        print(model_yaml)
        # SHAPE PARAMETER
        self.img_rows = 256
        self.img_cols = 256
        self.channels = CHANNEL
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        print(type(train_yaml["lr"]))
        # PATH
        self.model_name = model_yaml["model_name"]
        self.model_dir = train_yaml["training_dir"] + self.model_name + "/"
        self.this_training_dir = self.model_dir + "training_{}/".format(train_yaml["training_number"])
        self.saving_image_path = self.this_training_dir + "saved_training_images/"
        self.saving_logs_path = self.this_training_dir + "logs/"
        self.checkpoint_dir = self.this_training_dir + "checkpoints/"
        # TRAIN PARAMETER
        self.epoch = train_yaml["epoch"]
        self.batch_size = train_yaml["batch_size"]
        self.sess = sess
        self.learning_rate = train_yaml["lr"]
        self.fact_g_lr = train_yaml["fact_g_lr"]
        self.beta1 = train_yaml["beta1"]
        self.data_X, self.data_y = load_data(train_yaml["train_directory"])
        self.num_batches = self.data_X.shape[0] // self.batch_size
        self.model_yaml = model_yaml
        self.saving_step = train_yaml["saving_step"]

        # LOSSES
        self.wasserstein = train_yaml["wasserstein"]
        if self.wasserstein:
            self.generator_loss=load_loss("wasser_gene_loss")
            self.discriminator_loss=load_loss("wasser_discri_loss")
        else:
            self.generator_loss = load_loss(train_yaml["generator_loss"])
            self.discriminator_loss = load_loss(train_yaml["discriminator_loss"])
        print(self.discriminator_loss)
        # test
        self.sample_num = train_yaml["n_train_image_saved"]  # number of generated images to be saved

        # REDUCE THE DISCRIMINATOR PERFORMANCE
        self.val_lambda = train_yaml["lambda"]
        self.real_label_smoothing = tuple(train_yaml["real_label_smoothing"])
        self.fake_label_smoothing = tuple(train_yaml["fake_label_smoothing"])
        self.sigma_init = train_yaml["sigma_init"]
        self.sigma_step = train_yaml['sigma_step']
        self.sigma_decay = train_yaml["sigma_decay"]
        self.ite_train_g=train_yaml["train_g_multiple_time"]


    def discriminator(self, discri_input, model_yaml, print_summary=False, reuse=False, is_training=True):

        if model_yaml["d_activation"] == "lrelu":
            d_activation = lambda x: tf.nn.leaky_relu(x, alpha=model_yaml["lrelu_alpha"])
        else:
            d_activation = model_yaml["d_activation"]

        with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
            # discri_input=tf.keras.Input(shape=tuple(model_yaml["d_input_shape"]))
            if model_yaml["add_discri_noise"]:
                x=GaussianNoise(self.sigma_val, input_shape=self.model_yaml["dim_gt_image"],name="d_GaussianNoise")(discri_input)
            else:
                x=discri_input
            # layer 1
            x = ZeroPadding2D(
                padding=(1, 1),name="d_pad_0")(discri_input)
            x = Conv2D(64, 4, padding="valid", activation=d_activation, strides=(2, 2), name="d_conv1")(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn1")(x)
            # layer 2
            x = ZeroPadding2D(padding=(1, 1),name="d_pad2")(x)
            x = Conv2D(128, 4, padding="valid", activation=d_activation, strides=(2, 2), name="d_conv2")(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn2")(x)
            # layer 3
            x = ZeroPadding2D(padding=(1, 1),name="d_pad3")(x)
            x = Conv2D(256, 4, padding="valid",activation=d_activation, strides=(2, 2), name="d_conv3")(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn3")(x)
            # layer 4
            x = ZeroPadding2D(padding=(1, 1),name="d_pad4")(x)
            x = Conv2D(512, 4, padding="valid", activation=d_activation, strides=(1, 1), name="d_conv4")(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn4")(x)
            # layer 3
            x = ZeroPadding2D(padding=(1, 1),name="d_pad5")(x)
            x = Conv2D(1, 4, padding="valid", activation=d_activation, strides=(1, 1), name="d_conv5")(x)
            #x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn5")(x)
            if model_yaml["d_last_activ"]=="sigmoid":
                x_final=tf.keras.layers.Activation('sigmoid',name="d_last_activ")(x)
            else:
                x_final=x
        if print_summary:
            model = Model(discri_input, x, name="GAN_discriminator")
            model.summary()
        # self.model_discri=Model(discri_input, x, name="GAN_discriminator")

        return x,x_final

    def generator(self, img_input, model_yaml, is_training=True, print_summary=True, reuse=False):

        def build_resnet_block(input, id=0):
            """Define the ResNet block"""
            x = Conv2D(model_yaml["dim_resnet"], model_yaml["k_resnet"], padding=model_yaml["padding"],
                       strides=tuple(model_yaml["stride"]), name="g_block_{}_conv1".format(id))(input)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                   name="g_block_{}_bn1".format(id))(x)
            x = ReLU(name="g_block_{}_relu1".format(id))(x)
            x = Dropout(rate=model_yaml["do_rate"], name="g_block_{}_do".format(id))(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                   name="g_block_{}_bn2".format(id))(x)
            x = Add(name="g_block_{}_add".format(id))([x, input])
            x = ReLU(name="g_block_{}_relu2".format(id))(x)
            return x

        if model_yaml["last_activation"] == "tanh":
            print("use tanh keras")
            last_activ = lambda x: tf.keras.activations.tanh(x)
        else:
            last_activ = model_yaml["last_activation"]

        # img_input = tf.keras.Input(shape=tuple(model_yaml["input_shape"]))
        with tf.compat.v1.variable_scope("generator", reuse=reuse):
            x = img_input
            for i, param_lay in enumerate(
                    model_yaml["param_before_resnet"]):  # build the blocks before the Resnet Blocks
                x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),
                           padding=model_yaml["padding"], name="g_conv{}".format(i), activation="relu")(x)
                x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                       name="g_{}_bn".format(i))(x)
            for j in range(model_yaml["nb_resnet_blocs"]):  # add the Resnet blocks
                x = build_resnet_block(x, id=j)
            for i, param_lay in enumerate(model_yaml["param_after_resnet"]):
                x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),padding=model_yaml["padding"],activation="relu",name="g_conv_after_resnetblock{}".format(i))(x)
                x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                       name="g_after_resnetblock{}_bn2".format(i))(x)
            # The last layer
            x = Conv2D(model_yaml["last_layer"][0], model_yaml["last_layer"][1], strides=tuple(model_yaml["stride"]),
                       padding=model_yaml["padding"], name="g_final_conv", activation=last_activ)(x)
            # print("last layer gene", x)
            # print(type(img_input))
        if print_summary:
            model = Model(img_input, x, name='GAN_generator')
            model.summary()

        return x

    def build_model(self):

        # The input in the model graph
        self.g_input = tf.keras.backend.placeholder(
            shape=(self.batch_size, self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]),
            name="Input_data")  # the input of the label of the generator
        print("ginput", self.g_input)
        # the Ground truth images
        self.gt_images = tf.keras.backend.placeholder(shape=tuple([self.batch_size] + self.model_yaml["dim_gt_image"]),
                                                      name="GT_image")
        print("gt_image", self.gt_images)
        # the loss function
        G = self.generator(self.g_input, self.model_yaml, is_training=True, print_summary=False, reuse=False)
        print("output_g", G)
        self.sigma_val = tf.Variable(0.2)
        if self.model_yaml["add_discri_white_noise"]:
            print("We add Gaussian Noise")
            new_gt = GaussianNoise(self.sigma_val, input_shape=self.model_yaml["dim_gt_image"],name="d_inputGN")(self.gt_images)
            if self.model_yaml["add_relu_after_noise"]:
                new_gt = tf.keras.layers.Activation(lambda x: tf.keras.activations.tanh(x),name="d_before_activ")(new_gt)
        else:
            new_gt = self.gt_images

        D_input_real = tf.concat([new_gt, self.g_input],
                                 axis=-1)  # input in the discriminator correspond to a pair of s2 images
        D_input_fake = tf.concat([G, self.g_input],
                                 axis=-1)  # Input correpsond to the pair of images : Ground truth and synthetized
        # image from the generator
        D_output_real,D_output_real_final = self.discriminator(D_input_real, self.model_yaml, print_summary=False, reuse=False)
        D_output_fake,D_output_fake_final = self.discriminator(D_input_fake, self.model_yaml, print_summary=False, reuse=True)
        # print("concat res ",D_input_fake)
        self.noise_real = tf.Variable(1.0)
        self.noise_fake = tf.Variable(0.0)
        d_loss_real, d_loss_fake = self.discriminator_loss(D_output_real, D_output_fake, self.noise_real,
                                                           self.noise_fake)
        self.d_loss = d_loss_real + d_loss_fake
        # THE GENERATOR LOSS
        g_loss, cycle_loss = self.generator_loss(self.gt_images, G, D_output_fake, self.val_lambda)
        self.g_loss = g_loss + cycle_loss
        print("loss g", self.g_loss)
        print("loss d ", self.d_loss)
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.compat.v1.trainable_variables()
        # print("tvariable to optimize",t_vars)
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        # print(d_vars,g_vars)

        # optimizers
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.d_gradient=tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).compute_gradients(
            self.d_loss, var_list=d_vars)
            self.d_optim=tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).apply_gradients(self.d_gradient, name="apply_gradient")
            self.g_gradient=tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).compute_gradients(
            self.g_loss, var_list=g_vars)
            self.g_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).apply_gradients(self.g_gradient,name="g_apply_gradient")
            print("G gradient",self.g_gradient)

            #self.d_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
             #   .minimize(self.d_loss, var_list=d_vars)
            #self.g_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate * self.fact_g_lr, beta1=self.beta1) \
             #   .minimize(self.g_loss, var_list=g_vars)
        ##TO TES DEMAIN
        # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #     gradients_of_generator = gen_tape.gradient(self.d_loss, d_vars)
        #     gradients_of_discriminator = disc_tape.gradient(self.g_loss, g_vars)
        #     discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)
        #     generator_optimizer=tf.compat.v1.train.AdamOptimizer(self.learning_rate * self.fact_g_lr, beta1=self.beta1)
        #     generator_optimizer.apply_gradients(zip(gradients_of_generator, g_vars))
        #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,d_vars))

        print("D vars", d_vars)
        print("G_vars",g_vars)
        if self.wasserstein:
            #weight clipping
            self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
        # for test
        self.fake_images = self.generator(self.g_input, self.model_yaml, print_summary=False, is_training=False,
                                          reuse=True)
        """ Summary """
        g_grad0=tf.summary.histogram("g_gradient_init".format(self.g_gradient[0][1]),self.g_gradient[0][0])
        g_grad_final=tf.summary.histogram("g_gradient_fin".format(self.g_gradient[0][1]),self.g_gradient[-1][0])
        d_grad0 = tf.summary.histogram("d_gradient_init".format(self.d_gradient[0][1]), self.d_gradient[0][0])
        d_grad_final = tf.summary.histogram("d_gradient_fin".format(self.d_gradient[-1][1]), self.d_gradient[-1][0])
        d_gt_sum=tf.summary.histogram("d_input_gt",new_gt)
        d_noise_real_sum=tf.summary.scalar("d_loss_noise_real",self.noise_real)
        d_noise_fake_sum=tf.summary.scalar("d_loss_noise_fake",self.noise_fake)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        g_cycle_loss_sum = tf.summary.scalar("g_cycle_loss", cycle_loss)
        g_loss_sum_tot = tf.summary.scalar("g_loss_tot", self.g_loss)
        g_image_summary = tf.summary.image("image_gene", self.fake_images, max_outputs=self.batch_size)
        G_summary = tf.summary.image("G",G, max_outputs=self.batch_size)
        gt_image_summary = tf.summary.image("image_gt", self.gt_images, max_outputs=self.batch_size)
        d_fake_image_sum = tf.summary.image("d_output_fake", 255 * D_output_fake_final, max_outputs=self.batch_size)
        d_real_image_sum = tf.summary.image("d_output_real", 255 * D_output_real_final, max_outputs=self.batch_size)
        g_layer_one = tf.summary.histogram("g_layerone", self.g_input)
        g_layer_last = tf.summary.histogram("g_layer_last", G)
        d_layer_one_fake = tf.summary.histogram("d_layer_one_fake", D_input_fake)
        d_layer_one_real = tf.summary.histogram("d_layer_one_real", D_input_real)
        d_layer_last_real = tf.summary.histogram("d_layer_last_real", D_output_real_final)
        d_layer_last_fake = tf.summary.histogram("d_layer_last_fake", D_output_fake_final)
        d_sigma_val = tf.summary.scalar("d_sigma_val", self.sigma_val)
        list_g_sum = [g_loss_sum, g_cycle_loss_sum, g_loss_sum_tot, g_image_summary, g_layer_one, g_layer_last,
                      gt_image_summary,G_summary,g_grad0,g_grad_final]
        list_d_sum = [d_loss_fake_sum, d_loss_real_sum, d_loss_sum, d_layer_last_fake, d_layer_last_real,
                      d_layer_one_fake,
                      d_layer_one_real, d_real_image_sum, d_fake_image_sum, d_sigma_val,d_noise_fake_sum,d_noise_real_sum,
                      d_gt_sum,d_grad0,d_grad_final]

        # final summary operations
        self.g_sum = tf.summary.merge(list_g_sum)
        self.d_sum = tf.summary.merge(list_d_sum)

    def train(self):
        create_safe_directory(self.saving_image_path)
        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()
        # graph inputs for visualize training results
        self.sample_z = np.resize(self.data_X[0, :, :, :],
                                  (1, self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]))  # to visualize

        # saver to save model
        self.saver = tf.compat.v1.train.Saver()

        # summary writer
        self.writer = tf.compat.v1.summary.FileWriter(self.saving_logs_path, self.sess.graph)
        # ## Create the tensorboard logdir
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        sigma_val = self.sigma_init


        for epoch in range(start_epoch, self.epoch):
            # get batch data
            print("TOTAL number batch".format(self.num_batches))
            for idx in range(start_batch_id, self.num_batches):
                print(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]  # the input
                # print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                batch_gt = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]  # the Ground Truth images
                # print("GT",batch_gt.shape)
                # update D network
                d_noise_real = random.uniform(self.real_label_smoothing[0],
                                              self.real_label_smoothing[1])  # Add noise on the loss
                d_noise_fake = random.uniform(self.fake_label_smoothing[0],
                                              self.fake_label_smoothing[1])  # Add noise on the loss
                if epoch not in [i for i in self.ite_train_g]:
                    if self.wasserstein:
                        _,_, summary_str, d_loss= self.sess.run([self.d_optim,self.clip_D, self.d_sum, self.d_loss],
                                                               feed_dict={self.g_input: batch_input,
                                                                          self.gt_images: batch_gt,
                                                                          self.noise_real: d_noise_real,
                                                                          self.noise_fake: d_noise_fake,
                                                                          self.sigma_val: sigma_val})
                    else:
                        _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.g_input: batch_input, self.gt_images: batch_gt,
                                                                  self.noise_real: d_noise_real,
                                                                  self.noise_fake: d_noise_fake,
                                                                  self.sigma_val: sigma_val})
                    self.writer.add_summary(summary_str, counter)
                # update G network
                # print("Before G run ", self.g_input,batch_input.shape)
                _, summary_str, g_loss,fake_im = self.sess.run([self.g_optim, self.g_sum, self.g_loss,self.fake_images],
                                                       feed_dict={self.g_input: batch_input, self.gt_images: batch_gt})

                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every N steps
                if np.mod(counter, self.saving_step) == 0:
                    #samples = self.sess.run(self.fake_images, feed_dict={self.g_input: self.sample_z})
                    samples=fake_im
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    print("save_image at {}".format(self.saving_image_path))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], self.saving_image_path, ite=counter)
                if np.mod(counter, self.sigma_step):
                    sigma_val = sigma_val * self.sigma_decay
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            self.save(counter)

        # save model for final step
        self.save(counter)

    def load(self, checkpoint_dir):
        if checkpoint_dir is None:
            return False, 0

        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)  # TODO modify that

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train_2(self):

        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()
        # graph inputs for visualize training results
        self.sample_z = np.resize(self.data_X[0, :, :, :],
                                  (1, self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]))  # to visualize

        # saver to save model
        self.saver = tf.compat.v1.train.Saver()

        # summary writer
        self.writer = tf.compat.v1.summary.FileWriter(self.saving_logs_path, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        global_index = 0
        g_loss = 1000
        d_loss = 1000
        for epoch in range(start_epoch, self.epoch):
            for k in range(self.k_step):
                for idx in range(start_batch_id, self.num_batches):
                    print(idx * self.batch_size, (idx + 1) * self.batch_size)
                    batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]  # the input
                    # print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                    batch_gt = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]  # the Ground Truth images
                    # print("GT",batch_gt.shape)
                    # update D network
                    # TODO adapt so that the discriminator can be trained in more iteration than the generator
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                           feed_dict={self.g_input: batch_input,
                                                                      self.gt_images: batch_gt})
                    self.writer.add_summary(summary_str, counter)
                    # display training status
                    counter += 1
                    print("Epoch only discriminator : [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))
                    # save training results for every N steps
                    if np.mod(counter, self.saving_step) == 0:
                        samples = self.sess.run(self.fake_images, feed_dict={self.g_input: self.sample_z})
                        tot_num_samples = min(self.sample_num, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                        save_images(samples[:manifold_h * manifold_w, :, :, :], self.saving_image_path, ite=counter)
                start_batch_id = 0
            for idx in range(start_batch_id, self.num_batches):
                print(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]  # the input
                # print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                batch_gt = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]  # the Ground Truth images
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.g_input: batch_input, self.gt_images: batch_gt})
                self.writer.add_summary(summary_str, counter)
                counter += 1
                print("Epoch only generator : [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every N steps
                if np.mod(counter, self.saving_step) == 0:
                    #samples = self.sess.run(self.fake_images, feed_dict={self.g_input: self.sample_z})
                    samples=self.fake_images
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], self.saving_image_path, ite=counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            self.save(counter)

        # save model for final step
        self.save(counter)

    def save(self, step):
        checkpoint_dir = self.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def visualize_results(self, epoch):
        pass


