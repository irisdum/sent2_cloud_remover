# This script is inspired by https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
import time

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Add, Reshape, Flatten, Dropout, BatchNormalization, Conv2D, ReLU, add, \
    ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from constant.model_constant import CHANNEL
from utils.load_dataset import load_data, save_images
import numpy as np
from models.losses import modified_discriminator_loss,modified_generator_loss,total_generatot_loss,discriminator_loss
from ruamel import yaml
import os


class GAN():
    def __init__(self, train_yaml, model_yaml, sess):
        """:param train_yaml,model_yaml two dictionnaries"""
        self.k_step = train_yaml["k_step"]
        self.model_dir = train_yaml["model_dir"]
        print(train_yaml)
        print(model_yaml)
        self.img_rows = 256
        self.img_cols = 256
        self.channels = CHANNEL
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        print(type(train_yaml["lr"]))
        optimizer = Adam(train_yaml["lr"], train_yaml["beta1"])
        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator(model_yaml)
        # self.discriminator.compile(loss='binary_crossentropy',
        #                            optimizer=optimizer,
        #                            metrics=['accuracy'])

        # Build the generator
        # self.z = self.build_generator(model_yaml,)
        self.log_dir = train_yaml["logdir"]
        self.model_name=model_yaml["model_name"]
        self.epoch = train_yaml["epoch"]
        self.batch_size = train_yaml["batch_size"]
        self.checkpoint_dir = train_yaml["checkpoint_dir"]
        self.sess = sess
        self.learning_rate = train_yaml["lr"]
        self.fact_g_lr=train_yaml["fact_g_lr"]
        self.beta1 = train_yaml["beta1"]
        self.data_X, self.data_y = load_data(train_yaml["train_directory"])
        self.num_batches = self.data_X.shape[0] // self.batch_size
        self.model_yaml=model_yaml
        self.saving_step=train_yaml["saving_step"]
        # test
        self.sample_num = train_yaml["n_train_image_saved"]  # number of generated images to be saved
        self.result_dir=train_yaml["result_dir"]
        self.val_lambda=train_yaml["lambda"]



    def discriminator(self, discri_input, model_yaml, print_summary=True, reuse=False):

        if model_yaml["d_activation"] == "lrelu":
            d_activation = lambda x: tf.keras.activations.relu(x, alpha=model_yaml["lrelu_alpha"])
        else:
            d_activation = model_yaml["d_activation"]
        with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
            # discri_input=tf.keras.Input(shape=tuple(model_yaml["d_input_shape"]))
            # layer 1
            x = ZeroPadding2D(
                padding=(1, 1))(discri_input)
            x = Conv2D(64, 4, padding="valid", activation=d_activation, strides=(2, 2),name="d_conv1")(x)
            # layer 2
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(128, 4, padding="valid", activation=d_activation, strides=(2, 2),name="d_conv2")(x)
            # layer 3
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(256, 4, padding="valid", activation=d_activation, strides=(2, 2),name="d_conv3")(x)
            # layer 4
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(512, 4, padding="valid", activation=d_activation, strides=(1, 1),name="d_conv4")(x)
            # layer 3
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(1, 4, padding="valid", activation=d_activation, strides=(1, 1),name="d_conv5")(x)

        if print_summary:
            model = Model(discri_input, x, name="GAN_discriminator")
            model.summary()
        #self.model_discri=Model(discri_input, x, name="GAN_discriminator")

        return x

    def generator(self, img_input, model_yaml, is_training=True, print_summary=True, reuse=False):

        def build_resnet_block(input,id=0):
            """Define the ResNet block"""
            x = Conv2D(model_yaml["dim_resnet"], model_yaml["k_resnet"], padding=model_yaml["padding"],
                       strides=tuple(model_yaml["stride"]),name="g_block_{}_conv1".format(id))(input)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,name="g_block_{}_bn1".format(id))(x)
            x = ReLU(name="g_block_{}_relu1".format(id))(x)
            x = Dropout(rate=model_yaml["do_rate"],name="g_block_{}_do".format(id))(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,name="g_block_{}_bn2".format(id))(x)
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
                           padding=model_yaml["padding"],name="g_conv{}".format(i),activation="relu")(x)

            for j in range(model_yaml["nb_resnet_blocs"]):  # add the Resnet blocks
                x = build_resnet_block(x,id=j)
            for i, param_lay in enumerate(model_yaml["param_after_resnet"]):
                x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),
                           padding=model_yaml["padding"],
                           activation="relu")(x)
            # The last layer
            x = Conv2D(model_yaml["last_layer"][0], model_yaml["last_layer"][1], strides=tuple(model_yaml["stride"]),
                       padding=model_yaml["padding"],name="g_final_conv", activation=last_activ)(x)
            # print("last layer gene", x)
            # print(type(img_input))
        if print_summary:
            model = Model(img_input, x, name='GAN_generator')
            model.summary()



        return x



    def build_model(self):

        #The input in the model graph
        self.g_input=tf.keras.backend.placeholder(shape=(self.batch_size,self.data_X.shape[1],self.data_X.shape[2],self.data_X.shape[3]),name="Input_data") #the input of the label of the generator
        print("ginput",self.g_input)
        #the Ground truth images
        self.gt_images=tf.keras.backend.placeholder(shape=tuple([self.batch_size]+self.model_yaml["dim_gt_image"]),name="GT_image")
        print("gt_image",self.gt_images)
        #the loss function
        G=self.generator(self.g_input,self.model_yaml,is_training=True,print_summary=False,reuse=False)
        print("output_g",G)
        D_input_real=tf.concat([self.gt_images,self.gt_images],axis=-1)  #input in the discriminator correspond to a pair of s2 images
        D_input_fake=tf.concat([self.gt_images,G],axis=-1) #Input correpsond to the pair of images : Ground truth and synthetized image from the generator

        D_output_real=self.discriminator(D_input_real,self.model_yaml,print_summary=False,reuse=False)
        D_output_fake=self.discriminator(D_input_fake,self.model_yaml,print_summary=False,reuse=True)

        #print("concat res ",D_input_fake)

        d_loss_real,d_loss_fake=discriminator_loss(D_output_real, D_output_fake)
        self.d_loss=d_loss_real+d_loss_fake
        # THE GENERATOR LOSS
        #discri_output=self.discriminator(D_input_fake,self.model_yaml,print_summary=False)
        self.g_loss=total_generatot_loss(self.gt_images,G,D_output_fake,self.val_lambda)
        print("loss g",self.g_loss)
        print("loss d ",self.d_loss)
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.compat.v1.trainable_variables()
        #print("tvariable to optimize",t_vars)
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        #print(d_vars,g_vars)

        # optimizers
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate * self.fact_g_lr, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)
        print("D optim",d_vars)

        # for test
        self.fake_images = self.generator(self.g_input,self.model_yaml,print_summary=False, is_training=False, reuse=True)
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()
        # graph inputs for visualize training results
        self.sample_z = np.resize(self.data_X[0,:,:,:],(1,self.data_X.shape[1],self.data_X.shape[2],self.data_X.shape[3])) #to visualize


        # saver to save model
        self.saver = tf.compat.v1.train.Saver()

        # summary writer
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)
        # ## Create the tensorboard logdir
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir)

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
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            print("TOTAL numebr batch".format(self.num_batches))
            for idx in range(start_batch_id, self.num_batches):
                print(idx * self.batch_size,(idx + 1) * self.batch_size)
                batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size] # the input
                #print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                batch_gt=self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size] #the Ground Truth images
                #print("GT",batch_gt.shape)

                # update D network
                #TODO adapt so that the discriminator can be trained in more iteration than the generator
                _,summary_str, d_loss = self.sess.run([self.d_optim,self.d_sum, self.d_loss],
                                                       feed_dict={self.g_input: batch_input, self.gt_images: batch_gt})
                self.writer.add_summary(summary_str, counter)
                # update G network
                #print("Before G run ", self.g_input,batch_input.shape)
                _,summary_str,g_loss = self.sess.run([self.g_optim,self.g_sum, self.g_loss],
                                                       feed_dict={self.g_input: batch_input,self.gt_images:batch_gt})

                self.writer.add_summary(summary_str, counter)
                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every N steps
                if np.mod(counter, self.saving_step) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.g_input: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :],self.result_dir,ite=counter)

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
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name) #TODO modify that

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
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
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

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
        global_index=0
        g_loss=1000
        d_loss=1000
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
                    _,summary_str, d_loss = self.sess.run([self.d_optim,self.d_sum, self.d_loss],
                                                        feed_dict={self.g_input: batch_input, self.gt_images: batch_gt})
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
                        save_images(samples[:manifold_h * manifold_w, :, :, :], self.result_dir, ite=counter)
                start_batch_id = 0
            for idx in range(start_batch_id, self.num_batches):
                print(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]  # the input
                # print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                batch_gt = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]  # the Ground Truth images
                _,summary_str, g_loss = self.sess.run([self.g_optim,self.g_sum, self.g_loss],
                                                    feed_dict={self.g_input: batch_input, self.gt_images: batch_gt})
                self.writer.add_summary(summary_str, counter)
                counter += 1
                print("Epoch only generator : [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every N steps
                if np.mod(counter, self.saving_step) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.g_input: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], self.result_dir, ite=counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            self.save(counter)

        # save model for final step
        self.save(counter)

    def save(self, step):
        checkpoint_dir = os.path.join(self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def visualize_results(self, epoch):
        pass