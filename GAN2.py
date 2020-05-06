# Keras Implementation of GAN
import os
import random
import time
from ruamel import yaml
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout,Add
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D,ReLU, GaussianNoise
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import  Conv2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from processing import create_safe_directory
from utils.load_dataset import load_data, save_images
import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self,model_yaml,train_yaml):
        self.sigma_val = 0
        self.model_yaml = model_yaml
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        #self.latent_dim = 100
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
        #self.sess = sess
        self.learning_rate = train_yaml["lr"]
        self.fact_g_lr = train_yaml["fact_g_lr"]
        self.beta1 = train_yaml["beta1"]
        self.data_X, self.data_y = load_data(train_yaml["train_directory"])
        self.num_batches = self.data_X.shape[0] // self.batch_size
        self.model_yaml = model_yaml
        self.saving_step = train_yaml["saving_step"]

        # REDUCE THE DISCRIMINATOR PERFORMANCE
        self.val_lambda = train_yaml["lambda"]
        self.real_label_smoothing = tuple(train_yaml["real_label_smoothing"])
        self.fake_label_smoothing = tuple(train_yaml["fake_label_smoothing"])
        self.sigma_init = train_yaml["sigma_init"]
        self.sigma_step = train_yaml['sigma_step']
        self.sigma_decay = train_yaml["sigma_decay"]
        self.ite_train_g = train_yaml["train_g_multiple_time"]
        self.d_optimizer=Adam(self.learning_rate,self.beta1)
        self.g_optimizer=Adam(self.learning_rate*self.fact_g_lr,self.beta1)
        self.build_model()
    def build_model(self):

        # We use the discriminator
        self.discriminator = self.build_discriminator(self.model_yaml)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.d_optimizer,
                                   metrics=['accuracy'])
        self.generator=self.build_generator(self.model_yaml,is_training=True)
        print("Input G")
        g_input= Input(shape=(self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]),
                          name="g_build_model_input_data")
        G=self.generator(g_input)
        print("G",G)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        D_input=tf.concat([G, g_input],axis=-1)
        print("INPUT DISCRI ",D_input)
        # The discriminator takes generated images as input and determines validity
        D_output_fake= self.discriminator(D_input)
        #print(D_output)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(g_input, D_output_fake)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.g_optimizer)




    def build_generator(self,model_yaml,is_training=True):
        img_input = Input(shape=(self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]),
                          name="g_input_data")
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
        x = img_input

        for i, param_lay in enumerate(model_yaml["param_before_resnet"]):  # build the blocks before the Resnet Blocks
            x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),
                       padding=model_yaml["padding"], name="g_conv{}".format(i), activation="relu")(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                   name="g_{}_bn".format(i))(x)

        for j in range(model_yaml["nb_resnet_blocs"]):  # add the Resnet blocks
            x = build_resnet_block(x, id=j)

        for i, param_lay in enumerate(model_yaml["param_after_resnet"]):
            x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),
                       padding=model_yaml["padding"], activation="relu",
                       name="g_conv_after_resnetblock{}".format(i))(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                   name="g_after_resnetblock{}_bn2".format(i))(x)
        # The last layer
        x = Conv2D(model_yaml["last_layer"][0], model_yaml["last_layer"][1], strides=tuple(model_yaml["stride"]),
                   padding=model_yaml["padding"], name="g_final_conv", activation=last_activ)(x)
        model_gene=Model(img_input,x)
        model_gene.summary()
        return model_gene

    def build_discriminator(self,model_yaml,is_training=True):
        discri_input=Input(shape=tuple([256,256,12]),name="d_input")
        if model_yaml["d_activation"] == "lrelu":
            d_activation = lambda x: tf.nn.leaky_relu(x, alpha=model_yaml["lrelu_alpha"])
        else:
            d_activation = model_yaml["d_activation"]

        if model_yaml["add_discri_noise"]:
            x = GaussianNoise(self.sigma_val, input_shape=self.model_yaml["dim_gt_image"], name="d_GaussianNoise")(
                discri_input)
        else:
            x = discri_input
        # layer 1
        x = ZeroPadding2D(
            padding=(1, 1), name="d_pad_0")(x)
        x = Conv2D(64, 4, padding="valid", activation=d_activation, strides=(2, 2), name="d_conv1")(x)
        x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn1")(x)
        # layer 2
        x = ZeroPadding2D(padding=(1, 1), name="d_pad2")(x)
        x = Conv2D(128, 4, padding="valid", activation=d_activation, strides=(2, 2), name="d_conv2")(x)
        x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn2")(x)
        # layer 3
        x = ZeroPadding2D(padding=(1, 1), name="d_pad3")(x)
        x = Conv2D(256, 4, padding="valid", activation=d_activation, strides=(2, 2), name="d_conv3")(x)
        x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn3")(x)
        # layer 4
        x = ZeroPadding2D(padding=(1, 1), name="d_pad4")(x)
        x = Conv2D(512, 4, padding="valid", activation=d_activation, strides=(1, 1), name="d_conv4")(x)
        x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn4")(x)
        # layer 3
        x = ZeroPadding2D(padding=(1, 1), name="d_pad5")(x)
        x = Conv2D(1, 4, padding="valid", activation=d_activation, strides=(1, 1), name="d_conv5")(x)
        # x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training, name="d_bn5")(x)
        if model_yaml["d_last_activ"] == "sigmoid":
            x_final = tf.keras.layers.Activation('sigmoid', name="d_last_activ")(x)
        else:
            x_final = x
        model_discri=Model(discri_input,x_final)
        model_discri.summary()
        return model_discri


    def train(self):
        #self.build_model()
        create_safe_directory(self.saving_image_path)
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        # loop for epoch
        start_time = time.time()
        sigma_val = self.sigma_init
        start_batch_id = 0
        for epoch in range(0, self.epoch):
            print("starting epoch {}".format(epoch))
            for idx in range(start_batch_id, self.num_batches):
                d_noise_real = random.uniform(self.real_label_smoothing[0],self.real_label_smoothing[1])  # Add noise on the loss
                d_noise_fake = random.uniform(self.fake_label_smoothing[0],self.fake_label_smoothing[1])  # Add noise on the loss
                #print(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]  # the input
                # print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                batch_gt = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]  # the Ground Truth images
                # Generate a batch of new images
                gen_imgs = self.generator.predict(batch_input) #.astype(np.float32)
                print("gen_img",gen_imgs.dtype)
                D_input_real = tf.concat([batch_gt, batch_input], axis=-1)
                D_input_fake=  tf.concat([gen_imgs, batch_input], axis=-1)
                #print("SHAPE DISCRI INPUT",D_input_real.shape, D_input_fake.shape)
                d_loss_real = self.discriminator.train_on_batch(D_input_real, d_noise_real*valid)
                d_loss_fake = self.discriminator.train_on_batch(D_input_fake, d_noise_fake*fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(batch_input, valid)

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

                # Plot on tensorboard


                # If at save interval => save generated image samples
                if epoch % self.saving_step == 0:
                    gen_imgs = self.generator.predict(batch_input)
                    save_images(gen_imgs, self.saving_image_path,ite=epoch)

def saving_yaml(path_yaml,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system("cp {} {}".format(path_yaml,output_dir))

def open_yaml(path_yaml):
    with open(path_yaml) as f:
        return yaml.load(f)

if __name__ == '__main__':
    path_train="./GAN_confs/train.yaml"
    path_model="./GAN_confs/model.yaml"
    gan = GAN(open_yaml(path_model),open_yaml(path_train))
    gan.build_model()
    gan.train()