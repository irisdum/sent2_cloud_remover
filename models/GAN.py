# This script is inspired by https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Conv2D, ReLU, add,ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from constant.model_constant import CHANNEL, LEARNING_RATE, BETA_1, LOGDIR

import numpy as np
from ruamel import yaml


class GAN():
    def __init__(self, train_yaml, model_yaml):
        """:param train_yaml,model_yaml two dictionnaries"""
        print(train_yaml)
        print(model_yaml)
        self.img_rows = 256
        self.img_cols = 256
        self.channels = CHANNEL
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        print(type(train_yaml["lr"]))
        optimizer = Adam(train_yaml["lr"], train_yaml["beta1"])
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(model_yaml)
        # self.discriminator.compile(loss='binary_crossentropy',
        #                            optimizer=optimizer,
        #                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(model_yaml)

        # ##TODO the generator has as in input S1, S2, O1 and generate O2, work on the input
        # # The generator takes noise as input and generates imgs
        # z = Input(shape=(self.latent_dim,))
        # img = self.generator(z)
        #
        # # For the combined model we will only train the generator
        # self.discriminator.trainable = False  # TODO get what is the role of that !!!
        #
        # # The discriminator takes generated images as input and determines validity
        # validity = self.discriminator(img)
        #
        # # The combined model  (stacked generator and discriminator)
        # # Trains the generator to fool the discriminator
        # self.combined = Model(z, validity)
        # self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #
        # ## Create the tensorboard logdir
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=train_yaml[
        #     "logdir"])  # TODO change the name of the logdir make a function which gives a name to the train

    def build_discriminator(self,model_yaml,print_summary=True):
        if model_yaml["d_activation"]=="lrelu":
            d_activation= lambda x: tf.keras.activations.relu(x, alpha=model_yaml["lrelu_alpha"])
        else:
            d_activation= model_yaml["d_activation"]
        discri_input=tf.keras.Input(shape=tuple(model_yaml["d_input_shape"]))
        #layer 1
        x=ZeroPadding2D(
            padding=(1, 1))(discri_input)
        x=Conv2D(64,4,padding="valid",activation=d_activation,strides=(2,2))(x)
        #layer 2
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(128, 4, padding="valid", activation=d_activation,strides=(2,2))(x)
        #layer 3
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, 4, padding="valid", activation=d_activation,strides=(2,2))(x)
        # layer 4
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, 4, padding="valid", activation=d_activation, strides=(1, 1))(x)
        # layer 3
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(1, 4, padding="valid", activation=d_activation, strides=(1, 1))(x)
        print(x)
        model=Model(discri_input,x,name="GAN_discriminator")
        if print_summary:
            model.summary()
        return model

    def build_generator(self, model_yaml, is_training=True, print_summary=True):

        def build_resnet_block(input):
            """Define the ResNet block"""
            x = Conv2D(model_yaml["dim_resnet"], model_yaml["k_resnet"], padding=model_yaml["padding"],
                       strides=tuple(model_yaml["stride"]))(input)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training)(x)
            x = tf.keras.activations.relu(x)
            x = Dropout(rate=model_yaml["do_rate"])(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training)(x)
            x = add([x, input])
            return tf.keras.activations.relu(x)

        if model_yaml["last_activation"]=="tanh":
            print("use tanh keras")
            last_activ= lambda x: tf.keras.activations.tanh(x)
        else:
            last_activ=model_yaml["last_activation"]

        img_input = tf.keras.Input(shape=tuple(model_yaml["input_shape"]))
        x=img_input
        for i, param_lay in enumerate(model_yaml["param_before_resnet"]):  # build the blocks before the Resnet Blocks
            x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]), padding=model_yaml["padding"],
                       activation="relu")(x)

        for j in range(model_yaml["nb_resnet_blocs"]):  # add the Resnet blocks
            x = build_resnet_block(x)
        for i, param_lay in enumerate(model_yaml["param_after_resnet"]):
            x=Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]), padding=model_yaml["padding"],
                             activation="relu")(x)
        # The last layer
        x=Conv2D(model_yaml["last_layer"][0], model_yaml["last_layer"][1], strides=tuple(model_yaml["stride"]),
                         padding=model_yaml["padding"],
                         activation=last_activ)(x)
        print("last layer gene", x)
        print(type(img_input))
        model=Model(img_input, x, name='GAN_generator')
        if print_summary:
            model.summary()
        return model
