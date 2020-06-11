# Keras Implementation of GAN
import os
import random
import time
import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D, ReLU, GaussianNoise

from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.models import Sequential, Model, model_from_yaml
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from constant.gee_constant import LABEL_DIR, DICT_SHAPE
from models.callbacks import write_log, write_log_tf2
from models.losses import L1_loss
from utils.image_find_tbx import create_safe_directory, find_image_indir
from utils.load_dataset import load_data, save_images, load_from_dir
from utils.open_yaml import open_yaml, saving_yaml
from utils.metrics import batch_psnr, ssim_batch, compute_metric

import numpy as np


class GAN():
    def __init__(self, model_yaml, train_yaml):
        tf.test.gpu_device_name()
        self.sigma_val = 0
        self.model_yaml = model_yaml
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        if "dict_band_x" not in train_yaml:
            self.dict_band_X = None
            self.dict_band_label = None
            self.dict_rescale_type = None
        else:
            self.dict_band_X = train_yaml["dict_band_x"]
            self.dict_band_label = train_yaml["dict_band_label"]
            self.dict_rescale_type = train_yaml["dict_rescale_type"]
            assert type(self.dict_band_label) == type(
                {"u": 1}), "The argument {} of dict band label is not a dictionnary  but {}".format(
                self.dict_band_label, type(self.dict_band_label))
            self.path_csv = train_yaml["path_csv"]

        if "path_csv" not in train_yaml:
            print("path°csv undefined do not use global band min and max to normalize the data")
            self.path_csv = None
        else:
            self.path_csv=train_yaml["path_csv"]

        # self.latent_dim = 100
        # PATH
        self.model_name = model_yaml["model_name"]
        self.model_dir = train_yaml["training_dir"] + self.model_name + "/"
        self.this_training_dir = self.model_dir + "training_{}/".format(train_yaml["training_number"])
        self.saving_image_path = self.this_training_dir + "saved_training_images/"
        self.saving_logs_path = self.this_training_dir + "logs/"
        self.checkpoint_dir = self.this_training_dir + "checkpoints/"
        self.previous_checkpoint = train_yaml["load_model"]
        # TRAIN PARAMETER
        self.normalization = train_yaml["normalization"]
        self.epoch = train_yaml["epoch"]
        self.batch_size = train_yaml["batch_size"]
        # self.sess = sess
        self.learning_rate = train_yaml["lr"]
        self.fact_g_lr = train_yaml["fact_g_lr"]
        self.beta1 = train_yaml["beta1"]
        self.val_directory = train_yaml["val_directory"]
        self.data_X, self.data_y = load_data(train_yaml["train_directory"], normalization=self.normalization,
                                             dict_band_X=self.dict_band_X, dict_band_label=self.dict_band_label,dict_rescale_type=self.dict_rescale_type,dir_csv=self.path_csv)
        self.val_X, self.val_Y = load_data(self.val_directory, normalization=self.normalization,
                                           dict_band_X=self.dict_band_X, dict_band_label=self.dict_band_label,
                                           dict_rescale_type=self.dict_rescale_type,dir_csv=self.path_csv)
        print("Loading the data done dataX {} dataY ".format(self.data_X.shape, self.data_y.shape))
        self.num_batches = self.data_X.shape[0] // self.batch_size
        self.model_yaml = model_yaml
        self.im_saving_step = train_yaml["im_saving_step"]
        self.w_saving_step = train_yaml["weights_saving_step"]
        self.val_metric_step = train_yaml["metric_step"]
        # REDUCE THE DISCRIMINATOR PERFORMANCE
        self.val_lambda = train_yaml["lambda"]
        self.real_label_smoothing = tuple(train_yaml["real_label_smoothing"])
        self.fake_label_smoothing = tuple(train_yaml["fake_label_smoothing"])
        self.sigma_init = train_yaml["sigma_init"]
        self.sigma_step = train_yaml['sigma_step']
        self.sigma_decay = train_yaml["sigma_decay"]
        self.ite_train_g = train_yaml["train_g_multiple_time"]
        self.d_optimizer = Adam(self.learning_rate, self.beta1)
        self.g_optimizer = Adam(self.learning_rate * self.fact_g_lr, self.beta1)
        self.max_im = 10
        self.build_model()
        # self.data_X, self.data_y = load_data(train_yaml["train_directory"], normalization=self.normalization)
        # self.val_X, self.val_Y = load_data(train_yaml["val_directory"], normalization=self.normalization)

        self.model_writer=tf.summary.create_file_writer(self.saving_logs_path)
    def build_model(self):

        # We use the discriminator
        self.discriminator = self.build_discriminator(self.model_yaml)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.d_optimizer,
                                   metrics=['accuracy'])
        self.generator = self.build_generator(self.model_yaml, is_training=True)
        print("Input G")
        g_input = Input(shape=(self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]),
                        name="g_build_model_input_data")

        G = self.generator(g_input)
        print("G", G)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        D_input = tf.concat([G, g_input], axis=-1)
        print("INPUT DISCRI ", D_input)
        # The discriminator takes generated images as input and determines validity
        D_output_fake = self.discriminator(D_input)
        # print(D_output)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(g_input, [D_output_fake, G], name="Combined_model")
        self.combined.compile(loss=['binary_crossentropy', L1_loss], loss_weights=[1, self.val_lambda],
                              optimizer=self.g_optimizer)
        print("[INFO] combiend model loss are : ".format(self.combined.metrics_names))

    def build_generator(self, model_yaml, is_training=True):
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
        model_gene = Model(img_input, x, name="Generator")
        model_gene.summary()
        return model_gene

    def build_discriminator(self, model_yaml, is_training=True):
        discri_input = Input(shape=tuple([256, 256, 12]), name="d_input")
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
        model_discri = Model(discri_input, x_final, name="discriminator")
        model_discri.summary()
        return model_discri

    def produce_noisy_input(self, input, sigma_val):
        if self.model_yaml["add_discri_white_noise"]:
            # print("[INFO] On each batch GT label we add Gaussian Noise before training discri on labelled image")
            new_gt = GaussianNoise(sigma_val, input_shape=self.model_yaml["dim_gt_image"], name="d_inputGN")(
                input)
            if self.model_yaml["add_relu_after_noise"]:
                new_gt = tf.keras.layers.Activation(lambda x: tf.keras.activations.tanh(x), name="d_before_activ")(
                    new_gt)
        else:
            new_gt = input
        return new_gt

    def define_callback(self):
        # Define Tensorboard callbacks
        self.g_tensorboard_callback = TensorBoard(log_dir=self.saving_logs_path, histogram_freq=0,
                                                  batch_size=self.batch_size,
                                                  write_graph=True, write_grads=True)
        self.g_tensorboard_callback.set_model(self.combined)

    def train(self):
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 30, 30, 1))
        fake = np.zeros((self.batch_size, 30, 30, 1))
        if self.previous_checkpoint is not None:
            print("LOADING the model from step {}".format(self.previous_checkpoint))
            # TODO LOAD WEIGHTS FOR DISCRI COMBINED AND GENE
            start_epoch = int(self.previous_checkpoint) + 1
            self.load_from_checkpoint(self.previous_checkpoint)
        else:
            #create_safe_directory(self.saving_logs_path)
            create_safe_directory(self.saving_image_path)
            start_epoch = 0
        #self.define_callback()
        # loop for epoch
        start_time = time.time()
        sigma_val = self.sigma_init
        start_batch_id = 0
        # dict_metric={"epoch":[],"d_loss_real":[],"d_loss_fake":[],"d_loss":[],"g_loss":[]}
        d_loss_real = [100, 100]  # init losses
        d_loss_fake = [100, 100]
        d_loss = [100, 100]
        l_val_name_metrics, l_val_value_metrics = [], []
        for epoch in range(start_epoch, self.epoch):
            print("starting epoch {}".format(epoch))
            for idx in range(start_batch_id, self.num_batches):
                ###   THE INPUTS ##
                batch_input = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size].astype(
                    np.float32)  # the input
                # print("batch_input ite {} shape {} ".format(idx,batch_input.shape))
                batch_gt = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size].astype(
                    np.float32)  # the Ground Truth images

                ##  TRAIN THE DISCRIMINATOR

                d_noise_real = random.uniform(self.real_label_smoothing[0],
                                              self.real_label_smoothing[1])  # Add noise on the loss
                d_noise_fake = random.uniform(self.fake_label_smoothing[0],
                                              self.fake_label_smoothing[1])  # Add noise on the loss

                # Create a noisy gt images
                batch_new_gt = self.produce_noisy_input(batch_gt, sigma_val)
                # Generate a batch of new images
                # print("Make a prediction")
                gen_imgs = self.generator.predict(batch_input)  # .astype(np.float32)
                D_input_real = tf.concat([batch_new_gt, batch_input], axis=-1)
                D_input_fake = tf.concat([gen_imgs, batch_input], axis=-1)

                if epoch not in [i for i in self.ite_train_g]:
                    # print("Train the driscriminator real")
                    d_loss_real = self.discriminator.train_on_batch(D_input_real, d_noise_real * valid)
                    # print("Train the discri fake")
                    d_loss_fake = self.discriminator.train_on_batch(D_input_fake, d_noise_fake * fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # Train the generator (to have the discriminator label samples as valid)
                # print("Train combined")
                g_loss = self.combined.train_on_batch(batch_input, [valid, batch_gt])

                # Plot the progress
                print("%d iter %d [D loss: %f, acc.: %.2f%%] [G loss: %f %f]" % (epoch, self.num_batches * epoch + idx,
                                                                                 d_loss[0], 100 * d_loss[1], g_loss[0],
                                                                                 g_loss[1]))

                if epoch % self.im_saving_step == 0 and idx < self.max_im:  # to save some generated_images
                    gen_imgs = self.generator.predict(batch_input)
                    save_images(gen_imgs, self.saving_image_path, ite=self.num_batches * epoch + idx)
                # LOGS to print in Tensorboard
                if idx % self.val_metric_step == 0:
                    l_val_name_metrics, l_val_value_metrics = self.val_metric()
                    name_val_metric = ["val_{}".format(name) for name in l_val_name_metrics]
                    name_logs = self.combined.metrics_names + ["g_loss_tot", "d_loss_real", "d_loss_fake", "d_loss_tot",
                                                               "d_acc_real", "d_acc_fake", "d_acc_tot"]
                    val_logs = g_loss + [g_loss[0] + 100 * g_loss[1], d_loss_real[0], d_loss_fake[0], d_loss[0],
                                         d_loss_real[1], d_loss_fake[1], d_loss[1]]
                    # The metrics
                    l_name_metrics, l_value_metrics = compute_metric(batch_gt, gen_imgs)
                    assert len(val_logs) == len(
                        name_logs), "The name and value list of logs does not have the same lenght {} vs {}".format(
                        name_logs, val_logs)
                    write_log_tf2(self.model_writer,name_logs + l_name_metrics + name_val_metric,
                                  val_logs + l_value_metrics + l_val_value_metrics,self.num_batches * epoch + idx)
                   # write_log(self.g_tensorboard_callback, name_logs + l_name_metrics + name_val_metric,
                    #          val_logs + l_value_metrics + l_val_value_metrics,
                     #         self.num_batches * epoch + idx)

            if epoch % self.sigma_step == 0:  # update simga
                sigma_val = sigma_val * self.sigma_decay
            # save the models
            if epoch % self.w_saving_step == 0:
                self.save_model(epoch)

    def save_model(self, step):
        print("Saving model at {} step {}".format(self.checkpoint_dir,step))
        checkpoint_dir = self.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.isfile("{}model_generator.yaml".format(self.checkpoint_dir)):
            gene_yaml = self.generator.to_yaml()
            with open("{}model_generator.yaml".format(self.checkpoint_dir), "w") as yaml_file:
                yaml_file.write(gene_yaml)
        if not os.path.isfile("{}model_combined.yaml".format(self.checkpoint_dir)):
            comb_yaml = self.combined.to_yaml()
            with open("{}model_combined.yaml".format(self.checkpoint_dir), "w") as yaml_file:
                yaml_file.write(comb_yaml)
        if not os.path.isfile("{}model_discri.yaml".format(self.checkpoint_dir)):
            discri_yaml = self.discriminator.to_yaml()
            with open("{}model_discri.yaml".format(self.checkpoint_dir), "w") as yaml_file:
                yaml_file.write(discri_yaml)
        self.generator.save_weights("{}model_gene_i{}.h5".format(self.checkpoint_dir, step))
        self.discriminator.save_weights("{}model_discri_i{}.h5".format(self.checkpoint_dir, step))
        self.combined.save_weights("{}model_combined_i{}.h5".format(self.checkpoint_dir, step))

    def load_from_checkpoint(self, step):
        assert os.path.isfile("{}model_discri_i{}.h5".format(self.checkpoint_dir, step)), "No file at {}".format(
            "{}model_discri_i{}.h5".format(self.checkpoint_dir, step))
        self.discriminator.load_weights("{}model_discri_i{}.h5".format(self.checkpoint_dir, step))
        self.generator.load_weights("{}model_gene_i{}.h5".format(self.checkpoint_dir, step))
        self.combined.load_weights("{}model_combined_i{}.h5".format(self.checkpoint_dir, step))

    def load_generator(self, path_yaml, path_weight):
        # load YAML and create model
        yaml_file = open(path_yaml, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(path_weight)
        print("Loaded model from disk")
        return loaded_model

    def val_metric(self):
        val_pred = self.generator.predict(self.val_X)
        return compute_metric(self.val_Y, val_pred)

    def predict_on_iter(self, batch, path_save, l_image_id=None):
        """given an iter load the model at this iteration, returns the a predicted_batch but check if image have been saved at this directory"""
        if type(batch) == type("u"):  # the param is an string we load the bathc from this directory
            print("We load our data from {}".format(batch))
            batch, _ = load_data(batch, normalization=self.normalization, dict_band_X=self.dict_band_X,
                                 dict_band_label=self.dict_band_label, dict_rescale_type=self.dict_rescale_type)
            l_image_id = find_image_indir(batch, "npy")
        else:
            if l_image_id is None:
                print("We defined our own index for image name")
                l_image_id = [i for i in range(batch.shape[0])]
        assert len(l_image_id) == batch.shape[0], "Wrong size of the name of the images is {} should be {} ".format(
            len(l_image_id), batch.shape[0])
        if os.path.isdir(path_save):
            print("[INFO] the directory where to store the image already exists")
            data_array, path_tile,_ = load_from_dir(path_save, DICT_SHAPE[LABEL_DIR], self.path_csv)
            return data_array
        else:
            create_safe_directory(path_save)
            batch_res = self.generator.predict(batch)
            assert batch_res.shape[0]==batch.shape[0],"Wrong prediction should have shape {} but has shape {}".format(batch_res.shape,batch.shape)
            if path_save is not None:
                # we store the data at path_save
                for i in range(batch_res.shape[0]):
                    np.save("{}_image_{}".format(path_save, l_image_id[i].split("/")[-1]),batch_res[i,:,:,:])
        return batch_res


if __name__ == '__main__':
    path_train = "./GAN_confs/train.yaml"
    path_model = "./GAN_confs/model.yaml"
    gan = GAN(open_yaml(path_model), open_yaml(path_train))
    gan.train()
