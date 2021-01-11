# Keras Implementation of GAN
import os
import random
import time
import tensorflow as tf
from tensorflow.compat.v2.keras.utils import multi_gpu_model
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D, ReLU, GaussianNoise

from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.models import Sequential, Model, model_from_yaml
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from constant.gee_constant import DICT_RESCALE_REVERSE
from constant.storing_constant import XDIR, LABEL_DIR, DICT_SHAPE
from models.callbacks import write_log, write_log_tf2
from models.losses import L1_loss
from utils.image_find_tbx import create_safe_directory, find_image_indir
from utils.load_dataset import load_data, save_images, load_from_dir, csv_2_dictstat
from utils.normalize import rescale_on_batch, rescale_array
from utils.open_yaml import open_yaml, saving_yaml
from utils.metrics import batch_psnr, ssim_batch, compute_metric

import numpy as np
import time


class GAN():
    def __init__(self, model_yaml, train_yaml):
        """

        Args:
            model_yaml: dictionnary with the model parameters
            train_yaml: dictionnary the tran parameters
        """
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
        self.s1bands = train_yaml["s1bands"]
        self.s2bands = train_yaml["s2bands"]
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
        self.fact_s2 = train_yaml["s2_scale"]
        self.fact_s1 = train_yaml["s1_scale"]
        self.data_X, self.data_y, self.scale_dict_train = load_data(train_yaml["train_directory"],
                                                                    x_shape=model_yaml["input_shape"],
                                                                    label_shape=model_yaml["dim_gt_image"],
                                                                    normalization=self.normalization,
                                                                    dict_band_X=self.dict_band_X,
                                                                    dict_band_label=self.dict_band_label,
                                                                    dict_rescale_type=self.dict_rescale_type,
                                                                    fact_s2=self.fact_s2, fact_s1=self.fact_s1,
                                                                    s2_bands=self.s2bands, s1_bands=self.s1bands,
                                                                    lim=train_yaml["lim_train_tile"])
        self.val_X, self.val_Y, scale_dict_val = load_data(self.val_directory, x_shape=model_yaml["input_shape"],
                                                           label_shape=model_yaml["dim_gt_image"],
                                                           normalization=self.normalization,
                                                           dict_band_X=self.dict_band_X,
                                                           dict_band_label=self.dict_band_label,
                                                           dict_rescale_type=self.dict_rescale_type,
                                                           dict_scale=self.scale_dict_train, fact_s2=self.fact_s2,
                                                           fact_s1=self.fact_s1, s2_bands=self.s2bands,
                                                           s1_bands=self.s1bands, lim=train_yaml["lim_val_tile"])
        print("Loading the data done dataX {} dataY {}".format(self.data_X.shape, self.data_y.shape))
        self.gpu = train_yaml["n_gpu"]
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

        self.max_im = 10
        self.strategy= tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.d_optimizer = Adam(self.learning_rate, self.beta1)
            self.g_optimizer = Adam(self.learning_rate * self.fact_g_lr, self.beta1)
        self.buffer_size=self.data_X.shape[0]


        self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
        self.build_model()
        # self.data_X, self.data_y = load_data(train_yaml["train_directory"], normalization=self.normalization)
        # self.val_X, self.val_Y = load_data(train_yaml["val_directory"], normalization=self.normalization)

        self.model_writer = tf.summary.create_file_writer(self.saving_logs_path)
        #self.strategy = tf.distribute.MirroredStrategy()

    def build_model(self):
        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # with strategy.scope():
        with self.strategy.scope():
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
            # TO TRAIN WITH MULTIPLE GPU

            self.combined = Model(g_input, [D_output_fake, G], name="Combined_model")
            self.combined.compile(loss=['binary_crossentropy', L1_loss], loss_weights=[1, self.val_lambda],
                              optimizer=self.g_optimizer)
        print("[INFO] combined model loss are : ".format(self.combined.metrics_names))

    def build_generator(self, model_yaml, is_training=True):

        def build_resnet_block(input, id=0):
            """Define the ResNet block"""
            x = Conv2D(model_yaml["dim_resnet"], model_yaml["k_resnet"], padding=model_yaml["padding"],
                       strides=tuple(model_yaml["stride"]), name="g_block_{}_conv1".format(id))(input)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                   name="g_block_{}_bn1".format(id))(x)
            x = ReLU(name="g_block_{}_relu1".format(id))(x)
            x = Dropout(rate=model_yaml["do_rate"], name="g_block_{}_do".format(id))(x)
            x = Conv2D(model_yaml["dim_resnet"], model_yaml["k_resnet"], padding=model_yaml["padding"],
                       strides=tuple(model_yaml["stride"]), name="g_block_{}_conv2".format(id))(x)
            x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                   name="g_block_{}_bn2".format(id))(x)
            x = Add(name="g_block_{}_add".format(id))([x, input])
            x = ReLU(name="g_block_{}_relu2".format(id))(x)
            return x
        with self.strategy.scope():
            img_input = Input(shape=(self.data_X.shape[1], self.data_X.shape[2], self.data_X.shape[3]),
                              name="g_input_data")

            if model_yaml["last_activation"] == "tanh":
                print("use tanh keras")
                last_activ = lambda x: tf.keras.activations.tanh(x)
            else:
                last_activ = model_yaml["last_activation"]
            x = img_input

            for i, param_lay in enumerate(model_yaml["param_before_resnet"]):  # build the blocks before the Resnet Blocks
                x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),
                           padding=model_yaml["padding"], name="g_conv{}".format(i))(x)
                x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                       name="g_{}_bn".format(i))(x)
                x = ReLU(name="g_{}_lay_relu".format(i))(x)

            for j in range(model_yaml["nb_resnet_blocs"]):  # add the Resnet blocks
                x = build_resnet_block(x, id=j)

            for i, param_lay in enumerate(model_yaml["param_after_resnet"]):
                x = Conv2D(param_lay[0], param_lay[1], strides=tuple(model_yaml["stride"]),
                           padding=model_yaml["padding"],
                           name="g_conv_after_resnetblock{}".format(i))(x)
                x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                       name="g_after_resnetblock{}_bn2".format(i))(x)
                x = ReLU(name="g_after_resnetblock_relu_{}".format(i))(x)
            # The last layer
            x = Conv2D(model_yaml["last_layer"][0], model_yaml["last_layer"][1], strides=tuple(model_yaml["stride"]),
                       padding=model_yaml["padding"], name="g_final_conv", activation=last_activ)(x)
            model_gene = Model(img_input, x, name="Generator")
        model_gene.summary()
        return model_gene

    def build_discriminator(self, model_yaml, is_training=True):
        with self.strategy.scope():
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
            for i, layer_index in enumerate(model_yaml["dict_discri_archi"]):
                layer_val = model_yaml["dict_discri_archi"][layer_index]
                layer_key = model_yaml["layer_key"]
                layer_param = dict(zip(layer_key, layer_val))
                pad = layer_param["padding"]
                vpadding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])  # the last dimension is 12
                x = tf.pad(x, vpadding, model_yaml["discri_opt_padding"],
                           name="{}_padding_{}".format(model_yaml["discri_opt_padding"],
                                                       layer_index))  # the type of padding is defined the yaml,
                # more infomration  in https://www.tensorflow.org/api_docs/python/tf/pad
                #
                # x = ZeroPadding2D(
                #   padding=(layer_param["padding"], layer_param["padding"]), name="d_pad_{}".format(layer_index))(x)
                x = Conv2D(layer_param["nfilter"], layer_param["kernel"], padding="valid", activation=d_activation,
                           strides=(layer_param["stride"], layer_param["stride"]), name="d_conv{}".format(layer_index))(x)
                if i > 0:
                    x = BatchNormalization(momentum=model_yaml["bn_momentum"], trainable=is_training,
                                           name="d_bn{}".format(layer_index))(x)

            # x = Flatten(name="flatten")(x)
            # for i, dlayer_idx in enumerate(model_yaml["discri_dense_archi"]):
            #    dense_layer = model_yaml["discri_dense_archi"][dlayer_idx]
            #    x = Dense(dense_layer, activation=d_activation, name="dense_{}".format(dlayer_idx))(x)

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
        valid = np.ones((self.batch_size, 30, 30, 1))  # because of the shape of the discri
        fake = np.zeros((self.batch_size, 30, 30, 1))
        if self.previous_checkpoint is not None:
            print("LOADING the model from step {}".format(self.previous_checkpoint))
            start_epoch = int(self.previous_checkpoint) + 1
            self.load_from_checkpoint(self.previous_checkpoint)
        else:
            # create_safe_directory(self.saving_logs_path)
            create_safe_directory(self.saving_image_path)
            start_epoch = 0
        ## LOADING THE DATA ON MULTIPLE GPU
        strategy = tf.distribute.MirroredStrategy()

        train_dataset = tf.data.Dataset.from_tensor_slices((self.data_X, self.data_y)).shuffle(self.batch_size).batch(
            self.global_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((self.val_X, self.val_Y)).batch(self.global_batch_size)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        test_dist_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        # loop for epoch
        start_time = time.time()
        sigma_val = self.sigma_init
        start_batch_id = 0
        # dict_metric={"epoch":[],"d_loss_real":[],"d_loss_fake":[],"d_loss":[],"g_loss":[]}
        d_loss_real = [100, 100]  # init losses
        d_loss_fake = [100, 100]
        d_loss = [100, 100]
        l_val_name_metrics, l_val_value_metrics = [], []
        start_time=time.time()
        for epoch in range(start_epoch, self.epoch):

            # print("starting epoch {}".format(epoch))
            for idx,(batch_input,batch_gt) in enumerate(train_dist_dataset):


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

                d_loss_real = self.discriminator.train_on_batch(D_input_real, d_noise_real * valid)

                d_loss_fake = self.discriminator.train_on_batch(D_input_fake, d_noise_fake * fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

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
                    print(type(batch_gt),type(gen_imgs))
                    l_name_metrics, l_value_metrics = compute_metric(batch_gt.numpy(), gen_imgs)
                    assert len(val_logs) == len(
                        name_logs), "The name and value list of logs does not have the same lenght {} vs {}".format(
                        name_logs, val_logs)
                    write_log_tf2(self.model_writer, name_logs + l_name_metrics + name_val_metric+["time_in_sec"],
                                  val_logs + l_value_metrics + l_val_value_metrics+[start_time-time.time()], self.num_batches * epoch + idx,)
                # write_log(self.g_tensorboard_callback, name_logs + l_name_metrics + name_val_metric,
                #          val_logs + l_value_metrics + l_val_value_metrics,
                #         self.num_batches * epoch + idx)

            if epoch % self.sigma_step == 0:  # update simga
                sigma_val = sigma_val * self.sigma_decay
            # save the models
            if epoch % self.w_saving_step == 0:
                self.save_model(epoch)
    def train_multi_gpu(self):

        BUFFER_SIZE = self.data_X.shape[0]
        BATCH_SIZE_PER_REPLICA = self.batch_size
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync
        train_dataset = tf.data.Dataset.from_tensor_slices((self.data_X, self.data_y)).shuffle(BUFFER_SIZE).batch(
            GLOBAL_BATCH_SIZE)

        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)


        loss_object = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)
        #
        # def compute_loss(labels, predictions):
        #     per_example_loss = loss_object(labels, predictions)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        #
        # def train_step(inputs):
        #     features, labels = inputs
        #
        #     with tf.GradientTape() as tape:
        #         predictions = model(features, training=True)
        #         loss = compute_loss(labels, predictions)
        #
        #     gradients = tape.gradient(loss, model.trainable_variables)
        #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #     return loss
        #
        # @tf.function
        # def distributed_train_step(dist_inputs):
        #     per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
        #     return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
        #                                     axis=None)

    def save_model(self, step):
        print("Saving model at {} step {}".format(self.checkpoint_dir, step))
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
        test_dataset = tf.data.Dataset.from_tensor_slices((self.val_X, self.val_Y)).batch(self.val_X.shape[0])
        test_dist_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        for i,(x,y) in enumerate(test_dist_dataset):
            print("eval on {}".format(i))

            val_pred = self.generator.predict(x)
            print("type  {} {}".format(type(y),type(val_pred)))
            label=y
        return compute_metric(label.numpy(), val_pred)

    def predict_on_iter(self, batch, path_save, l_image_id=None, un_rescale=True):
        """given an iter load the model at this iteration, returns the a predicted_batch but check if image have been saved at this directory
        :param dataset:
        :param batch could be a string : path to the dataset  or an array corresponding to the batch we are going to predict on
        """
        if type(batch) == type("u"):  # the param is an string we load the bathc from this directory
            print("We load our data from {}".format(batch))

            l_image_id = find_image_indir(batch + XDIR, "npy")
            batch, _ = load_data(batch, x_shape=self.model_yaml["input_shape"],
                                 label_shape=self.model_yaml["dim_gt_image"], normalization=self.normalization,
                                 dict_band_X=self.dict_band_X, dict_band_label=self.dict_band_label,
                                 dict_rescale_type=self.dict_rescale_type, dict_scale=self.scale_dict_train,
                                 fact_s2=self.fact_s2, fact_s1=self.fact_s1, s2_bands=self.s2bands,
                                 s1_bands=self.s1bands, clip_s2=False)
        else:
            if l_image_id is None:
                print("We defined our own index for image name")
                l_image_id = [i for i in range(batch.shape[0])]
        assert len(l_image_id) == batch.shape[0], "Wrong size of the name of the images is {} should be {} ".format(
            len(l_image_id), batch.shape[0])
        if os.path.isdir(path_save):
            print("[INFO] the directory where to store the image already exists")
            data_array, path_tile, _ = load_from_dir(path_save, self.model_yaml["dim_gt_image"])
            return data_array
        else:
            create_safe_directory(path_save)
            batch_res = self.generator.predict(batch)
            # if un_rescale:  # remove the normalization made on the data

            # _, batch_res, _ = rescale_array(batch, batch_res, dict_group_band_X=self.dict_band_X,
            #                                 dict_group_band_label=self.dict_band_label,
            #                                 dict_rescale_type=self.dict_rescale_type,
            #                                 dict_scale=self.scale_dict_train, invert=True, fact_scale2=self.fact_s2,
            #                                 fact_scale1=self.fact_s1,clip_s2=False)
            assert batch_res.shape[0] == batch.shape[
                0], "Wrong prediction should have shape {} but has shape {}".format(batch_res.shape,
                                                                                    batch.shape)
            if path_save is not None:
                # we store the data at path_save
                for i in range(batch_res.shape[0]):
                    np.save("{}_image_{}".format(path_save, l_image_id[i].split("/")[-1]), batch_res[i, :, :, :])
        return batch_res


if __name__ == '__main__':
    path_train = "./GAN_confs/train.yaml"
    path_model = "./GAN_confs/model.yaml"
    gan = GAN(open_yaml(path_model), open_yaml(path_train))
    gan.train()
