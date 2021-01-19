import os


def load_from_checkpoint(checkpoint_dir, step):
    assert os.path.isfile("{}model_discri_i{}.h5".format(checkpoint_dir, step)), "No file at {}".format(
        "{}model_discri_i{}.h5".format(checkpoint_dir, step))
    # self.discriminator.load_weights("{}model_discri_i{}.h5".format(self.checkpoint_dir, step))
    # self.generator.load_weights("{}model_gene_i{}.h5".format(self.checkpoint_dir, step))
    # self.combined.load_weights("{}model_combined_i{}.h5".format(self.checkpoint_dir, step))
    discriminator = tf.keras.models.load_model("{}model_discri_i{}.h5".format(checkpoint_dir, step))
    generator = tf.keras.models.load_model("{}model_gene_i{}.h5".format(checkpoint_dir, step))
    combined = tf.keras.models.load_model("{}model_combined_i{}.h5".format(checkpoint_dir, step))
    return discriminator, generator, combined