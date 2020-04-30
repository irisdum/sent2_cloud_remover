import tensorflow as tf
import random
#loss from https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py

def minimax_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Original minimax discriminator loss for GANs, with label smoothing.
  Note that the authors don't recommend using this loss. A more practically
  useful loss is `modified_discriminator_loss`.
  L = - real_weights * log(sigmoid(D(x)))
      - generated_weights * log(1 - sigmoid(D(G(z))))
  See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
  details.
  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data`, and must be broadcastable to `real_data` (i.e., all
      dimensions must be either `1`, or the same as the corresponding
      dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with tf.compat.v1.name_scope(
      scope, 'discriminator_minimax_loss',
      (discriminator_real_outputs, discriminator_gen_outputs, real_weights,
       generated_weights, label_smoothing)) as scope:

    # -log((1 - label_smoothing) - sigmoid(D(x)))
    loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs),
        discriminator_real_outputs,
        real_weights,
        label_smoothing,
        scope,
        loss_collection=None,
        reduction=reduction)
    # -log(- sigmoid(D(G(x))))
    loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_gen_outputs),
        discriminator_gen_outputs,
        generated_weights,
        scope=scope,
        loss_collection=None,
        reduction=reduction)

    loss = loss_on_real + loss_on_generated
    tf.compat.v1.losses.add_loss(loss, loss_collection)

    if add_summaries:
      tf.compat.v1.summary.scalar('discriminator_gen_minimax_loss',
                                  loss_on_generated)
      tf.compat.v1.summary.scalar('discriminator_real_minimax_loss',
                                  loss_on_real)
      tf.compat.v1.summary.scalar('discriminator_minimax_loss', loss)

  return loss


def minimax_generator_loss(
    discriminator_gen_outputs,
    label_smoothing=0.0,
    weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Original minimax generator loss for GANs.
  Note that the authors don't recommend using this loss. A more practically
  useful loss is `modified_generator_loss`.
  L = log(sigmoid(D(x))) + log(1 - sigmoid(D(G(z))))
  See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
  details.
  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with tf.compat.v1.name_scope(scope, 'generator_minimax_loss') as scope:
    loss = - minimax_discriminator_loss(
        tf.ones_like(discriminator_gen_outputs),
        discriminator_gen_outputs, label_smoothing, weights, weights, scope,
        loss_collection, reduction, add_summaries=False)

  if add_summaries:
    tf.compat.v1.summary.scalar('generator_minimax_loss', loss)

  return loss




def modified_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Same as minimax discriminator loss.
  See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
  details.
  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  return minimax_discriminator_loss(
      discriminator_real_outputs,
      discriminator_gen_outputs,
      label_smoothing,
      real_weights,
      generated_weights,
      scope or 'discriminator_modified_loss',
      loss_collection,
      reduction,
      add_summaries)


def modified_generator_loss(
    discriminator_gen_outputs,
    label_smoothing=0.0,
    weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Modified generator loss for GANs.
  L = -log(sigmoid(D(G(z))))
  This is the trick used in the original paper to avoid vanishing gradients
  early in training. See `Generative Adversarial Nets`
  (https://arxiv.org/abs/1406.2661) for more details.
  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to `labels` (i.e.,
      all dimensions must be either `1`, or the same as the corresponding
      dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with tf.compat.v1.name_scope(scope, 'generator_modified_loss',
                               [discriminator_gen_outputs]) as scope:
    loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_gen_outputs), discriminator_gen_outputs,
        weights, label_smoothing, scope, loss_collection, reduction)

    if add_summaries:
      tf.compat.v1.summary.scalar('generator_modified_loss', loss)

  return loss

def L1_modified_generator_loss(discriminator_gen_outputs,
    label_smoothing=0.0,
    weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.compat.v1.name_scope(scope, 'generator_modified_loss',
                                 [discriminator_gen_outputs]) as scope:
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_gen_outputs), discriminator_gen_outputs,
            weights, label_smoothing, scope, loss_collection, reduction)

        if add_summaries:
            tf.compat.v1.summary.scalar('generator_modified_loss', loss)

    return loss


def generator_loss(D_output_fake):
    g_loss=tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_output_fake, labels=tf.ones_like(D_output_fake)))
    return g_loss

def calc_cycle_loss(real_image, fake_image, val_lambda):
    loss1 = tf.reduce_mean(tf.abs(real_image - fake_image))
    return val_lambda * loss1


def total_generatot_loss(real_image,fake_image,D_output_fake,val_lambda):
    return generator_loss(D_output_fake)+calc_cycle_loss(real_image,fake_image,val_lambda)


def discriminator_loss(D_output_real,D_output_fake):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_output_real, labels=tf.ones_like(D_output_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_output_fake, labels=tf.zeros_like(D_output_fake)))
    return d_loss_real , d_loss_fake


def noisy_discriminator_loss(D_output_real,D_output_fake):
    noise=random.uniform(0.9,1)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_output_real, labels=noise*tf.ones_like(D_output_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_output_fake, labels=tf.zeros_like(D_output_fake)))
    return d_loss_real, d_loss_fake
