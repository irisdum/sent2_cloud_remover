import tensorflow as tf
from constant.gee_constant import PREFIX_HIST,PREFIX_IM


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def write_log_tf2(writer,names,logs,batch_no):
    with writer.as_default():
      for name, value in zip(names, logs):
        # other model code would go here
        tf.summary.scalar(name, value, step=batch_no)
    writer.flush()
