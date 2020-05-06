import tensorflow as tf



def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()