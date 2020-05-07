import tensorflow as tf
from constant.gee_constant import PREFIX_HIST,PREFIX_IM


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        if PREFIX_IM in name: #FOR AN IMAGE:
            summary_value.image=value
        elif PREFIX_HIST in name:
            summary_value.histo=value
        else:
            summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
