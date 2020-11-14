import tensorflow as tf


class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x = tf.reduce_max(x, axis=1, keepdims=True)
        ws = tf.where(inputs == x, x, 0.0)
        ws = tf.reduce_sum(ws, axis=2)
        x = tf.squeeze(x, axis=1)
        return x, ws
