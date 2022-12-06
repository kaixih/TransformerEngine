import tensorflow as tf
import _pywrap_transformer_engine as te

a = tf.random.normal([16, 32], dtype=tf.float32)
a_scale = tf.constant(0.5, dtype=tf.float32)
a_casted = tf.zeros([16, 32], dtype=tf.int8)
a_scale_inv = tf.zeros([], dtype=tf.float32)
a_amax = tf.zeros([], dtype=tf.float32)
te.cast_to_fp8(a, a_scale, a_casted, a_scale_inv, a_amax, te.FP8_E4M3)

b = tf.random.normal([16, 16], dtype=tf.float32)
b_scale = tf.constant(0.5, dtype=tf.float32)
b_casted = tf.zeros([16, 16], dtype=tf.int8)
b_scale_inv = tf.zeros([], dtype=tf.float32)
b_amax = tf.zeros([], dtype=tf.float32)
te.cast_to_fp8(b, b_scale, b_casted, b_scale_inv, b_amax, te.FP8_E4M3)

d = tf.zeros([a.shape[0], b.shape[0]], dtype=tf.float32)

use_bias = False
bias = tf.zeros(())
te.fp8_gemm(b_casted, b_scale_inv, te.FP8_E4M3, a_casted, a_scale_inv,
            te.FP8_E4M3, d, use_bias, bias, True, False, False, False, False)

print("d = ", d)
