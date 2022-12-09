import tensorflow as tf
import _pywrap_transformer_engine as te

a = tf.random.normal([16, 32], dtype=tf.float32)
a_scale = tf.constant(0.5, dtype=tf.float32)
a_amax = tf.zeros([], dtype=tf.float32)
a_scale_inv = tf.zeros([], dtype=tf.float32)
a_casted = te.cast_to_fp8(a, a_scale, a_amax, a_scale_inv, te.Float8E4M3)

b = tf.random.normal([16, 16], dtype=tf.float32)
b_scale = tf.constant(0.5, dtype=tf.float32)
b_amax = tf.zeros([], dtype=tf.float32)
b_scale_inv = tf.zeros([], dtype=tf.float32)
b_casted = te.cast_to_fp8(b, b_scale, b_amax, b_scale_inv, te.Float8E4M3)

d = tf.zeros([a.shape[0], b.shape[0]], dtype=tf.float32)
use_bias = False
bias = tf.zeros(())
workspace = tf.zeros([33_554_432], dtype=tf.int8)
te.fp8_gemm(b_casted, b_scale_inv, te.Float8E4M3, a_casted, a_scale_inv,
            te.Float8E4M3, d, workspace, use_bias, bias, True, False, False, False, False)

print("d =", d)
