import tensorflow as tf
import _pywrap_transformer_engine as tex

a = tf.random.normal([16, 32], dtype=tf.float32)
a_scale = tf.constant(0.5, dtype=tf.float32)
a_amax = tf.zeros([], dtype=tf.float32)
a_casted, a_amax, a_scale_inv = tex.cast_to_fp8(a, a_scale, a_amax, tex.DType.kFloat8E4M3)

b = tf.random.normal([16, 16], dtype=tf.float32)
b_scale = tf.constant(0.5, dtype=tf.float32)
b_amax = tf.zeros([], dtype=tf.float32)
b_casted, b_amax, b_scale_inv = tex.cast_to_fp8(b, b_scale, b_amax, tex.DType.kFloat8E4M3)

use_bias = False
bias = tf.zeros(())
workspace = tf.zeros([33_554_432], dtype=tf.int8)
d = tex.fp8_gemm(b_casted, b_scale_inv, tex.DType.kFloat8E4M3, a_casted, a_scale_inv,
             tex.DType.kFloat8E4M3, workspace, use_bias, bias, True, False,
             False, False, False)

print("d =", d)
