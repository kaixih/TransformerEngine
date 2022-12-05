import module as te
import tensorflow as tf

from module import Dense, DelayedScaling, Format

input_shape = (16, 32)
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=12)

my_dense = Dense(16, kernel_initializer=initializer)
my_dense.build(input_shape=input_shape)

my_dense_ref = tf.keras.layers.Dense(16, kernel_initializer=initializer,
                                     use_bias=False)
my_dense_ref.build(input_shape=input_shape)

fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.HYBRID,
                            amax_compute_algo='max', amax_history_len=3)

def train_step(x, use_fp8, use_ref):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    if use_ref:
      y = my_dense_ref(x)
      kernel = my_dense_ref.kernel
    else:
      kernel = my_dense.kernel
      with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        y = my_dense(x, training=True)
    loss = tf.reduce_sum(y)
  dx, dweight = tape.gradient(loss, [x, kernel])
  return y, dx, dweight

for i in range(4):
  x = tf.random.normal(input_shape)

  y, dx, dw = train_step(x, True, False)
  y_ref, dx_ref, dw_ref = train_step(x, False, True)
  
  tf.debugging.assert_near(y, y_ref, rtol=1e0, atol=1e0)
  print("Results:", y[0])
  print("Reference Results:", y_ref[0])

  tf.debugging.assert_near(dx, dx_ref, rtol=1e0, atol=1e0)
  print("Bwd dx:", dx[0])
  print("Reference dx:", dx_ref[0])

  tf.debugging.assert_near(dw, dw_ref, rtol=1e0, atol=1e0)
  print("Bwd dweight:", dw[0:2])
  print("Reference Bwd dweight:", dw_ref[0:2])
