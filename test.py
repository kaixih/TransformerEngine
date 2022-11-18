import tensorflow as tf
from module import MyDense, DelayedScaling, Format
import module as te
#from layer_demo import MyDense

my_dense = MyDense(16)
fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.HYBRID,
                            amax_compute_algo='max', amax_history_len=3)

for i in range(4):
  x = tf.random.normal((16, 32))

  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      y = my_dense(x, training=True)
    loss = tf.reduce_sum(y)
  dx = tape.gradient(loss, x)
  dweight = tape.gradient(loss, my_dense.kernel)

  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y_ref = my_dense(x, training=True)
    loss = tf.reduce_sum(y_ref)
  dx_ref = tape.gradient(loss, x)
  dweight_ref = tape.gradient(loss, my_dense.kernel)
  print("Results:", y)
  print("Reference Results:", y_ref)
  print("Bwd dx:", dx[0])
  print("Reference dx:", dx_ref[0])
  print("Bwd dweight:", dweight[0:2])
  print("Reference Bwd dweight:", dweight_ref[0:2])
