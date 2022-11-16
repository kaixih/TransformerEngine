import tensorflow as tf
from module import MyDense, DelayedScaling, Format
import module as te
#from layer_demo import MyDense

my_dense = MyDense(16)
fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.E4M3,
                            amax_compute_algo='max', amax_history_len=3)

for i in range(5):
  x = tf.random.normal((3, 32))

  with tf.GradientTape(persistent=True) as tape:
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      y = my_dense(x, training=True)
    loss = tf.reduce_sum(y)
  tape.gradient(loss, x)
  print(y)
