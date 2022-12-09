"""Tests for the fp8 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import transformer_engine.tensorflow.module as te
import tensorflow as tf

from transformer_engine.tensorflow.module import Dense, DelayedScaling, Format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

def train_step(x, model, use_fp8=False, fp8_recipe=None, use_bias=False):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
      y = model(x, training=True)
    loss = tf.reduce_sum(y)
  dx, dvars = tape.gradient(loss, [x, model.variables])
  return y, dx, dvars

class Fp8LayersTest(test.TestCase):
  @test_util.run_gpu_only
  def testDenseLayer(self):
    for use_bias in [True, False]:
      with self.cached_session(use_gpu=True):
        input_shape = (16, 16)
        units = 16
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        # Use the inputs with lower precision by rounding them to the nearest
        # integers so that we can compare the results from the fp32 and fp8
        # computations.
        def int_initializer(shape, dtype=None):
          return tf.round(initializer(shape, dtype))

        dense = Dense(units, kernel_initializer=int_initializer,
                      use_bias=use_bias)
        dense.build(input_shape=input_shape)
        fp8_recipe = DelayedScaling(margin=0, interval=1,
                                    fp8_format=Format.HYBRID,
                                    amax_compute_algo='max', amax_history_len=3)
        
        x = tf.round(tf.random.normal(input_shape))

        y_ref, dx_ref, dvars_ref = train_step(x, dense, use_bias=use_bias)
        y, dx, dvars= train_step(x, dense, True, fp8_recipe, use_bias=use_bias)

        self.assertAllEqual(y_ref, y)
        self.assertAllEqual(dx_ref, dx)
        self.assertAllEqual(dvars_ref[0], dvars[0])
        if use_bias:
          self.assertAllEqual(dvars_ref[1], dvars[1])


if __name__ == '__main__':
  test.main()

      


