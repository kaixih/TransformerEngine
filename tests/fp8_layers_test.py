"""Tests for the fp8 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Adding the path to the module.py.
# TODO: remove this when the building process is improved.
import os
import sys
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(directory))

import numpy as np
import module as te
import tensorflow as tf

from module import Dense, DelayedScaling, Format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

def train_step_ref(x, model):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = model(x)
    loss = tf.reduce_sum(y)
  dx, dweight = tape.gradient(loss, [x, model.kernel])
  return y, dx, dweight

def train_step_fp8(x, model, fp8_recipe):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      y = model(x, training=True)
    loss = tf.reduce_sum(y)
  dx, dweight = tape.gradient(loss, [x, model.kernel])
  return y, dx, dweight

class Fp8LayersTest(test.TestCase):
  @test_util.run_gpu_only
  def testDenseLayer(self):
    with self.cached_session(use_gpu=True):
      input_shape = (16, 16)
      units = 16
      initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

      # Use the inputs with lower precision by rounding them to the nearest
      # integers so that we can compare the results from the fp32 and fp8
      # computations.
      def int_initializer(shape, dtype=None):
        return tf.round(initializer(shape, dtype))

      fp8_dense = Dense(units, kernel_initializer=int_initializer)
      fp8_dense.build(input_shape=input_shape)
      fp8_recipe = DelayedScaling(margin=0, interval=1,
                                  fp8_format=Format.HYBRID,
                                  amax_compute_algo='max', amax_history_len=3)
      
      fp32_dense = tf.keras.layers.Dense(
          units, kernel_initializer=int_initializer, use_bias=False)
      fp32_dense.build(input_shape=input_shape)

      x = tf.round(tf.random.normal(input_shape))

      y, dx, dw = train_step_fp8(x, fp8_dense, fp8_recipe)
      y_ref, dx_ref, dw_ref = train_step_ref(x, fp32_dense)
      
      self.assertAllEqual(y_ref, y)
      self.assertAllEqual(dx_ref, dx)
      self.assertAllEqual(dw_ref, dw)


if __name__ == '__main__':
  test.main()

      


