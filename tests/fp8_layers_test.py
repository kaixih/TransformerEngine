"""Tests for the fp8 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import transformer_engine.tensorflow as te
import tensorflow as tf

from transformer_engine.tensorflow import Dense, DelayedScaling, Format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

def train_step(x, model, use_fp8=False, fp8_recipe=None):
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

        y_ref, dx_ref, dvars_ref = train_step(x, dense)
        y, dx, dvars= train_step(x, dense, True, fp8_recipe)

        self.assertAllEqual(y_ref, y)
        self.assertAllEqual(dx_ref, dx)
        self.assertAllEqual(dvars_ref[0], dvars[0])
        if use_bias:
          self.assertAllEqual(dvars_ref[1], dvars[1])

  @test_util.run_gpu_only
  def testDenseLayerAmaxBookkeeping(self):
    with self.cached_session(use_gpu=True):
      input_shape = (16, 16)
      units = 16
      initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

      # Use the inputs with lower precision by rounding them to the nearest
      # integers so that we can compare the results from the fp32 and fp8
      # computations.
      def int_initializer(shape, dtype=None):
        return tf.round(initializer(shape, dtype))

      dense = Dense(units, kernel_initializer=int_initializer)
      dense.build(input_shape=input_shape)
      fp8_recipe = DelayedScaling(margin=0, interval=1,
                                  fp8_format=Format.HYBRID,
                                  amax_compute_algo='max', amax_history_len=3)

      # Intentionally shift the random values to enforce different amax values.
      inputs = [tf.round(tf.random.normal(input_shape)),
                tf.round(tf.random.normal(input_shape) + 1.0),
                tf.round(tf.random.normal(input_shape) + 2.0)]

      x_max = [tf.math.reduce_max(x).numpy() for x in inputs]
      k_max = [tf.math.reduce_max(dense.kernel).numpy()] * len(inputs)

      fwd_amax_history_refs = [
          [[x_max[0], k_max[0]], [0., 0.], [0., 0.]],
          [[x_max[1], k_max[1]], [0., 0.], [x_max[0], k_max[0]]],
          [[x_max[2], k_max[2]], [x_max[0], k_max[0]], [x_max[1], k_max[1]]]]
      fwd_scale_refs = [[1., 1.], [128., 128.], [64., 128.]]
      fwd_scale_inv_refs = [[1.0 / x, 1.0 / y] for x, y in fwd_scale_refs]

      # Since we use reduce_sum as the loss function, the grad is full of 1.0.
      bwd_amax_history_refs = [
          [[1.], [0.], [0.]],
          [[1.], [0.], [1.]],
          [[1.], [1.], [1.]]]
      bwd_scale_refs = [[1.], [32768.], [32768]]
      bwd_scale_inv_refs = [[1.0 / x[0]] for x in bwd_scale_refs]

      for step_idx in range(len(inputs)):
        y, dx, dvars= train_step(inputs[step_idx], dense, True, fp8_recipe)

        self.assertAllEqual(dense.fp8_meta['scaling_fwd']['amax_history'],
                            fwd_amax_history_refs[step_idx])
        self.assertAllEqual(dense.fp8_meta['scaling_fwd']['scale'],
                            fwd_scale_refs[step_idx])
        self.assertAllEqual(dense.fp8_meta['scaling_fwd']['scale_inv'],
                            fwd_scale_inv_refs[step_idx])
        self.assertAllEqual(dense.fp8_meta['scaling_bwd']['amax_history'],
                            bwd_amax_history_refs[step_idx])
        self.assertAllEqual(dense.fp8_meta['scaling_bwd']['scale'],
                            bwd_scale_refs[step_idx])
        self.assertAllEqual(dense.fp8_meta['scaling_bwd']['scale_inv'],
                            bwd_scale_inv_refs[step_idx])
      

if __name__ == '__main__':
  test.main()

      


