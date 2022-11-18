import ctypes
import numpy as np
import tensorflow as tf

from contextlib import contextmanager
from typing import Callable, Optional, Dict, Any, Tuple, Union

from enum import Enum
from typing import Literal, Optional, Union, Callable, NamedTuple
from pydantic.dataclasses import dataclass

from keras import backend
from tensorflow.keras.layers import Layer

nvlib = ctypes.cdll.LoadLibrary("./cu_demo.so")

_FP8_ENABLED = False
_FP8_RECIPE = None
_FP8_DISTRIBUTED_GROUP = None
_IS_FIRST_FP8_MODULE = False
_FP8_AUTOCAST_COUNTER = 0
_FP8_CURRENT_CONTEXT_ID = 0
_FP8_AUTOCAST_DEPTH = 0
_global_fp8_buffer = {}
_amax_forward_global_reduce_func = lambda : None
_buffer_delete_key_fwd = None
_buffer_delete_key_bwd = None

def is_fp8_enabled():
  return _FP8_ENABLED

def is_first_fp8_module():
  """Returns `True` only the first time when called multiple
  times from within the same `fp8_autocast` context.
  """
  global _IS_FIRST_FP8_MODULE
  tmp = _IS_FIRST_FP8_MODULE
  _IS_FIRST_FP8_MODULE = False
  return tmp

def get_fp8_recipe():
  return _FP8_RECIPE

def new_fp8_context_id() -> int:
  return _FP8_AUTOCAST_COUNTER

def get_fp8_context_id() -> int:
  return _FP8_CURRENT_CONTEXT_ID

def set_fp8_context_id(ctx_id: int) -> None:
  global _FP8_CURRENT_CONTEXT_ID
  _FP8_CURRENT_CONTEXT_ID = ctx_id

def get_meta_tensor_key(forward: bool = True) -> str:
  if forward:
    return "scaling_fwd"
  return "scaling_bwd"

def get_buffer_position_key(forward: bool = True) -> str:
  if forward:
    return "global_fp8_buffer_pos_fwd"
  return "global_fp8_buffer_pos_bwd"

def get_autocast_key(forward: bool = True) -> str:
  if forward:
    return "autocast_id_fwd"
  return "autocast_id_bwd"

def get_amax_buffer_key(fp8_meta: Dict[str, Any], forward: bool = True) -> str:
  if forward:
    return f"FWD_AMAX_{fp8_meta['autocast_id_fwd']}"
  return f"BWD_AMAX_{fp8_meta['autocast_id_bwd']}"

def copy_amax_from_global_buffer(fp8_meta: Dict[str, Any],
                                 forward: bool = True) -> None:
  fp8_meta_tensor_key = get_meta_tensor_key(forward=forward)
  buffer_position_key = get_buffer_position_key(forward=forward)
  if buffer_position_key not in fp8_meta:
    return
  amax_buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)
  update_tensor = _global_fp8_buffer[amax_buffer_key][
      fp8_meta[buffer_position_key]
  ]
  new_tensor = tf.tensor_scatter_nd_update(
      fp8_meta[fp8_meta_tensor_key]["amax_history"], [[0]], [update_tensor])
  fp8_meta[fp8_meta_tensor_key]["amax_history"] = new_tensor

def update_amax_history(amax_history: tf.Tensor) -> tf.Tensor:
  amax_history = tf.roll(amax_history, -1, 0)
  zeros = tf.zeros(shape=amax_history[0].shape)
  amax_history = tf.tensor_scatter_nd_update(amax_history, [[0]], [zeros])
  return amax_history

def _default_get_amax(amax_history: tf.Tensor,
                      amax_compute_algo: str) -> Tuple[tf.Tensor, tf.Tensor]:
  if amax_compute_algo == "max":
    amax = tf.reduce_max(amax_history, axis=0)
  else:
    assert amax_compute_algo == "most_recent"
    amax = amax_history[0]

  amax_history = update_amax_history(amax_history)
  return amax_history, amax

def _default_sf_compute(amax: tf.Tensor,
                        scale: tf.Tensor,
                        fp8_max: float,
                        margin: int) -> tf.Tensor:
  exp = tf.math.floor(tf.experimental.numpy.log2(fp8_max / amax)) - margin
  sf = tf.math.round(tf.math.pow(2, tf.math.abs(exp)))
  sf = tf.where(amax > 0.0, sf, scale)
  sf = tf.where(tf.math.is_inf(amax), sf, scale)
  sf = tf.where(exp < 0, 1 / sf, sf)

  return sf

def fused_amax_and_scale_update(
    amax_history: tf.Tensor,
    scale: tf.Tensor,
    fp8_max: float,
    margin: int,
    amax_compute_algo: str,
) -> Tuple[tf.Tensor, tf.Tensor]:
  # Get amax from history.
  amax_history, amax = _default_get_amax(
      amax_history,
      amax_compute_algo,
  )

  # Calculate new scaling factor.
  return amax_history, _default_sf_compute(
      amax,
      scale,
      fp8_max,
      margin,
  )

def amax_and_scale_update(
    fp8_meta: Dict[str, Any],
    fwd_update: bool,
) -> None:
  amax_compute = fp8_meta["recipe"].amax_compute_algo
  sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo
  fp8_meta_tensor_key = "scaling_fwd" if fwd_update else "scaling_bwd"
  fp8_max_key = "fp8_max_fwd" if fwd_update else "fp8_max_bwd"
  print("XXX amax_history", fp8_meta_tensor_key,
        fp8_meta[fp8_meta_tensor_key]["amax_history"])

  if not callable(amax_compute) and sf_compute is None:
    (
        fp8_meta[fp8_meta_tensor_key]["amax_history"],
        fp8_meta[fp8_meta_tensor_key]["scale"],
    ) = fused_amax_and_scale_update(
        fp8_meta[fp8_meta_tensor_key]["amax_history"],
        fp8_meta[fp8_meta_tensor_key]["scale"],
        fp8_meta[fp8_max_key],
        fp8_meta["recipe"].margin,
        fp8_meta["recipe"].amax_compute_algo,
    )

def set_amax_buffer_key_deletion(
    fp8_meta: Dict[str, Any], forward: bool = True
) -> None:
  if get_autocast_key(forward=forward) not in fp8_meta:
    return
  global _buffer_delete_key_fwd, _buffer_delete_key_bwd
  if forward:
    _buffer_delete_key_fwd = get_amax_buffer_key(fp8_meta, forward=forward)
  else:
    _buffer_delete_key_bwd = get_amax_buffer_key(fp8_meta, forward=forward)

def add_amax_to_global_buffer(
    fp8_meta: Dict[str, Any],
    forward: bool = True
) -> None:
  """Append 1D tensor `amax` to global buffer."""
  global _global_fp8_buffer
  buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)
  fp8_meta_tensor_key = get_meta_tensor_key(forward=forward)
  buffer_position_key = get_buffer_position_key(forward=forward)

  if buffer_key not in _global_fp8_buffer:
    _global_fp8_buffer[buffer_key] = \
        [fp8_meta[fp8_meta_tensor_key]["amax_history"][0]]
  else:
    _global_fp8_buffer[buffer_key].append(
        fp8_meta[fp8_meta_tensor_key]["amax_history"][0]
    )

  if buffer_position_key not in fp8_meta:
    fp8_meta[buffer_position_key] = len(_global_fp8_buffer[buffer_key]) - 1

def delete_key_from_amax_buffer(
    forward: bool = True
) -> None:
  global _global_fp8_buffer, _buffer_delete_key_fwd, _buffer_delete_key_bwd
  if forward:
    if (
        _buffer_delete_key_fwd is not None
        and _buffer_delete_key_fwd in _global_fp8_buffer
    ):
      del _global_fp8_buffer[_buffer_delete_key_fwd]
  else:
    if (
        _buffer_delete_key_bwd is not None
        and _buffer_delete_key_bwd in _global_fp8_buffer
    ):
      del _global_fp8_buffer[_buffer_delete_key_bwd]

class _FormatHelper(NamedTuple):
  max_fwd: float
  max_bwd: float

class Format(Enum):
  E4M3 = _FormatHelper(max_fwd=448, max_bwd=448)
  E5M2 = _FormatHelper(max_fwd=57344, max_bwd=57344)
  HYBRID = _FormatHelper(max_fwd=E4M3.max_fwd, max_bwd=E5M2.max_bwd)

class _OverrideLinearPrecision(NamedTuple):
  fprop: bool = False
  dgrad: bool = False
  wgrad: bool = False

@dataclass()
class DelayedScaling:
  margin: int = 0
  interval: int = 1
  fp8_format: Format = Format.HYBRID
  amax_history_len: int = 1
  amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = \
      "most_recent"
  override_linear_precision: _OverrideLinearPrecision = \
      _OverrideLinearPrecision()
  scaling_factor_compute_algo: Optional[Callable] = None

  def __post_init__(self) -> None:
    assert self.fp8_format != Format.E5M2, \
           "Pure E5M2 training is not supported."
    assert self.override_linear_precision in (
        (False, False, False),
        (False, False, True),
    ), "Only wgrad GEMM override is currently supported."

def get_default_fp8_recipe():
  return DelayedScaling()

def get_fp8_te_dtype(
  fp8_recipe: DelayedScaling, fprop_tensor: bool = True):
  """Get fp8 data type according to recipe and tensor"""
  if fp8_recipe.fp8_format == Format.E4M3 or (
      fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
  ):
    return 0
  return 1

@contextmanager
def fp8_autocast(
    enabled: bool = False,
    fp8_recipe: Optional[DelayedScaling] = None,
) -> None:
  global _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP, _FP8_AUTOCAST_DEPTH
  global _IS_FIRST_FP8_MODULE, _FP8_AUTOCAST_COUNTER
  global _global_fp8_buffer, _buffer_delete_key_fwd
  fp8_state = (_FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP)
  try:
    _FP8_ENABLED = enabled
    _FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe

    if _FP8_AUTOCAST_DEPTH == 0:
      _IS_FIRST_FP8_MODULE = True
      _FP8_AUTOCAST_COUNTER += 1
    _FP8_AUTOCAST_DEPTH += 1

    yield
  finally:
    _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP = fp8_state
    _IS_FIRST_FP8_MODULE = False
    _FP8_AUTOCAST_DEPTH -= 1

    if _FP8_AUTOCAST_DEPTH == 0:
      if callable(_amax_forward_global_reduce_func):
        _amax_forward_global_reduce_func()
      delete_key_from_amax_buffer(forward=True)

def cast_to_fp8_wrapper(inp, fp8_meta, amax_index, fwd, dtype):
  scaling_key = "scaling_fwd" if fwd else "scaling_bwd"
  inp_shape = inp.shape
  x = inp
  x_cpu = np.copy(x).flatten()
  scale = fp8_meta[scaling_key]["scale"][amax_index]
  scale_cpu = np.copy(scale).flatten()
  amax = fp8_meta[scaling_key]["amax_history"][0][amax_index]
  amax_cpu = np.copy(amax).flatten()
  scale_inv = fp8_meta[scaling_key]["scale_inv"][amax_index]
  scale_inv_cpu = np.copy(scale_inv).flatten()
  x_fp8 = tf.zeros(inp.shape, dtype=tf.int8)
  x_fp8_cpu = np.copy(x_fp8).flatten()

  nvlib.cast_to_fp8(ctypes.c_void_p(x_cpu.ctypes.data),
                    ctypes.c_void_p(scale_cpu.ctypes.data),
                    ctypes.c_void_p(amax_cpu.ctypes.data),
                    ctypes.c_void_p(scale_inv_cpu.ctypes.data),
                    ctypes.c_void_p(x_fp8_cpu.ctypes.data),
                    ctypes.c_size_t(x.shape[0]),
                    ctypes.c_size_t(x.shape[1]),
                    ctypes.c_int(dtype))

  x_fp8 = tf.convert_to_tensor(x_fp8_cpu.reshape(inp_shape))

  scale_inv = tf.convert_to_tensor(scale_inv_cpu)
  new_tensor = tf.tensor_scatter_nd_update(
      fp8_meta[scaling_key]["scale_inv"], [[amax_index]], scale_inv)
  fp8_meta[scaling_key]["scale_inv"] = new_tensor

  amax = tf.convert_to_tensor(amax_cpu)
  new_tensor = tf.tensor_scatter_nd_update(
      fp8_meta[scaling_key]["amax_history"], [[0, amax_index]], amax)
  fp8_meta[scaling_key]["amax_history"] = new_tensor
  return x_fp8


def fp8_matmul_wrapper(inp, weight, fp8_meta, mode, A_dtype, B_dtype):
  A = inp
  A_cpu = np.copy(A).flatten()
  if mode == 'fwd':
    A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"][0]
    B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"][1]
  elif mode == 'bwd_input':
    A_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"][0]
    B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"][1]
  elif mode == 'bwd_weight':
    A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"][0]
    B_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"][0]

  A_scale_inv_cpu = np.copy(A_scale_inv).flatten()

  B = weight
  B_cpu = np.copy(B).flatten()
  B_scale_inv_cpu = np.copy(B_scale_inv).flatten()

  # weight is actually transposed weight
  D_shape = (A.shape[0], B.shape[0])
  D = tf.zeros(D_shape, dtype=tf.float32)
  D_cpu = np.copy(D).flatten()

  nvlib.fp8_gemm(ctypes.c_void_p(B_cpu.ctypes.data),
                 ctypes.c_void_p(B_scale_inv_cpu.ctypes.data),
                 ctypes.c_int(B_dtype),
                 ctypes.c_void_p(A_cpu.ctypes.data),
                 ctypes.c_void_p(A_scale_inv_cpu.ctypes.data),
                 ctypes.c_int(A_dtype),
                 ctypes.c_void_p(D_cpu.ctypes.data),
                 ctypes.c_int(B.shape[0]),
                 ctypes.c_int(B.shape[1]),
                 ctypes.c_int(A.shape[0]),
                 ctypes.c_int(A.shape[1]),
                 ctypes.c_bool(True),
                 ctypes.c_bool(False),
                 ctypes.c_bool(False),
                 ctypes.c_bool(False),
                 ctypes.c_bool(False))

  D = tf.convert_to_tensor(D_cpu.reshape(D_shape))
  return D


class MyDense(Layer):
  def __init__(self, units, kernel_initializer, **kwargs):
    super().__init__(**kwargs)
    self.units = int(units) if not isinstance(units, int) else units
    self.kernel_initializer = kernel_initializer

    # fp8 related
    self.fp8 = False
    self.fp8_meta = {}
    self.fp8_meta["recipe"] = get_default_fp8_recipe()
    self.fp8_meta_tensors_initialized = False
    self.fp8_weight_shapes = []


  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            dtype=self.dtype,
            trainable=True,
            initializer=self.kernel_initializer,
        )

    # fp8 related
    self.fp8_weight_shapes.append((last_dim, self.units))

    self.built = True

  def set_meta_tensor(self, fwd):
    """Init scales and amaxes for fwd | bwd."""
    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
    num_fp8_tensors = (
        self.fp8_meta["num_gemms"] * 2 if fwd else self.fp8_meta["num_gemms"]
    )

    self.fp8_meta[fp8_meta_tensor_key] = {}
    self.fp8_meta[fp8_meta_tensor_key]["scale"] = tf.ones(
        (num_fp8_tensors), dtype=tf.float32)
    self.fp8_meta[fp8_meta_tensor_key]["scale_inv"] = tf.ones(
        (num_fp8_tensors), dtype=tf.float32)
    self.fp8_meta[fp8_meta_tensor_key]["amax_history"] = tf.zeros(
        (self.fp8_meta["recipe"].amax_history_len, num_fp8_tensors),
        dtype=tf.float32)

  def init_fp8_meta_tensors(self):
    """Init scales and amaxes."""
    # Checkpoint loaded
    if self.fp8_meta_tensors_initialized:
      return

    self.set_meta_tensor(True)
    self.set_meta_tensor(False)

  def fp8_init(self, num_gemms=1):
    if not is_fp8_enabled():
      self.fp8 = False
      return

    # FP8 is already enabled and recipe is the same, don't do anything.
    if self.fp8 and get_fp8_recipe() == self.fp8_meta["recipe"]:
      return

    # Set FP8, recipe, and other FP8 metadata
    self.fp8 = True
    self.fp8_meta["recipe"] = get_fp8_recipe()
    self.fp8_meta["num_gemms"] = num_gemms

    # Set FP8_MAX per tensor according to recipe
    self.fp8_meta["fp8_max_fwd"] = \
        self.fp8_meta["recipe"].fp8_format.value.max_fwd
    self.fp8_meta["fp8_max_bwd"] = \
        self.fp8_meta["recipe"].fp8_format.value.max_bwd

    # Allocate scales and amaxes
    self.init_fp8_meta_tensors()

  def _get_training_value(self, training=None):
    if training is None:
      training = backend.learning_phase()
    if isinstance(training, int):
      training = bool(training)
    if not self.trainable:
      # When the layer is not trainable, it overrides the value passed
      # from model.
      training = False
    return training

  def pre_forward(self, inputs, training, num_gemms=1):
    self.fp8_init(num_gemms=num_gemms)

    if self.fp8:
      if self.fp8_meta.get("update_amax_and_scale_fwd", False):
        # Previous iteration was grad_enabled
        #copy_amax_from_global_buffer(self.fp8_meta, forward=True)
        amax_and_scale_update(self.fp8_meta, True)
        set_amax_buffer_key_deletion(self.fp8_meta, forward=True)

      if training:
        self.fp8_meta["first_module"] = is_first_fp8_module()

        if self.fp8_meta["first_module"]:
          self.fp8_meta["autocast_id_fwd"] = new_fp8_context_id()
          set_fp8_context_id(self.fp8_meta["autocast_id_fwd"])
        else:
          self.fp8_meta["autocast_id_fwd"] = get_fp8_context_id()

        #add_amax_to_global_buffer(self.fp8_meta, forward=True)
        self.fp8_meta["update_amax_and_scale_fwd"] = True

        # Create an empty tensor as a placeholder for the backprop to correctly
        # know how many tensors to autograd.
        self.fp8_meta["autocast_id_bwd"] = -1
      else:
        self.fp8_meta["update_amax_and_scale_fwd"] = False

  def pre_backward(self, fp8_meta):
    # From previous iteration
    #copy_amax_from_global_buffer(fp8_meta, forward=False)
    amax_and_scale_update(fp8_meta, False)
    set_amax_buffer_key_deletion(fp8_meta, forward=False)

    # Get new backward key.
    #if "autocast_id_bwd" not in fp8_meta:
    if fp8_meta["autocast_id_bwd"] == -1:
      fp8_meta["autocast_id_bwd"] = fp8_meta["autocast_id_fwd"]
    else:
      fp8_meta["autocast_id_bwd"] += 1

    #add_amax_to_global_buffer(fp8_meta, forward=False)

  @tf.custom_gradient
  def fp8_matmul(
      self,
      inp: tf.Tensor,
      fp8_meta: Dict[str, Any],
      fp8_dtype_forward: bool,
      fp8_dtype_backward: bool,
      ) -> tf.Tensor:
    weight = self.kernel

    x_fp8 = cast_to_fp8_wrapper(inp, fp8_meta, 0, True, fp8_dtype_forward)
    x_t_fp8 = tf.transpose(x_fp8)
    weight_fp8 = cast_to_fp8_wrapper(weight, fp8_meta, 1, True,
                                     fp8_dtype_forward)
    weight_t_fp8 = tf.transpose(weight_fp8)
  
    outputs = fp8_matmul_wrapper(x_fp8, weight_t_fp8, fp8_meta, 'fwd',
                                 fp8_dtype_forward, fp8_dtype_forward)

    def grad_fn(upstream, variables):
      self.pre_backward(self.fp8_meta)
      grad_fp8 = cast_to_fp8_wrapper(upstream, fp8_meta, 0, False,
                                     fp8_dtype_backward)
      grad_t_fp8 = tf.transpose(grad_fp8)

      grad_x = fp8_matmul_wrapper(grad_fp8, weight_fp8, fp8_meta, 'bwd_input',
                                  fp8_dtype_backward, fp8_dtype_forward)
      grad_weight = fp8_matmul_wrapper(x_t_fp8, grad_t_fp8, fp8_meta,
                                       'bwd_weight',
                                       fp8_dtype_forward, fp8_dtype_backward)

      # The fp8_meta contains 16 tensors which don't need grads.
      grad_inputs = [grad_x]
      grad_inputs.extend([None] * 15)
      grad_vars = []
      grad_vars.append(grad_weight)
      return grad_inputs, grad_vars
  
    return outputs, grad_fn


  def call(self, inputs, training=None):
    training = self._get_training_value(training)
    self.pre_forward(inputs, training)

    if self.fp8:
      # Recipe is an object of DelayedScaling which is not supported when TF
      # does the autograd. Since it is not used in the computation, we
      # temporarily remove it.
      recipe_copy = self.fp8_meta['recipe']
      fp8_dtype_forward = get_fp8_te_dtype(self.fp8_meta["recipe"],
                                           fprop_tensor=True)
      fp8_dtype_backward = get_fp8_te_dtype(self.fp8_meta["recipe"],
                                            fprop_tensor=False)
      del self.fp8_meta['recipe']
      outputs = self.fp8_matmul(inputs, self.fp8_meta, fp8_dtype_forward,
                                fp8_dtype_backward)
      self.fp8_meta['recipe'] = recipe_copy
    else:
      outputs = tf.matmul(a=inputs, b=self.kernel)

    return outputs
