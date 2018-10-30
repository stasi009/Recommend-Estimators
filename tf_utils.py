import tensorflow as tf

from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def to_sparse_input_and_drop_ignore_values(input_tensor, ignore_value=None):
    """Converts a `Tensor` to a `SparseTensor`, dropping ignore_value cells.

    If `input_tensor` is already a `SparseTensor`, just return it.

    Args:
      input_tensor: A string or integer `Tensor`.
      ignore_value: Entries in `dense_tensor` equal to this value will be
        absent from the resulting `SparseTensor`. If `None`, default value of
        `dense_tensor`'s dtype will be used ('' for `str`, -1 for `int`).

    Returns:
      A `SparseTensor` with the same shape as `input_tensor`.

    Raises:
      ValueError: when `input_tensor`'s rank is `None`.
    """
    input_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        input_tensor)
    if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
        return input_tensor
    with ops.name_scope(None, 'to_sparse_input', (input_tensor, ignore_value,)):
        if ignore_value is None:
            if input_tensor.dtype == dtypes.string:
                # Exception due to TF strings are converted to numpy objects by default.
                ignore_value = ''
            elif input_tensor.dtype.is_integer:
                ignore_value = -1  # -1 has a special meaning of missing feature
            else:
                # NOTE: `as_numpy_dtype` is a property, so with the parentheses this is
                # constructing a new numpy object of the given type, which yields the
                # default value for that type.
                ignore_value = input_tensor.dtype.as_numpy_dtype()
        ignore_value = math_ops.cast(
            ignore_value, input_tensor.dtype, name='ignore_value')
        indices = array_ops.where(
            math_ops.not_equal(input_tensor, ignore_value), name='indices')
        return sparse_tensor_lib.SparseTensor(
            indices=indices,
            values=array_ops.gather_nd(input_tensor, indices, name='values'),
            dense_shape=array_ops.shape(
                input_tensor, out_type=dtypes.int64, name='dense_shape'))



