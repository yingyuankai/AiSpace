# -*- coding: utf-8 -*-
# @Time    : 2019-08-21 14:05
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tf_utils.py

import tensorflow as tf
import numbers
import numpy as np
import six
from collections import OrderedDict

from aispace.layers import activations

__all__ = [
    "get_initializer",
    "get_bias_initializer",
    "get_shape",
    "assert_rank",
    "pack_inputs",
    "unpack_inputs",
    "create_attention_mask_from_input_mask"
]


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
      Args:
        initializer_range: float, initializer range for stddev.
      Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
      """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def get_bias_initializer(type='conv'):
    """
    create bias constant initializer
    :param type: if type equals to conv, return 0 constant initializer, if equals to dense, return 1 constant initializer
    :return: tensorflow constant initializer
    """
    if type == 'dense':
        return tf.constant_initializer(1)
    else:
        return tf.constant_initializer(0)


def pack_inputs(inputs):
    """Pack a list of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is None, replace it with a special constant
    tensor.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if x is None:
            outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
        else:
            outputs.append(x)
    return tuple(outputs)


def unpack_inputs(inputs):
    """unpack a tuple of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is a special constant tensor, replace it
    with None.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if is_special_none_tensor(x):
            outputs.append(None)
        else:
            outputs.append(x)
    x = tuple(outputs)

    # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
    # from triggering.
    if len(x) == 1:
        return x[0]
    return tuple(outputs)


def is_special_none_tensor(tensor):
    """Checks if a tensor is a special None Tensor."""
    return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


def get_shape(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank)))


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
    from_shape = get_shape(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
        dtype=from_tensor.dtype)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=from_tensor.dtype)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

  It checks string first and if it is one of customized activation not in TF,
  the corresponding activation will be returned. For non-customized activation
  names and callable identifiers, always fallback to tf.keras.activations.get.

  Args:
    identifier: String name of the activation function or callable.

  Returns:
    A Python function corresponding to the activation function.
  """
    if isinstance(identifier, six.string_types):
        name_to_fn = {
            "gelu": activations.gelu,
            "simple_swish": activations.simple_swish,
            "hard_swish": activations.hard_swish,
            "identity": activations.identity,
        }
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


def get_sequence_length(sequence):
    length = tf.reduce_sum(tf.sign(tf.abs(sequence)), -1)
    return length


def tf_gather(from_tensor, indices):
    """

    :param from_tensor: [batch_size, seq_length, hidden_size]
    :param indices: [batch_size, max_indices_num]
    :return: [batch_size, max_indices_num, hidden_size]
    """
    seq_length = get_shape(from_tensor)[1]
    # [batch_size, hidden_size, seq_length]
    from_tensor_transposed = tf.cast(tf.transpose(from_tensor, [0, 2, 1]), tf.float32)
    # [batch_size, seq_length, max_indices_num]
    indices_one_hot = tf.cast(tf.transpose(tf.one_hot(indices, seq_length), [0, 2, 1]), tf.float32)
    # [batch_size, max_indices_num, hidden_size]
    gather_res = tf.transpose(tf.matmul(from_tensor_transposed, indices_one_hot), [0, 2, 1])
    return gather_res


# @tf.function(experimental_relax_shapes=True)
def generate_relative_positions_matrix(length, max_relative_position,
                                       cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat = range_mat - tf.transpose(range_mat)
    else:
        distance_mat = tf.expand_dims(tf.range(-length + 1, 1, 1), 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


# @tf.function(experimental_relax_shapes=True)
def generate_relative_positions_embeddings(length, depth,
                                           max_relative_position, name,
                                           cache=False):
    """
    Generates tensor of size [1 if cache else length, length, depth].
    example:
        # `relation_keys` = [F|T, F|T, H]
           relations_keys = _generate_relative_positions_embeddings(
        to_seq_length, size_per_head, max_relative_position, "relative_positions_keys",
        cache=False)
      relations_keys = tf.saturate_cast(relations_keys, compute_type)
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
      length = to_seq_length
      depth = size_per_head
      max_relative_position
      name = "relative_positions_keys"
    """
    # '''
    relative_positions_matrix = generate_relative_positions_matrix(
        length, max_relative_position, cache=cache)
    vocab_size = max_relative_position * 2 + 1
    # Generates embedding for each relative position of dimension depth.
    embeddings_table = np.zeros([vocab_size, depth])

    position = tf.range(0.0, vocab_size, 1.0)
    position = tf.reshape(position, [vocab_size, -1])

    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = tf.convert_to_tensor(embeddings_table, tf.float32)
    flat_relative_positions_matrix = tf.reshape(relative_positions_matrix, [-1])
    # [length * length?, vocab_size]
    one_hot_relative_positions_matrix = tf.one_hot(flat_relative_positions_matrix, depth=vocab_size)

    embeddings = tf.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)

    my_shape = get_shape(relative_positions_matrix)
    my_shape.append(depth)

    embeddings = tf.reshape(embeddings, my_shape)
    return embeddings


###################
# < tf 2.0

def sequence_length_3D(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def sequence_length_2D(sequence):
    used = tf.sign(tf.abs(sequence))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def minimum_bounding_box(input):
    input_shape = get_shape(input)
    if len(input_shape) == 2:
        length = sequence_length_2D(input)
        max_len = tf.reduce_max(length, axis=-1)
        mbb_shape = (input_shape[0], max_len)
        output = tf.slice(input, [0, 0], mbb_shape)
        return output
    elif len(input_shape) == 3:
        used = tf.sign(tf.abs(input))
        max_len_d1 = tf.reduce_max(tf.reduce_sum(used, axis=1))
        max_len_d2 = tf.reduce_max(tf.reduce_sum(used, axis=2))
        mbb_shape = (input_shape[0], max_len_d1, max_len_d2)
        output = tf.slice(input, [0, 0, 0], mbb_shape)
        return output
    else:
        raise ValueError('input must be 2 or 3 dim.')


# Convert a dense matrix into a sparse matrix (for e.g. edit_distance)
def to_sparse(tensor, lengths, max_length):
    mask = tf.sequence_mask(lengths, max_length)
    indices = tf.to_int64(tf.where(tf.equal(mask, True)))
    values = tf.to_int32(tf.boolean_mask(tensor, mask))
    shape = tf.to_int64(tf.shape(tensor))
    return tf.SparseTensor(indices, values, shape)


def get_tf_config(gpus=None, gpu_fraction=1, horovod=None,
                  allow_parallel_threads=True):
    intra_op_parallelism_threads = 2  # defult in tensorflow
    inter_op_parallelism_threads = 5  # defult in tensorflow
    if not allow_parallel_threads:
        # this is needed for reproducibility
        intra_op_parallelism_threads = 1
        inter_op_parallelism_threads = 1

    if gpus is not None:
        if 0 < gpu_fraction < 1:
            # this is the source of freezing in tensorflow 1.3.1
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_fraction,
                allow_growth=True)
        else:
            gpu_options = tf.GPUOptions()
        if isinstance(gpus, int):
            gpus = [gpus]
        gpu_options.visible_device_list = ','.join(str(g) for g in gpus)
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=False,
                                   intra_op_parallelism_threads=intra_op_parallelism_threads,
                                   inter_op_parallelism_threads=inter_op_parallelism_threads,
                                   gpu_options=gpu_options)
    else:
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=False,
                                   intra_op_parallelism_threads=intra_op_parallelism_threads,
                                   inter_op_parallelism_threads=inter_op_parallelism_threads)

    if horovod is not None:
        tf_config.gpu_options.visible_device_list = str(horovod.local_rank())

    return tf_config


def reshape(tensor, dims_list):
    """Reshape the given tensor by collapsing dimensions."""
    shape = get_shape(tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, numbers.Number):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(tf.prod([shape[d] for d in dims]))
    tensor = tf.reshape(tensor, dims_prod)
    return tensor


def expand_dims(tensor, dims):
    """Expand the rank of a tensor by inserting singular dimensions."""
    if isinstance(dims, numbers.Number):
        dims = [dims]
    for dim in dims:
        tensor = tf.expand_dims(tensor, dim)
    return tensor


def tile_like(tensor, like):
    """Tile a tensor to match another."""
    tensor = tf.tile(tensor, tf.shape(like) / tf.shape(tensor))
    return tensor


def dropout(x, keep_prob, training, noise_shape=None):
    if keep_prob >= 1.0:
        return x
    if isinstance(training, bool):
        training = tf.constant(training, dtype=tf.bool)
    return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape), lambda: x)


def weighted_sum(seq, prob):
    return tf.reduce_sum(seq * tf.expand_dims(prob, axis=2), axis=1)


def masked_softmax(logits, mask, is_training):
    # if is_training:
    #     return tf.nn.softmax(logits, axis=-1)
    if len(logits.shape.as_list()) != len(mask.shape.as_list()):
        mask = tf.sequence_mask(mask, tf.shape(logits)[1], dtype=tf.float32)
    mask = tf.cast(mask, tf.float32)

    return tf.nn.softmax(logits * mask + (1.0 - mask) * tf.float32.min, axis=-1)


def masked_log_softmax(logits, mask):
    if len(logits.shape.as_list()) != len(mask.shape.as_list()):
        mask = tf.sequence_mask(mask, tf.shape(logits)[1], dtype=tf.float32)
    mask = tf.cast(mask, tf.float32)

    return tf.nn.log_softmax(logits * mask + (1.0 - mask) * tf.float32.min, axis=-1)


def mask_logits(logits, mask):
    if len(logits.shape.as_list()) != len(mask.shape.as_list()):
        mask = tf.sequence_mask(mask, tf.shape(logits)[1], dtype=tf.float32)
    mask = tf.cast(mask, tf.float32)
    return logits * mask + (1.0 - mask) * tf.float32.min


def add_seq_mask(inputs, seq_len, mode='mul', max_len=None):
    mask = tf.expand_dims(tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32), 2)
    if mode == 'mul':
        return inputs * mask
    if mode == 'add':
        mask = (1 - mask) * tf.float32.min
        return inputs + mask


def generate_onehot_label(input_data, input_depth):
    """Generate one-hot label"""
    return tf.one_hot(input_data, depth=input_depth, on_value=1.0, off_value=0.0, dtype=tf.float32)


def pack_inputs(inputs):
    """Pack a list of `inputs` tensors to a tuple.
  Args:
    inputs: a list of tensors.
  Returns:
    a tuple of tensors. if any input is None, replace it with a special constant
    tensor.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if x is None:
            outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
        else:
            outputs.append(x)
    return tuple(outputs)


def unpack_inputs(inputs):
    """unpack a tuple of `inputs` tensors to a tuple.
  Args:
    inputs: a list of tensors.
  Returns:
    a tuple of tensors. if any input is a special constant tensor, replace it
    with None.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if is_special_none_tensor(x):
            outputs.append(None)
        else:
            outputs.append(x)
    x = tuple(outputs)

    # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
    # from triggering.
    if len(x) == 1:
        return x[0]
    return tuple(outputs)


def is_special_none_tensor(tensor):
    """Checks if a tensor is a special None Tensor."""
    return tensor.shape.ndims == 0 and tensor.dtype == tf.int32
