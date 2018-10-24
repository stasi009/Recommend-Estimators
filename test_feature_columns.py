import pandas as pd
import tensorflow as tf
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_cate_featcol_with_vocablist():
    # ================== prepare input
    # 1. 为什么要用b前缀，这是因为通过input_fn读进来的字符串都有b前缀
    # 而我要测试的是，如果我传入的vocab list是普通的str，能否与这些b匹配成功
    # 2. '' represents missing, feature_column treats them "ignored in sparse tensor, and 0 in dense tensor"
    # 3. 'z' represents OOV, feature_column treats them "-1 in sparse tensor, and 0 in dense tensor"
    # 4. duplicates should be merged in dense tensor by summing up their occurrence
    x_values = {'x': [[b'a', b'z', b'a', b'c'],
                      [b'b', b'', b'd', b'b']]}
    builder = _LazyBuilder(x_values)  # lazy representation of input

    # ================== define ops
    sparse_featcol = feature_column.categorical_column_with_vocabulary_list(
        'x', ['a', 'b', 'c'], dtype=tf.string, default_value=-1)
    x_sparse_tensor = sparse_featcol._get_sparse_tensors(builder)
    # 尽管一行中有重复，但是并没有合并，所以压根就没有weight
    # 只是导致id_tensor中会出现重复数值而已，而导致embedding_lookup_sparse时出现累加
    assert x_sparse_tensor.weight_tensor is None

    # 将sparse转换成dense，注意第一行有重复，所以结果应该是multi-hot
    dense_featcol = feature_column.indicator_column(sparse_featcol)
    x_dense_tensor = feature_column.input_layer(x_values, [dense_featcol])

    # ================== run
    with tf.Session() as sess:
        # 必须initialize table，否则报错
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        print("************************* sparse tensor")
        # 结果证明：
        # 1. 输入数据用b前缀的字符串，能够匹配上vocab list中的str
        # 2. 注意第二行只有两个元素，''在sparse tensor中被忽略掉了
        # 3. 'z','d'代表oov，sparse tensor中被映射成-1
        # 4. sparse tensor的dense_shape与原始输入的shape相同
        # [SparseTensorValue(indices=array([[0, 0],
        #                                   [0, 1],
        #                                   [0, 2],
        #                                   [0, 3],
        #                                   [1, 0],
        #                                   [1, 2],
        #                                   [1, 3]]), values=array([0, -1, 0, 2, 1, -1, 1]), dense_shape=array([2, 4]))]
        print(sess.run([x_sparse_tensor.id_tensor]))

        print("************************* dense MHE tensor")
        # 结果证明：
        # 1. 在dense表示中，duplicates的出现次数被加和，使用MHE
        # 2. 无论是原始的missing（或许是由padding造成的），还是oov，在dense结果中都不出现
        # 3. dense_tensor的shape=[#batch_size, vocab_size]
        # [array([[2., 0., 1.],
        #         [0., 2., 0.]], dtype=float32)]
        print(sess.run([x_dense_tensor]))
