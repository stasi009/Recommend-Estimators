import tensorflow as tf
import tf_utils

COLUMNS_MAX_TOKENS = [('numeric', 13), ('categorical', 26)]
DEFAULT_VALUES = [[0], [''], ['']]


def _decode_tsv(line):
    columns = tf.decode_csv(line, record_defaults=DEFAULT_VALUES, field_delim='\t')
    y = columns[0]

    feat_columns = dict(zip((t[0] for t in COLUMNS_MAX_TOKENS), columns[1:]))
    X = {}
    for colname, max_tokens in COLUMNS_MAX_TOKENS:
        # 调用string_split时，第一个参数必须是一个list，所以要把columns[colname]放在[]中
        # 这时每个kv还是'k:v'这样的字符串
        kvpairs = tf.string_split([feat_columns[colname]], ',').values[:max_tokens]

        # k,v已经拆开, kvpairs是一个SparseTensor，因为每个kvpair格式相同，都是"k:v"
        # 既不会出现"k"，也不会出现"k:v1:v2:v3:..."
        # 所以，这时的kvpairs实际上是一个满阵
        kvpairs = tf.string_split(kvpairs, ':')

        # kvpairs是一个[n_valid_pairs,2]矩阵
        kvpairs = tf.reshape(kvpairs.values, kvpairs.dense_shape)

        feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

        # 不能调用squeeze, squeeze的限制太多, 当原始矩阵有1行或0行时，squeeze都会报错
        X[colname + "_ids"] = tf.reshape(feat_ids, shape=[-1])
        X[colname + "_values"] = tf.reshape(feat_vals, shape=[-1])

    return X, y


def input_fn(data_file, n_repeat, batch_size, batches_per_shuffle):
    # ----------- prepare padding
    pad_shapes = {}
    pad_values = {}
    for c, max_tokens in COLUMNS_MAX_TOKENS:
        pad_shapes[c + "_ids"] = tf.TensorShape([max_tokens])
        pad_shapes[c + "_values"] = tf.TensorShape([max_tokens])

        pad_values[c + "_ids"] = -1  # 0 is still valid token-id, -1 for padding
        pad_values[c + "_values"] = 0.0

    # no need to pad labels
    pad_shapes = (pad_shapes, tf.TensorShape([]))
    pad_values = (pad_values, 0)

    # ----------- define reading ops
    dataset = tf.data.TextLineDataset(data_file).skip(1)  # skip the header
    dataset = dataset.map(_decode_tsv, num_parallel_calls=4)

    if batches_per_shuffle > 0:
        dataset = dataset.shuffle(batches_per_shuffle * batch_size)

    dataset = dataset.repeat(n_repeat)
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=pad_shapes,
                                   padding_values=pad_values)

    iterator = dataset.make_one_shot_iterator()
    dense_Xs, ys = iterator.get_next()

    # ----------- convert dense to sparse
    sparse_Xs = {}
    for c, _ in COLUMNS_MAX_TOKENS:
        for suffix in ["ids", "values"]:
            k = "{}_{}".format(c, suffix)
            sparse_Xs[k] = tf_utils.to_sparse_input_and_drop_ignore_values(dense_Xs[k])

    # ----------- return
    return sparse_Xs, ys
