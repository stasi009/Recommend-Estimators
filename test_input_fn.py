import tensorflow as tf


def test_load_field_multi_tokens():
    # 文件内容如下：
    # a:1,b:2	c:3,b:4,e:5
    # a:-11	a:10,c:-3,e:5,m:8,n:99
    # b:22	b:49,a:15
    # a:1,b:2,z:49,x:15,c:-99	e:-8
    max_tokens = {
        'col1': 2,
        'col2': 3
    }
    col_defaults = [[''] for c in max_tokens.keys()]

    # ------------- parse
    def _parse_tsv(line):
        columns = tf.decode_csv(line, record_defaults=col_defaults, field_delim='\t')
        tmp = dict(zip(max_tokens.keys(), columns))

        features = {}
        for colname, max_pkgs_per_user in max_tokens.items():
            value = tmp[colname]
            # 按最大长度进行截断
            features[colname] = tf.string_split([value], ',').values[:max_pkgs_per_user]

        return features

    # ------------- dataset
    dataset = tf.data.TextLineDataset('dataset/test/test1.tsv').skip(1)  # skip the header
    dataset = dataset.map(_parse_tsv, num_parallel_calls=4)

    dataset = dataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=2,
                                                                  row_shape=max_tokens))


def test_parse_field_feature_value():
    """
    输入文件是一个tsv文件，每列是一个field
    每列的内容是一个csv，格式是: feature1:value1,feature2:value2,...,featureN:valueN
    """
    # -------- prepare input
    segments = ['a:1,b:2', 'c:3,d:4,e:5']
    line = '\t'.join(segments)

    # -------- define ops


if __name__ == "__main__":
    test_load_field_multi_tokens()
