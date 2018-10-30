from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self):
        # There are 13 integer features and 26 categorical features
        self._cols_int = ['I{}'.format(idx) for idx in range(1, 14)]
        self._cols_category = ['C{}'.format(idx) for idx in range(1, 27)]

        print("loading original data, ......")
        df = pd.read_csv("dataset/criteo/raw.tsv", sep='\t', header=None)
        df.columns = ['label'] + self._cols_int + self._cols_category

        # fillna是为了让处理tag时，不会被nan所干扰
        # 不会影响稀疏性，反正nan和0，在导出时，都会被剔除
        df.fillna(value={c: '' for c in self._cols_category}, inplace=True)
        print("original data[{}] loaded".format(df.shape))

        self._y = df['label']
        self._X = df.loc[:, df.columns != 'label']
        self._preproced_df = None

    def _normalize_numerics(self, upper_bound):
        numeric_features = self._X.loc[:, self._cols_int].copy()

        numeric_features.clip_upper(upper_bound, axis=1, inplace=True)

        # I2有小于0的值
        # -1    204968(占10%左右)
        # -2      1229
        # -3         1
        numeric_features['I2'] = (numeric_features['I2'] + 1).clip_lower(0)

        numeric_features = np.log1p(numeric_features)

        # 既然不能用zero mean,unit variance scaling，因为那样会破坏数据的稀疏性
        # 最简单的就是用min-max-scaling
        # 理论做法，应该先split train and test，再做scaling
        # 这里就不那么讲究了，差别也没有那么大
        # (因为numeric_features不是ndarray，没有被minmax_scale inplace modify的可能性，也就没设置copy=False)
        col_min = numeric_features.min()
        col_max = numeric_features.max()
        return (numeric_features - col_min) / (col_max - col_min)

    def _build_catval_vocab(self, min_occur):
        vocab = []
        for c in self._cols_category:
            cat_counts = self._X[c].value_counts()

            valid_catcounts = cat_counts.loc[cat_counts >= min_occur]
            print("column[{}] has {} tags occur >= {} times".format(c, valid_catcounts.shape[0], min_occur))

            # 尽管tag都已经hash过了，但是以防万一，还是加上所属列名，以防止重复
            vocab.extend("{}/{}".format(c, tag) for tag in valid_catcounts.index)

        print("*** there are totally {} categorical tags occur >= {} times ***".format(len(vocab), min_occur))
        # idx从1开始计数，0号位置为'罕见的tag'(出现不足min_occur次)预留
        return {tag: idx for idx, tag in enumerate(vocab, start=1)}

    def _transform_numerics_row(self, row):
        txts = []

        # int型特征，只有13列，没有OOV的问题，idx可以从0开始
        for idx, c in enumerate(self._cols_int):
            value = row[c]

            # 0和nan都不会加入特征，以保持稀疏性
            if np.isnan(value) or abs(value) < 1e-6:
                continue

            txts.append("{}:{:.6f}".format(idx, value))

        return ",".join(txts)

    def _transform_categorical_row(self, row, tag2idx):
        txts = []

        for c in self._cols_category:
            tag = row[c]
            if len(tag) == 0:  # ignore whitespace
                continue

            # 尽管tag都已经hash过了，但是以防万一，还是加上所属列名，以防止重复
            idx = tag2idx.get("{}/{}".format(c, tag), 0)  # 0号位置为OOV预留

            txts.append("{}:1".format(idx))

        return ",".join(txts)

    def run(self, int_upper_bound, cat_min_occur):
        print("\n============ preprocessing numeric features, ......")
        normed_numeric_feats = self._normalize_numerics(int_upper_bound)
        merged_normed_numeric_feats = normed_numeric_feats.progress_apply(self._transform_numerics_row,
                                                                          axis=1)
        print("\n============ numeric features preprocessed")

        print("\n============ preprocessing categorical features, ......")
        tag2idx = self._build_catval_vocab(cat_min_occur)
        merged_categorical_feats = self._X.progress_apply(lambda row: self._transform_categorical_row(row, tag2idx),
                                                          axis=1)
        print("\n============ categorical features preprocessed")

        self._preproced_df = pd.concat([self._y, merged_normed_numeric_feats, merged_categorical_feats],
                                       keys=['label', 'numeric', 'categorical'],
                                       axis=1)

    def split_save(self, prefix, test_ratio=0.2, sample_ratio=1.0):
        df = self._preproced_df
        if sample_ratio < 1:
            df = self._preproced_df.sample(frac=sample_ratio)

        train_df, test_df = train_test_split(df, test_size=test_ratio)

        outfname = 'dataset/criteo/{}_train.tsv'.format(prefix)
        train_df.to_csv(outfname, sep='\t', index=False)
        print("data[{}] saved to '{}'".format(train_df.shape, outfname))

        outfname = 'dataset/criteo/{}_test.tsv'.format(prefix)
        test_df.to_csv(outfname, sep='\t', index=False)
        print("data[{}] saved to '{}'".format(test_df.shape, outfname))


if __name__ == "__main__":
    tqdm.pandas()

    preproc = DataPreprocessor()

    # int_upper_bound: derived from the 95% quantile of the total values in each feature
    # and they are copied from PaddlePaddle preprocessing codes
    int_upper_bound = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]

    # 因为所有样本总共2000000，其0.0001（万分之一）是200次
    # !!! 其实这个cutoff，还是抛弃了大量的长尾，实际系统中不可取
    # !!! 不过，对于测试来说，也算可以接受了
    cat_min_occur = 200

    preproc.run(int_upper_bound=int_upper_bound, cat_min_occur=cat_min_occur)

    # 一分抽样后的小数据，快速测试用
    preproc.split_save(prefix='small', test_ratio=0.2, sample_ratio=0.025)

    # 全量数据
    preproc.split_save(prefix='whole', test_ratio=0.2, sample_ratio=1.0)
