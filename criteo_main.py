import tensorflow as tf
from deepfm import model_fn
from criteo_data_load import input_fn


def get_hparams():
    vocab_sizes = {
        'numeric': 13,
        # there are totally 14738 categorical tags occur >= 200
        # since 0 is reserved for OOV, so total vocab_size=14739
        'categorical': 14739
    }

    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001)

    return {
        'embed_dim': 128,
        'vocab_sizes': vocab_sizes,
        # 在这个case中，没有多个field共享同一个vocab的情况，而且field_name和vocab_name相同
        'field_vocab_mapping': {'numeric': 'numeric', 'categorical': 'categorical'},
        'dropout_rate': 0.3,
        'batch_norm': False,
        'hidden_units': [64, 32],
        'optimizer': optimizer
    }


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(999)

    hparams = get_hparams()
    deepfm = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir='models/criteo',
                                    params=hparams)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data_file='dataset/criteo/whole_train.tsv',
                                                                  n_repeat=10,
                                                                  batch_size=128,
                                                                  batches_per_shuffle=10))

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(data_file='dataset/criteo/whole_test.tsv',
                                                                n_repeat=1,
                                                                batch_size=128,
                                                                batches_per_shuffle=-1))

    tf.estimator.train_and_evaluate(deepfm, train_spec, eval_spec)
