import tensorflow as tf
import os


class GetData:
    def __init__(self):
        self.N_KOSPI_TRAIN = 738498
        self.N_KOSPI_TEST = 280398
        self.N_KOSDAQ_TRAIN = 1232292
        self.N_KOSDAQ_TEST = 442590
    # tfrecord file을 data로 parsing해주는 function
    def _parse_function(self, tfrecord_serialized):
        features = {
            "class_label": tf.io.FixedLenFeature([], tf.float32),
            "high_label": tf.io.FixedLenFeature([], tf.float32),
            "close_label": tf.io.FixedLenFeature([], tf.float32),
            "arr": tf.io.FixedLenFeature([], tf.string),
            "info": tf.io.FixedLenFeature([], tf.string),
            "var": tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

        arr = tf.io.decode_raw(parsed_features["arr"], tf.float32)
        info = tf.io.decode_raw(parsed_features["info"], tf.float32)
        var = tf.io.decode_raw(parsed_features["var"], tf.float32)

        arr = tf.cast(arr, tf.float32)  # n,2(min,day),24,5
        info = tf.cast(info, tf.float32)  # n,2(min,day),24,6
        var = tf.cast(var, tf.float32)  # n,11

        arr = tf.reshape(arr, (2, 24, 5))
        info = tf.reshape(info, (2, 24, 6))

        min_arr, day_arr = tf.split(arr, [1, 1], axis=0)
        min_info, day_info = tf.split(info, [1, 1], axis=0)

        enc_data = tf.reshape(
            tf.concat([day_info, day_arr], axis=-1), (24, 11)
        )  # 24,11
        dec_data = tf.reshape(tf.concat([min_info, min_arr], axis=-1), (24, 11))

        data = tf.concat([enc_data, dec_data, var[tf.newaxis, :]], axis=0)  # 49,11

        class_label = tf.cast(parsed_features["class_label"], tf.float32)
        high_label = tf.cast(parsed_features["high_label"], tf.float32)
        close_label = tf.cast(parsed_features["close_label"], tf.float32)

        return data, class_label, high_label, close_label

    def DataSet(self, train_batch=512, test_batch=5000):
        cur_dir = os.curdir
        tfr_dir = os.path.join(cur_dir, "tfrecord")

        kospi_tfr_train_dir = os.path.join(tfr_dir, "kospi/cls_train.tfr")
        kospi_tfr_test_dir = os.path.join(tfr_dir, "kospi/cls_test.tfr")

        ## kospi train dataset 만들기
        kospi_train_dataset = tf.data.TFRecordDataset(kospi_tfr_train_dir)
        kospi_train_dataset = kospi_train_dataset.map(
            self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        kospi_train_dataset = (
            kospi_train_dataset.shuffle(self.N_KOSPI_TRAIN,seed=2020)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .batch(train_batch)
        )

        kospi_test_dataset = tf.data.TFRecordDataset(kospi_tfr_test_dir)
        kospi_test_dataset = (
            kospi_test_dataset.map(
                self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .shuffle(self.N_KOSPI_TEST,seed=2020)
            .batch(test_batch)
        )

        kosdaq_tfr_train_dir = os.path.join(tfr_dir, "kosdaq/cls_train.tfr")
        kosdaq_tfr_test_dir = os.path.join(tfr_dir, "kosdaq/cls_test.tfr")

        ## kosdaq train dataset 만들기
        kosdaq_train_dataset = tf.data.TFRecordDataset(kosdaq_tfr_train_dir)
        kosdaq_train_dataset = kosdaq_train_dataset.map(
            self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        kosdaq_train_dataset = (
            kosdaq_train_dataset.shuffle(self.N_KOSDAQ_TRAIN,seed=2020)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .batch(train_batch)
        )

        kosdaq_test_dataset = tf.data.TFRecordDataset(kosdaq_tfr_test_dir)
        kosdaq_test_dataset = (
            kosdaq_test_dataset.map(
                self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .shuffle(self.N_KOSDAQ_TEST,seed=2020)
            .batch(test_batch)
        )
        return (
            kospi_train_dataset,
            kospi_test_dataset,
            kosdaq_train_dataset,
            kosdaq_test_dataset,
        )
    def AllTestSet(self):
        cur_dir = os.curdir
        tfr_dir = os.path.join(cur_dir, "tfrecord")

        kospi_tfr_train_dir = os.path.join(tfr_dir, "kospi/cls_train.tfr")
        kospi_tfr_test_dir = os.path.join(tfr_dir, "kospi/cls_test.tfr")

        kospi_test_dataset = tf.data.TFRecordDataset(kospi_tfr_test_dir)
        kospi_test_dataset = kospi_test_dataset.map(
            self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.N_KOSPI_TEST)

        kosdaq_tfr_test_dir = os.path.join(tfr_dir, "kosdaq/cls_test.tfr")

        kosdaq_test_dataset = tf.data.TFRecordDataset(kosdaq_tfr_test_dir)
        kosdaq_test_dataset = kosdaq_test_dataset.map(
            self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.N_KOSDAQ_TEST)
        return (
            kospi_test_dataset,
            kosdaq_test_dataset,
        )
