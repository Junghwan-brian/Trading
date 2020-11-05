import tensorflow as tf

# 특정 클래스에 대한 정밀도(TP/(TP+FP))
def single_class_precision(interesting_class_id):
    def prec(y_true, y_pred):
        class_id_true = tf.cast(y_true, tf.int64)
        class_id_pred = tf.math.argmax(y_pred, axis=-1)
        # mask 는 모델의 예측과 보고자 하는 클래스 아이디가 같은 것을 1로 해서 저장함
        # 틀리면 0이 됨. -> 양성이라고 판정한 수
        precision_mask = tf.cast(
            tf.math.equal(class_id_pred, interesting_class_id), "int32"
        )
        # tensor 는 모델이 맞춘 것들과 예측하는 것이 일치하고
        # id 가 예측하고자 하는 것과 같은 것을 저장함. => 즉, 예측값중 실제 양성수(TP)
        class_prec_tensor = (
            tf.cast(tf.math.equal(class_id_true, class_id_pred), "int32")
            * precision_mask
        )
        # 실제 양성수 / 양성이라고 판정한 수
        class_prec = tf.cast(
            tf.math.reduce_sum(class_prec_tensor), "float32"
        ) / tf.cast(tf.math.maximum(tf.math.reduce_sum(precision_mask), 1), "float32")
        return class_prec

    return prec


# 특정 클래스에 대한 재현율(TP/(TP+FN))
def single_class_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = tf.cast(y_true, tf.int64)
        class_id_pred = tf.math.argmax(y_pred, axis=-1)
        # 전체 양성수
        recall_mask = tf.cast(
            tf.math.equal(class_id_true, interesting_class_id), "int32"
        )
        # 검출 양성수
        class_recall_tensor = (
            tf.cast(tf.math.equal(class_id_true, class_id_pred), "int32") * recall_mask
        )
        # 검출 양성수/전체 양성수
        class_recall = tf.cast(
            tf.math.reduce_sum(class_recall_tensor), "float32"
        ) / tf.cast(tf.math.maximum(tf.math.reduce_sum(recall_mask), 1), "float32")
        return class_recall

    return recall


# critical success index - TP/(TP+FP+FN)
def single_class_csi(interesting_class_id):
    def csi(y_true, y_pred):
        class_id_true = tf.cast(y_true, tf.int64)
        class_id_pred = tf.math.argmax(y_pred, axis=-1)
        # 전체 양성수(TP+FN)
        real_positive = tf.cast(
            tf.math.equal(class_id_true, interesting_class_id), "int32"
        )
        # 양성이라고 판정한 수(TP+FP)
        pred_positive = tf.cast(
            tf.math.equal(class_id_pred, interesting_class_id), "int32"
        )
        # 검출 양성수(TP)
        true_positive = (
            tf.cast(tf.math.equal(class_id_true, class_id_pred), "int32")
            * real_positive
        )
        tp_fp_fn = real_positive + pred_positive - true_positive
        # 검출 양성수/전체 양성수
        class_csi = tf.cast(tf.math.reduce_sum(true_positive), "float32") / tf.cast(
            tf.math.maximum(tf.math.reduce_sum(tp_fp_fn), 1), "float32"
        )
        return class_csi

    return csi


# Pred = 0 인 것중에 라벨이 1인 것의 비율을 구한다.
def ratio_of_Minus(interesting_class_id=0, minus_class_id=1):
    def isMinus(y_true, y_pred):
        class_id_true = tf.cast(y_true, tf.int64)
        class_id_pred = tf.math.argmax(y_pred, axis=-1)
        pred_positive = tf.cast(
            tf.math.equal(class_id_pred, interesting_class_id), "int32"
        )
        minus_mask = tf.cast(tf.math.equal(class_id_true, minus_class_id), "int32")
        pred_minus_class = tf.cast(
            tf.math.reduce_sum(pred_positive * minus_mask), "float32"
        )
        total_num = tf.maximum(tf.cast(tf.math.reduce_sum(pred_positive), "float32"), 1)
        minus_ratio = tf.cast(pred_minus_class / total_num, "float32")

        return minus_ratio

    return isMinus
