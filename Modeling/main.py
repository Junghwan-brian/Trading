#%%
from precision_recall import *
from conv_transformer import ConvTransformer
import tensorflow as tf
from DataSet import GetData
import matplotlib.pyplot as plt
from time import time

batch_size = 1024
m_dim = 128
getdata = GetData()
kospi_train_ds, kospi_test_ds, kosdaq_train_ds, kosdaq_test_ds = getdata.DataSet(
    train_batch=batch_size, test_batch=30000
)
model = ConvTransformer(num_heads=4, m_dim=m_dim)
model.build(input_shape=(batch_size, 25, 12))
print(model.summary())

#%%


@tf.function
def train_step(
    model,
    inputs,
    cls_labels,
    reg_labels,
    optimizer,
    cls_loss_obj,
    reg_loss_obj,
    train_loss,
    train_accuracy,
    precision,
    recall,
    csi,
    is_minus,
    train_prec,
    train_rec,
    train_csi,
    train_isMinus,
):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        cls_prediction = prediction[:, :3]
        reg_prediction = prediction[:, -1]
        cls_loss = cls_loss_obj(cls_labels, cls_prediction)
        reg_loss = reg_loss_obj(reg_labels, reg_prediction)
        loss = 2 * cls_loss + reg_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(cls_loss)
    train_accuracy(cls_labels, cls_prediction)
    train_prec(precision(cls_labels, cls_prediction))
    train_rec(recall(cls_labels, cls_prediction))
    train_csi(csi(cls_labels, cls_prediction))
    train_isMinus(is_minus(cls_labels, cls_prediction))


@tf.function
def test_step(
    model,
    inputs,
    cls_labels,
    reg_labels,
    cls_loss_obj,
    reg_loss_obj,
    test_loss,
    test_accuracy,
    precision,
    recall,
    csi,
    is_minus,
    test_prec,
    test_rec,
    test_csi,
    test_isMinus,
):
    prediction = model(inputs, training=False)
    cls_prediction = prediction[:, :3]
    cls_loss = cls_loss_obj(cls_labels, cls_prediction)
    test_loss(cls_loss)
    test_accuracy(cls_labels, cls_prediction)
    test_prec(precision(cls_labels, cls_prediction))
    test_rec(recall(cls_labels, cls_prediction))
    test_csi(csi(cls_labels, cls_prediction))
    test_isMinus(is_minus(cls_labels, cls_prediction))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


#%%
learning_rate = CustomSchedule(m_dim)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-9
)
cls_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
reg_loss_obj = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

precision = single_class_precision(0)
recall = single_class_recall(0)
csi = single_class_csi(0)
is_minus = ratio_of_Minus(0, 1)

train_prec = tf.keras.metrics.Mean()
test_prec = tf.keras.metrics.Mean()

train_rec = tf.keras.metrics.Mean()
test_rec = tf.keras.metrics.Mean()

train_csi = tf.keras.metrics.Mean()
test_csi = tf.keras.metrics.Mean()

train_isMinus = tf.keras.metrics.Mean()
test_isMinus = tf.keras.metrics.Mean()

train_ac_list = []
train_loss_list = []
train_prec_list = []
train_rec_list = []
train_fscore_list = []
train_csi_list = []
train_minus_list = []

test_ac_list = []
test_loss_list = []
test_prec_list = []
test_rec_list = []
test_fscore_list = []
test_csi_list = []
test_minus_list = []

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, "./att_ckpts", max_to_keep=3)


def train_and_checkpoint(manager):
    ckpt.restore(manager.latest_checkpoint)  # check point를 불러오는 함수인것 같다.
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


train_and_checkpoint(manager)

compare_value = 0.2
#%%
t_inputs, t_class_label, t_high_label, t_close_label = next(iter(kosdaq_test_ds))
for epoch in range(20):
    for inputs, class_label, high_label, close_label in kosdaq_train_ds:
        train_step(
            model,
            inputs,
            class_label,
            close_label,
            optimizer,
            cls_loss_obj,
            reg_loss_obj,
            train_loss,
            train_accuracy,
            precision,
            recall,
            csi,
            is_minus,
            train_prec,
            train_rec,
            train_csi,
            train_isMinus,
        )
    train_recall = train_rec.result()
    train_precision = train_prec.result()
    if train_recall != 0 or train_precision != 0:
        train_fscore = (
            2 * (train_recall * train_precision) / (train_recall + train_precision)
        )
    else:
        train_fscore = 0

    test_step(
        model,
        t_inputs,
        t_class_label,
        t_close_label,
        cls_loss_obj,
        reg_loss_obj,
        test_loss,
        test_accuracy,
        precision,
        recall,
        csi,
        is_minus,
        test_prec,
        test_rec,
        test_csi,
        test_isMinus,
    )
    # ckpt.step.assign_add(1)

    test_recall = test_rec.result()
    test_precision = test_prec.result()

    if test_recall != 0 or test_precision != 0:
        test_fscore = (
            2 * (test_recall * test_precision) / (test_recall + test_precision)
        )
    else:
        test_fscore = 0
    print("Epoch : ", epoch + 1)
    if test_csi.result() > compare_value and test_precision > 0.55:
        #     save_path = manager.save()
        #     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        compare_value = test_csi.result()

    print(
        f"train_loss : {train_loss.result()} , train_accuracy : {train_accuracy.result() * 100}"
    )
    print(
        f"train_precision : {train_prec.result()} , train_recall : {train_rec.result()}"
    )
    print(f"train_fscore = {train_fscore} , train_csi = {train_csi.result()}")
    print(f"train_Minus_ratio = {train_isMinus.result()} \n")

    print(
        f"test_loss : {test_loss.result()} , test_accuracy : {test_accuracy.result() * 100}"
    )
    print(f"test_precision : {test_prec.result()} , test_recall : {test_rec.result()}")
    print(f"test_fscore = {test_fscore}, test_csi = {test_csi.result()}")
    print(f"test_Minus_ratio = {test_isMinus.result()} \n")

    train_loss_list.append(train_loss.result())
    train_ac_list.append(train_accuracy.result() * 100)
    train_prec_list.append(train_prec.result())
    train_rec_list.append(train_rec.result())
    train_fscore_list.append(train_fscore)
    train_csi_list.append(train_csi.result())
    train_minus_list.append(train_isMinus.result())

    test_loss_list.append(test_loss.result())
    test_ac_list.append(test_accuracy.result() * 100)
    test_prec_list.append(test_prec.result())
    test_rec_list.append(test_rec.result())
    test_fscore_list.append(test_fscore)
    test_csi_list.append(test_csi.result())
    test_minus_list.append(test_isMinus.result())

    test_loss.reset_states()
    test_accuracy.reset_states()
    test_rec.reset_states()
    test_prec.reset_states()
    test_csi.reset_states()
    test_isMinus.reset_states()

    train_loss.reset_states()
    train_accuracy.reset_states()
    train_rec.reset_states()
    train_prec.reset_states()
    train_csi.reset_states()
    train_isMinus.reset_states()
print("MAX CSI: ", compare_value)
#%%

plt.figure(figsize=(12, 15))

plt.subplot(4, 2, 1)
plt.plot(test_ac_list, label="test")
plt.plot(train_ac_list, label="train")
plt.title("Accuracy")
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(test_loss_list, label="test")
plt.plot(train_loss_list, label="train")
plt.title("Loss")
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(test_prec_list, label="test")
plt.plot(train_prec_list, label="train")
plt.title("Precision")
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(test_rec_list, label="test")
plt.plot(train_rec_list, label="train")
plt.title("Recall")
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(test_fscore_list, label="test")
plt.plot(train_fscore_list, label="train")
plt.title("Fscore")
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(test_csi_list, label="test")
plt.plot(train_csi_list, label="train")
plt.title("Critical success index")
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(test_minus_list, label="test")
plt.plot(train_minus_list, label="train")
plt.title("Minus Ratio")
plt.legend()

plt.show()

#%%
model1 = tf.keras.models.load_model("convTransformer_light.h5")
model2 = tf.keras.models.load_model("convTransformer.h5")

#%%
getdata = GetData()
all_kospi_ds, all_kosdaq_ds = getdata.AllTestSet()

t_inputs, t_labels, _, _ = next(iter(all_kospi_ds))
t = time()
pred1 = model1.predict(t_inputs)
pred2 = model2.predict(t_inputs)
pred = (pred1 + pred2) / 2
print(t_labels.shape[0], " 개의 데이터")
print(f"{time() - t:.2f}초")
print("Kospi recall0 - ", single_class_recall(0)(t_labels, pred).numpy())
print("Kospi precision0 - ", single_class_precision(0)(t_labels, pred).numpy())
print("Kospi csi0 - ", single_class_csi(0)(t_labels, pred).numpy())
print("Kospi isMinus", ratio_of_Minus(0, 1)(t_labels, pred).numpy())

t_inputs, t_labels, _, _ = next(iter(all_kosdaq_ds))
t = time()
pred1 = model1.predict(t_inputs)
pred2 = model2.predict(t_inputs)
pred = (pred1 + pred2) / 2
print(t_labels.shape[0], " 개의 데이터")
print(f"{time() - t:.2f}초")
print("Kosdaq recall0 - ", single_class_recall(0)(t_labels, pred).numpy())
print("Kosdaq precision0 - ", single_class_precision(0)(t_labels, pred).numpy())
print("Kosdaq csi0 - ", single_class_csi(0)(t_labels, pred).numpy())
print("Kosdaq isMinus", ratio_of_Minus(0, 1)(t_labels, pred).numpy())
#%%
num = -1000
t = time()
pred1 = model1.predict(t_inputs[num:])
pred2 = model2.predict(t_inputs[num:])
pred = (pred1 + 2 * pred2) / 3
print(f"{time() - t:.2f}초")
print("Kosdaq recall0 - ", single_class_recall(0)(t_labels[num:], pred).numpy())
print("Kosdaq precision0 - ", single_class_precision(0)(t_labels[num:], pred).numpy())
print("Kosdaq csi0 - ", single_class_csi(0)(t_labels[num:], pred).numpy())
print("Kosdaq isMinus", ratio_of_Minus(0, 1)(t_labels[num:], pred).numpy())
