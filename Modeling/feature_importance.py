#%%
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from precision_recall import *
import matplotlib.pyplot as plt

cur_dir = os.path.curdir
path = os.path.join(cur_dir, "data/검증데이터")
arr = np.load(open(os.path.join(path, "arr.npy"), "rb"))
info = np.load(open(os.path.join(path, "info.npy"), "rb"))
var = np.load(open(os.path.join(path, "var.npy"), "rb"))
class_labels = np.load(open(os.path.join(path, "class_labels.npy"), "rb"))
#%%
kospi_light_model = load_model("kospi_convTransformer_light.h5")
# total_light_model = load_model("total_convTransformer_light.h5")
#%%
min_arr, _, day_arr = np.split(arr, [1, 1], axis=1)
min_info, _, day_info = np.split(info, [1, 1], axis=1)
enc_data = np.reshape(
    np.concatenate([day_info, day_arr], axis=-1), (-1, 24, 11)
)  # 24,11
dec_data = np.reshape(np.concatenate([min_info, min_arr], axis=-1), (-1, 24, 11))
data = np.concatenate([enc_data, dec_data, var[:, np.newaxis, :]], axis=1)  # 49,11
labels = np.reshape(class_labels, (-1,))
#%%
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
precision = single_class_precision(0)
recall = single_class_recall(0)
csi = single_class_csi(0)
is_minus = ratio_of_Minus(0, 1)

test_prec = tf.keras.metrics.Mean()
test_rec = tf.keras.metrics.Mean()
test_csi = tf.keras.metrics.Mean()
test_isMinus = tf.keras.metrics.Mean()
#%%
origin_pred1 = kospi_light_model.predict(data)
# origin_pred2 = total_light_model.predict(data)
accuracy(labels, origin_pred1)
test_prec(precision(labels, origin_pred1))
test_rec(recall(labels, origin_pred1))
test_csi(csi(labels, origin_pred1))
test_isMinus(is_minus(labels, origin_pred1))

origin_accuracy = accuracy.result() * 100
origin_precision = test_prec.result()
origin_recall = test_rec.result()
origin_csi = test_csi.result()
origin_minus_ratio = test_isMinus.result()

print(f"Accuracy : {origin_accuracy}")
print(f"Precision : {origin_precision}")
print(f"Recall : {origin_recall}")
print(f"CSI : {origin_csi}")
print(f"Minus Ratio : {origin_minus_ratio}")
#%%
"""
enc : 0~23, dec : 24~48 , var : 49
info: close_open, high_open, open_low, high_low, high_close, low_close
arr : "open", "high", "low", "close", "vol"
var :시간,거래량*6(30분거래량 값과 scale을 맞추기 위해 6 곱함)/30일평균거래량,
     30분거래량/30일평균거래량 , 5분종가(현재가)/최근30일고점 및 저점,
     현재가/분단위 이평선,현재가/시초가,현재가/하루 고점=>실시간업데이트, 현재가/하루 저점, 현재가/30일 평균가
     현재가/ 2시간 평균값(시가,종가,저가,고가 모두 고려한 값)
"""
day_acc_result = []
day_prec_result = []
day_rec_result = []
day_csi_result = []

min_acc_result = []
min_prec_result = []
min_rec_result = []
min_csi_result = []

var_acc_result = []
var_prec_result = []
var_rec_result = []
var_csi_result = []

total_acc_result = []
total_prec_result = []
total_rec_result = []
total_csi_result = []


np.random.seed(0)
for i in [23, 47, 48]:
    for j in range(11):
        copy_data = data.copy()
        copy_data[:, i, j] = np.random.permutation(copy_data[:, i, j])

        pred = kospi_light_model.predict(copy_data)
        accuracy(labels, pred)
        test_prec(precision(labels, pred))
        test_rec(recall(labels, pred))
        test_csi(csi(labels, pred))

        perm_accuracy = accuracy.result() * 100
        perm_precision = test_prec.result()
        perm_recall = test_rec.result()
        perm_csi = test_csi.result()
        if i < 24:
            day_acc_result.append(origin_accuracy - perm_accuracy)
            day_prec_result.append(origin_precision - perm_precision)
            day_rec_result.append(origin_recall - perm_recall)
            day_csi_result.append(origin_csi - perm_csi)
        elif i < 48:
            min_acc_result.append(origin_accuracy - perm_accuracy)
            min_prec_result.append(origin_precision - perm_precision)
            min_rec_result.append(origin_recall - perm_recall)
            min_csi_result.append(origin_csi - perm_csi)
        else:
            var_acc_result.append(origin_accuracy - perm_accuracy)
            var_prec_result.append(origin_precision - perm_precision)
            var_rec_result.append(origin_recall - perm_recall)
            var_csi_result.append(origin_csi - perm_csi)
        total_acc_result.append(origin_accuracy - perm_accuracy)
        total_prec_result.append(origin_precision - perm_precision)
        total_rec_result.append(origin_recall - perm_recall)
        total_csi_result.append(origin_csi - perm_csi)

        accuracy.reset_states()
        test_prec.reset_states()
        test_csi.reset_states()
#%%
"""
enc : 0~23, dec : 24~48 , var : 49
info: close_open, high_open, open_low, high_low, high_close, low_close
arr : "open", "high", "low", "close", "vol"
var :시간,거래량*6(30분거래량 값과 scale을 맞추기 위해 6 곱함)/30일평균거래량,
     30분거래량/30일평균거래량 , 5분종가(현재가)/최근30일고점 및 저점,
     현재가/분단위 이평선,현재가/시초가,현재가/하루 고점=>실시간업데이트, 현재가/하루 저점, 현재가/30일 평균가
     현재가/ 2시간 평균값(시가,종가,저가,고가 모두 고려한 값)
"""
data_index = [
    "close_open",
    "high_open",
    "open_low",
    "high_low",
    "high_close",
    "low_close",
    "open",
    "high",
    "low",
    "close",
    "vol",
]
var_index = [
    "시간",
    "거래량*6/30일평균거래량",
    "30분거래량/30일평균거래량",
    "5분종가(현재가)/최근30일고점",
    "5분종가(현재가)/최근30일저점",
    "현재가/분단위 이평선",
    "현재가/시초가",
    "현재가/하루 고점",
    "현재가/하루 저점",
    "현재가/30일 평균가",
    "현재가/ 2시간 평균값",
]

day_acc_sort = np.argsort(day_acc_result)[::-1]
day_prec_sort = np.argsort(day_prec_result)[::-1]
day_rec_sort = np.argsort(day_rec_result)[::-1]
day_csi_sort = np.argsort(day_csi_result)[::-1]

min_acc_sort = np.argsort(min_acc_result)[::-1]
min_prec_sort = np.argsort(min_prec_result)[::-1]
min_rec_sort = np.argsort(min_rec_result)[::-1]
min_csi_sort = np.argsort(min_csi_result)[::-1]

var_acc_sort = np.argsort(var_acc_result)[::-1]
var_prec_sort = np.argsort(var_prec_result)[::-1]
var_rec_sort = np.argsort(var_rec_result)[::-1]
var_csi_sort = np.argsort(var_csi_result)[::-1]


total_acc_sort = np.argsort(total_acc_result)[::-1]
total_prec_sort = np.argsort(total_prec_result)[::-1]
total_rec_sort = np.argsort(total_rec_result)[::-1]
total_csi_sort = np.argsort(total_csi_result)[::-1]

print(f"일별 데이터 accuracy 기준 중요도 1위 : {data_index[day_acc_sort[0]]}")
print(f"일별 데이터 precision 기준 중요도 1위 : {data_index[day_prec_sort[0]]}")
print(f"일별 데이터 recall 기준 중요도 1위 : {data_index[day_rec_sort[0]]}")
print(f"일별 데이터 csi 기준 중요도 1위 : {data_index[day_csi_sort[0]]}")

print(f"분별 데이터 accuracy 기준 중요도 1위 : {data_index[min_acc_sort[0]]}")
print(f"분별 데이터 precision 기준 중요도 1위 : {data_index[min_prec_sort[0]]}")
print(f"분별 데이터 recall 기준 중요도 1위 : {data_index[min_rec_sort[0]]}")
print(f"분별 데이터 csi 기준 중요도 1위 : {data_index[min_csi_sort[0]]}")

print(f"변수 데이터 accuracy 기준 중요도 1위 : {var_index[var_acc_sort[0]]}")
print(f"변수 데이터 precision 기준 중요도 1위 : {var_index[var_prec_sort[0]]}")
print(f"변수 데이터 recall 기준 중요도 1위 : {var_index[var_rec_sort[0]]}")
print(f"변수 데이터 csi 기준 중요도 1위 : {var_index[var_csi_sort[0]]}")

day_acc_index = [data_index[idx] for idx in day_acc_sort]
day_prec_index = [data_index[idx] for idx in day_prec_sort]
day_rec_index = [data_index[idx] for idx in day_rec_sort]
day_csi_index = [data_index[idx] for idx in day_csi_sort]

min_acc_index = [data_index[idx] for idx in min_acc_sort]
min_prec_index = [data_index[idx] for idx in min_prec_sort]
min_rec_index = [data_index[idx] for idx in min_rec_sort]
min_csi_index = [data_index[idx] for idx in min_csi_sort]

var_acc_index = [var_index[idx] for idx in var_acc_sort]
var_prec_index = [var_index[idx] for idx in var_prec_sort]
var_rec_index = [var_index[idx] for idx in var_rec_sort]
var_csi_index = [var_index[idx] for idx in var_csi_sort]

day_acc_final = [day_acc_result[idx] for idx in day_acc_sort]
day_prec_final = [day_prec_result[idx] for idx in day_prec_sort]
day_rec_final = [day_rec_result[idx] for idx in day_rec_sort]
day_csi_final = [day_csi_result[idx] for idx in day_csi_sort]

min_acc_final = [min_acc_result[idx] for idx in min_acc_sort]
min_prec_final = [min_prec_result[idx] for idx in min_prec_sort]
min_rec_final = [min_rec_result[idx] for idx in min_rec_sort]
min_csi_final = [min_csi_result[idx] for idx in min_csi_sort]

var_acc_final = [var_acc_result[idx] for idx in var_acc_sort]
var_prec_final = [var_prec_result[idx] for idx in var_prec_sort]
var_rec_final = [var_rec_result[idx] for idx in var_rec_sort]
var_csi_final = [var_csi_result[idx] for idx in var_csi_sort]

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.bar(day_acc_index, day_acc_final)
plt.xticks(rotation=45)
plt.title("Day Accuracy")

plt.subplot(2, 2, 2)
plt.bar(day_prec_index, day_prec_final)
plt.xticks(rotation=45)
plt.title("Day Precision")

plt.subplot(2, 2, 3)
plt.bar(day_rec_index, day_rec_final)
plt.xticks(rotation=45)
plt.title("Day Recall")

plt.subplot(2, 2, 4)
plt.bar(day_csi_index, day_csi_final)
plt.xticks(rotation=45)
plt.title("Day CSI")

plt.show()

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.bar(min_acc_index, min_acc_final)
plt.xticks(rotation=45)
plt.title("Minutes Accuracy")

plt.subplot(2, 2, 2)
plt.bar(min_prec_index, min_prec_final)
plt.xticks(rotation=45)
plt.title("Minutes Precision")

plt.subplot(2, 2, 3)
plt.bar(min_rec_index, min_rec_final)
plt.xticks(rotation=45)
plt.title("Minutes Recall")

plt.subplot(2, 2, 4)
plt.bar(min_csi_index, min_csi_final)
plt.xticks(rotation=45)
plt.title("Minutes CSI")

plt.show()


plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.bar(var_acc_index, var_acc_final)
plt.xticks(rotation=45)
plt.title("VAR Accuracy")

plt.subplot(2, 2, 2)
plt.bar(var_prec_index, var_prec_final)
plt.xticks(rotation=45)
plt.title("VAR Precision")

plt.subplot(2, 2, 3)
plt.bar(var_rec_index, var_rec_final)
plt.xticks(rotation=45)
plt.title("VAR Recall")

plt.subplot(2, 2, 4)
plt.bar(var_csi_index, var_csi_final)
plt.xticks(rotation=45)
plt.title("VAR CSI")

plt.show()
