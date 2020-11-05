#%%
import tensorflow as tf
from DataSet import GetData
from precision_recall import *
from time import time
import matplotlib.pyplot as plt


#%%
data = GetData()
kospi, kosdaq = data.AllTestSet()

model1 = tf.keras.models.load_model("convTransformer.h5")
model2 = tf.keras.models.load_model("convTransformer_light.h5")
kosdaq_model = tf.keras.models.load_model("kosdaq_convTransformer.h5")
kospi_model = tf.keras.models.load_model("kospi_convTransformer.h5")

#%%
models = [model1, model2, kosdaq_model, kospi_model]
times = []
recalls = []
precisions = []
csis = []
minus = []
t_inputs, t_labels, _, _ = next(iter(kospi))
for model in models:
    t = time()
    pred = model.predict(t_inputs)
    times.append(time() - t)
    recalls.append(single_class_recall(0)(t_labels, pred).numpy())
    precisions.append(single_class_precision(0)(t_labels, pred).numpy())
    csis.append(single_class_csi(0)(t_labels, pred).numpy())
    minus.append(ratio_of_Minus(0, 1)(t_labels, pred).numpy())
#%%
plt.figure(figsize=(10, 12))
plt.subplot(3, 2, 1)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=times)
plt.title("time")
plt.subplot(3, 2, 2)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=recalls)
plt.title("recall")
plt.subplot(3, 2, 3)
plt.bar(
    x=["first model", "light model", "kosdaq model", "kospi model"], height=precisions
)
plt.title("precision")
plt.subplot(3, 2, 4)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=csis)
plt.title("csi")
plt.subplot(3, 2, 5)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=minus)
plt.title("minus ratio")
plt.show()
#%%
models = [model1, model2, kosdaq_model, kospi_model]
times = []
recalls = []
precisions = []
csis = []
minus = []
t_inputs, t_labels, _, _ = next(iter(kosdaq))
for model in models:
    t = time()
    pred = model.predict(t_inputs)
    times.append(time() - t)
    recalls.append(single_class_recall(0)(t_labels, pred).numpy())
    precisions.append(single_class_precision(0)(t_labels, pred).numpy())
    csis.append(single_class_csi(0)(t_labels, pred).numpy())
    minus.append(ratio_of_Minus(0, 1)(t_labels, pred).numpy())
#%%
plt.figure(figsize=(10, 12))
plt.subplot(3, 2, 1)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=times)
plt.title("time")
plt.subplot(3, 2, 2)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=recalls)
plt.title("recall")
plt.subplot(3, 2, 3)
plt.bar(
    x=["first model", "light model", "kosdaq model", "kospi model"], height=precisions
)
plt.title("precision")
plt.subplot(3, 2, 4)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=csis)
plt.title("csi")
plt.subplot(3, 2, 5)
plt.bar(x=["first model", "light model", "kosdaq model", "kospi model"], height=minus)
plt.title("minus ratio")
plt.show()
