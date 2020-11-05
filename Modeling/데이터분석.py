#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
kosdaq_close = np.load(open("data/검증데이터/KOSDAQ_close_price.npy", "rb"))
kosdaq_pre_close = np.load(open("data/검증데이터/KOSDAQ_pre_close_price.npy", "rb"))
kosdaq_low = np.load(open("data/검증데이터/KOSDAQ_low_price.npy", "rb"))
kosdaq_high = np.load(open("data/검증데이터/KOSDAQ_high_price.npy", "rb"))
kospi_close = np.load(open("data/검증데이터/KOSPI_close_price.npy", "rb"))
kospi_pre_close = np.load(open("data/검증데이터/KOSPI_pre_close_price.npy", "rb"))
kospi_low = np.load(open("data/검증데이터/KOSPI_low_price.npy", "rb"))
kospi_high = np.load(open("data/검증데이터/KOSPI_high_price.npy", "rb"))
#%%
close_low_ratio = []
close_close_ratio = []
close_high_ratio = []
for i in range(len(kospi_low)):
    close_low_ratio.append(
        (kospi_low[i][0] - kospi_pre_close[i][0]) / kospi_pre_close[i][0] * 100
    )
    close_high_ratio.append(
        (-kospi_pre_close[i][0] + kospi_high[i][0]) / kospi_pre_close[i][0] * 100
    )
    close_close_ratio.append(
        (-kospi_pre_close[i][0] + kospi_close[i][0]) / kospi_pre_close[i][0] * 100
    )
for i in range(len(kosdaq_low)):
    close_low_ratio.append(
        (kosdaq_low[i][0] - kosdaq_pre_close[i][0]) / kosdaq_pre_close[i][0] * 100
    )
    close_close_ratio.append(
        (-kosdaq_pre_close[i][0] + kosdaq_high[i][0]) / kosdaq_pre_close[i][0] * 100
    )
    close_high_ratio.append(
        (-kosdaq_pre_close[i][0] + kosdaq_close[i][0]) / kosdaq_pre_close[i][0] * 100
    )
close_close = pd.DataFrame(close_close_ratio, columns=["ratio"])
close_low = pd.DataFrame(close_low_ratio, columns=["ratio"])
close_high = pd.DataFrame(close_high_ratio, columns=["ratio"])
#%%
plt.figure(figsize=(12, 10))
plt.subplot(1, 3, 1)
sns.boxplot(y="ratio", data=close_close)
plt.title("total close close ratio")
plt.subplot(1, 3, 2)
sns.boxplot(y="ratio", data=close_low)
plt.title("total close low ratio")
plt.subplot(1, 3, 3)
sns.boxplot(y="ratio", data=close_high)
plt.title("total close high ratio")
plt.show()
#%%
higher_than_1_close_ratio = []
higher_than_1_low_ratio = []
for i in range(len(close_high_ratio)):
    if close_high_ratio[i] > 1.0:
        higher_than_1_close_ratio.append(close_close_ratio[i])
        higher_than_1_low_ratio.append(close_low_ratio[i])
#%%
higer_low = pd.DataFrame(higher_than_1_low_ratio, columns=["ratio"])
higer_close = pd.DataFrame(higher_than_1_close_ratio, columns=["ratio"])
plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
sns.boxplot(y="ratio", data=higer_low)
plt.title("total higher low ratio")
plt.subplot(1, 2, 2)
sns.boxplot(y="ratio", data=higer_close)
plt.title("total higher close ratio")
plt.show()
