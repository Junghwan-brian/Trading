"""
호가단위
                            코스피          코스닥
1000원 미만 -                  1원           1원
1000원 이상 ~ 5000원 미만 -     5원           5원
5000원 이상 ~ 10000원 미만 -    10원          10원
10000원 이상 ~ 50000원 미만 -   50원          50원
50000원 이상 ~ 100000원 미만 -  100원         100원
100000원 이상 ~ 500000원 미만 - 500원         100원
500000원 이상 -                1000원        100원
"""
#%%
import numpy as np
from tensorflow.keras.models import load_model
from DataSet import GetData
from time import time

#%%
total_close = np.load(open("data/검증데이터/close_price.npy", "rb"))
total_pre_close = np.load(open("data/검증데이터/pre_close_price.npy", "rb"))
total_low = np.load(open("data/검증데이터/low_price.npy", "rb"))
total_high = np.load(open("data/검증데이터/high_price.npy", "rb"))
#%%
data = GetData()
test_ds = data.TotalTestSet()
kospi_light_model = load_model("kospi_convTransformer_light.h5")
#%%
total_inputs, total_cls, _ = next(iter(test_ds))
t = time()
total_pred1 = kospi_light_model.predict(
    total_inputs, workers=-1, use_multiprocessing=True
)
print(f"{time()-t} sec")
#%%
total_close_low_ratio = []
total_close_close_ratio = []
total_close_high_ratio = []
for i in range(len(total_low)):
    total_close_low_ratio.append(
        (total_low[i][0] - total_pre_close[i][0]) / total_pre_close[i][0]
    )
    total_close_high_ratio.append(
        (-total_pre_close[i][0] + total_high[i][0]) / total_pre_close[i][0]
    )
    total_close_close_ratio.append(
        (-total_pre_close[i][0] + total_close[i][0]) / total_pre_close[i][0]
    )
# 거래세도 고려해야 한다.
# 팔때 : 0.265%
# 살때 : 0.015%
origin_balance = 500000
win_ratio = 0.6
fee_ratio = 0.02
lose_ratio = 0.02
kelly_ratio = (
    origin_balance * fee_ratio * win_ratio
    - origin_balance * lose_ratio * (1 - win_ratio)
) / (origin_balance * fee_ratio)
print(f"{kelly_ratio:.2f}")

#%%

balance_seq = []
balance = 500000
final_close_ratio = []  # high 가 1퍼센트를 안 넘었을 때의 비율들을 모은다.
rest_sell_ratio = []  # 절반 매도 후 이익의 비율을 모은다.
minus_check = 0
buy_tax_ratio = 0.00015  # 살 때의 세금 비율
sell_tax_ratio = 0.00265  # 팔 때의 세금 비율
loss_rate = -0.05  # 손절 비율
profit_rate1 = 0.02  # 1차 익절 비율
profit_rate2 = 0.05  # 2차 익절 비율
profit_rate3 = 0.1  # 3차 익절 비율
sell_ratio1 = 1 / 5  # 일부 매도할 때의 비율
sell_ratio2 = 1 / 5
sell_ratio3 = 2 / 5
loss_cut_num = 0  # 손절 횟수
total_loss_count = 0  # 손실 일 때의 총 횟수
is_minus = False  # 연속해서 마이너스가 될 때 최대 손실을 구한다.
is_previous_minus = False
max_consecutive_loss_count = 0
consecutive_loss_count = 0
betting_price = kelly_ratio * origin_balance


def get_tax(income_ratio, sell_ratio):
    return -(1 + income_ratio) * sell_ratio * betting_price * sell_tax_ratio


for i in range(total_inputs.shape[0]):
    pred_value = total_pred1[i]
    if np.argmax(pred_value) != 0 or pred_value[0] < 0.75:
        continue
    balance -= buy_tax_ratio * betting_price  # 살 때의 세금
    if total_close_low_ratio[i] < loss_rate:
        balance += -(1 + loss_rate) * betting_price * sell_tax_ratio  # 번 돈의 세금 비율을 빼준다.
        balance += loss_rate * betting_price
        is_minus = True
        total_loss_count += 1
        loss_cut_num += 1
    elif total_close_high_ratio[i] >= profit_rate1:
        balance += get_tax(profit_rate1, sell_ratio1)
        balance += profit_rate1 * sell_ratio1 * betting_price  # 일부 매도
        rest_balance = betting_price * (1 - sell_ratio1)
        profit_ratio = profit_rate1 * sell_ratio1  # 최종 이익을 봤는지 손실을 봤는지 판단하기 위함.
        if total_close_high_ratio[i] >= profit_rate2:
            balance += get_tax(profit_rate2, sell_ratio2)
            balance += profit_rate2 * sell_ratio2 * betting_price  # 일부 매도
            rest_balance = betting_price * (1 - (sell_ratio1 + sell_ratio2))
            profit_ratio += profit_rate2 * sell_ratio2
        if total_close_high_ratio[i] >= profit_rate3:
            balance += get_tax(profit_rate3, sell_ratio3)
            balance += profit_rate3 * sell_ratio3 * betting_price  # 일부 매도
            rest_balance = betting_price * (
                1 - (sell_ratio1 + sell_ratio2 + sell_ratio3)
            )
            profit_ratio += profit_rate3 * sell_ratio3
        close_profit_rate = rest_balance / betting_price
        balance += get_tax(total_close_close_ratio[i], close_profit_rate)
        balance += total_close_close_ratio[i] * rest_balance  # 종가에 매도
        profit_ratio += total_close_close_ratio[i] * close_profit_rate
        rest_sell_ratio.append(total_close_close_ratio[i])
        if profit_ratio > 0:
            is_minus = False
        else:
            total_loss_count += 1
            is_minus = True

    else:
        balance += -(1 + total_close_close_ratio[i]) * betting_price * sell_tax_ratio
        balance += total_close_close_ratio[i] * betting_price
        final_close_ratio.append(total_close_close_ratio[i])
        if total_close_close_ratio[i] < 0:
            is_minus = True
            total_loss_count += 1
        else:
            is_minus = False
    if balance <= 0:
        minus_check += 1
    balance_seq.append(balance)
    if is_minus is True and is_previous_minus is True:
        consecutive_loss_count += 1
        if max_consecutive_loss_count < consecutive_loss_count:
            max_consecutive_loss_count = consecutive_loss_count
    if is_minus == True:
        is_previous_minus = True
    else:
        consecutive_loss_count = 0
        is_previous_minus = False
print(f"마이너스 찍은 횟수: {minus_check}")
print(
    f"가장 큰 손실 : {np.minimum(loss_rate, np.minimum(np.min(rest_sell_ratio), np.min(final_close_ratio))) * 100}%"
)
print(
    f"가장 큰 이익 : {np.maximum(profit_rate1, np.maximum(np.max(rest_sell_ratio), np.max(final_close_ratio))) * 100:.2f}%"
)
print(f"연속 최대 손실 횟수 : {max_consecutive_loss_count}회")
print(f"손절 횟수 : {loss_cut_num}회")
print(f"최저 점 : {int(np.min(balance_seq))}원")
print(f"예측이 틀렸을 때의 종가 평균 값 : {np.mean(final_close_ratio)*100:.2f}%")
print(f"예측이 틀렸을 때의 종가 min 값 : {np.min(final_close_ratio)*100:.2f}%")
print(f"일정 비율만 매도 후 종가 매도를 했을 때의 평균 값 : {np.mean(rest_sell_ratio)*100:.2f}%")
print(f"일정 비율만 매도 후 종가 매도를 했을 때의 min 값 : {np.min(rest_sell_ratio)*100:.2f}%")
print(f"일정 비율만 매도 후 종가 매도를 했을 때의 max 값 : {np.max(rest_sell_ratio)*100:.2f}%")
print(f"총 손실 횟수 : {total_loss_count}회")
print(f"전체 손실 발생 비율 : {total_loss_count/len(balance_seq)*100:.2f}%")
print(f"총 거래 횟수 : {len(balance_seq)}회")
print(f"하루 평균 거래 횟수 : {len(balance_seq)/(3396/13):.2f}회")
print(f"최종 남은 돈: {int(balance)}원")
print(f"수익률 : {(balance-origin_balance)/origin_balance*100:.2f}%")
