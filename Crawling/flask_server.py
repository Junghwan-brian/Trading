from tensorflow.keras.models import load_model
from flask import Flask
from crawling2 import GetRealData
from tensorflow import constant, float32
from numpy import argsort


def get_inputs():
    real_dict_list = real_data.get_real_data()
    comp_dict = {}
    for elem in real_dict_list:
        if elem is None:
            continue
        comp_dict.update(elem)
    inputs = list(comp_dict.values())
    codes = list(comp_dict.keys())
    inputs = constant(inputs, dtype=float32)
    return codes, inputs


def get_increase_company():
    """
    거래량 상위종목을 60까지 불러온다.
    종목명과 거래량을 함께 가져오는데 이때 60 종목이 넘어가면 거래량을 비교하여
    상위 종목만을 추출한다.
    :return: 거래량 상위종목
    """
    increase_company = real_data.get_investing_increase_company()
    selected_company = {}
    for company in increase_company:
        selected_company.update(company) if company is not None else 0

    if len(selected_company) < 60:
        selected_company = selected_company.keys()
    else:
        res = sorted(selected_company.items(), key=(lambda x: x[1]), reverse=True)
        selected_company = [r[0] for r in res]
        selected_company = selected_company[:60]
    return list(selected_company)


load = load_model("kospi_convTransformer_light.h5")
load_inference = load.signatures["serving_default"]
while True:
    try:
        real_data = GetRealData()
        break
    except Exception as e:
        print(e)
        print("에러발생 재시작")
app = Flask(__name__)

# 거래 상위 종목을 불러오고 decrease_list_from_increase_company 함수를 한 번만 호출하면 되므로
# 따로 넣어줬다.
@app.route("/update", methods=["POST"])
def update():
    company = get_increase_company()
    real_data.decrease_list_from_increase_company(company)
    return str(company)


@app.route("/naver_update", methods=["POST"])
def naver_update():
    company1 = real_data.get_naver_increase_company(0)
    company2 = real_data.get_naver_increase_company(1)
    company = company1 + company2
    real_data.decrease_list_from_increase_company(company)
    return str(company)


@app.route("/inference", methods=["POST"])
def inference():
    comp_list, inputs = get_inputs()
    result = load_inference(inputs)["output_1"].numpy()[:, 0]
    arg_result = []
    sort_pred = argsort(result)
    for idx in sort_pred[-5:]:
        arg_result.append(comp_list[idx]) if result[idx] >= 0.75 else 0
    return str(arg_result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2431, threaded=False)
