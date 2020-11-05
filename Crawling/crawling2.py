import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from time import mktime
from re import compile
from json import loads
from pandas.tseries.offsets import BusinessDay
import numpy as np
from urllib.request import Request, urlopen
import asyncio

# 다른건 괜찮지만 실시간 데이터를 불러오는 경우는 시간이 지체시 치명적일 수 있으므로
# 에러가 나면 5분 동안은 거래를 하지 않는 것으로 처리해준다.
class GetRealData:
    def __init__(self):
        ###############
        # 전날 구해놓는 변수들
        kosdaq_data = pd.read_pickle("data/index/코스닥선정종목데이터.pkl")
        kospi_data = pd.read_pickle("data/index/코스피선정종목데이터.pkl")
        self.total_data = pd.concat([kosdaq_data, kospi_data], axis=1)
        kosdaq_investing_info = pd.read_pickle(
            "data/index/KOSDAQ_selected_krx_info.pkl"
        )[["Symbol", "investing_id", "Name"]]
        kospi_investing_info = pd.read_pickle("data/index/KOSPI_selected_krx_info.pkl")[
            ["Symbol", "investing_id", "Name"]
        ]
        self.total_investing_info = pd.concat(
            [kosdaq_investing_info, kospi_investing_info], axis=0
        )
        self.mean_vol = None
        self.highest_price = None
        self.lowest_price = None
        self.mean_price = None
        ###############

        ###############
        # 실시간 변수들
        self.index_time = None
        self.close_p = None
        self.open_p = None
        self.high_p = None
        self.low_p = None
        self.vol = None
        self.arr = None
        self.scale_arr = None
        self.hour = None
        self.selected_increase_dict = None  # 당일 시가가 담겨있는 거래량 상위 종목 dict
        self.selected_total_investing_info = None
        self.selected_total_data = None
        self.times = None  # 하루동안의 모든 시간들을 담아 놓는 리스트
        self.increase_company = []
        ################

        ################
        # 이벤트 루프
        self.open_loop = None
        self.loop = None
        self.increase_loop = None
        ###############

        ################
        # 당일 아침에 미리 뽑아 놓는 변수들
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"
        }
        self.carrier, self.time_stamp = self.get_carrier_time()
        selected_list = self.total_data.columns.to_list()
        self.now = datetime.now()
        self.today_year = self.now.year
        self.today_month = self.now.month
        self.today_day = self.now.day
        self.measure = mktime(
            datetime(self.now.year, self.now.month, self.now.day, 9, 0).timetuple()
        )
        self.open_selected_dict = self.get_open_price(selected_list)
        # open_selected_dict 는 인베스팅 홈페이지에 당일 시가가 있는 목록이다.
        # print(f"총 {len(self.open_selected_dict)}개의 데이터 사용.")
        ##################

    def decrease_list_from_increase_company(self, increase_company):
        """
        거래량 상위 종목을 뽑고 탐색시간을 줄이기 위해 open_selected_dict,kosdaq_investing_info,kosdaq_data 를 줄인다.
        :return:
        """
        self.increase_company = []
        self.selected_increase_dict = {}
        self.selected_total_investing_info = pd.DataFrame()
        self.selected_total_data = pd.DataFrame()
        for company_code in increase_company:
            if company_code in self.open_selected_dict:
                self.increase_company.append(company_code)
                self.selected_increase_dict.update(
                    {company_code: self.open_selected_dict[company_code]}
                )
                company_info = self.total_investing_info.query(
                    f"Symbol=='{company_code}'"
                )
                self.selected_total_investing_info = pd.concat(
                    [self.selected_total_investing_info, company_info], axis=0
                )
                company_data = self.total_data[company_code]
                self.selected_total_data = pd.concat(
                    [self.selected_total_data, company_data], axis=1
                )
        self.selected_total_investing_info = self.selected_total_investing_info.reset_index(
            drop=True
        )

    def get_naver_increase_company(self, market=1):
        """
        네이버에서 거래 상위 종목을 가져온다.
        :param market:  0: 코스피, 1: 코스닥
        :return: 거래 상위 30개에 대한 종목명
        """
        url = f"https://finance.naver.com/sise/sise_quant.nhn?sosok={market}"
        req = requests.get(url, headers=self.headers).text
        bs = BeautifulSoup(req, "lxml")
        table = bs.find("table", {"class": "type_2"})
        table = pd.read_html(str(table))[0].dropna(how="all")[:30]["종목명"].values
        company_list = []
        for name in table:
            try:
                code = self.total_investing_info.query(f"Name == '{name}'")[
                    "Symbol"
                ].iloc[0]
                if code in self.open_selected_dict:
                    company_list.append(code)
            except IndexError:
                pass
        return company_list

    def get_carrier_time(self):
        """
        carrier(일종의 id 인 듯한데 계속 바뀌지만 하나로 사용해도 되는 것 같다.),
        time_stamp(접속할 때의 시간인데 그냥 현재 시각을 넣으면 안되서 값을 받아온다.)
        두 변수를 처음 시작할 때 계속 받아오고 사용한다.
        :return: carrier, time_stamp
        """
        carrier_re = compile("carrier=[0-9a-z]+")
        time_re = compile("time=[0-9a-z]+")
        url = "https://kr.investing.com/equities/seegene-inc-chart"
        req = requests.get(url, headers=self.headers).text
        bs = BeautifulSoup(req, "lxml")
        carrier = carrier_re.search(str(bs)).group()[8:]
        time_stamp = time_re.search(str(bs)).group()[5:]
        return carrier, time_stamp

    def plus_info(self):
        # open,high,low,close
        close_open = (self.scale_arr[:, -1] - self.scale_arr[:, 0]).reshape([24, 1])
        high_open = (self.scale_arr[:, 1] - self.scale_arr[:, 0]).reshape([24, 1])
        open_low = (self.scale_arr[:, 0] - self.scale_arr[:, 2]).reshape([24, 1])
        high_low = (self.scale_arr[:, 1] - self.scale_arr[:, 2]).reshape([24, 1])
        high_close = (self.scale_arr[:, 1] - self.scale_arr[:, -1]).reshape([24, 1])
        low_close = (self.scale_arr[:, 2] - self.scale_arr[:, -1]).reshape([24, 1])
        day_plus_info = np.concatenate(
            [close_open, high_open, open_low, high_low, high_close, low_close], axis=1
        )  # 24,6
        return day_plus_info

    @staticmethod
    def min_max_scaler(arr, _min, _max):
        regul_arr = (arr - _min) / (_max - _min + 0.0001)
        return regul_arr

    def get_close_ma_ratio(self):
        ma = np.sum(self.close_p[-20:]) / 20
        close_ma_ratio = self.close_p[-1] / ma
        return close_ma_ratio

    def get_hour_index(self):
        if self.hour == 9:
            hour_index = 0.3
        elif self.hour == 14 or self.hour == 10:
            hour_index = 0.2
        else:
            hour_index = 0.1
        return hour_index

    def get_var_seq(self, company_code):
        """
        var 에 대한 데이터를 구한다.
        :return:  [ 시간,거래량*6(30분거래량 값과 scale 을 맞추기 위해 6 곱함)/30일평균거래량,
         30분거래량/30일평균거래량 , 5분종가(현재가)/최근30일고점 및 저점,\
         현재가/분단위 이평선,현재가/시초가,현재가/하루 고점=>실시간업데이트, 현재가/하루 저점, 현재가/30일 평균가
        현재가/ 2시간 평균값(시가,종가,저가,고가 모두 고려한 값) ]
        """
        self.hour = datetime.fromtimestamp(self.index_time[-1]).hour
        hour_index = self.get_hour_index()
        now_price = self.close_p[-1]
        close_ma_ratio = self.get_close_ma_ratio()
        day_open_value = self.selected_increase_dict[company_code]
        today = datetime.fromtimestamp(self.index_time[-1]).day
        if datetime.fromtimestamp(self.index_time[0]).day == today:
            the_day_max_value, the_day_min_value = (
                np.max(self.high_p),
                np.min(self.low_p),
            )
        else:
            time_arr = np.array(self.index_time)
            leng = time_arr[time_arr > self.measure].shape[0]
            the_day_max_value, the_day_min_value = (
                np.max(self.high_p[-leng:]),
                np.min(self.low_p[-leng:]),
            )
        min30_vol = np.mean(self.vol[-6:])
        now_volume = self.vol[-1]
        two_hours_mean = np.mean(self.arr)
        var = [
            hour_index,
            now_volume * 6 / self.mean_vol,
            min30_vol / self.mean_vol,
            now_price / self.highest_price,
            now_price / self.lowest_price,
            close_ma_ratio,
            now_price / day_open_value,
            now_price / the_day_max_value,
            now_price / the_day_min_value,
            now_price / self.mean_price,
            now_price / two_hours_mean,
        ]
        var_seq = np.array(var).reshape(1, 11).astype(np.float32)
        return var_seq

    @staticmethod
    def get_real_timestamp(now):
        """
        실시간 타임스탬프를 구한다. 따라서 __init__ 에서 생성한 now 가 아닌 실시간 now 를 생성함.
        :return:
        """
        # 시간이 조금 빨리 올 수 있어서 5초를 더해준다.

        if now.hour < 11:
            # 월요일이면 지난 금요일의 날짜를 가져온다.
            from_time_stamp = mktime(
                (now - BusinessDay() + pd.Timedelta(3, "H")).timetuple()
            )
        else:
            from_time_stamp = mktime((now - pd.Timedelta(3, "H")).timetuple())

        # 12:10분일 경우 12:05분 까지의 데이터를 구해야한다. 10분이면 10~15분의 데이터를 의미하기 때문.
        to_time_stamp = mktime(
            (now - pd.Timedelta(5 + now.minute % 5, "m")).timetuple()
        )
        return from_time_stamp, to_time_stamp

    async def increase_fetch(self, code, url):
        result = {}
        request = Request(url, headers=self.headers)  # UA가 없으면 403 에러 발생
        response = await self.increase_loop.run_in_executor(
            None, urlopen, request
        )  # run_in_executor 사용
        page = await self.increase_loop.run_in_executor(
            None, response.read
        )  # run in executor 사용
        js = loads(page)
        try:
            index_time = js["t"][-24:]
        except KeyError:
            return None
        # 종목을 선정할 때 미리 길이가 24 이상인 것을 뽑아 놓는다.
        if len(index_time) != 24:
            return None
        vol = np.array(js["v"][-6:])
        sum_vol = np.sum(vol)
        if sum_vol > 400000:
            result.update({code: sum_vol})
            return result
        else:
            return None

    async def increase_gather_data(self, company_urls):
        futures = [
            asyncio.ensure_future(self.increase_fetch(code, url))
            for code, url in company_urls.items()
        ]
        # 태스크(퓨처) 객체를 리스트로 만듦
        results = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴
        return results

    def get_increase_url(self, company_list, from_time_stamp, to_time_stamp):
        company_urls = {}
        for code in company_list:
            symbol = self.total_investing_info.query(f"Symbol=='{code}'")[
                "investing_id"
            ].iloc[0]
            url = f"https://tvc4.forexpros.com/{self.carrier}/{self.time_stamp}/18/18/88/history?symbol={symbol}&resolution=5&from={from_time_stamp}&to={to_time_stamp}"
            company_urls.update({code: url})
        return company_urls

    def get_investing_increase_company(self):
        """
        실시간 거래량이 높은 종목을 고르기 위한 함수.
        :return:
        """
        now = datetime.now()
        now = now - pd.Timedelta(5 + now.minute % 5, "m")
        from_time_stamp, to_time_stamp = self.get_real_timestamp(now)
        company_urls = self.get_increase_url(
            self.open_selected_dict.keys(), from_time_stamp, to_time_stamp
        )
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.increase_loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
        result = self.increase_loop.run_until_complete(
            self.increase_gather_data(company_urls)
        )  # gather_data 끝날 때까지 기다림
        self.increase_loop.close()  # 이벤트 루프를 닫음
        return result

    async def fetch(self, company_code, url):
        """
        모델에 넣을 데이터를 구한다.
        :param company_code: 상승 종목을 미리 구해서 변수로 넣어줘야 한다.
        :param url: 링크
        :return:
        arr = ("open", "high", "low", "close")
        info = np.concatenate([close_open, high_open, open_low, high_low, high_close, low_close], axis=1)  # 24,6
        var = { 시간,거래량*6(30분거래량 값과 scale 을 맞추기 위해 6 곱함)/30일평균거래량,
                30분거래량/30일평균거래량 , 5분종가(현재가)/최근30일고점 및 저점,
            현재가/분단위 이평선,현재가/시초가,현재가/하루 고점=>실시간업데이트, 현재가/하루 저점, 현재가/30일 평균가
            현재가/ 2시간 평균값(시가,종가,저가,고가 모두 고려한 값) }
        최종데이터 = (enc,dec,var) , shape = (batch,49,11)
        """
        result = {}
        request = Request(url, headers=self.headers)  # UA가 없으면 403 에러 발생
        response = await self.loop.run_in_executor(
            None, urlopen, request
        )  # run_in_executor 사용
        page = await self.loop.run_in_executor(
            None, response.read
        )  # run in executor 사용
        js = loads(page)
        self.index_time = js["t"][-24:]
        # 증권사 api 를 사용하여 거래량 상위 종목을 불러올 경우 index_time 이 24보다 낮을 수 있다.
        if len(self.index_time) != 24:
            return None
        day_dict = self.selected_total_data[company_code]
        min_price = day_dict["min_price"]
        max_price = day_dict["max_price"]
        min_vol = day_dict["min_vol"]
        max_vol = day_dict["max_vol"]
        self.mean_vol = day_dict["mean_vol"]
        self.highest_price = day_dict["highest"]
        self.lowest_price = day_dict["lowest"]
        self.mean_price = day_dict["mean_price"]
        enc_data = day_dict["enc_data"]
        self.close_p = np.array(js["c"][-24:])
        self.open_p = np.array(js["o"][-24:])
        self.high_p = np.array(js["h"][-24:])
        self.low_p = np.array(js["l"][-24:])
        self.vol = np.array(js["v"][-24:])
        self.arr = np.stack(
            [self.open_p, self.high_p, self.low_p, self.close_p], axis=-1
        )  # 24,4
        self.scale_arr = self.min_max_scaler(
            self.arr, _min=min_price, _max=max_price
        )  # 24,4
        info = self.plus_info()  # 24, 6
        scale_vol_arr = self.min_max_scaler(
            self.vol, _min=min_vol, _max=max_vol
        ).reshape(
            24, 1
        )  # 24,1
        min_arr = np.concatenate([self.scale_arr, scale_vol_arr], axis=-1)
        dec_data = np.concatenate([info, min_arr], axis=-1).astype(np.float32)
        var = self.get_var_seq(company_code)
        final_data = np.concatenate([enc_data, dec_data, var], axis=0)
        result.update({company_code: final_data})
        return result

    async def gather_data(self, company_urls):
        futures = [
            asyncio.ensure_future(self.fetch(company_code, url))
            for company_code, url in company_urls.items()
        ]
        # 태스크(퓨처) 객체를 리스트로 만듦
        results = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴
        return results

    def get_url(self, from_time_stamp, to_time_stamp):
        company_urls = {}
        for company_code in self.increase_company:
            symbol = self.selected_total_investing_info.query(
                f"Symbol =='{company_code}'"
            )["investing_id"].iloc[0]
            url = f"https://tvc4.forexpros.com/{self.carrier}/{self.time_stamp}/18/18/88/history?symbol={symbol}&resolution=5&from={from_time_stamp}&to={to_time_stamp}"
            company_urls.update({company_code: url})
        return company_urls

    def get_real_data(self):
        now = datetime.now() + pd.Timedelta(10, "s")
        from_time_stamp, to_time_stamp = self.get_real_timestamp(now)
        company_urls = self.get_url(from_time_stamp, to_time_stamp)
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
        result = self.loop.run_until_complete(
            self.gather_data(company_urls)
        )  # gather_data 끝날 때까지 기다림
        self.loop.close()  # 이벤트 루프를 닫음
        return result

    async def open_fetch(self, code, url):
        result = {}
        request = Request(url, headers=self.headers)  # UA가 없으면 403 에러 발생
        response = await self.open_loop.run_in_executor(
            None, urlopen, request
        )  # run_in_executor 사용
        page = await self.open_loop.run_in_executor(
            None, response.read
        )  # run in executor 사용
        js = loads(page)
        try:
            open_p = js["o"][0]
        except:
            return None
        result.update({code: open_p})
        return result

    async def open_gather_data(self, urls):
        futures = [
            asyncio.ensure_future(self.open_fetch(code, url))
            for code, url in urls.items()
        ]
        # 태스크(퓨처) 객체를 리스트로 만듦
        results = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴
        return results

    def get_open_url(self, increase_company, from_time_stamp, to_time_stamp):
        company_urls = {}
        for code in increase_company:
            symbol = self.total_investing_info.query(f"Symbol =='{code}'")[
                "investing_id"
            ].iloc[0]
            url = f"https://tvc4.forexpros.com/{self.carrier}/{self.time_stamp}/18/18/88/history?symbol={symbol}&resolution=5&from={from_time_stamp}&to={to_time_stamp}"
            company_urls.update({code: url})
        return company_urls

    def get_open_price(self, company_list):
        """
        종목에 대한 당일 시가를 미리 구하기 위한 함수.
        :param company_list: 종목들
        :return: 당일 시가
        """
        open_from_time_stamp = mktime(
            datetime(self.now.year, self.now.month, self.now.day, 9, 0).timetuple()
        )
        open_to_time_stamp = mktime(
            datetime(self.now.year, self.now.month, self.now.day, 9, 5).timetuple()
        )
        company_urls = self.get_open_url(
            company_list, open_from_time_stamp, open_to_time_stamp
        )
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.open_loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
        result = self.open_loop.run_until_complete(
            self.open_gather_data(company_urls)
        )  # gather_data 끝날 때까지 기다림
        self.open_loop.close()  # 이벤트 루프를 닫음
        open_selected_dict = {}
        for company in result:
            if company is not None:
                open_selected_dict.update(company)
        return open_selected_dict
