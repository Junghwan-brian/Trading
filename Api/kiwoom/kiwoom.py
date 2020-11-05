from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
from config.errorCode import *
from config.kiwoomType import *
from sys import exit
from time import time
from subprocess import Popen, PIPE
from json import loads
from requests import post
from datetime import datetime
from PyQt5.QtTest import *

"""
증권사에 시그널을 보내 요청 -> 이벤트가 연결다리를 한다. (OnReceiveRealData 등)
-> 결과값이 큐에 쌓이면 이벤트 루프를 통하여 슬롯에서 데이터를 반환받음.
PyQt5의 이벤트루프는 다음 코드가 실행되는 것을 막는다.+
실시간 주시를 하며 매수/매도를 하며 장시간체크를 한다.
TR 요청시 3.6초 마다 하면 요청이 끊기지 않는다. 이때 사용하는 함수는 QTest.qWait(3600)

주문 -> 접수 -> 확인 -> 체결 -> 잔고 ..
"""


class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()

        print("Kiwoom class")
        self.realType = RealType()
        ##########
        # event loop 모음
        self.login_event_loop = None
        self.account_info_event_loop = QEventLoop()
        ##########

        ##########
        # 스크린 번호 모음
        # 하나의 스크린 번호에 100개 까지 등록 가능하나 애초에 그렇게 많이 등록하지 않을 것이므로 몇개 사용하지 않는다.
        self.screen_my_info = "2000"
        self.screen_real_stock = (
            "5000"  # 종목별로 할당할 스크린 번호 -> 거래량 상위 종목을 불러올 때 마다 1씩 증가시켜 사용한다.
        )
        self.previous_screen_number = None  # 거래량 상위 종목을 바꿀 때 이전 스크린번호를 저장할 변수
        self.screen_meme_stock = "6000"  # 종목별 매매에 사용할 스크린 번호
        self.screen_start_stop_real = "1000"
        #########

        ##########
        # 변수 모음
        self.account_num = None  # 계좌번호
        self.proc = None  # subprocess 통신에 이용하는 변수
        self.account_stock_dict = (
            {}
        )  # 계좌평가 잔고내역 (원래 가지고 있는 종목) => 오버나잇이 없는 한 실제로는 사용하지 않을 듯.
        self.not_account_stock_dict = {}  # 미체결 종목
        self.jango_dict = {}  # 오늘 거래한 내역

        self.env64 = "C:/Users/brian/Anaconda3/envs/trading/python.exe"
        self.top_volume_company_py = (
            "C:/Users/brian/Python_programming/billion/update_server.py"
        )
        self.is_ing_get_volume_company = (
            False  # subprocess 로 거래량 상위 종목을 구하고 있는지 여부에 대한 bool
        )
        self.is_get_predict = False
        self.is_need_disconnect_screen = False  # 거래량 상위 종목을 구하는 시간인지 여부에 대한 bool
        self.is_time_to_restart = False
        self.is_time_to_remove = True
        self.volume_increase_company_list = []  # 거래량 상위 종목 리스트
        self.volume_increase_company_dict = {}  # subprocess 로 구해온 거래량 상위 종목
        self.portfolio_stock_dict = (
            {}
        )  # 예측 종목을 불러오고 매수를 할 때 등록해 놓는다. (매수수량,매도시도,매수가격,추가매도수량,손절완료,1차거래,2차거래)
        self.past_portfolio_stock_dict = (
            {}
        )  # 매도를 미처 다 못하고 넘어갈 때를 대비하여 이전 dict 를 copy 해 놓기 위한 용도.
        self.specified_price_count = {
            "time": datetime.now(),
            "count": 0,
        }  # 지정가격을 보낼 때 1초에 5번 이상 보내지 못하므로 숫자를 세줘야 한다.
        self.call_top_volume_companies_time1 = 21  # 거래량 상위 종목을 불러올 시간 1
        self.call_top_volume_companies_time2 = 51  # 거래량 상위 종목을 불러올 시간 2
        ##########

        ##########
        # 계좌관련 변수
        self.use_money = 0  # 투자에 사용할 금액
        self.use_money_percent = 0.9  # 투자에 사용할 비율
        ##########

        self.get_ocx_instance()  # ocx 방식을 파이썬에 사용할 수 있게 변환해 주는 함수
        self.event_slots()  # 키움과 연결하기 위한 시그널 / 슬롯 모음
        self.real_event_slots()  # 실시간 이벤트 시그널 / 슬롯 연결
        self.signal_login_commconnect()  # 로그인 요청 시그널 포함
        self.get_account_info()  # 계좌번호 가져오기

        self.detail_account_info()  # 예수금 요청 시그널 포함
        self.detail_account_mystock()  # 계좌평가 잔고내역 요청 시그널 포함
        self.volume_increase_company_list = self.get_volume_increase_company_naver()
        print(
            f"네이버 거래 상위 종목 {len(self.volume_increase_company_list)}개 : {self.volume_increase_company_list}"
        )
        for code in self.volume_increase_company_list:
            self.volume_increase_company_dict.update({code: {}})
        self.dynamicCall(
            "SetRealReg(QString,QString,QString,QString)",
            self.screen_start_stop_real,
            "",
            self.realType.REALTYPE["장시작시간"]["장운영구분"],
            "0",  # 처음만 0으로 등록.
        )
        self.register_volume_increase_company()

    def get_ocx_instance(self):
        # 응용프로그램 제어 , 경로지정
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")  # 레지스트리에 저장된 api 모듈 불러오기

    def event_slots(self):
        self.OnEventConnect.connect(self.login_slot)  # 로그인 관련 이벤트
        self.OnReceiveTrData.connect(self.trdata_slot)
        self.OnReceiveMsg.connect(self.msg_slot)

    def real_event_slots(self):
        self.OnReceiveRealData.connect(self.realdata_slot)
        self.OnReceiveChejanData.connect(self.chejan_slot)

    def login_slot(self, errcode):
        print(errors(errcode))  # 0이면 성공
        # login 이 끝나면 이벤트루프를 종료시킨다.
        self.login_event_loop.exit()

    def signal_login_commconnect(self):
        # pyqt 의 함수로 프로그램과 연결하여 실행시켜줌
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        # login 이 될 때까지 기다리게 함.
        self.login_event_loop.exec_()

    def get_account_info(self):
        account_list = self.dynamicCall("GetLoginInfo(String)", "ACCNO")  # 계좌번호
        self.account_num = account_list.split(";")[0]  # ;를 기준으로 리스트를 만듬
        print(f"나의 계좌번호 : {self.account_num}")

    def detail_account_info(self):
        print("예수금 요청")
        self.dynamicCall("SetInputValue(String,String)", "계좌번호", self.account_num)
        self.dynamicCall(
            "SetInputValue(String,String)", "비밀번호", "0000"
        )  # TODO : 비밀번호를 실전에는 바꿔야 한다.
        self.dynamicCall("SetInputValue(String,String)", "비밀번호입력매체구분", "00")
        self.dynamicCall("SetInputValue(String,String)", "조회구분", "2")
        # 화면번호(screen number)는 여러개의 요청에 대한 그룹을 지어주는 것이다. 하나의 화면번호에는 100개까지가 한계다.
        # 총 200개의 화면번호를 만들 수 있으며 번호는 마음대로 생성가능.
        # 화면번호에 대한 그룹핑으로 각 번호마다 종목 100개를 요청하는 것으로 코스피 코스닥을 나누거나
        # 주문 요청용 번호만 모아 놓은 주분번호를 만드는 등 요청별로 구분해서 만들 수 있다.
        # 예수금 상세 현황 요청은 아무이름이나 가능하나 구분 가능하게 사용한다.
        self.dynamicCall(
            "CommRqData(String,String,int,String)",
            "예수금상세현황요청",
            "opw00001",
            "0",
            self.screen_my_info,
        )

        self.account_info_event_loop.exec_()

    # sPrevNext = 0 이면 마지막 페이지라는 의미 ( 보유종목이 20개 이하 )
    def detail_account_mystock(self, sPrevNext="0"):
        print("계좌평가잔고내역요청")
        self.dynamicCall("SetInputValue(String,String)", "계좌번호", self.account_num)
        self.dynamicCall(
            "SetInputValue(String,String)", "비밀번호", "0000"
        )  # TODO : 비밀번호를 실전에는 바꿔야 한다.
        self.dynamicCall("SetInputValue(String,String)", "비밀번호입력매체구분", "00")
        self.dynamicCall("SetInputValue(String,String)", "조회구분", "2")
        self.dynamicCall(
            "CommRqData(String,String,int,String)",
            "계좌평가잔고내역요청",
            "opw00018",
            sPrevNext,
            self.screen_my_info,
        )
        # 페이지가 여러개(보유종목이 20개 이상) 일 경우 이 함수를 한번 더 실행하는데 그 때 이벤트 루프가 한번 더
        # 발생하는 경우가 생길 수 있으므로 이벤트 루프를 여기서 정의하지 않고 init 에서 정의 했다.
        # 실행도 두번이 될 것이어서 에러가 날 수 있으나 상관없다고 함.
        # self.account_info_event_loop2 = QEventLoop()
        self.account_info_event_loop.exec_()

    def trdata_slot(self, sScrNo, sRQName, sTrCode, sRecordName, sPrevNext):
        """
        tr 요청을 받는 구역
        :param sScrNo: 스크린 번호
        :param sRQName: 내가 요청했을 때 지은 이름
        :param sTrCode: 요청 trcode
        :param sRecordName: 사용안함
        :param sPrevNext: 연속조회 유무를 판단하는 값 0: 연속(추가조회)데이터 없음, 2:연속(추가조회) 데이터 있음
        :return:
        """

        if sRQName == "예수금상세현황요청":
            # TR 데이터 목록의 예수금 상세 현황 요청 부분에 있는 데이터들을 요청할 수 있다.
            deposit = self.dynamicCall(
                "GetCommData(String,String,int,String)", sTrCode, sRQName, 0, "예수금"
            )
            ok_deposit = self.dynamicCall(
                "GetCommData(String,String,int,String)", sTrCode, sRQName, 0, "출금가능금액"
            )
            available_deposit = self.dynamicCall(
                "GetCommData(String,String,int,String)", sTrCode, sRQName, 0, "주문가능금액"
            )
            # 사용할 돈은 총 사용가능한 돈의 일정 비율만 사용한다.
            self.use_money = int(available_deposit) * self.use_money_percent

            print(f"예수금 : {int(deposit)}")
            print(f"출금가능금액 : {int(ok_deposit)}")
            print(f"주문가능금액 : {int(available_deposit)}")
            print(f"투자에 사용할 금액 : {self.use_money}")

            self.account_info_event_loop.exit()
        elif sRQName == "계좌평가잔고내역요청":
            total_buy_money = self.dynamicCall(
                "GetCommData(String,String,int,String)", sTrCode, sRQName, 0, "총매입금액"
            )
            print(f"총매입금액 : {int(total_buy_money)}")
            total_profit_rate = self.dynamicCall(
                "GetCommData(String,String,int,String)", sTrCode, sRQName, 0, "총수익률(%)"
            )
            print(f"총수익률 : {float(total_profit_rate):.2f}%")

            # GetRepeatCnt 를 사용하면 멀티데이터를 가져오겠다는 것이다.
            # cnt 는 최대 20이다. 그 이상 보유하고 있으면 sPrevNext 가 2가 되어 다음 페이지로 넘어가게 됨.

            cnt = self.dynamicCall(
                "GetRepeatCnt(QString,QString)", sTrCode, sRQName
            )  # 보유 종목 수
            print(f"보유하고 있는 종목 수: {cnt}")
            for i in range(cnt):
                code = self.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    sTrCode,
                    sRQName,
                    i,
                    "종목번호",
                )  # 출력 : A039423 // 알파벳 A는 장내주식, J는 ELW종목, Q는 ETN종목
                code_nm = self.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    sTrCode,
                    sRQName,
                    i,
                    "종목명",
                )  # 출럭 : 한국기업평가
                stock_quantity = self.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    sTrCode,
                    sRQName,
                    i,
                    "보유수량",
                )  # 보유수량 : 000000000000010
                buy_price = self.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    sTrCode,
                    sRQName,
                    i,
                    "매입가",
                )  # 매입가 : 000000000054100
                learn_rate = self.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    sTrCode,
                    sRQName,
                    i,
                    "수익률(%)",
                )  # 수익률 : -000000001.94
                possible_quantity = self.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    sTrCode,
                    sRQName,
                    i,
                    "매매가능수량",
                )
                code = code.strip()[1:]
                if code in self.account_stock_dict:
                    pass
                else:
                    self.account_stock_dict[code] = {}

                code_nm = code_nm.strip()
                stock_quantity = int(stock_quantity.strip())
                buy_price = int(buy_price.strip())
                learn_rate = float(learn_rate.strip())
                possible_quantity = int(possible_quantity.strip())

                order_success = self.dynamicCall(
                    "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                    [
                        "신규매도",
                        self.screen_meme_stock,
                        self.account_num,
                        2,  # 신규 매도 : 2
                        code,
                        stock_quantity,
                        0,
                        self.realType.SENDTYPE["거래구분"]["시장가"],
                        "",
                    ],
                )
                if order_success == 0:
                    print(f"{code} 보유종목 매도주문 전달 성공")
                else:
                    print(f"{code} 보유종목 매도주문 전달 실패")
                # 한번 지정해 놓고 하면 계속 찾는 연산을 하지 않아도 된다.
                update_dict = self.account_stock_dict[code]

                update_dict.update({"종목명": code_nm})
                update_dict.update({"보유수량": stock_quantity})
                update_dict.update({"매입가": buy_price})
                update_dict.update({"수익률(%)": learn_rate})
                update_dict.update({"매매가능수량": possible_quantity})
                self.portfolio_stock_dict.update(
                    {
                        code: {
                            "매수수량": stock_quantity,
                            "매도시도": False,
                            "추가매도수량": 0,
                            "손절완료": False,
                            "1차거래": False,
                            "2차거래": False,
                        }
                    }
                )
                print(f"{code_nm}의 수익률 : {learn_rate}")

            if sPrevNext == "2":
                self.detail_account_mystock(sPrevNext="2")
            else:
                self.account_info_event_loop.exit()

    @staticmethod
    def get_volume_increase_company_naver():
        headers = {"Content-Type": "application/json"}
        address = "http://127.0.0.1:2431/naver_update"
        result = post(address, headers=headers)
        encode = str(result.content, encoding="utf-8")
        encode = loads(encode.replace("'", '"'))
        return encode

    def get_real_data(self):

        headers = {"Content-Type": "application/json"}
        address = "http://127.0.0.1:2431/inference"
        result = post(address, headers=headers)
        encode = str(result.content, encoding="utf-8")
        encode = loads(encode.replace("'", '"'))
        return encode

    def sell_stock(self, sCode, quantity, sell_price):
        order_success = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            [
                "신규매도",
                self.screen_meme_stock,
                self.account_num,
                2,  # 신규 매도 : 2
                sCode,
                quantity,
                sell_price,
                self.realType.SENDTYPE["거래구분"]["지정가"],
                "",
            ],
        )
        print(f"지정가 매도 - {sCode} - {quantity} - {sell_price}")
        self.specified_price_count["count"] += 1
        if self.specified_price_count["count"] == 5:
            print(f"=====================지정가 매도 5번 초과로 1초 쉼====================")
            QTest.qWait(1000)
            self.specified_price_count["count"] = 0
        if order_success == 0:
            print(f"{sCode} 지정가 매도주문 전달 성공")
            return 0
        else:
            print(f"{sCode} 지정가 매도주문 전달 실패")
            return quantity

    def register_volume_increase_company(self):
        """
        거래 상위 종목을 구하고 데이터를 받기로 등록을 한다.
        :return:
        """
        fids = self.realType.REALTYPE["주식체결"]["체결시간"]  # 틱데이터(체결이 이루어질 때)마다 체결시간을 알려줌.
        self.screen_real_stock = str(int(self.screen_real_stock) + 1)
        for code in self.volume_increase_company_list:
            self.dynamicCall(
                "SetRealReg(QString,QString,QString,QString)",
                self.screen_real_stock,
                code,
                fids,
                "1",
            )

    def buy_stock_at_specified_price(self, pred_company):
        """
        예측 결과 리스트를 받으면 그 종목을 매수한다.
        :param pred_company: 예측 종목
        :return:
        """
        self.past_portfolio_stock_dict = self.portfolio_stock_dict.copy()
        self.portfolio_stock_dict.clear()
        count = max(3, len(pred_company))
        use_money = int(self.use_money / count)
        for code in pred_company:
            try:
                now_price = self.volume_increase_company_dict[code]["현재가"]
            except KeyError:
                continue
            buy_quantity = int(use_money / now_price)
            if buy_quantity == 0:
                print(f"돈이 부족하여 매수 불가 : {code} - {now_price}원")
                continue
            order_success = self.dynamicCall(
                "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                [
                    "신규매수",  # 사용자 지정명
                    self.screen_meme_stock,
                    self.account_num,
                    1,  # 신규 매수 : 1
                    code,
                    buy_quantity,
                    0,
                    self.realType.SENDTYPE["거래구분"]["시장가"],
                    "",
                ],
            )
            self.portfolio_stock_dict.update(
                {
                    code: {
                        "매수수량": buy_quantity,
                        "매도시도": False,
                        "추가매도수량": 0,
                        "손절완료": False,
                        "1차거래": False,
                        "2차거래": False,
                    }
                }
            )
            if order_success == 0:
                print(f"{code} 매수주문 전달 성공")
            else:
                print(f"{code} 매수주문 전달 실패")

    def remove_real_reg(self):
        """
        5분이 되기 전에 모든 종목을 종가 매도 하여야 하는데 실제로는 종가는 못하고 50초 쯤 매도 주문을 넣는다.
        매도 정정으로 아직 체결이 안 된 것들을 모두 판다.
        :return:
        """
        for order_num in self.not_account_stock_dict.keys():
            the_not_account_dict = self.not_account_stock_dict[order_num]
            code = the_not_account_dict["종목코드"]
            if the_not_account_dict["매도취소여부"] is False:
                self.sell_cancel(
                    the_not_account_dict, order_num, code,
                )

    def restart(self):
        """
        5분이 되기 전에 모든 종목을 종가 매도 하여야 하는데 실제로는 종가는 못하고 50초 쯤 매도 주문을 넣는다.
        매도 정정으로 아직 체결이 안 된 것들을 모두 판다.
        :return:
        """

        is_finish = True
        for code in self.jango_dict.keys():
            the_jango_dict = self.jango_dict[code]
            if the_jango_dict["매도여부"] is False:
                is_finish = False
                self.stop_loss(
                    the_jango_dict, code,
                )
        return is_finish

    def sell_cancel(self, the_not_account_dict, order_num, sCode, is_loss_cut=False):
        """
        가격이 손절가에 도달하거나 시간이 다 되었을 경우 가진 주식을 모두 파는 함수다.
        :param the_not_account_dict: 해당 종목의 미체결 dict
        :param order_num: 주문번호
        :param sCode: 해당 코드
        :return:
        """
        print(f"{sCode} - {order_num} 매도 취소")
        order_success = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            [
                "매도취소",
                self.screen_meme_stock,
                self.account_num,
                4,  # 매도 취소
                sCode,
                0,  # 전량 취소
                0,  # 주문 가격
                # 지정가를 시장가로 바꿔서 정정을 하는 것은 불가하며 이렇게 하면 상하한가 오류가 발생하게 된다.
                self.realType.SENDTYPE["거래구분"]["지정가"],
                order_num,
            ],
        )

        if order_success == 0:
            the_jango_dict = self.jango_dict[sCode]
            the_not_account_dict["매도취소여부"] = True
            the_jango_dict[
                "매도여부"
            ] = False  # 한 종목을 여러번에 나눠서 매도 하므로 매도 취소가 되면 False 로 바꿔줘야 한다.
            if is_loss_cut:
                the_jango_dict["손절"] = True
            print("매도취소 전달 성공")
        else:
            print("매도취소 전달 실패")

    def stop_loss(self, the_jango_dict, sCode, is_loss_cut=False):
        available_sell_count = the_jango_dict["주문가능수량"]
        if available_sell_count == 0:
            return
        order_success = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            [
                "신규매도",
                self.screen_meme_stock,
                self.account_num,
                2,  # 신규 매도
                sCode,
                available_sell_count,
                0,  # 주문 가격
                # 지정가를 시장가로 바꿔서 정정을 하는 것은 불가하며 이렇게 하면 상하한가 오류가 발생하게 된다.
                self.realType.SENDTYPE["거래구분"]["시장가"],
                "",
            ],
        )

        if order_success == 0:
            the_jango_dict["매도여부"] = True
            self.portfolio_stock_dict[sCode]["손절완료"] = True
            if is_loss_cut:
                the_jango_dict["손절"] = False
            print("매도취소 후 매도 전달 성공")
        else:
            print("매도취소 후 매도 전달 실패")

    def manage_rest_stock(
        self, the_jango_dict, sCode, sell_count, the_portfolio, first_trade
    ):
        print(f"{sCode} 지정가 매도 실패 종목 시장가 매도 -  {sell_count}")
        order_success = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            [
                "신규매도",
                self.screen_meme_stock,
                self.account_num,
                2,  # 신규 매도
                sCode,
                sell_count,
                0,  # 주문 가격
                # 지정가를 시장가로 바꿔서 정정을 하는 것은 불가하며 이렇게 하면 상하한가 오류가 발생하게 된다.
                self.realType.SENDTYPE["거래구분"]["시장가"],
                "",
            ],
        )

        if order_success == 0:
            the_portfolio["추가매도수량"] -= sell_count
            if first_trade:
                the_portfolio["1차거래"] = True
            else:
                the_portfolio["2차거래"] = True
            if the_portfolio["추가매도수량"] == 0:
                the_jango_dict["매도여부"] = True
            print("매도취소 후 매도 전달 성공")
        else:
            print("매도취소 후 매도 전달 실패")

    def realdata_slot(self, sCode, sRealType, sRealData):
        if sRealType == "장시작시간":
            fid = self.realType.REALTYPE[sRealType]["장운영구분"]
            value = self.dynamicCall("GetCommRealData(QString,int)", sCode, fid)

            if value == "0":
                print("장 시작 전")
            elif value == "3":
                print("장 시작")  # 9시
            elif value == "2":
                while True:
                    is_finish = self.restart()  # 모두 시장가 매도를 하고 종료한다.
                    if is_finish:
                        break
                print("장 종료 후 동시호가로 넘어감")
                self.dynamicCall("DisconnectRealData(QString)", self.screen_real_stock)
                exit()
            elif value == "4":
                print("3시 30분 장 종료")
                exit()
            else:
                print(f"장 시작 시간 value : {value}")

        elif sRealType == "주식체결":
            execute_time = self.dynamicCall(
                "GetCommRealData(QString, int)",
                sCode,
                self.realType.REALTYPE[sRealType]["체결시간"],
            )  # 출력 HHMMSS -> string
            now_price = self.dynamicCall(
                "GetCommRealData(QString, int)",
                sCode,
                self.realType.REALTYPE[sRealType]["현재가"],
            )  # 출력 : +(-)2520
            now_price = abs(int(now_price))

            if sCode not in self.volume_increase_company_dict:
                if sCode in self.past_portfolio_stock_dict:
                    print(
                        f"==============past_portfolio_stock_dict 에 있어서 {sCode} 추가 !! =============="
                    )
                    self.volume_increase_company_dict.update({sCode: {}})
                else:
                    return
            the_volume_dict = self.volume_increase_company_dict[sCode]
            the_volume_dict.update({"현재가": now_price})
            execute_minute = int(execute_time[2:4])

            # 5분 단위와 2초이상 경과하지 않았을 경우 상승 예측 종목을 불러온다.
            # ToDO: 연속 중복 거래를 막고 싶으나 그렇게 하면 새로운 종목을 살 돈이 없게 되므로 일단은 스킵
            if (
                execute_minute % 5 == 0
                and int(execute_time[-2:]) < 2
                and self.is_get_predict is False
            ):
                t = time()
                try:
                    predict_company = self.get_real_data()
                    if len(predict_company) != 0:
                        self.buy_stock_at_specified_price(predict_company)
                    print(f" 가격 상승 예상 종목을 불러오는 데 걸리는 시간 : {time() - t}")
                except Exception as e:
                    print(f"예측 종목을 불러오는 데 에러 발생 - {e}")
                    predict_company = None
                self.is_get_predict = True
                self.is_time_to_remove = True
                print(f"예상 종목 : {predict_company}")
                QTest.qWait(800)
            # 가격 상승 예상 종목을 불러올 때, 실시간 체결을 모두 불러와서 시간이 지연되게 된다. 그래서 추가적인 boolean 을 변수로 넣음.
            if execute_minute % 5 != 0 and self.is_get_predict is True:
                self.is_get_predict = False
                # 기존의 스크린을 끊고 재등록.
                if self.is_need_disconnect_screen:
                    print("screen 을 바꿔야 하는 조건 통과")
                    self.dynamicCall(
                        "DisconnectRealData(QString)", self.previous_screen_number
                    )  # 해당 화면번호에 등록되어 있는 모든 종목의 연결을 끊는다.
                    self.dynamicCall(
                        "DisconnectRealData(QString)", self.screen_real_stock
                    )  # 해당 화면번호에 등록되어 있는 모든 종목의 연결을 끊는다.
                    self.volume_increase_company_dict.clear()

                    self.register_volume_increase_company()
                    # 이전 스크린 연결을 끊고 새로 등록한다.
                    for company_code in self.volume_increase_company_list:
                        self.volume_increase_company_dict.update({company_code: {}})
                    print(
                        f"screen 바꾼 후 등록 갯수 : {len(self.volume_increase_company_dict.keys())}"
                    )
                    self.is_need_disconnect_screen = False

            # 12분과 42분에 거래량 상위 종목을 불러오는 subprocess 를 실행한다.
            if (
                execute_minute == self.call_top_volume_companies_time1
                or execute_minute == self.call_top_volume_companies_time2
            ) and self.is_ing_get_volume_company is False:
                self.proc = Popen([self.env64, self.top_volume_company_py], stdout=PIPE)
                self.is_ing_get_volume_company = True

            # subprocess 를 실행하고 약 2분뒤 거래량 상위 종목을 미리 불러온다.
            if (
                execute_minute == self.call_top_volume_companies_time1 + 2
                or execute_minute == self.call_top_volume_companies_time2 + 2
            ) and self.is_ing_get_volume_company is True:
                try:
                    self.volume_increase_company_list, _ = self.proc.communicate()
                    self.volume_increase_company_list = loads(
                        self.volume_increase_company_list.decode("utf-8").replace(
                            "'", '"'
                        )
                    )
                except Exception as e:
                    self.volume_increase_company_list = (
                        self.get_volume_increase_company_naver()
                    )
                    print(f"communicate 중 에러 발생 - {e} , 네이버 상위 종목으로 대체.")
                self.is_need_disconnect_screen = True
                self.is_ing_get_volume_company = False
                self.previous_screen_number = (
                    self.screen_real_stock
                )  # 이전 스크린 번호를 미리 등록.
                for company_code in self.volume_increase_company_list:
                    self.volume_increase_company_dict.update({company_code: {}})
                print(
                    f"거래량 상위 종목 {len(self.volume_increase_company_list)}개: {self.volume_increase_company_list}"
                )
                self.register_volume_increase_company()

            # 5분이 되기 10초 전 가지고 있는 모든 종목을 매도한다.
            # TODO : 5분 이후에 매도하지 못한 종목처리를 어떻게 해야될까??
            if (
                execute_minute % 5 == 4
                and int(execute_time[-2:]) > 50
                and self.is_time_to_remove
            ):
                self.remove_real_reg()
                if len(self.not_account_stock_dict) == 0:
                    self.is_time_to_remove = False
                    self.is_time_to_restart = True
                    print("매도 취소 성공 !!!")
            if self.is_time_to_restart and execute_minute % 5 == 4:
                is_finish = self.restart()
                if is_finish:
                    self.is_time_to_restart = False
                    print(f"매도 취소 후 매도 성공 !!!")
                else:
                    self.is_time_to_restart = True
                # if len(self.jango_dict) == 0:
                #     self.is_time_to_restart = False

            # list 를 감싸게 되면 주소값을 공유하지 않음.
            # self.not_account_stock_dict 가 계속 업데이트 되어 바뀔 수 있어서 for 문이 이상해지므로 미리 copy 를 해놓음.
            if sCode in self.jango_dict:
                the_jango_dict = self.jango_dict[sCode]
                buy_price = the_jango_dict["매입단가"]
                # 4% 이하로 내려가면 손절 , 50~59초 사이에 손절을 하면 매도 취소를 하게 되어 조건 추가.
                if (
                    buy_price * (1 - 0.05) >= now_price
                    and self.is_time_to_restart is False
                    and self.portfolio_stock_dict[sCode]["손절완료"] is False
                ):
                    not_meme_list = list(self.not_account_stock_dict)
                    for order_num in not_meme_list:
                        not_account_info = self.not_account_stock_dict[order_num]
                        code = not_account_info["종목코드"]
                        if sCode == code and not_account_info["매도취소여부"] is False:
                            print(f"{sCode} 손절 = {buy_price} ,{now_price}")
                            self.sell_cancel(
                                not_account_info, order_num, sCode, is_loss_cut=True
                            )
                if (
                    the_jango_dict["매도여부"] is False
                    and the_jango_dict["손절"] is True
                    and self.portfolio_stock_dict[sCode]["매수수량"]
                    == the_jango_dict["주문가능수량"]  # 한번에 다 팔기 위한 조건.
                ):
                    print(f"손절 매도 : {sCode}")
                    self.stop_loss(the_jango_dict, sCode, is_loss_cut=True)

                if sCode in self.portfolio_stock_dict:
                    the_portfolio = self.portfolio_stock_dict[sCode]
                    plus_sell_quan = the_portfolio["추가매도수량"]
                    # 이전에 지정가 주문을 실패했을 경우 다음과 같이 처리한다.
                    if plus_sell_quan != 0 and the_jango_dict["매도여부"] is False:
                        if buy_price * (1 - 0.05) >= now_price:
                            print(f"{sCode} 지정가 주문 실패 후 시장가 손절 처리")
                            self.stop_loss(the_jango_dict, sCode)
                        if the_portfolio["1차거래"] is False:  # 한번만 거래하도록 한다.
                            hour = datetime.now().hour
                            sell_count = (
                                int(plus_sell_quan * 0.4)
                                if plus_sell_quan >= 2
                                else plus_sell_quan
                            )
                            if hour == 9:
                                if buy_price * (1 + 0.03) <= now_price:

                                    self.manage_rest_stock(
                                        the_jango_dict,
                                        sCode,
                                        sell_count,
                                        the_portfolio,
                                        first_trade=True,
                                    )
                            else:
                                if buy_price * (1 + 0.02) <= now_price:

                                    self.manage_rest_stock(
                                        the_jango_dict,
                                        sCode,
                                        sell_count,
                                        the_portfolio,
                                        first_trade=True,
                                    )
                        elif the_portfolio["2차거래"] is False:  # 한번만 거래하도록 한다.
                            hour = datetime.now().hour
                            if hour == 9:
                                if buy_price * (1 + 0.07) <= now_price:
                                    self.manage_rest_stock(
                                        the_jango_dict,
                                        sCode,
                                        plus_sell_quan,
                                        the_portfolio,
                                        first_trade=False,
                                    )
                            else:
                                if buy_price * (1 + 0.05) <= now_price:
                                    self.manage_rest_stock(
                                        the_jango_dict,
                                        sCode,
                                        plus_sell_quan,
                                        the_portfolio,
                                        first_trade=False,
                                    )

    def revise_scale(self, price, rate):
        """
        코스닥을 기준으로 호가 단위를 맞춰서 가격을 리턴한다.
        :param price:
        :param rate:
        :return:
        """
        sell_p = int(price * rate)
        if price < 1000:
            return sell_p
        elif price < 5000:
            return sell_p - (sell_p % 5)
        elif price < 10000:
            return sell_p - (sell_p % 10)
        elif price < 50000:
            return sell_p - (sell_p % 50)
        else:
            return sell_p - (sell_p % 100)

    def chejan_slot(self, sGubun, nItemCnt, sFidList):

        if int(sGubun) == 0:  # 주문접수와 체결 통보
            # 주문체결시에는 GetChejanData 함수를 이용한다. 실시간 타입에 포함된 FID 를 인풋으로 준다.
            sCode = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["주문체결"]["종목코드"]
            )[1:]
            order_number = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["주문체결"]["주문번호"]
            )  # 출럭: 0115061 마지막 주문번호

            not_chegual_quan = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["주문체결"]["미체결수량"]
            )  # 출력: 15, default: 0
            not_chegual_quan = int(not_chegual_quan)

            # 새로 들어온 주문이면 주문번호 할당
            if order_number not in self.not_account_stock_dict.keys():
                self.not_account_stock_dict.update({order_number: {}})
            if not_chegual_quan == 0:
                del self.not_account_stock_dict[order_number]
            else:
                not_account_dict = self.not_account_stock_dict[order_number]
                not_account_dict.update({"종목코드": sCode})
                not_account_dict.update({"미체결수량": not_chegual_quan})
                not_account_dict.update({"매수취소여부": False})
                not_account_dict.update({"매도취소여부": False})

            # 손절시 매도가 되야 가능하므로 체결이 되었을 때, 손절을 하게 한다.
            if sCode in self.jango_dict:
                the_jango_dict = self.jango_dict[sCode]
                if (
                    the_jango_dict["매도여부"] is False
                    and the_jango_dict["손절"] is True
                    and self.portfolio_stock_dict[sCode]["매수수량"]
                    == the_jango_dict["주문가능수량"]
                ):  # 한번에 다 팔기 위한 조건.
                    print(f"{sCode} chejan slot 에서 손절!")
                    print(f"{the_jango_dict}")
                    self.stop_loss(the_jango_dict, sCode, is_loss_cut=True)

        elif int(sGubun) == 1:  # 잔고 통보
            sCode = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["잔고"]["종목코드"]
            )[1:]

            current_price = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["잔고"]["현재가"]
            )
            current_price = abs(int(current_price))

            stock_quan = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["잔고"]["보유수량"]
            )
            stock_quan = int(stock_quan)

            like_quan = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["잔고"]["주문가능수량"]
            )
            like_quan = int(like_quan)
            # 주문 가능 수량은 매수 후 매도 처리를 하고자 할 때 이미 매도를 절반을 했다면 나머지 절반을 의미한다.

            buy_price = self.dynamicCall(
                "GetChejanData(int)", self.realType.REALTYPE["잔고"]["매입단가"]
            )
            buy_price = abs(int(buy_price))

            if sCode not in self.jango_dict:
                self.jango_dict.update({sCode: {}})
            the_dict = self.jango_dict[sCode]
            the_dict.update({"현재가": current_price})
            the_dict.update({"종목코드": sCode})
            the_dict.update({"보유수량": stock_quan})
            the_dict.update({"주문가능수량": like_quan})
            the_dict.update({"매입단가": buy_price})
            the_dict.update({"매도여부": False})
            the_dict.update({"손절": False})
            if (
                sCode not in self.portfolio_stock_dict
                and sCode in self.past_portfolio_stock_dict
            ):
                self.portfolio_stock_dict.update(
                    {sCode: self.past_portfolio_stock_dict[sCode]}
                )

                print(
                    f"=========================================={sCode}가 past_portfolio_stock_dict 에 있어 추가해준다.================================================"
                )
                print(self.past_portfolio_stock_dict)
            the_portfolio = self.portfolio_stock_dict[sCode]
            print(
                f"chejan slot - {sCode} 매수수량 : {the_portfolio['매수수량']}, 보유수량 : {stock_quan}, 주문가능수량 : {like_quan}"
            )
            if stock_quan == the_portfolio["매수수량"] and the_portfolio["매도시도"] is False:
                now = datetime.now()
                if (
                    now - self.specified_price_count["time"]
                ).seconds >= 2:  # 2초 이상 지나면 카운트 초기화
                    self.specified_price_count["count"] = 0
                self.specified_price_count["time"] = now
                hour = now.hour
                if stock_quan < 3:
                    if hour == 9:
                        sell_price = self.revise_scale(buy_price, 1.05)
                        add_quantity = self.sell_stock(sCode, stock_quan, sell_price)
                    else:
                        sell_price = self.revise_scale(buy_price, 1.04)
                        add_quantity = self.sell_stock(sCode, stock_quan, sell_price)

                elif stock_quan < 5:
                    if hour == 9:
                        sell_price = self.revise_scale(buy_price, 1.04)
                        quan1 = self.sell_stock(sCode, 2, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.07)
                        quan2 = self.sell_stock(sCode, stock_quan - 2, sell_price)
                        add_quantity = quan1 + quan2
                    elif hour == 10 or hour == 14:
                        sell_price = self.revise_scale(buy_price, 1.03)
                        quan1 = self.sell_stock(sCode, 2, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.06)
                        quan2 = self.sell_stock(sCode, stock_quan - 2, sell_price)
                        add_quantity = quan1 + quan2
                    else:
                        sell_price = self.revise_scale(buy_price, 1.03)
                        quan1 = self.sell_stock(sCode, 2, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.05)
                        quan2 = self.sell_stock(sCode, stock_quan - 2, sell_price)
                        add_quantity = quan1 + quan2

                elif stock_quan < 10:
                    if hour == 9:
                        sell_price = self.revise_scale(buy_price, 1.03)
                        quan1 = self.sell_stock(sCode, 2, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.07)
                        quan2 = self.sell_stock(sCode, 3, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.10)
                        quan3 = self.sell_stock(sCode, stock_quan - 5, sell_price)
                        add_quantity = quan1 + quan2 + quan3
                    elif hour == 10 or hour == 14:
                        sell_price = self.revise_scale(buy_price, 1.02)
                        quan1 = self.sell_stock(sCode, 2, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.05)
                        quan2 = self.sell_stock(sCode, 3, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.08)
                        quan3 = self.sell_stock(sCode, stock_quan - 5, sell_price)
                        add_quantity = quan1 + quan2 + quan3
                    else:
                        sell_price = self.revise_scale(buy_price, 1.02)
                        quan1 = self.sell_stock(sCode, 3, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.045)
                        quan2 = self.sell_stock(sCode, 2, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.075)
                        quan3 = self.sell_stock(sCode, stock_quan - 5, sell_price)
                        add_quantity = quan1 + quan2 + quan3
                else:
                    first_quan = int(stock_quan * 0.2)
                    second_quan = int(stock_quan * 0.2)
                    third_quan = int(stock_quan * 0.3)
                    gap_quan = stock_quan - (first_quan + second_quan + third_quan)
                    if hour == 9:
                        sell_price = self.revise_scale(buy_price, 1.02)
                        quan1 = self.sell_stock(sCode, first_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.04)
                        quan2 = self.sell_stock(sCode, second_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.08)
                        quan3 = self.sell_stock(sCode, third_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.10)
                        quan4 = self.sell_stock(sCode, gap_quan, sell_price)
                        add_quantity = quan1 + quan2 + quan3 + quan4
                    elif hour == 10 or hour == 14:
                        sell_price = self.revise_scale(buy_price, 1.02)
                        quan1 = self.sell_stock(sCode, first_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.03)
                        quan2 = self.sell_stock(sCode, second_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.05)
                        quan3 = self.sell_stock(sCode, third_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.08)
                        quan4 = self.sell_stock(sCode, gap_quan, sell_price)
                        add_quantity = quan1 + quan2 + quan3 + quan4
                    else:
                        sell_price = self.revise_scale(buy_price, 1.015)
                        quan1 = self.sell_stock(sCode, first_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.03)
                        quan2 = self.sell_stock(sCode, second_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.05)
                        quan3 = self.sell_stock(sCode, third_quan, sell_price)
                        sell_price = self.revise_scale(buy_price, 1.07)
                        quan4 = self.sell_stock(sCode, gap_quan, sell_price)
                        add_quantity = quan1 + quan2 + quan3 + quan4
                the_portfolio["추가매도수량"] = add_quantity
                the_portfolio["매도시도"] = True

            elif the_portfolio["매도시도"] is True:
                # 상하한가 오류로 지정가 매도에 성공했지만 실제로는 되지 않는 경우가 있다. 그래서 따로 지정해줬다.
                the_portfolio["추가매도수량"] = like_quan

            if stock_quan == 0:
                del self.jango_dict[sCode]

    # 송수신 메세지 get, 서버통신 후 수신한 메시지를 알려준다.
    def msg_slot(self, sScrNo, sRQName, sTrCode, msg):
        print(f"스크린: {sScrNo}, 요청이름: {sRQName}, tr코드: {sTrCode} --- {msg}")
