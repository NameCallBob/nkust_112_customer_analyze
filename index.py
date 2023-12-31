import pandas as pd
import matplotlib.pyplot as plt
from data import Data

class Main(Data):
    """
    資料探索性分析
    """
    def __init__(self) -> None:
        self.data = super().read()
        self.data_s = super().read_standardize()

    def test(self):
        # 呼叫 -> 拿取資料進行處理
        # self.age()
        # self.turnover()
        # self.count()
        self.money()

    def age(self):
        # 呼叫 -> 拿取資料進行處理
        df = self.data
        df["TRANSACTION_DT"] = pd.to_datetime(df["TRANSACTION_DT"])

        # 圓餅圖
        pie_data = df["AGE_GROUP"].value_counts()
        plt.figure(figsize=(10, 8))
        plt.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
        plt.title("年齡組分佈")
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 長條圖
        bar_data = df["AGE_GROUP"].value_counts().sort_index()
        plt.figure(figsize=(10, 6))

        bars = plt.bar(bar_data.index, bar_data, color="skyblue")

        # 在長條圖上標示數字
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

        plt.title("年齡組分佈")
        plt.xlabel("年齡組")
        plt.ylabel("人數")
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

    def turnover(self):
        df = self.data_s
       # 將日期轉換為日期類型
        df["TRANSACTION_DT"] = pd.to_datetime(df["TRANSACTION_DT"])

        # 按月份加總營業額
        monthly_sales = df.groupby(df["TRANSACTION_DT"].dt.to_period("M"))["AMOUNT"].sum()

        # 折線圖
        plt.figure(figsize=(15, 8))
        line = plt.plot(monthly_sales.index.astype(str), monthly_sales, marker="o", color="green")[0]
        plt.title("每月營業額變化")
        plt.xlabel("日期")
        plt.ylabel("營業額")

        # 在每個點上標示數字
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            plt.text(x, y, f'{y:.0f}', ha='right', va='bottom')
        plt.show()

    def count(self):
        df = self.data_s
        # 產品購買次數統計
        product_counts = df["PRODUCT_ID"].value_counts()

        # 取前10個最受歡迎的產品
        top_products = product_counts.head(10)

        # 條形圖
        plt.figure(figsize=(15, 10))
        bars = plt.bar(top_products.index.astype(str), top_products, color="skyblue")

        # 在每個條形上標示數字
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

        plt.title("前10個最受歡迎的產品")
        plt.xlabel("產品ID")
        plt.ylabel("購買次數")
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

    def money(self):
        data = self.data
        data_Total = data[["CUSTOMER_ID","TOTAL"]].groupby("CUSTOMER_ID").sum()['TOTAL']
        # print(data_Total)
        plt.plot(data_Total)
        plt.show()
        data_Quality =data[["CUSTOMER_ID","AMOUNT"]].groupby("CUSTOMER_ID").sum()['AMOUNT']
        # print(data_Quality)
        plt.plot(data_Quality)
        plt.show()

Main().test()
