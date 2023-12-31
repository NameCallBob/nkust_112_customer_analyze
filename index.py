import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Main().test()
    
class Customer(Data):
    """顧客分析"""
    def __init__(self) -> None :
        # 設定中文字型
        # plt.rcParams['font.family'] = 'Microsoft YaHei'
        plt.rcParams['font.family'] = 'Arial Unicode'
        plt.rcParams['axes.unicode_minus'] = False

    def main(self):
        # 載入資料集
        df = super().read()

        # 繪製年齡分佈圖
        plt.figure(figsize=(10, 6))
        df['AGE_GROUP'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('顧客年齡分佈')
        plt.xlabel('年齡組別')
        plt.ylabel('顧客數量')
        plt.show()

        # 繪製每個年齡組別的總銷售金額
        plt.figure(figsize=(10, 6))
        df.groupby('AGE_GROUP')['AMOUNT'].sum().sort_index().plot(kind='bar', color='lightgreen')
        plt.title('每個年齡組別的總銷售金額')
        plt.xlabel('年齡組別')
        plt.ylabel('總銷售金額')
        plt.show()

        # 繪製每個PIN碼的顧客數量
        plt.figure(figsize=(12, 8))
        df['PIN_CODE'].value_counts().head(20).plot(kind='bar', color='orange')
        plt.title('每個地區的顧客數量')
        plt.xlabel('郵遞區號')
        plt.ylabel('顧客數量')
        plt.show()
        
        # 將 TRANSACTION_DT 轉換為日期類型
        df['TRANSACTION_DT'] = pd.to_datetime(df['TRANSACTION_DT'])
        # 計算每位顧客的購買次數
        customer_purchase_count = df.groupby('CUSTOMER_ID')['TRANSACTION_DT'].count()

        # 初次客與回流客分布
        plt.figure(figsize=(12, 6))

        # 初次客與回流客分布的長條圖
        plt.subplot(1, 2, 1)
        returning_customers = (customer_purchase_count > 1).sum() / len(customer_purchase_count)
        new_customers = 1 - returning_customers
        sns.barplot(x=['初次客', '回流客'], y=[new_customers, returning_customers])
        plt.title('初次客與回流客比例 (長條圖)')

        # 初次客與回流客分布的圓餅圖
        plt.subplot(1, 2, 2)
        labels = ['初次客', '回流客']
        sizes = [new_customers, returning_customers]
        colors = ['#ff9999','#66b3ff']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('初次客與回流客比例 (圓餅圖)')
        plt.tight_layout()
        plt.show()

        # 計算每位顧客的購買次數和總交易金額
        customer_summary = df.groupby('CUSTOMER_ID').agg({'TRANSACTION_DT': 'count', 'AMOUNT': 'sum'})

        # 初次客和回流客的交易金額
        new_customers = customer_summary[customer_summary['TRANSACTION_DT'] == 1]
        returning_customers = customer_summary[customer_summary['TRANSACTION_DT'] > 1]

        # 計算平均交易金額
        avg_amount_new_customers = new_customers['AMOUNT'].mean()
        avg_amount_returning_customers = returning_customers['AMOUNT'].mean()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=['初次客', '回流客'], y=[avg_amount_new_customers, avg_amount_returning_customers])
        plt.title('初次客與回流客的平均交易金額')
        plt.xlabel('顧客類別')
        plt.ylabel('平均交易金額 (元)')
        plt.show()


Customer().main()