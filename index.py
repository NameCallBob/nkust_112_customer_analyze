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

    def age(self):
        # 呼叫 -> 拿取資料進行處理
        df = self.data
        df["TRANSACTION_DT"] = pd.to_datetime(df["TRANSACTION_DT"])
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

        # 折線圖
        line_data = df.groupby("TRANSACTION_DT")["AGE_GROUP"].value_counts().unstack().fillna(0)
        line_data.plot(kind='line', marker='', figsize=(10, 6))  # 將 marker 設置為空字符串
        plt.title("年齡組變化折線圖")
        plt.xlabel("日期")
        plt.ylabel("人數")
        plt.legend(title="年齡組")
        plt.show()

        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

    def turnover(self):
        df = self.data_s
        # 將日期轉換為日期類型
        df["TRANSACTION_DT"] = pd.to_datetime(df["TRANSACTION_DT"])

        # 按周加總營業額
        weekly_sales = df.groupby(df["TRANSACTION_DT"].dt.to_period("W-Mon"))["AMOUNT"].sum()

        # 折線圖（每周）
        plt.figure(figsize=(15, 8))
        line_weekly = plt.plot(weekly_sales.index.astype(str), weekly_sales, marker="o", color="blue")[0]
        plt.title("每周營業額變化")
        plt.xlabel("日期")
        plt.ylabel("營業額")
        plt.xticks(rotation='vertical')
        # 在每個點上標示數字
        for x, y in zip(line_weekly.get_xdata(), line_weekly.get_ydata()):
            plt.text(x, y, f'{y:.0f}', ha='right', va='bottom')
        plt.show()

    def count(self):
        df = self.data_s
        # 產品購買次數統計
        product_counts = df["PRODUCT_ID"].value_counts()

        # 取前10個最受歡迎的產品
        top_products = product_counts.head(15)


        # 橫長條圖
        plt.figure(figsize=(15, 10))
        bars_horizontal = plt.barh(top_products.index.astype(str), top_products, color="skyblue")

        # 在每個條形上標示數字
        for bar in bars_horizontal:
            xval = bar.get_width()
            plt.text(xval, bar.get_y() + bar.get_height()/2, round(xval, 1), ha='left', va='center')

        plt.title("前15個最受歡迎的產品")
        plt.xlabel("購買次數")
        plt.ylabel("產品ID")
        plt.show()

        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False
    def two(self):
        df = self.data
        df_1 = pd.read_excel('12345.xlsx')
        # 在共同列上合併數據框（在這種情況下為“CUSTOMER_ID”）
        merged_df = pd.merge(df, df_1, left_on="CUSTOMER_ID", right_on="Customer_ID")

        # 識別每個客戶群體的前三名產品
        top_products_by_segment = merged_df.groupby(["Recency", "Frequency", "Monetary"]).apply(lambda x: x.nlargest(3, "SALES_PRICE"))

        # 顯示結果
        print(top_products_by_segment)
    def show(self):
        self.age();self.turnover();self.count();self.two

class Customer(Data):
    def __init__(self) -> None:
        super().__init__()
        plt.rcParams['font.family'] = 'Microsoft YaHei'
        plt.rcParams['axes.unicode_minus'] = False
    def Anaylize(self):
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
    def New_back(self):
        import seaborn as sns
        df = super().read()

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
    def show(self):
        self.Anaylize();self.New_back()

Main().show()
Customer().show()
