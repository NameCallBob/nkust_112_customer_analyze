from data import Data
import pandas as pd ; import numpy as np ; import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from datetime import datetime
# model
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
class Model(Data):
    """
    主要用於模型訓練的code
    """
    def __init__(self) -> None:
        self.data = super().read()
        self.data_s = super().read_standardize()

    def Cluster(self):
        """分群"""
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # 讀取資料
        df = self.__RFM_data()
        print(df)

        # 繪製散點圖 - Kmeans
        # 找出最好效果的群數
        # self.__kmeans_group_choice(df[['Recency', 'Frequency', 'Monetary']],10)
        # train_df, test_df = train_test_split(df[['Recency', 'Frequency', 'Monetary']], test_size=0.2, random_state=42)\

        kmeans = KMeans(n_clusters=4,random_state=42)
        df['Cluster'] = kmeans.fit_predict(df)
        # 設定圖表風格
        from mpl_toolkits.mplot3d import Axes3D
        colors=['purple', 'blue', 'green', 'gold']
        fig = plt.figure()
        fig.set_size_inches(15,15)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(kmeans.n_clusters):
            df_cluster=df[df['Cluster']==i]
            ax.scatter(df_cluster['Recency'], df_cluster['Monetary'],df_cluster['Frequency'],s=50,label='Cluster'+str(i+1), c=colors[i])
        # 添加標籤
        ax.set_xlabel('Recency')
        ax.set_ylabel('Monetary')
        ax.set_zlabel('Frequency')

        # 添加標題
        ax.set_title('3D K-Means Clustering')
        # 顯示圖例
        plt.legend()
        ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=200,marker='^', c='red', alpha=0.7, label='Centroids')
        # 顯示圖形
        plt.show()



    def Linear(self):
        data = self.day(name=2)
        # 將時間轉換為日期格式
        data['TRANSACTION_DT'] = pd.to_datetime(data['TRANSACTION_DT'])
        df = data
       # 選擇相應的特徵和目標
        for i in self.TOP5_product():
            df = df[df["PRODUCT_SUBCLASS"] == i]
            X = df[['AMOUNT']]
            y = df['SALES_PRICE']
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            # 創建 ColumnTransformer 來處理類別變數
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', OneHotEncoder(), ['PRODUCT_SUBCLASS'])
                ],
                remainder='passthrough'
            )

            # 創建 Pipeline，將處理器和模型包裝在一起
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', LinearRegression())
            ])

            # 配適模型
            pipeline.fit(X, y)

            # 繪製結果
            plt.scatter(X['AMOUNT'], y, color='blue')
            plt.plot(X['AMOUNT'], pipeline.predict(X), color='red', linewidth=3)
            plt.xlabel('AMOUNT')
            plt.ylabel('SALES_PRICE')
            plt.title('簡單線性回歸（包含類別變數）')
            plt.show()

        # from sklearn import metrics
        # # 計算回歸指標
        # mae = metrics.mean_absolute_error(y_test, y_pred)
        # mse = metrics.mean_squared_error(y_test, y_pred)
        # r_squared = metrics.r2_score(y_test, y_pred)
        # # 輸出回歸指標
        # print(f'Mean Absolute Error: {mae}')
        # print(f'Mean Squared Error: {mse}')
        # print(f'R-squared: {r_squared}')
        # """
        # Mean Absolute Error (MAE) - 平均絕對誤差：

        # 意義： 衡量預測值和實際值之間的平均絕對誤差。
        # 解釋： MAE 的值越低越好。在這個案例中，MAE 約為 10963 元，這表示每次預測的平均誤差約為 10963 元。如果 MAE 過高，可能意味著模型尚有改進的空間。
        # Mean Squared Error (MSE) - 均方誤差：

        # 意義： 衡量預測值和實際值之間平方誤差的平均值。
        # 解釋： MSE 的值越低越好。在這個案例中，MSE 約為 469527280，這表示預測值和實際值之間的誤差平方的平均值。MSE 對較大的誤差給予更高的權重。
        # R-squared - 決定係數：

        # 意義： 衡量模型解釋目標變量變異性的能力，介於 0 和 1 之間。
        # 解釋： R-squared 約為 0.76，表示模型能夠解釋目標變量約 76% 的變異性。較高的 R-squared 表示模型更能解釋目標變量的變異。
        # """

    def __find_best_params(self):
        """找尋model最佳參數"""
        pass

    def __kmeans_group_choice(self,transformed_data,group_num):
        """找尋最佳成效分群"""
        from sklearn.metrics import silhouette_score
        res = []
        for i in range(2,group_num):
            kmeans = KMeans(n_clusters=int(i), random_state=0,n_init=100,max_iter=1000,init="k-means++")
            # 適應模型
            kmeans.fit(transformed_data)
            res.append(silhouette_score(transformed_data,kmeans.labels_))
        plt.plot(range(2,group_num),res)
        plt.title('elbow');plt.xlabel('No. cluster')
        plt.show()

    def __RFM_data(self):
        """依照顧客資料，輸出RFM結果"""
        data = self.read() ; res = pd.DataFrame()
        data = data[["CUSTOMER_ID","AMOUNT","SALES_PRICE","TRANSACTION_DT","TOTAL"]]
        # 處理R
        R = data[["CUSTOMER_ID","TRANSACTION_DT"]].groupby("CUSTOMER_ID").max()
        target = pd.to_datetime("2001-2-28")
        R['Day'] = (target - R["TRANSACTION_DT"]).dt.days
        print(R['Day'].mean())
        res['Recency'] = R.apply(self.__R_score,axis=1)
        # print(res['Recency'])
        # 處理F
        F = data[["CUSTOMER_ID","AMOUNT"]].groupby("CUSTOMER_ID").count().sort_values(by="AMOUNT",ascending=False)
        # print(F)
        res['Frequency'] = F.apply(self.__F_score,axis=1)
        # print(res['frequency'])
        # 處理M
        M = data[["CUSTOMER_ID","TOTAL"]].groupby("CUSTOMER_ID").sum().sort_values(by="TOTAL",ascending=False)
        # print(M)
        res['Monetary'] = M.apply(self.__M_score,axis=1)
        # print(res['monetary'])
        print(res.value_counts().to_excel("12345.xlsx"))
        # res.to_excel("1234.xlsx")
        # tmp
        # res['Frequency'] = F;res['Monetary'] = M
        # res.to_excel("123.xlsx")
        return res

    def show(self):
        """展現成果"""

        print("-"*50)
        print("RFM分群結果")
        self.Cluster()
        print("-"*50)
        # print("Kmeans分群結果")
        # self.regress()
        # print("輸出結束")
    def __R_score(self,row):
        """
        將R進行分5等分

        分等解釋 day = [9,18,36,96,122]

        <= 24 -> 24天內回購

        以此類推
        """
        day = [9,18,36,96,122]
        for i in range(len(day)):
            if row['Day'] <= day[i]:
                return len(day) - i
    def __F_score(self,row):
        t = [5,7,25,100,1246]
        # t = sorted(t)
        # print(t)
        for i in range(0,5):
            if row['AMOUNT'] <= t[i]:
                return i+1
            elif row['AMOUNT'] >= t[4]:
                return 5
    def __M_score(self,row):
        t = [1000,2000,3000,60000,355221564]
        # t = sorted(t,)
        # print(t)
        for i in range(0,5):
            if row['TOTAL'] <= t[i]:
                return i+1
            elif row['TOTAL'] >= t[4]:
                return 5

    def test(self):
        print(self.__RFM_data())
Model().Linear()
# Model().Cluster()
# Model().test()


# class Cluster():
#     def __init__(self):
#         self.R_max =