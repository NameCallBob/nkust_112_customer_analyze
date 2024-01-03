from data import Data
import pandas as pd ; import numpy as np ; import seaborn as sns
import matplotlib.pyplot as plt ; import matplotlib
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from datetime import datetime
# model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression , Lasso ,ElasticNet,HuberRegressor,LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report,mean_absolute_error,make_scorer, r2_score ,accuracy_score,explained_variance_score

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

class Model(Data):
    """
    主要用於模型訓練的code
    """
    def __init__(self) -> None:
        self.data = super().read()
        self.data_s = super().read_standardize()
        matplotlib.rc('font', family='Microsoft JhengHei')

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
    def Logistic(self):
        df = super().read_s_l()
        # 使用分類器進行預處理
        kmeans = KMeans(n_clusters=13, random_state=17)
        kmeans_labels = kmeans.fit_predict(df[['TOTAL', 'AGE_GROUP']])

        # 將 KMeans 分群後的標籤加入 DataFrame
        df['TOTAL_CLUSTER'] = kmeans_labels

        # 視覺化分組結果
        plt.scatter(df['AGE_GROUP'], df['TOTAL'], c=df['TOTAL_CLUSTER'], cmap='viridis')
        plt.xlabel('AGE_GROUP')
        plt.ylabel('TOTAL')
        plt.title('KMeans Clustering')
        plt.colorbar(label='Cluster')
        plt.show()

        # 重設索引以確保資料合併時索引的一致性
        df.reset_index(drop=True, inplace=True)
        kmeans_labels_df = pd.DataFrame({'TOTAL_CLUSTER_ORIG': kmeans_labels})

        # 合併資料
        df_with_labels = pd.concat([df, kmeans_labels_df], axis=1)

        X = df_with_labels[['TOTAL_CLUSTER_ORIG']]
        y = df['AGE_GROUP']
        # 資料切割成訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

        # 建立並訓練邏輯斯迴歸模型
        log_reg = LogisticRegression(C=1.0, solver='liblinear', max_iter=100, random_state=17)
        log_reg.fit(X_train, y_train)

        # 在測試集上進行預測
        y_pred = log_reg.predict(X_test)

        # 評估模型
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"模型準確度：{accuracy}")
        print(f"分類報告：\n{report}")

    def Linear_Net(self):
        df = self.day(name=1)
        # 保留 AGE_GROUP、PRODUCT_SUBCLASS 和 SALES_PRICE 這三列進行分析
        data = df[['AMOUNT','AGE_GROUP', 'PRODUCT_SUBCLASS', 'SALES_PRICE']]

        preprocessor = ColumnTransformer(
            transformers=[
                ('age', OneHotEncoder(), ['AGE_GROUP']),
                ('product', OneHotEncoder(), ['PRODUCT_SUBCLASS'])
            ],
            remainder='passthrough'
        )

        # 將資料分為訓練集和測試集
        X = data[['AMOUNT','AGE_GROUP', 'PRODUCT_SUBCLASS']]
        y = data['SALES_PRICE']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 初始化 Elastic Net 模型，alpha 是正則化項的強度，l1_ratio 是 L1 正則化的比例
        model = ElasticNet(alpha=0.01,l1_ratio=0.3)
        # 使用Pipeline串聯預處理和模型
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        # self.__find_best_params(pipeline,X_train,y_train)
        # 訓練模型
        pipeline.fit(X_train, y_train)

        # 預測測試集
        y_pred = pipeline.predict(X_test)
        # 視覺化預測結果及趨勢線
        plt.scatter(y_test, y_pred, color='black')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='blue', linewidth=2, label='趨勢線')
        plt.xlabel('實際銷售額')
        plt.ylabel('預測銷售額')
        plt.title('實際銷售額 vs. 預測銷售額_ElasticNet')
        plt.legend()
        plt.show()
        print(y_test) ; print(y_pred)
        # 計算評估指標
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(y_test.head(20).values) ; print(y_pred[0:20])
        # 輸出結果
        print('ElasticNet')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R²): {r2}')
        # 計算解釋變異量
        explained_variance = explained_variance_score(y_test, y_pred)
        print(f'Explained Variance: {explained_variance}')
        
    def Linear_Huber(self):
        df = self.day(name=1)
        # 保留 AGE_GROUP、PRODUCT_SUBCLASS 和 SALES_PRICE 這三列進行分析
        data = df[['AMOUNT','AGE_GROUP', 'PRODUCT_SUBCLASS', 'SALES_PRICE']]

        # 將 AGE_GROUP 轉換成數值型態，可以使用 get_dummies 或者自訂的映射函數
        # 將 PRODUCT_SUBCLASS 轉換成虛擬變數（dummy variables）
        preprocessor = ColumnTransformer(
            transformers=[
                ('age', OneHotEncoder(), ['AGE_GROUP']),
                ('product', OneHotEncoder(), ['PRODUCT_SUBCLASS'])
            ],
            remainder='passthrough'
        )

        # 將資料分為訓練集和測試集
        X = data[['AMOUNT','AGE_GROUP', 'PRODUCT_SUBCLASS']]
        y = data['SALES_PRICE']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # self.__find_best_params(X_train,y_train)
        # 初始化 Elastic Net 模型，alpha 是正則化項的強度，l1_ratio 是 L1 正則化的比例
        model = HuberRegressor(epsilon=1.4)  # 試著調整 alpha 和 l1_ratio 的值

        # 使用Pipeline串聯預處理和模型
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        # self.__find_best_params(pipeline,X_train,y_train)
        # 訓練模型
        pipeline.fit(X_train, y_train)

        # 預測測試集
        y_pred = pipeline.predict(X_test)
        # 視覺化預測結果及趨勢線
        plt.scatter(y_test, y_pred, color='black')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='blue', linewidth=2, label='趨勢線')
        plt.xlabel('實際銷售額')
        plt.ylabel('預測銷售額')
        plt.title('實際銷售額 vs. 預測銷售額_Huber')
        plt.legend()
        plt.show()
        # 計算評估指標
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(y_pred[0:10])

        # 輸出結果
        print('Huber')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R²): {r2}')
        # 計算解釋變異量
        explained_variance = explained_variance_score(y_test, y_pred)
        print(f'Explained Variance: {explained_variance}')

    def Linear_Simple(self):
        """
        簡單線性回歸
        """
        df = self.day(name=1)
        # 將時間轉換為日期格式
        # 保留 AGE_GROUP、PRODUCT_SUBCLASS 和 SALES_PRICE 這三列進行分析
        data = df[['AMOUNT','AGE_GROUP', 'PRODUCT_SUBCLASS', 'SALES_PRICE']]

        # 將 AGE_GROUP 轉換成數值型態，可以使用 get_dummies 或者自訂的映射函數
        # 將 PRODUCT_SUBCLASS 轉換成虛擬變數（dummy variables）
        preprocessor = ColumnTransformer(
            transformers=[
                ('age', OneHotEncoder(), ['AGE_GROUP']),
                ('product', OneHotEncoder(), ['PRODUCT_SUBCLASS'])
            ],
            remainder='passthrough'
        )

        # 將資料分為訓練集和測試集
        X = data[['AMOUNT','AGE_GROUP', 'PRODUCT_SUBCLASS']]
        y = data['SALES_PRICE']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 初始化線性回歸模型
        model = LinearRegression()

        # 使用Pipeline串聯預處理和模型
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # 訓練模型
        pipeline.fit(X_train, y_train)

        # 預測測試集
        y_pred = pipeline.predict(X_test)
        print(y_pred[0:20])
        # 視覺化預測結果及趨勢線
        plt.scatter(y_test, y_pred, color='black')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='blue', linewidth=2, label='趨勢線')
        plt.xlabel('實際銷售額')
        plt.ylabel('預測銷售額')
        plt.title('實際銷售額 vs. 預測銷售額_Linear')
        plt.legend()
        plt.show()

        # 計算MSE、MAE和R²
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 輸出結果
        print('-'*50) ; print('Simple')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R²): {r2}')
        explained_variance = explained_variance_score(y_test, y_pred)
        print(f'Explained Variance: {explained_variance}')
        print('-'*50)

        """
        均方誤差（Mean Squared Error，MSE）：

        MSE 是預測值和實際值之間的平方誤差的平均值。
        MSE 的數值越小，表示模型的預測越接近實際值。在這個例子中，MSE 為 7852545.97，這表示預測值和實際值的平方誤差的平均值為約 7852545.97。這個值的大小取決於銷售額的單位。
        平均絕對誤差（Mean Absolute Error，MAE）：

        MAE 是預測值和實際值之間絕對誤差的平均值。
        MAE 的數值越小，表示模型的預測越接近實際值。在這個例子中，MAE 為 1558.03，這表示預測值和實際值的絕對誤差的平均值為約 1558.03。這個值的大小取決於銷售額的單位。
        決定係數（R²）：

        R² 表示模型解釋目標變數方差的比例。R² 的範圍在0到1之間。
        R² 越接近1，表示模型能夠更好地解釋目標變數的變異性。在這個例子中，R² 為 0.7768，這表示模型能夠解釋目標變數銷售額變異性的約 77.68%。
        """

    def __find_best_params(self,pipeline,X_train,y_train):
        """找尋model最佳參數"""
        # 定義超參數範圍-Lesso
        # param_grid = {
        #     'model__alpha': [0.001, 0.01, 0.1, 1.0],
        #     'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        # }
        # 定義超參數範圍-Huber
        param_grid = {'model__epsilon': [1.1, 1.2, 1.3, 1.35, 1.4]}
        # 定義評估指標（explained variance）
        scorer = make_scorer(explained_variance_score)

        # 使用GridSearchCV進行超參數搜索
        grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=5)
        grid_search.fit(X_train, y_train)

        # 找到最佳超參數組合
        best_params = grid_search.best_params_
        print(best_params)

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
        # print(res.value_counts().to_excel("12345.xlsx"))

            # res.to_excel("1234.xlsx")
        # tmp
        # res['Frequency'] = F;res['Monetary'] = M
        # res.to_excel("123.xlsx")
        return res
    def RFM_cluster_pic(self):
        """利用ＲＦＭ分數處理出的分群圖"""
        excel_file_path = 'RFM族群加總.xlsx'

        # 讀取檔案
        df = pd.read_excel(excel_file_path)

        rfm_data = df[['Recency', 'Frequency', 'Monetary']]

        # 計算 RFM 分數，這裡假設已經有了 'RFMScore' 這一欄
        # 根據需要修改計算 RFM 分數的方式
        rfm_data['RFMScore'] = (rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']) / 15

        # 定義分群邏輯
        def rfm_grouping(score):
            if score >= 0.8: 
                return '頂級忠誠客戶'
            elif score >= 0.6:  
                return '忠誠客戶'
            elif score >= 0.4:  
                return '潛在忠誠客戶'
            else:  
                return '有錢途潛力的客戶'

        # RFMGroup 欄位
        rfm_data['RFMGroup'] = rfm_data['RFMScore'].apply(rfm_grouping)

        # 選取資料
        group1 = rfm_data[rfm_data['RFMGroup'] == '頂級忠誠客戶']
        group2 = rfm_data[rfm_data['RFMGroup'] == '忠誠客戶']
        group3 = rfm_data[rfm_data['RFMGroup'] == '潛在忠誠客戶']
        group4 = rfm_data[rfm_data['RFMGroup'] == '有錢途潛力的客戶']

        # 3D 散點圖 
        #RFM基本資料及人口樣態
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 繪製散點圖
        ax.scatter3D(group1['Recency'], group1['Frequency'], group1['Monetary'], label='頂級忠誠客戶', c='red', s=50, alpha=0.8)
        ax.scatter3D(group2['Recency'], group2['Frequency'], group2['Monetary'], label='忠誠客戶', c='green', s=50, alpha=0.8)
        ax.scatter3D(group3['Recency'], group3['Frequency'], group3['Monetary'], label='潛在忠誠客戶', c='blue', s=50, alpha=0.8)
        ax.scatter3D(group4['Recency'], group4['Frequency'], group4['Monetary'], label='有錢途潛力的客戶', c='purple', s=50, alpha=0.8)

        # 分成四大群

        # 頂級忠誠客戶：
        # 特徵：最近購買（Recency）、高頻率購買（Frequency）、高金額購買（Monetary）。
        # 行銷策略：提供獨家優惠、早期存取新產品、定期回饋感謝，保持高品質的客戶服務。

        # 忠誠客戶：
        # 特徵：較近期購買（Recency）、中等頻率購買（Frequency）、中等金額購買（Monetary）。
        # 行銷策略：推薦相關產品、提供忠誠計畫獎勵、優先存取新功能或產品。

        # 潛在忠誠客戶：
        # 特徵：相對近期購買（Recency）、低頻率購買（Frequency）、低金額購買（Monetary）。
        # 行銷策略：特價促銷、新客戶優惠、增加產品知名度。

        # 有錢途潛力的客戶：
        # 特徵：較久未購買（Recency）、低頻率購買（Frequency）、高金額購買（Monetary）。
        # 行銷策略：重新吸引注意、提供獨家新客戶禮遇、推薦高價值產品。

        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('RFM模型3D散點圖')
        ax.legend()

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

    def show(self):
        """呈現結果"""
        print("線性")
        self.Linear_Net() ; self.Linear_Simple() ; self.Linear_Huber()
        print('羅吉斯')
        self.Logistic()
        print('分群')
        self.Cluster()
        self.Customer_A()
        print('行銷方案_產品')
        self.Recommand()
        print('ＲＦＭ分群圖整理')
        self.RFM_cluster_pic()

    def Customer_A(self):
        """
        先找出四群代表的RFM分數，將前三名購買產品進行輸出
        頂級忠誠客戶 - 544 -> 3666人
        忠誠客戶 - 334 -> 873人
        淺在忠誠客戶 - 311 -> 705人
        有錢讀淺力客戶 - 214 -> 285人
        """
        data = self.__RFM_data()
        print(data)
        R = [5,3,3,2]
        F = [4,3,1,1]
        M = [4,4,1,4]
        res = []
        for i in range(4):
            print("R:{},F:{},M:{}".format(R[i],F[i],M[i]))
            tmp = data[
                (data['Recency'] == R[i])
                & (data['Frequency'] == F[i])
                & (data['Monetary'] == M[i])
               ]
            res.append(tmp.index.to_list())
        d_r = self.read()
        print(d_r)
        for i in range(4):
            tmp = d_r[d_r['CUSTOMER_ID'].isin(res[i])]
            print(tmp['PRODUCT_ID'].value_counts(sort=True).head(10))

    def Recommand(self):
        """行銷方案-產品4711271000014、4714981010038 """
        data = self.read()
        p = [4711271000014,4714981010038]
        for i in p :
            tmp = data[data["PRODUCT_ID"] == i]
            # print(tmp)
            plt.figure(figsize=(12,8))
            plt.title("產品編號{}的價格分布".format(i))
            plt.scatter(tmp["TRANSACTION_DT"],tmp["SALES_PRICE"],color="black")
            plt.xlabel('時間');plt.ylabel('銷售金額')
            plt.show()
            tmp_1 = tmp['SALES_PRICE'].value_counts().sort_index()
            tmp_2 = tmp['SALES_PRICE'].value_counts().sort_values()
            print(tmp_1);print(tmp_2)
            print("最高值:{}".format(max(tmp_1.index.to_list())))
            print("平均值:{}".format(sum(tmp_1.index.to_list())/len(tmp_1.index.to_list())))
            print("最低值:{}".format(min(tmp_1.index.to_list())))
            tmp = tmp_1.reset_index()
            # 購買數量趨勢
            model = LinearRegression()
            X = tmp['SALES_PRICE'].values
            y = tmp['count'].values
            X_reshaped = X.reshape(-1, 1)

            model.fit(X_reshaped, y)
            # 繪製數據點
            plt.scatter(X, y, color='blue', label='購買點')

            # 繪製趨勢線
            plt.plot(X, model.predict(X_reshaped), color='red', linewidth=2, label='趨勢線')

            # 添加標題和標籤
            plt.title(f'產品{i}的數量與價格趨勢')
            plt.xlabel('價格')
            plt.ylabel('數量')
            plt.grid(axis='both')

            # 顯示圖例
            plt.legend()

            # 顯示圖表
            plt.show()
            print("模型係數（斜率）:", model.coef_)
            print("模型截距:", model.intercept_)
    def RFM_data(self):
        return self.__RFM_data()
# Model().show() 