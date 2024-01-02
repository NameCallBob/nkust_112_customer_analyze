import pandas as pd

class Data():

    def read(self) -> pd.DataFrame:
        """
        用於資料預處理，將資料進行特徵化及處理異常值、空值

        output:
        DataFrame
        """
        try:
            df = pd.read_csv("./ta_feng_all_months_merged.csv")
            # self.__check_null(df)
            # 型態改變
            # 將文字型態轉為數值、日期
            df['TRANSACTION_DT'] = pd.to_datetime(df['TRANSACTION_DT'])
            df['AMOUNT'] = pd.to_numeric(df["AMOUNT"],downcast="integer")
            df['ASSET'] = pd.to_numeric(df['ASSET'],downcast="integer")
            df['SALES_PRICE'] = pd.to_numeric(df['SALES_PRICE'],downcast="integer")
            df['TOTAL'] = df['AMOUNT'] * df['SALES_PRICE']
            # 空值處理 -> 補出現最多次AGE_GROUP
            df['AGE_GROUP'] = df['AGE_GROUP'].fillna("35-39")
            # 刪除重複值 -> 無重複值
            df.drop_duplicates()
            return df
        except ImportError:
            raise ImportError("你路徑打錯了，請確定或自己改路徑，預設=./ta_feng_all_months_merged.csv")
        except Exception as e:
            raise SystemError(f"資料預處理出現問題，{e}")


    def read_standardize(self) -> pd.DataFrame:
        """
        資料預處理後將資料進行標準化

        output:
        DataFrame
        """
        from sklearn.preprocessing import LabelEncoder,StandardScaler
        data = self.read()
        # 文字
        LE = LabelEncoder() ;
        data["AGE_GROUP"] = LE.fit_transform(data['AGE_GROUP'])
        data['PIN_CODE'] = LE.fit_transform(data['PIN_CODE'])
        data["PRODUCT_SUBCLASS"] = LE.fit_transform(data["PRODUCT_SUBCLASS"])
        # 數字暫時無處理
        return data
    #文獻：https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/preprocessing-data-%E6%95%B8%E6%93%9A%E7%89%B9%E5%BE%B5%E6%A8%99%E6%BA%96%E5%8C%96%E5%92%8C%E6%AD%B8%E4%B8%80%E5%8C%96-9bd3e5a8f2fc

    def read_s_l(self):
        """
        資料預處理後將資料進行標準化

        output:
        DataFrame
        """
        from sklearn.preprocessing import LabelEncoder,StandardScaler
        data = self.read()
        data = data[data["TOTAL"]< 2000000]
        # 文字
        LE = LabelEncoder() ;
        data["AGE_GROUP"] = LE.fit_transform(data['AGE_GROUP'])
        data['PIN_CODE'] = LE.fit_transform(data['PIN_CODE'])
        data["PRODUCT_SUBCLASS"] = LE.fit_transform(data["PRODUCT_SUBCLASS"])
        # 數值特徵的標準化
        numerical_features = ['AMOUNT', 'ASSET', 'SALES_PRICE', 'TOTAL']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        return data

    def __check_null(self,df:pd.DataFrame) -> pd.DataFrame :
        """
        輸出每欄位的null加總（count）
        """
        for i in df.columns:
            print(df[i].isnull().value_counts())
        "最後得出 AGE_GROUP 22362個空值，其他都有值"

    def getInfo(self):
        df = self.read()
        # 輸出欄位名稱
        print(df.columns.to_list())
        # 輸出資料大小
        print(df.shape)

    def test(self):

        print("資料清洗（未特徵化輸出）")
        df = self.read()
        print(df)

        print("資料清洗（特徵化）")
        df = self.read_standardize()
        print(df)

    def TOP5_product(self):
        """輸出前5名的產品類別總銷售額"""
        data = self.read()[['PRODUCT_SUBCLASS','AMOUNT','SALES_PRICE']]
        data = data.groupby("PRODUCT_SUBCLASS").sum().sort_values(by='SALES_PRICE',ascending=False).head(5)
        return(data.index.to_list())

    def day(self,name):
        """
        將資料已經以日期進行加總(前五名產品)

        params;

        name = 1 -> 有計算年齡的前五名收入、數量合計

        name = 2 -> 純時間的前五名收入、數量合計

        return;

        pd.DataFrame
        """
        data = self.read()
        data = data[data['PRODUCT_SUBCLASS'].isin(self.TOP5_product())]
        if name == 1:
            data = data.pivot_table(
                columns=["PRODUCT_SUBCLASS"],
                index = ["TRANSACTION_DT","AGE_GROUP"],
                values=["AMOUNT","SALES_PRICE"],
                aggfunc="sum",
                fill_value=0,
            ).stack().reset_index()
        elif name == 2:
            data = data.pivot_table(
                columns=["PRODUCT_SUBCLASS"],
                index = ["TRANSACTION_DT"],
                values=["AMOUNT","SALES_PRICE"],
                aggfunc="sum",
                fill_value=0,
            ).stack().reset_index()
        else:
            raise KeyError("你輸入錯參數，產生資料失敗！")
        # print(data[['TRANSACTION_DT']].values)
        # print(data)
        return(data)



# Data().test()
# Data().getInfo()
# Data().TOP5_product()
# Data().day(name="age")
