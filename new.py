from data import Data ; from model import Model
class New(Data):
    
    def __init__(self):
        self.data = super().read()
    
    def Product(self):
        """依照RFM分群找尋四族群的產品關聯規則"""
        
        Origin_data = self.data
        RFM_data = Model().RFM_data()
        RFM
        print(RFM_data)
    
New().Product()
    