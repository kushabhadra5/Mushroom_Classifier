import pandas as pd
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from mushroom.exception import Mushroom_Exception

class ColumnEncoder:

def column_encoder(self, df:pd.DataFrame)->Optional[pd.DataFrame]:
        try:
            label_encoder = LabelEncoder()
            copy_dataframe = df.copy()

            for colname,col in df.items():
                copy_dataframe[colname] = label_encoder.fit_transform(col)
            
            return copy_dataframe

        except Exception as e:
            raise Mushroom_Exception(e,sys)