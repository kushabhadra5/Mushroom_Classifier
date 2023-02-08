#Import required dependencies and packages
import pymongo
import pandas as pd
import json
from dotenv import load_dotenv


#Configure the data:
DATA_FILE_PATH = "/config/workspace/mushrooms.csv"
DATABASE_NAME = 'Mushroom'
COLLECTION_NAME = 'Mushroom_Data'
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #insert converted json record to mongo db
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

    print("\nData successfully uploaded to MongoDB.")