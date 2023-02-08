from mushroom import utils 
from mushroom.entity import config_entity
from mushroom.entity import artifact_entity
from mushroom.exception import Mushroom_Exception
from mushroom.logger import logging
import os,sys
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig ):
        """
        Description:
        This function is inheriting DataIngestionConfig class from config_entity to take input for data ingestion
        component of training pipeline.
        """
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Mushroom_Exception(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        """
        Description:
        Step 1: This function is creating DataFrame from the dataset and saving it in csv format.
        Step 2: Storing the DataFrame in artifact/time_stapm/data_ingestion/featurestore, if this directory is not present then the directory will be created.
        Step 3: Now from the created DataFrame train and test DataFrame is created using sklearn.model_selection.train_test_split function.
        Step 4: Then both train and test DataFrame are to be saved in the artifact/time_stapm/data_ingestion/dataset directory in csv format.
        Step 5: Preparing the artifact of data_ingestion component those are:
                feaure_store file path
                train_file_path
                test_file_path
        """
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)
            logging.info("Save data in feature_store.")

            #Create feature store folder if not available
            logging.info("Create feature store folder if not available.")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")

            #Saving DataFrame to feature_store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)

            #split dataset into train and test set    
            logging.info("split dataset into train and test set")
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size, random_state=42)
            
            #create dataset directory folder if not available
            logging.info("create dataset directory folder if not available")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            #Save df to feature store folder
            logging.info("Save df to feature store folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            
            #Prepare artifact
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise Mushroom_Exception(error_message=e, error_detail=sys)