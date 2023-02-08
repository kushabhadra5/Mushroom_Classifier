import os, sys
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import ks_2samp

from mushroom import utils
from mushroom.logger import logging
from mushroom.exception import Mushroom_Exception
from mushroom.entity import artifact_entity, config_entity

class DataValidation:
    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact):

        try: 
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error=dict()

        except Exception as e:
            raise Mushroom_Exception(e, sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_columns = []

            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available in current dataframe.]")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True
        except Exception as e:
            raise Mushroom_Exception(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        
        try:
            drift_report=dict()
            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                #Null hypothesis is that both column data drawn from same distrubtion

                logging.info(f"Hypothesis testing on column '{base_column}'")
                logging.info(f"Column: '{base_column}' || D.type in base dataframe:{base_data.dtype} | D.type in current dataframe:{current_data.dtype} ")
                same_distribution =ks_2samp(base_data,current_data)

                if same_distribution.pvalue>0.05:
                    #We are accepting null hypothesis
                    logging.info(f"In both the dataframe the column '{base_column}' is having similar distribution")
                    logging.info(f"No data drift detected in the column '{base_column}'")
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    #We will reject the null hypothesis
                    logging.info(f"In both the dataframe the column {base_column} is not having similar distribution")
                    logging.info(f"Data drift detected in the column '{base_column}'")
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
                    #different distribution

            self.validation_error[report_key_name]=drift_report

        except Exception as e:
            raise Mushroom_Exception(e, sys)


    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:

        try:
            #Loading base dataframe.
            logging.info("Loading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            #Loading train dataframe:
            logging.info("Loading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)

            #Loading test dataframe:
            logging.info("Loading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Validating columns in train dataframe
            logging.info(f"Is all required columns present in train df")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df,
                                                                        current_df=train_df,
                                                                        report_key_name="missing_columns_within_train_dataset")
            
            #Validating columns in train dataframe
            logging.info(f"Is all required columns present in test df")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, 
                                                                        current_df=test_df,
                                                                        report_key_name="missing_columns_within_test_dataset")

            #Looking for any type of data drift present in the dataframe
            if train_df_columns_status:
                logging.info(f"As all column are available in train df hence checking for data drift")
                self.data_drift(base_df=base_df, current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info(f"As all column are available in test df hence checking for data drift")
                self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_within_test_dataset")
            
            #Preparing the report in yaml file
            logging.info("Preparing the report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise Mushroom_Exception(e, sys)

            