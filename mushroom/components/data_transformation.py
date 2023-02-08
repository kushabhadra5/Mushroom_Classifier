from mushroom.entity import artifact_entity,config_entity
from mushroom.exception import Mushroom_Exception
from mushroom.logger import logging
from mushroom import utils
from mushroom.config import TARGET_COLUMN

from typing import Optional
import os,sys 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class DataTransformation:

    def __init__(self, data_transformation_config : config_entity.DataTransformationConfig,
                        data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise Mushroom_Exception(e, sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
            logging.info('Applying simple_imputer and standard scalar')
            scaler =  StandardScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('StandardScaler',scaler)
                ])
            return pipeline
        except Exception as e:
            raise Mushroom_Exception(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #Loading training and testing file
            logging.info("Loading train dataframe from ingestion_artifact")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Loading test dataframe from ingestion_artifact")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Loading input feature from train and test dataframe
            logging.info("Loading input feature from train dataframe")
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            logging.info("Loading input feature from test dataframe")
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #Loading target feature from train and test dataframe
            logging.info("Loading target feature from train dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info("Loading target feature from test dataframe")
            target_feature_test_df = test_df[TARGET_COLUMN]


            #Label encoding input_feature_train dataframe 
            logging.info("Label encoding input_feature_train dataframe")
            encoded_input_feature_train_df = utils.column_encoder(input_feature_train_df)
            #Label encoding input_feature_test dataframe 
            logging.info("Label encoding input_feature_test dataframe")
            encoded_input_feature_test_df = utils.column_encoder(input_feature_test_df)

            
            #Simple imputer
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(encoded_input_feature_train_df)
            logging.info(f"Features used for the transformation: {[feature for feature in transformation_pipleine.feature_names_in_]}")
            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(encoded_input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(encoded_input_feature_test_df)
            

            #Label encoder
            Label_Encoder = LabelEncoder()
            Label_Encoder.fit(target_feature_train_df)
            #Label encoding target_feature_train dataframe
            logging.info("Label encoding target_feature_train dataframe")
            target_feature_train_arr = Label_Encoder.transform(target_feature_train_df)
            #Label encoding target_feature_train dataframe
            logging.info("Label encoding target_feature_test dataframe")
            target_feature_test_arr = Label_Encoder.transform(target_feature_test_df)

            '''#Standard Scaling the data
            logging.info("Creating instance of Standard Scalar")
            standard_scalar = StandardScaler()
            #Fitting and transforming the train data
            logging.info("Fitting and transforming the train data using Standard Scalar")
            standard_scalar.fit_transform(x_train)
            #Fitting transforming the test model
            logging.info("Fitting and transforming the test data using Standard Scalar")
            standard_scalar.transform(x_test)'''

            #Concatenating input_feature_train_arr and target_feature_train_arr as train_arr.
            train_arr = np.c_[encoded_input_feature_train_df, target_feature_train_arr ]
            #Concatenating input_feature_test_arr and target_feature_test_arr as test_arr.
            test_arr = np.c_[encoded_input_feature_test_df, target_feature_test_arr]

            #Saving it as numpy arrays:
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=Label_Encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            
            return data_transformation_artifact
        
        except Exception as e:
            raise Mushroom_Exception(e, sys)



