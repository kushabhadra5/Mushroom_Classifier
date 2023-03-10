from mushroom.entity import config_entity, artifact_entity
from mushroom.predictor import ModelResolver
from mushroom.exception import Mushroom_Exception
from mushroom.logger import logging
from mushroom.utils import load_object
from mushroom.utils import column_encoder

from sklearn.metrics import accuracy_score
import pandas  as pd
import sys,os
import pickle
from mushroom.config import TARGET_COLUMN

class ModelEvaluation:

    def __init__(self, model_eval_config:config_entity.ModelEvaluationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                    model_trainer_artifact:artifact_entity.ModelTrainerArtifact
                    ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise Mushroom_Exception(e,sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #This will compare the latest model present in folder saved_models and the trained model.
            logging.info("if folder name 'saved_model' has model then compare the base_model and trained model and return the best trained model")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            
            #Condition if no model is present.
            #Generally this happens in initial stage of the model.
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            #Finding location of latest transformer_path, model_path and target_encoder
            logging.info("Locating following objects: transformer, model and target_encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info("Loading previously trained objects: transformer, model and target encoder")
            #Previous trained  objects
            transformer = load_object(file_path=transformer_path)
            logging.info(f"Objects in Transformer file: {transformer}")
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            logging.info("Loading currently trained model objects: transformer, model and target encoder")
            #Currently trained model objects
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path) 
            #test_df = column_encoder(df=test_df.drop(columns='class',axis=1))
            #logging.info(f"Following are the first 5 records: {test_df[0:5]}")
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)


            # accuracy using previously trained model
            input_feature_name = list(transformer.feature_names_in_)
            input_test_df = column_encoder(df=test_df[input_feature_name])
            input_arr =transformer.transform(input_test_df)
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            logging.info(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = accuracy_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")
           
            # accuracy using current trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_test_df = column_encoder(df=test_df[input_feature_name])
            input_arr =current_transformer.transform(input_test_df)
            y_pred = current_model.predict(input_arr)
            y_true =current_target_encoder.transform(target_df)
            print(f"Prediction using trained model: {current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = accuracy_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
            
        except Exception as e:
            raise Mushroom_Exception(e,sys)