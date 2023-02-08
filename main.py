from mushroom.logger import logging
from mushroom.exception import Mushroom_Exception
from mushroom.utils import get_collection_as_dataframe
import sys,os
from mushroom.entity import config_entity
from mushroom.components.data_ingestion import DataIngestion
from mushroom.components.data_validation import DataValidation
from mushroom.components.data_transformation import DataTransformation
from mushroom.components.model_trainer import ModelTrainer
from mushroom.components.model_evaluation import ModelEvaluation
from mushroom.components.model_pusher import ModelPusher

if __name__=="__main__":
     try:
          #Initiated training pipeline:
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          
          #Data Ingestion component:
          #Preparing input object for Data Ingestion component
          data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
          print(data_ingestion.initiate_data_ingestion())
          #Preparing output object from Data Ingestion component
          data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


          #Data Validation component:
          #Preparing input object for Data Validation component
          data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
          data_validation = DataValidation(data_validation_config=data_validation_config,
                         data_ingestion_artifact=data_ingestion_artifact)
          #Preparing output object from Data Validation component
          data_validation_artifact = data_validation.initiate_data_validation()

          #Data Trasformation component:
          #Preparing input object for Data Trasformation component
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config = training_pipeline_config)
          data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                         data_ingestion_artifact=data_ingestion_artifact)
          ##Preparing output object from Data Trasformation component
          data_transformation_artifact = data_transformation.initiate_data_transformation()

          #Model Training component:
          #Preparing input object for Model Training component
          model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config = training_pipeline_config)
          model_trainer = ModelTrainer(model_trainer_config = model_trainer_config,
                                        data_transformation_artifact = data_transformation_artifact)
          ##Preparing output object from Model Training component
          model_trainer_artifact = model_trainer.initiate_model_trainer()

          #Model Evaluation component:
          #Preparing input object for Model Evaluation component
          model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
          model_eval = ModelEvaluation(model_eval_config = model_eval_config,
                                        data_ingestion_artifact = data_ingestion_artifact,
                                        data_transformation_artifact = data_transformation_artifact,
                                        model_trainer_artifact = model_trainer_artifact)
          ##Preparing output object from Model Evaluation component                             
          model_eval_artifact = model_eval.initiate_model_evaluation()

          #Model Pusher Component:
          #Preparing input object for Model Pusher component
          model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)

          model_pusher = ModelPusher(model_pusher_config = model_pusher_config,
                                        data_transformation_artifact = data_transformation_artifact,
                                        model_trainer_artifact = model_trainer_artifact)
          ##Preparing output object from Model Pusher component
          model_pusher_artifact = model_pusher.initiate_model_pusher()

     except Exception as e:
          print(e)