from mushroom.predictor import ModelResolver
from mushroom.entity.config_entity import ModelPusherConfig
from mushroom.exception import Mushroom_Exception
import os,sys
from mushroom.utils import load_object,save_object
from mushroom.logger import logging
from mushroom.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact


class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
                    data_transformation_artifact:DataTransformationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            #load current objects
            logging.info(f"Loading following current objects: transformer, model and target_encoder")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            logging.info(f"Loaded following current objects: transformer, model and target_encoder")

            # Saving objects to model_pusher dir
            logging.info(f"Saving following objects in model_pusher directory: transformer, model and target_encoder into model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)
            logging.info(f"Saved following objects in model_pusher directory: transformer, model and target_encoder into model pusher directory")

            #saved model dir
            logging.info(f"Saving model in saved_model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()
            target_encoder_path=self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=target_encoder_path, obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
             saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise Mushroom_Exception(e, sys)