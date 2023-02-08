from mushroom.entity import artifact_entity,config_entity
from mushroom.exception import Mushroom_Exception
from mushroom.logger import logging
from typing import Optional
import os,sys 
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from mushroom import utils
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Training {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise Mushroom_Exception(e,sys)

    
    def train_model(self,x,y):
        try:
            logging.info("Creating Instance of XG Boost Classifier")
            xgb_clf =  XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf

        except Exception as e:
            raise Mushroom_Exception(e,sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            '''#Standard Scaling the data
            logging.info("Creating instance of Standard Scalar")
            standard_scalar = StandardScaler()
            #Fitting and transforming the train data
            logging.info("Fitting and transforming the train data using Standard Scalar")
            standard_scalar.fit_transform(x_train)
            #Fitting transforming the test model
            logging.info("Fitting and transforming the test data using Standard Scalar")
            standard_scalar.transform(x_test)'''

            logging.info(f"Training the model using XG Boost")
            model = self.train_model(x=x_train,y=y_train)
            logging.info(f"Using XG Boost model is ready")

            logging.info(f"Applying model on train dataframe")
            yhat_train = model.predict(x_train)
            logging.info(f"Calculating f1 score of train dataframe")
            f1_train_score  =f1_score(y_true=y_train, y_pred=yhat_train)
            accuracy_train_score = accuracy_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Applying model on test dataframe")
            yhat_test = model.predict(x_test)
            logging.info(f"Calculating f1 score of train dataframe")
            f1_test_score  =f1_score(y_true=y_test, y_pred=yhat_test)
            accuracy_test_score = accuracy_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"f1 score|| train data: {f1_train_score} | test data: {f1_test_score}")
            logging.info(f"Accuracy score|| train data: {accuracy_train_score} | test data: {accuracy_test_score}")
            
            
            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score} | model actual score: {f1_test_score}")
            
            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving the trained model as an object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #Preparing artifact:
            logging.info(f"Preparing artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
                                                                            f1_train_score=f1_train_score, 
                                                                            f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact prepared: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise Mushroom_Exception(e,sys)
        