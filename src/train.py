''' In this code we will write code for training the model'''

import config
import feature_engg
import create_folds
import argparse

def run(dataset, fold):
    '''

    :param dataset: training data
    :param fold: for skfold
    :return: trained model
    '''
    return

def inference_stage(dataset, model):
    '''
    Run model on test set and get performance metrics
    :param dataset: test set
    :param model: the model
    :return:
    '''

    return

if __name__ == "__main__":

    # parsing the arguments
    parser = argparse.ArgumentParser()

    # adding the arguments
    parser.add_argument('--clean', typ=str,
                        help='Provide argument "--clean dataset" to get'
             " clean train and test split."
    )

    parser.add_argument('--train', typ=str,
                        help='Provide argument "--train skfold" to train the model using Stratified'
                             ' Kfold cross validation'
                        )

    parser.add_argument('--test', typ=str,
                        help='Provide argument "--test inference" to test the model and'
                             ' obtain performance metrics.'
                        )

    # code when argument is cleaning data

    # code when training the model

    # code when testing the model