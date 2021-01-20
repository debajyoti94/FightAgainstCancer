''' In this code we will write code for training the model'''

import config
import feature_engg
import create_folds

import argparse
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run(dataset, fold):
    '''

    :param dataset: training data
    :param fold: for skfold
    :return: trained model
    '''

    # create train and valid splits
    train_set = dataset[dataset.kfold != fold]
    validation_set = dataset[dataset.kfold == fold]

    # get X train, y train
    X_train = train_set.drop([1, 'kfold'], axis=1, inplace=False).values
    y_train = train_set[1].values

    # get X valid, y valid
    X_valid = validation_set.drop([1, 'kfold'], axis=1, inplace=False).values
    y_valid = validation_set[1].values

    # model instantiation here
    model = SVC()

    # train here
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, preds)
    p,r,f1,support = precision_recall_fscore_support(y_valid, preds)

    print(
        "---Fold={}---\nAccuracy={}\nPrecision={}\nRecall={}\nF1={}".format(
            fold, accuracy, p, r, f1
        )
    )

    dl_obj = feature_engg.DumpLoadFile()
    dl_obj.dump_file(model, str(config.MODEL_NAME) + str(fold) + ".pickle")

    # return here


def inference_stage(dataset, model):
    '''
    Run model on test set and get performance metrics
    :param dataset: test set
    :param model: the model
    :return:
    '''
    print(dataset.shape)
    X_test = dataset.drop([1], axis=1, inplace=False).values
    y_test = dataset[[1]].values

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    p,r,f1,support = precision_recall_fscore_support(y_test, preds)

    print(
        "\nAccuracy={}\nPrecision={}\nRecall={}\nF1={}".format(
           accuracy, p, r, f1
        )
    )
    return

if __name__ == "__main__":

    # parsing the arguments
    parser = argparse.ArgumentParser()

    # adding the arguments
    parser.add_argument('--clean', type=str,
                        help='Provide argument \"--clean dataset\" to get clean train and test split.'
    )

    parser.add_argument('--train', type=str,
                        help='Provide argument \"--train skfold\" to train the model using Stratified'
                             ' Kfold cross validation'
                        )

    parser.add_argument('--test', type=str,
                        help='Provide argument \"--test inference\" to test the model and'
                             ' obtain performance metrics.'
                        )

    args = parser.parse_args()
    dl_obj = feature_engg.DumpLoadFile()

    # code when argument is cleaning data
    if args.clean == 'dataset':

        fr_obj = feature_engg.FeatureEngineering()

        # get the train data
        raw_train_set = dl_obj.load_file(config.RAW_TRAIN_FILENAME)
        # call the cleaning data function from feature_engg
        clean_train_set = fr_obj.cleaning_data(raw_train_set)
        dl_obj.dump_file(clean_train_set, config.CLEAN_TRAIN_FILENAME)

        # get the test data
        raw_test_set = dl_obj.load_file(config.RAW_TEST_FILENAME)
        # call the cleaning data function from feature_engg
        clean_test_set = fr_obj.cleaning_data(raw_test_set)
        dl_obj.dump_file(clean_test_set, config.CLEAN_TEST_FILENAME)

    # code when training the model
    elif args.train == 'skfold':

        # get the train data
        if os.path.isfile(config.CLEAN_TRAIN_FILENAME):
            clean_train_set = dl_obj.load_file(config.CLEAN_TRAIN_FILENAME)

            # for stratified k fold cross validation
            clean_train_set['kfold'] = -1
            skfold_obj = create_folds.SKFolds()
            clean_train_set = skfold_obj.create_folds(clean_train_set)

            # train the model
            for fold in range(config.NUM_FOLDS):
                run(clean_train_set, fold)

        else:
            print(
                "Training set does not exist. Please obtain the train set first.\n"
                'Use "python train.py --clean dataset" to get the train and test set.'
            )

        # call run function for each fold

    # code when testing the model
    elif args.test == 'inference':

        # get the test data
        if os.path.isfile(config.CLEAN_TEST_FILENAME):
            clean_test_set = dl_obj.load_file(config.CLEAN_TEST_FILENAME)
            # get the model
            model = dl_obj.load_file(config.BEST_MODEL)

            # call the inference stage function
            inference_stage(clean_test_set, model)

        else:
            print(
                "Test set does not exist. Please obtain the Test set first.\n"
                'Use "python train.py --clean dataset" to get the train and test set.'
            )
        # and get test results