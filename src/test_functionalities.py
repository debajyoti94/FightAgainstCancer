'''In this code we will create unit tests for checking data sanity and also
unit tests for the functions that we create in this project.'''

import config
import feature_engg

import csv
import os
import pandas as pd

class TestFunctionalities:

    def test_null_checks(self):
        '''
        column null checks
        :return: True if all is okay
        '''
        dl_obj = feature_engg.DumpLoadFile()
        train_set = dl_obj.load_file(config.CLEAN_TRAIN_FILENAME)
        test_set = dl_obj.load_file(config.CLEAN_TEST_FILENAME)

        assert False if  False in pd.isnull(train_set) or \
                       False in pd.isnull(test_set) \
                       else True

    def test_data_shape_check(self):
        '''
        check if the shape of the original data matches
         with what is expected
        :return: True if all is okay
        '''
        dataset = pd.read_csv(config.ORIGINAL_DATASET_FILENAME)
        assert True if dataset.shape == config.DATASET_SHAPE else False


    def test_train_data_check(self):
        '''
        check if the train and test data exists
        :return: True if all is okay
        '''
        assert True if os.path.isfile(config.CLEAN_TRAIN_FILENAME) \
                       and os.path.isfile(config.CLEAN_TEST_FILENAME)\
                       else False

    def test_delimiter_check(self):
        '''
        check if the delimiter of the file matches
         with what is expected
        :return: True if all is okay
        '''
        with open(config.ORIGINAL_DATASET_FILENAME, "r") as csv_file:
            file_contents = csv.Sniffer().sniff(csv_file.readline())

        assert True if file_contents.delimiter == config.FILE_DELIMITER \
                                                            else False
