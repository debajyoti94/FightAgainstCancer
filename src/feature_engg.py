''' In this code we will perform data cleaning operations'''

import abc
import pickle
import seaborn as sns
import pandas as pd

# create an abc for feature engineering class
# doing this so that i do not forget the absolute necessary functions

class MustHaveForFeatureEngineering:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def cleaning_data(self):
        '''
        Apply feature engineering techniques and clean data
        :return: cleaned data
        '''
        return

    @abc.abstractmethod
    def plot_null_values(self):
        '''
        To check the if any null values exist in the dataset
        :return:
        '''
        return

class FeatureEngineering(MustHaveForFeatureEngineering):

    def cleaning_data(self, dataset):
        '''
        overriding the cleaning data function from abc class
        :param dataset: dataset to be cleaned
        :return: cleaned data
        '''
        return

    def plot_null_values(self, dataset):
        '''
        Create a heatmap to verify if any null value exists
        :param dataset: dataset to be verified
        :return: heatmap plot
        '''
        return

class LoadDumpFile:

    def load_file(self, filename):
        '''
        Pickled file that you want to load
        :param filename: consists of filename + path
        :return: unpickled file
        '''
        return

    def dump_file(self, file, filename):
        '''
        Create a pickled file
        :param file: file that you want to pickle
        :param filename: Filename for that pickled file
        :return: nothing
        '''
        return
