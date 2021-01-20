''' In this code we will perform data cleaning operations'''

import abc
import pickle
import seaborn as sns
import pandas as pd
import sklearn.preprocessing as preproc

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

    def label_encoder(self, dataset, features_to_encode):
        '''
        For handling categorical features. In this case, for handling target feature.
        :param data:
        :param features_to_encode:
        :return: encoded feature
        '''
        le = preproc.LabelEncoder()
        encoded_feature = le.fit_transform(dataset[features_to_encode])

        return encoded_feature

    def scaling_data(self, dataset):
        '''
        Applying minmax scaling
        :param dataset: input data
        :return: scaled data
        '''
        for feature in dataset.columns:
            dataset[feature] = preproc.minmax_scale(
                dataset[[feature]]
            )
        return dataset

    def cleaning_data(self, dataset):
        '''
        overriding the cleaning data function from abc class
        :param dataset: dataset to be cleaned
        :return: cleaned data X, cleaned Y
        '''

        # apply label encoding to target class
        # 1 is the target class
        # print(type(dataset))
        encoded_features = self.label_encoder(dataset,[1])
        dataset[1] = encoded_features
        # print(type(dataset))
        # print(dataset[[0,1]])
        dataset.drop(['index',0], axis=1, inplace=True)
        # apply minmax scaling to remaining features
        scaled_dataset = self.scaling_data(dataset)


        return scaled_dataset



    def plot_null_values(self, dataset):
        '''
        Create a heatmap to verify if any null value exists
        :param dataset: dataset to be verified
        :return: heatmap plot
        '''
        sns_heatmap_plot = sns.heatmap(dataset.isnull(), cmap="Blues", yticklabels=False)
        sns_heatmap_plot.figure.savefig(config.NULL_CHECK_HEATMAP)


class DumpLoadFile:

    def load_file(self, filename):
        '''
        Pickled file that you want to load
        :param filename: consists of filename + path
        :return: unpickled file
        '''
        with open(filename, 'rb') as pickle_handle:
            return pickle.load(pickle_handle)

    def dump_file(self, file, filename):
        '''
        Create a pickled file
        :param file: file that you want to pickle
        :param filename: Filename for that pickled file
        :return: nothing
        '''
        with open(filename, 'wb') as pickle_handle:
            pickle.dump(file, pickle_handle)

