'''Here we write the code for stratified k fold cross validation'''

from sklearn import model_selection
import config


class SKFolds:

    def create_folds(self, dataset_df):
        '''
        In this code we will assign values to the column kfold
        and use that during training the model
        :param dataset_df:
        :return: dataset with kfold values
        '''

        kf = model_selection.StratifiedKFold(n_splits=config.NUM_FOLDS,
                                             shuffle=True, random_state=0)

        y = dataset_df[config.OUTPUT_FEATURE].values

        for fold_value,(t_,y_index) in enumerate(kf.split(X=dataset_df,y=y)):
            dataset_df.loc[y_index, config.KFOLD_COLUMN_NAME] = fold_value

        return dataset_df