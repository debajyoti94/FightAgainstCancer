# Fight Against Cancer
***

Building a predictive model to identify which patient has malignant or benign cancerous cells. Using the [Wisconsin Diagnostic Breast Cancer (WDBC)](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from [UCI repository](https://archive.ics.uci.edu/ml/index.php).

The repository has multiple directories, with each serving a different purpose:
- input/: contains the raw dataset, split into train and test files.
- model/: consists of:
    - baseline models built using Support Vector Machines
    - improved models by using Grid Search to find the optimum set of hyper-parameters.
    Stratified Kfold validation was applied while training the model, with number of folds=5, hence you see 5 models. 
- notebooks/: Consists of one jupyter notebook. It was used for EDA purpose and also experiment with some functions used for feature engineering.
- src/: this directory consists of the source code for the project.
    - config.py: consists of variables which are used all across the code.
    - create_folds.py: used for implementing stratified kfold cross validation.
    - feature_engg.py: used for cleaning the dataset and applying feature engineering techniques.
    - test_functionalities: using pytest module, i define some data sanity checks on the training data.
    - train.py: this file contains the code for implementing the model. The train and the inference stage.

### To obtain clean data and split it into train and test set, use the following command:
  ```python train.py --clean dataset```
  
### To train the model without Grid Search, use following command:
  ```python train.py --train skfold```

### To train the model by using Grid Search, use following command:
  ```python train.py --train gridcv```
  
### For inference stage, use:
  ```python train.py --test inference```

### For more information use:
  ```python train.py --help```
