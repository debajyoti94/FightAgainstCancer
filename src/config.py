''' Here all the configuration variables are kept
which is used throughout this project'''

# dataset information
OUTPUT_FEATURE = 1
CATEGORICAL_VARIABLES = [1]

ORIGINAL_DATASET_FILENAME = "../input/wdbc.data"

RAW_TRAIN_FILENAME = "../input/train_set.pickle"
RAW_TEST_FILENAME = "../input/test_set.pickle"

CLEAN_TRAIN_FILENAME = "../input/clean_train_set.pickle"
CLEAN_TEST_FILENAME = "../input/clean_test_set.pickle"

# for plots
NULL_CHECK_HEATMAP = "../plots/null_check_heatmap.png"

# for cross validation
NUM_FOLDS = 5
KFOLD_COLUMN_NAME = "kfold"

# model name
BASELINE_MODEL_NAME = "../models/SVM_wdbc_Baseline_"
GRIDCV_MODEL_NAME = "../models/SVM_wdbc_GRIDCV_"
BEST_MODEL = "../models/SVM_wdbc_Baseline_0.pickle"

FILE_DELIMITER = ","
ENCODING_TYPE = "UTF-8"
DATASET_SHAPE = (569, 32)
