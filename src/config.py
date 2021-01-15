''' Here all the configuration variables are kept
which is used throughout this project'''

# dataset information
OUTPUT_FEATURE = 1
CATEGORICAL_VARIABLES = [1]
ORIGINAL_DATASET_FILENAME = "../input/wdbc.data"
TRAIN_FILENAME = "../input/train_set.pickle"
TEST_FILENAME = "../input/test_set.pickle"

# for plots
NULL_CHECK_HEATMAP = "../plots/null_check_heatmap.png"

# for cross validation
NUM_FOLDS = 5

# model name
MODEL_NAME = "../models/SVM_wdbc_Baseline_"
# BEST_MODEL = "../models/SVM_wdbc"

FILE_DELIMITER = ","
ENCODING_TYPE = "UTF-8"
DATASET_SHAPE = (569, 32)
