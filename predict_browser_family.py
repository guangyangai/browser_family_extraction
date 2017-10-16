import os, argparse, logging, csv
import pandas as pd
import numpy as np
from abc import ABCMeta
from sklearn.pipeline import Pipeline
from sklearn.model_selection  import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

DATA_PATH = "datasets"
RESULTS_PATH = "results"
NULL_VAL = "None"
OCCURRENCE_THRES = 2
SEPARATOR = '\t'
NGRAM_RANGE = (2, 2)
SVM_ALPHA_RANGE = (1e-2, 1e-3)

fh = logging.FileHandler('logs/app.log')
logger = logging.getLogger(__name__)
logger.addHandler(fh)

class AgentPredictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, training_file_path, test_file_path, pred_results_file_path):
        self.training_file_path_ = training_file_path
        self.test_file_path_ = test_file_path
        if not os.path.isdir(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)
        self.pred_results_file_path_ = os.path.join(RESULTS_PATH, pred_results_file_path)

    @property
    def header_row(self):
        return ["Agent", "AgentFamily", "Version"]

    @property
    def prediction_cols(self):
        return ["AgentFamily", "Version"]

    @property
    def feature_cols(self):
        return ["Agent"]

    def load_data(self, file_path, data_path = DATA_PATH):
        file_path = os.path.join(data_path, file_path)
        return pd.read_csv(file_path, sep = SEPARATOR, names = self.header_row)

    def filter_dataset(self,raw_dataset, column_name):
        """filter records with only one occurrence for sampling"""
        logger.info("{} records before filtering".format(len(raw_dataset)))
        filter_null_value = raw_dataset.loc[raw_dataset[column_name] != NULL_VAL]
        logger.info("{} records after excluding null values in column {}".format(len(filter_null_value),column_name))
        value_counts_dict = filter_null_value[column_name].value_counts().to_dict()
        records_w_few_occurrence = [k for k in value_counts_dict if value_counts_dict[k] < OCCURRENCE_THRES]
        logger.info("In column {}, {} has less occurrences than {} and will be dropped."
                    .format(column_name, records_w_few_occurrence, OCCURRENCE_THRES))
        filterd_dataset = filter_null_value.loc[~filter_null_value[column_name].isin(records_w_few_occurrence)]
        logger.info("{} records after excluding records with few occurrences in column {}"
                    .format(len(filter_null_value), column_name))
        return filterd_dataset

    def split_dataset(self, raw_dataset, column_name, n_splits=1, test_size=0.2, random_state=42):
        """split training data into test set and training set"""
        split = StratifiedShuffleSplit(n_splits, test_size, random_state)
        filterd_dataset = self.filter_dataset(raw_dataset, column_name)
        for train_index, test_index in split.split(filterd_dataset, filterd_dataset[column_name]):
            strat_train_set = filterd_dataset.loc[train_index]
            strat_test_set = filterd_dataset.loc[test_index]
        logger.info('split into {} training set and {} test set'.format(len(strat_train_set),len(strat_test_set)))
        return strat_train_set, strat_test_set


    def train_classifier(self, strat_train_set, pred_column):
        """Train a SVM classifier"""
        cls = Pipeline([
            ("vect", CountVectorizer(ngram_range=NGRAM_RANGE)),
            ("tfidf", TfidfTransformer()),
            ("clf-svm", SGDClassifier(loss='hinge', max_iter=5, random_state=42)),
        ])

        parameters_svm = {
            'clf-svm__alpha': SVM_ALPHA_RANGE,
        }

        gs_clf = GridSearchCV(cls, parameters_svm, n_jobs=-1)
        clf= gs_clf.fit(strat_train_set[self.feature_cols].values.astype('U'),
                                          strat_train_set[pred_column].values.astype('U'))

        logger.info('Best score within the search is {}'.format(clf.best_score_))
        logger.info('Best found parameters are {}'.format(clf.best_params_))

        return clf

    def eval_performance(self, strat_test_set, clf, pred_column):
        predicted_svm_test = clf.predict(strat_test_set[pred_column].values.astype('U'))
        test_score = np.mean(predicted_svm_test == strat_test_set[pred_column])
        logger.info('score on test set data within training data is {} for {}'.format(test_score, pred_column))

    def predict_column(self, test_dataset, clf, pred_column):
        predicted_svm_test = clf.predict(test_dataset[pred_column].values.astype('U'))
        test_score = np.mean(predicted_svm_test == test_dataset[pred_column])
        logger.info('score on new test data is {} for {}'.format(test_score, pred_column))
        return predicted_svm_test

    def form_results(self, pred_results, test_data):
        results_reorg = []
        for idx in range(len(test_data)):
            row = {}
            for col in self.feature_cols:
                row[col] = test_data[col][idx]
            for col in self.prediction_cols:
                row[col] = pred_results[col][idx]
            results_reorg.append(row)
        return results_reorg

    def write_to_file(self, pred_results, test_data):
        with open(self.pred_results_file_path_, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.header_row)
            writer.writeheader()
            rows = self.form_results(pred_results, test_data)
            for row in rows:
                writer.writerow(row)

    def run(self):
        """main function"""
        results = {}
        training_data = self.load_data(self.training_file_path_)
        test_data = self.load_data(self.test_file_path_)
        for col in self.prediction_cols:
            strat_train_set, strat_test_set = self.split_dataset(training_data, col)
            logger.info('Begin training for col {}'.format(col))
            clf = self.train_classifier(strat_train_set, col)
            self.eval_performance(strat_test_set, clf, col)
            predicted_col_test = self.predict_column(test_data, clf, col)
            results[col] = predicted_col_test
        self.write_to_file(results, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify settings for the prediction')
    parser.add_argument('--training', dest='training_file_path', required=True,
                        help='File path of the training file')
    parser.add_argument('--test', dest='test_file_path', required=True,
                        help='File path of the test file')
    parser.add_argument('--prediction-results', dest='pred_results_file_path', required=True,
                        help='File path saving the results')

    args = parser.parse_args()
    training_file_path = args.training_file_path
    test_file_path = args.test_file_path
    pred_results_file_path = args.pred_results_file_path
    agent_predictor = AgentPredictor(AgentPredictor,test_file_path,pred_results_file_path)
    agent_predictor.run()


