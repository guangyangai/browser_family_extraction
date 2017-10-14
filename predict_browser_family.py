import os,tarfile, argparse
import pandas as pd
from sklearn.model_selection  import StratifiedShuffleSplit


DATA_PATH = "datasets"


def load_data(data_path = DATA_PATH, data = training_file_path):
    file_path = os.path.join(data_path, data)
    return pd.read_csv(file_path, sep = '\t', names = ["Agent", "AgentFamily", "Version"])


def predict_agent_family(training_file_path, test_file_path, pred_results_file_path):
    agent = load_data(training_file_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify settings for the prediction')
    parser.add_argument('--training', dest='training_file_path', required=True,
                        help='Filepath of the training file')
    parser.add_argument('--test', dest='test_file_path', required=True,
                        help='Filepath of the test file')
    parser.add_argument('--prediction-results', dest='pred_results_file_path', required=False,
                        help='Filepath saving the results')

    args = parser.parse_args()
    training_file_path = args.training_file_path
    test_file_path = args.test_file_path
    pred_results_file_path = args.pred_results_file_path

    predict_agent_family(training_file_path, test_file_path, pred_results_file_path)

