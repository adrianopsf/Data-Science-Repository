import NaiveBayesSolver
import sys

# ***** Esta é a versão 2.0 deste script, atualizado em 02/07/2017 *****

def build_NaiveBayes_model(dataset_directory, model_file):
    nb = NaiveBayesSolver.NaiveBayesSolver()
    nb.train(dataset_directory, model_file)

def predict_with_NaiveBayes(dataset_directory, model_file):
    nb = NaiveBayesSolver.NaiveBayesSolver()
    nb.predict(dataset_directory, model_file)

if __name__ == "__main__":
    (mode, technique, dataset_directory, model_file) = sys.argv[1:5]

    if mode == 'train' and technique == 'bayes':
        build_NaiveBayes_model(dataset_directory, model_file)

    elif mode == 'test' and technique == 'bayes':
        predict_with_NaiveBayes(dataset_directory, model_file)
