import preprocess
import yaml
import argparse
global args
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

parser = argparse.ArgumentParser(description='SVD Representation')
parser.add_argument('--config', default='config.yml')
args = parser.parse_args()

NGRAM_RANGE = (1, 1)  # use unigrams for cuis
MIN_DF = 0.0

# Load yaml configs into configs dictionary
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)
    f.close()


if __name__ == "__main__":
    big_query = configs['data']['bq']

    dataset = preprocess.DatasetProvider(
        configs['data']['notes'],
        configs['data']['codes'],
        configs['args']['min_token_freq'],
        configs['args']['max_tokens_in_file'],
        configs['args']['min_examples_per_code'],
        big_query
    )

    x_train, y_train = dataset.load_raw()

    tf = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        min_df=MIN_DF,
        vocabulary=None,
        use_idf=True
    )
    train_tfidf_matrix = tf.fit_transform(x_train)

    pickle_tfidf = open("tfidf.pkl", 'wb')
    pickle.dump(tf, pickle_tfidf)
    pickle_tfidf.close()

    svd = TruncatedSVD(n_components=1000)
    svd.fit(train_tfidf_matrix)

    pickle_svd = open("svd.pkl", 'wb')
    pickle.dump(svd, pickle_svd)
    pickle_svd.close()

