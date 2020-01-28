import logging
import os

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

log = logging.getLogger('utility_functions')

def load_embeddings(embedding_path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    log.debug('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vector = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vector.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    log.debug('Embedding file loaded...')
    word_idx[u'-LRB-'] = word_idx.pop(u'(')
    word_idx[u'-RRB-'] = word_idx.pop(u')')
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(
        -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    return np.stack(weight_vectors), word_idx


# Combine and split the data into train and test
def read_data(rawDataFile, sentimentLabelFile):
    # read dictionary into df
    df_data_sentence = pd.read_table(rawDataFile)
    df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})

    # read sentiment labels into df
    df_data_sentiment = pd.read_table(sentimentLabelFile)
    df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})

    # combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')

    return df_processed_all


def training_data_split(all_data, splitPercent, splitDataDir):
    msk = np.random.rand(len(all_data)) < splitPercent
    train_only = all_data[msk]
    test_and_dev = all_data[~msk]

    msk_test = np.random.rand(len(test_and_dev)) < 0.5
    test_only = test_and_dev[msk_test]
    dev_only = test_and_dev[~msk_test]

    dev_only.to_csv(os.path.join(splitDataDir, 'dev.csv'))
    test_only.to_csv(os.path.join(splitDataDir, 'test.csv'))
    train_only.to_csv(os.path.join(splitDataDir, 'train.csv'))

    return train_only, test_only, dev_only


def maxSeqLen(training_data):
    sequence_length = training_data['Phrase'].apply(lambda sentence: len(sentence.split()))
    return max(sequence_length), sequence_length


def tf_data_pipeline_nltk(data, word_idx, weight_matrix, max_seq_len):
    # training_data = training_data[0:50]

    maxSeqLength = max_seq_len  # Maximum length of sentence
    no_rows = data.shape[0]
    ids = np.zeros((no_rows, maxSeqLength), dtype='int32')
    # conver keys in dict to lower case
    word_idx_lwr = {k.lower(): v for k, v in word_idx.items()}
    idx = 0

    for index, row in data.iterrows():

        sentence = (row['Phrase'])
        # print (sentence)
        tokenizer = RegexpTokenizer(r'\w+')
        sentence_words = tokenizer.tokenize(sentence)
        # print (sentence_words)
        i = 0
        for word in sentence_words:
            # print(index)
            word_lwr = word.lower()
            try:
                # print (word_lwr)
                ids[idx][i] = word_idx_lwr[word_lwr]

            except Exception as e:
                # print (e)
                # print (word)
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i = i + 1
        idx = idx + 1

    return ids


def labels_matrix(data):
    labels = data['sentiment_values']

    lables_float = labels.astype(float)

    cats = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    labels_mult = (lables_float * 10).astype(int)
    dummies = pd.get_dummies(labels_mult, prefix='', prefix_sep='')
    dummies = dummies.T.reindex(cats).T.fillna(0)
    labels_matrix = dummies.as_matrix()

    return labels_matrix