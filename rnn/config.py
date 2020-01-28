import os

data_dir = '../Data'
output_dir = os.path.join(data_dir, 'output')
glove_File = os.path.join(data_dir, 'glove/glove_6B_100d.txt')
label_data_file = os.path.join(data_dir, 'dictionary.txt')
sentiment_labels_file = os.path.join(data_dir, 'sentiment_labels.txt')
split_data_dir = os.path.join(data_dir, 'TrainingData')
batch_size = 2000  # batch size for training
EMBEDDING_DIM = 100  # size of the word embeddings
saved_model_file = os.path.join(output_dir, 'best_model.h5')
checkpoint_file = os.path.join(output_dir, 'best_weights.hdf5')