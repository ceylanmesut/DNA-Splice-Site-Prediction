
import tensorflow as tf
import pandas as pd
import collections
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, resample

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Data_Reader_Processor:
    
    def __init__(self, experiment_type):
        
        self.experiment = experiment_type
        
        # Reading the datasets acc. to the experiment type, a.k.a dataset type
        if self.experiment =="HUMAN":
            
            self.dataset_train = pd.read_csv(".\\data\\human\\dna_train.csv")
            self.dataset_val = pd.read_csv(".\\data\\human\\dna_validation.csv")
            self.dataset_test = pd.read_csv(".\\data\\human\\dna_test.csv")
            
        elif self.experiment =="WORM":
            
            self.dataset_train = pd.read_csv(".\\data\\worm\\worm_train.csv")
            self.dataset_val = pd.read_csv(".\\data\\worm\\worm_val.csv")
            self.dataset_test = pd.read_csv(".\\data\\worm\\worm_test.csv")

    def _read_summarize_data(self, dataset):
        """This function reads and summarizes a dataset w.r.t label distribution
        and DNA sequence lenght."""
        
        label_count = dataset["labels"].value_counts()
        seq_length = len(dataset["sequences"][0])

        pos_label = label_count[1]
        neg_label = label_count[-1]
        total = pos_label+neg_label

        print("Label Distribution:")
        print("+1 Label:{:.3f}".format(pos_label/total), "-1 Label:{:.3f}".format(neg_label/total))
        print("Sequence Length:", seq_length)

        return dataset

    def _sampler(self, dataset, sample_amount):
        """Removes duplicates and downsamples the majority class with sample amount desired."""

        # Lets first remove the duplicates
        print("---Length of Dataset Prior Duplicate Removal:", len(dataset))
        print("---Lenght of Majority Class:", len(dataset[dataset.labels==-1]))
        print("---Lenght of Minority Class:", len(dataset[dataset.labels==1]))

        print("---Removing Duplicates---")
        dataset = dataset.drop_duplicates()

        # Separating minority and majority class into different dataframes
        majority_df = dataset[dataset.labels==-1]
        minority_df = dataset[dataset.labels==1]

        print("---Lenght of Majority Class:", len(majority_df))
        print("---Lenght of Minority Class:", len(minority_df))

        # Downsampling majority class.
        majority_df_scaled = resample(majority_df, replace=False,  n_samples=sample_amount, random_state=49)

        # Upsampling minority class.
        minority_df_scaled = resample(minority_df, replace=True,  n_samples=sample_amount, random_state=49)        

        # Combine minority class with downsampled majority class
        data = pd.concat([majority_df_scaled, minority_df_scaled])
        data = data.reset_index(drop=True)
        data = shuffle(data, random_state=32)

        return data

    def _dataset_sampler(self, dataset, sample_amount):
        """Umbrella function for data reading and downsampling"""

        
        print("---Resampling The Dataset---")
        dataset = self._sampler(dataset=dataset, sample_amount=sample_amount)

        return dataset


    def _label_encoder(self, dataset_train, dataset_val, dataset_test):
        """Encodes labels with Label Encoder."""

        y_train = dataset_train["labels"]
        y_val = dataset_val["labels"]
        y_test = dataset_test["labels"]

        # Defining label encoder 
        label_encoder = LabelEncoder()
        # Fitting and transforming training labels
        y_train_coded = label_encoder.fit_transform(y_train)
        y_val_coded= label_encoder.transform(y_val)
        #y_val_coded= label_encoder.fit_transform(y_val) # earlier
        y_test_coded = label_encoder.transform(y_test)

        print("Train Label Encoding: ", collections.Counter(y_train_coded))
        print("Validation Label Encoding: ", collections.Counter(y_val_coded))
        print("Test Label Encoding: ", collections.Counter(y_test_coded))

        return y_train_coded, y_val_coded, y_test_coded

    def _generate_DNA_corpus(self, train_set, validation_set, test_set):
        """Generates DNA corpus from datasets and prints their lenght."""

        print("---Building DNA Corpuses---")
        corpus_train = train_set["sequences"].str.cat(sep=",")
        corpus_validation= validation_set["sequences"].str.cat(sep=",")
        corpus_test= test_set["sequences"].str.cat(sep=",")

        print("Train Corpus Length: ", len(corpus_train.split(",")))
        print("Validation Corpus Length: ", len(corpus_validation.split(",")))
        print("Test Corpus Length: ", len(corpus_test.split(",")))
        
        return corpus_train, corpus_validation, corpus_test

    def _processor(self, data, corpus):
        """Processor function to process corpus."""
        
        data_list = []
        corpus_whole = corpus.split(",")
        for i in tqdm(range(len(data))):

            seq = corpus_whole[i]
            data_list.append(seq)
        
        return(data_list)

    def _process_corpus(self, train_set, validation_set, test_set, 
                        train_corpus, validation_corpus, test_corpus):
        """Processes DNA corpus."""
        
        print("Train Corpus Processing")
        data_train = self._processor(train_set, train_corpus)
        
        print("Validation Corpus Processing")
        data_val = self._processor(validation_set, validation_corpus)
        
        print("Test Corpus Processing")
        data_test = self._processor(test_set, test_corpus)
    
        
        return data_train, data_val, data_test
    
    def _getKmers(self, sequence, size):
        """Generates K-Mers from DNA sequences."""
    
        return [sequence[x:x+size].upper() for x in range(len(sequence) - size + 1)]

    def _tokenize(self, kmer_data_train, kmer_data_val, kmer_data_test):
        """Tokenizes the K-Mers obtained from datasets and computes vocabulary size."""

        # Defining the tokenizer and fitting the training set
        tokenizer = Tokenizer(oov_token=True) # In case of missing vocab item in test set.
        tokenizer.fit_on_texts(kmer_data_train)

        encoded_docs_train = tokenizer.texts_to_sequences(kmer_data_train)
        max_length = max([len(s.split()) for s in kmer_data_train])
        X_train = pad_sequences(encoded_docs_train, maxlen = max_length, padding = 'post')
        print("Train Max Length:", max_length)

        # Encoding and processing validation data
        encoded_docs_val = tokenizer.texts_to_sequences(kmer_data_val)
        max_length = max([len(s.split()) for s in kmer_data_val])
        X_val = pad_sequences(encoded_docs_val, maxlen = max_length, padding = 'post')
        print("Val Max Length:", max_length)

        # Encoding and processing test data
        encoded_docs_test = tokenizer.texts_to_sequences(kmer_data_test)
        max_length = max([len(s.split()) for s in kmer_data_test])
        X_test = pad_sequences(encoded_docs_test, maxlen = max_length, padding = 'post')
        print("Test Max Length:", max_length)

        # Computing the vocabulary size
        vocab_size = len(tokenizer.word_index) + 1

        # TODO: This is wrong.!
        print("MAX_LENGTH", max_length)
        print("VOCAB_SIZE", vocab_size)

        return X_train, X_val, X_test, max_length, vocab_size

    def process_datasets(self, SAMPLE_AMOUNT_TRAIN=None, SAMPLE_AMOUNT_VAL=None, KMERS=None):
        """Processes the datasets and make them ready for model optimization."""

        dataset = self._read_summarize_data(self.dataset_train)

        if self.experiment =="HUMAN":
          # Sampling training dataset 
          dataset_train = self._dataset_sampler(self.dataset_train, sample_amount=SAMPLE_AMOUNT_TRAIN)

          # Sampling validation dataset
          dataset_val = self._dataset_sampler(self.dataset_val, sample_amount=SAMPLE_AMOUNT_VAL)

          dataset_test = self.dataset_test
        
        else:
          dataset_train = self.dataset_train
          dataset_val = self.dataset_val
          dataset_test = self. dataset_test


        y_train, y_val, y_test = self._label_encoder(dataset_train, dataset_val, dataset_test)
        corpus_train, corpus_validation, corpus_test = self._generate_DNA_corpus(dataset_train, dataset_val, dataset_test)
        data_train, data_val, data_test = self._process_corpus(dataset_train, dataset_val, dataset_test,
                                                                    corpus_train, corpus_validation, corpus_test)

        kmer = KMERS
        kmer_data_train = [' '.join(self._getKmers(i, kmer)) for i in data_train]
        kmer_data_val = [' '.join(self._getKmers(i, kmer)) for i in data_val]
        kmer_data_test = [' '.join(self._getKmers(i, kmer)) for i in data_test]

        X_train, X_validation, X_test, max_length, vocab_size = self._tokenize(kmer_data_train, kmer_data_val, kmer_data_test)

        return X_train, X_validation, X_test, y_train, y_val, y_test, max_length, vocab_size


class AUC_K(tf.keras.metrics.AUC):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(AUC_K, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(AUC_K, self).update_state(y_true, y_pred, sample_weight)
            
class Precision_K(tf.keras.metrics.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision_K, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision_K, self).update_state(y_true, y_pred, sample_weight)