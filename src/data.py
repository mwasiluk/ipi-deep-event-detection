import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from keras.layers import Embedding
from keras.models import Sequential
from keras.utils import to_categorical

from liner2 import Liner2
from utils import sliding_window, balanced_split


class Data(object):

    def __init__(self, config, input_files_index, window_size,
                 irrelevant_class, single_class, token_sequences_file_name,
                 file_to_save_token_sequences, cv=False, input_format="batch:ccl"):
        self.config = config
        self.cv = cv
        self.file_to_save_token_sequences = file_to_save_token_sequences
        self.input_files_index = input_files_index
        self.input_format = input_format
        self.irrelevant_class = irrelevant_class
        self.single_class = single_class
        self.token_sequences_file_name = token_sequences_file_name
        self.window_size = window_size
        self.max_sequence_length = self.window_size * 2 + 1

        self.liner = Liner2(config['liner']['config'])

        self.w2v_embeddings = {}
        self.ft_embeddings = {}
        self.indexed_embedding_dicts = {}
        self.position_embedding_dict ={}
        self.input_dim = 0
        self.input_dims = [0]
        self.num_classes = 0

        self.max_sentence_len=0
        self.indexed_features = {}
        self.labels_index = {}

        self.features_to_index = []
        if config['indexed_embeddings'] and len(config['indexed_embeddings']):
            self.features_to_index = [e[0] for e in config['indexed_embeddings']]

        self.annotation_to_label_mapper = None
        if single_class:
            self.annotation_to_label_mapper = lambda (a): "event"




    def load(self, indexed_features_file=None):

        self.data = []

        seq_dict = self.load_token_sequences(indexed_features_file)
        sequences = seq_dict['sequences']


        max_sentence_len = seq_dict['max_sentence_len']
        self.num_classes = len(self.labels_index)
        print('Found %s classes.' %  self.num_classes)
        print('max_sentence_len %s.' % max_sentence_len)

        self.load_embeddings()

        data = []
        labels = []
        if self.cv:
            for fold_index, fold in enumerate(sequences):
                data.append([])
                labels.append([])
                for seq in fold:
                    data[fold_index].append(self.get_vector_sequence(seq))

                    labels[fold_index].append(
                        self.map_label_cv(seq[self.window_size], self.labels_index, self.irrelevant_class, self.config['binary']))

            self.data = data
            self.labels = labels
            self.sequences = sequences
        else:
            for seq in sequences:
                data.append(self.get_vector_sequence(seq))
                labels.append(self.get_label(seq))

            self.data = [np.asarray(data)]
            self.labels = np.asarray(labels)

        self.input_dims = [self.input_dim]

    def get_label(self, seq):
        return self.map_label(seq[self.window_size], self.labels_index, self.irrelevant_class, self.config['binary'])

    def get_vector_sequence(self, seq):
        return [self.map_token(t, self.input_dim, i, True) for i, t in enumerate(seq)]

    def load_token_sequences(self, indexed_features_file):
        if not self.token_sequences_file_name and self.input_files_index:

            self.indexed_features = None
            if indexed_features_file:
                self.load_indexed_features(indexed_features_file)

            seq_dict = self.get_token_sequences(self.input_files_index, cv=self.cv)

            if self.file_to_save_token_sequences:
                print("Saving sequences to file")

                with open(self.file_to_save_token_sequences, 'wb') as f:
                    pickle.dump(seq_dict, f)
        else:
            print("Loading sequences from file")
            with open(self.token_sequences_file_name, "rb") as fp:
                seq_dict = pickle.load(fp)
            self.labels_index = seq_dict['labels_index']
            self.indexed_features = seq_dict['indexed_features']
            self.sequences = seq_dict['sequences']
            self.max_sentence_len = seq_dict['max_sentence_len']

        return seq_dict

    def map_token(self, t, input_d, position=None, add_position_feat=False):
        if not t:
            return np.zeros(input_d)

        vectors = []
        for e_name, e_conf in self.config['w2v_embeddings'].items():
            if not e_conf['enabled']:
                continue

            for attr in e_conf['attributes']:

                if isinstance(attr, list):
                    value = '.'.join([t[a] if t[a] else '' for a in attr])
                else:
                    value = t[attr]

                vector = self.get_w2v_vector(value, self.w2v_embeddings[e_name])
                vectors.append(vector)

        for e_name, e_conf in self.config['ft_embeddings'].items():
            if not e_conf['enabled']:
                continue

            for attr in e_conf['attributes']:

                if isinstance(attr, list):
                    value = '.'.join([t[a] if t[a] else '' for a in attr])
                else:
                    value = t[attr]

                vector = self.get_w2v_vector(value, self.ft_embeddings[e_name])
                vectors.append(vector)

        for attr, dim in self.config['indexed_embeddings']:
            vectors.append(self.indexed_embedding_dicts[attr][t[attr + '_index']])

        if add_position_feat:
            vectors.append(self.position_embedding_dict[position])

        return np.concatenate(vectors)

    def get_training_data(self, validation_split):

        num_samples = self.data[0].shape[0]
        val_indices, train_indices = balanced_split(self.labels, validation_split)
        num_validation_samples = int(validation_split * num_samples)


        x_val = self.data[0][val_indices]
        x_train = self.data[0][train_indices]


        print('x_train shape:', x_train.shape)
        print('x_val shape:', x_val.shape)

        y_val = self.labels[val_indices]
        y_train = self.labels[train_indices]

        print('y_train shape:', y_train.shape)
        print('y_val shape:', y_val.shape)

        return {
            'num_samples': num_samples,
            'num_validation_samples': num_validation_samples,
            'x_train': x_train,
            'x_val': x_val,
            'y_train': y_train,
            'y_val': y_val,

        }

    def get_cv_folds_data(self, validation_split):
        cv_results = []
        acc_list = []
        n_folds = len(self.sequences)
        for fold_index in range(n_folds):
            test_index = fold_index
            all_train_indices = [f for f in range(n_folds) if f != test_index]
            y_train = []
            for i in all_train_indices:
                y_train.extend(self.labels[i])

            val_indices, train_indices = balanced_split(y_train, validation_split)

            x_train = []
            for i in all_train_indices:
                x_train.extend(self.data[i])
            x_train = np.asarray(x_train)
            x_test = np.asarray(self.data[test_index])
            print('x_train shape:', x_train.shape)
            x_val = x_train[val_indices]
            x_train = x_train[train_indices]

            y_train = np.asarray(y_train)
            y_val = y_train[val_indices]
            y_train = y_train[train_indices]

            y_test = np.asarray(self.labels[test_index])

            yield {
                'fold_index': fold_index,
                'x_train': x_train,
                'x_test' : x_test,
                'x_val': x_val,
                'y_train': y_train,
                'y_test': y_test,
                'y_val': y_val,
        }



    def get_w2v_vector(self, word, w2v):
        if word and word in w2v:
            return w2v[word]
        else:
            return np.zeros(w2v.vector_size)

    def map_label(self, t, labels_index, irrelevant_class, binary_not_categorical=False):
        num_classes = len(labels_index)

        if num_classes == 2 and binary_not_categorical:
            if t:
                return t['label_index']
            else:
                return labels_index[irrelevant_class]
        if t:
            return to_categorical(t['label_index'], num_classes)[0]
        else:
            return to_categorical(labels_index[irrelevant_class], num_classes)[0]

    def map_label_cv(self, t, labels_index, irrelevant_class, binary_not_categorical=False):
        num_classes = len(labels_index)

        if num_classes == 2 and binary_not_categorical:
            if t:
                return t['label_index']
            else:
                return labels_index[irrelevant_class]
        if t:
            return to_categorical(t['label_index'], num_classes)[0]
        else:
            return to_categorical(labels_index[irrelevant_class], num_classes)[0]


    def load_folds(self, input_file):

        folds = []
        with open(input_file) as f:
            lines = f.readlines()

        root = os.path.dirname(input_file)

        for line in lines:
            line_data = line.strip().split('\t')
            if len(line_data) != 2:
                print("Incorrect line in folds file: " + input_file, line)
                continue
            file_name, fold = line_data
            if not file_name.startswith("/"):
                file_name = root + "/" + file_name
            while len(folds) < int(fold):
                folds.append([])

            folds[int(fold) - 1].append(file_name)

        return folds

    def get_training_set(fold, folds):
        pass

    # get sequences of dicts of token features
    def get_token_sequences(self, input_files_index,
                            document_limit=None, cv=False):

        readers = []
        if cv:
            for index, fold in enumerate(self.load_folds(input_files_index)):
                readers.append(self.liner.get_batch_reader("\n".join(fold), "", "cclrel"))
        else:
            readers.append(self.get_reader(input_files_index))

        gen = self.get_token_feature_generator()


        if not self.indexed_features:
            self.indexed_features = {}
            for f in self.features_to_index:
                self.indexed_features[f] = {}

        # labels_index = {}  # dictionary mapping label name to numeric id
        if not len(self.labels_index):
            self.labels_index[self.irrelevant_class] = 0

        all_sequences = []
        for reader in readers:
            all_sequences.append(self.process_reader(reader, gen, document_limit))

        if len(readers) == 1:
            all_sequences = all_sequences[0]

        return {
            'labels_index': self.labels_index,
            'indexed_features': self.indexed_features,
            'sequences': all_sequences,
            'max_sentence_len': self.max_sentence_len
        }

    def get_reader(self, input_files_index=None):
        if not input_files_index:
            input_files_index = self.input_files_index
        return self.liner.get_reader(input_files_index, self.input_format)

    def get_token_feature_generator(self):
        return self.liner.get_token_feature_generator()

    def process_reader(self, reader, token_features_generator = None, document_limit=None):

        if not token_features_generator:
            token_features_generator = self.get_token_feature_generator()

        sequences = []
        ii = 0
        while True:
            ii += 1
            if document_limit and ii > document_limit:
                break

            document = reader.nextDocument()
            if document is None:
                break

            self.process_document(document, token_features_generator, sequences=sequences)
        return sequences

    def process_document(self, document, token_features_generator = None, sequences=None):

        if not token_features_generator:
            token_features_generator = self.get_token_feature_generator()

        if sequences is None:
            sequences = []

        token_features_generator.generateFeatures(document)
        for sentence in document.getSentences():
            sentence_seq = self.process_sentence(sentence)

            for w in sliding_window(sentence_seq, self.window_size, self.window_size):
                sequences.append(w)

        return sequences

    def process_sentence(self, sentence):
        annotations = sentence.getAnnotations(self.liner.options.getTypes())
        tokens = sentence.getTokens()
        annotationBeginMap = {}
        annotationEndMap = {}
        sentence_len = len(tokens)
        self.max_sentence_len = max(sentence_len, self.max_sentence_len)
        seq = []
        for a in annotations:
            annotationBeginMap[a.getBegin()] = a.getType()
            annotationEndMap[a.getEnd()] = a.getType()
        annotation = None
        for idx, token in enumerate(tokens):
            t = {}

            for attr in token.getAttributeIndex().indexes:
                t[attr] = token.getAttributeValue(attr)

            for f in self.features_to_index:
                value = t[f]
                f_index_dict = self.indexed_features[f]
                if value in f_index_dict:
                    val_index = f_index_dict[value]
                else:
                    val_index = len(f_index_dict)
                    f_index_dict[value] = val_index

                t[f + '_index'] = val_index

            if idx in annotationBeginMap:
                annotation = annotationBeginMap[idx]

            label = self.irrelevant_class
            if annotation:
                if self.annotation_to_label_mapper:
                    label = self.annotation_to_label_mapper(annotation)
                else:
                    label = annotation

            if label in self.labels_index:
                label_index = self.labels_index[label]
            else:
                label_index = len(self.labels_index)
                self.labels_index[label] = label_index

            t['label'] = label
            t['label_index'] = label_index

            seq.append(t)

            if idx in annotationEndMap:
                annotation = None
        return seq

    def get_categorical_embedding_dict(self, values_index, embedding_dim):
        embedding = Sequential()
        embedding.add(Embedding(len(values_index), embedding_dim, input_length=1))

        embedding.compile('rmsprop', 'mse')
        embedding.summary()

        class_embedding_dict = {}

        for c in values_index:
            class_embedding_dict[values_index[c]] = embedding.predict(np.asarray([values_index[c]]))[0][0]

        return class_embedding_dict

    def get_pos_class_embedding_dict(self, classes_index, class_embedding_dim):
        class_embedding = Sequential()
        class_embedding.add(Embedding(len(classes_index), class_embedding_dim, input_length=1))

        class_embedding.compile('rmsprop', 'mse')
        class_embedding.summary()

        class_embedding_dict = {}

        for c in classes_index:
            class_embedding_dict[classes_index[c]] = class_embedding.predict(np.asarray([classes_index[c]]))[0][0]

        return class_embedding_dict

    def get_position_embedding_dict(self, window_size, embedding_dim):
        embedding = Sequential()
        seq_len = window_size * 2 + 1
        embedding.add(Embedding(seq_len, embedding_dim, input_length=1))

        embedding.compile('rmsprop', 'mse')
        embedding.summary()

        embedding_dict = {}

        for p in range(seq_len):
            embedding_dict[p] = embedding.predict(np.asarray([p]))[0][0]

        return embedding_dict

    def save_features(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump({
                'indexed_features':self.indexed_features,
                'labels_index': self.labels_index
            }, f)

    def load_indexed_features(self, indexed_features_file):
        print('loading indexed features from ' + indexed_features_file)
        with open(indexed_features_file, "rb") as fp:
            loaded = pickle.load(fp)
            self.indexed_features = loaded['indexed_features']
            self.labels_index = loaded['labels_index']
        print(self.indexed_features)


    def load_embeddings(self):
        self.input_dim = 0
        for name, e_conf in self.config['w2v_embeddings'].items():
            if not e_conf['enabled']:
                continue

            print("Loading w2v %s embedding ..." % name)
            self.w2v_embeddings[name] = Word2Vec.load(e_conf['path'])
            e_dim = self.w2v_embeddings[name].vector_size
            print("w2v %s embedding dim %s." % (name, e_dim))
            self.input_dim += e_dim * len(e_conf['attributes'])
        for name, e_conf in self.config['ft_embeddings'].items():
            if not e_conf['enabled']:
                continue

            print("Loading fasttext %s embedding ..." % name)
            self.ft_embeddings[name] = FastText.load(e_conf['path'])
            e_dim = self.ft_embeddings[name].vector_size
            print("fasttext %s embedding dim %s." % (name, e_dim))
            self.input_dim += e_dim * len(e_conf['attributes'])

        for i, (attr, dim) in enumerate(self.config['indexed_embeddings']):
            print(attr, "indexed embedding dim", dim)
            self.indexed_embedding_dicts[attr] = self.get_categorical_embedding_dict(self.indexed_features[attr],
                                                                                     dim)
            self.input_dim += dim
        print('position_embedding_dim %s.' % self.config['position_embedding_dim'])
        self.position_embedding_dict = self.get_position_embedding_dict(self.window_size, self.config['position_embedding_dim'])
        self.input_dim = self.input_dim + self.config['position_embedding_dim']