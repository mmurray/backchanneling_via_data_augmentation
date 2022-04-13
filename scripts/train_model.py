import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import argparse
import datetime
import math
import os
import random
import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeavePGroupsOut
from tensorflow import keras
import pandas as pd
import tensorflow_addons as tfa

from active_listening.models.lstm import LSTM


class RecordParser:
    def __init__(self, labeled=True):
        self.ctx_descr = {
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'speaker': tf.io.FixedLenFeature([], tf.int64),
        }
        self.features_descr = {
            'head_pose': tf.io.FixedLenSequenceFeature([3], tf.float32),
            'f0': tf.io.FixedLenSequenceFeature([1], tf.float32),
            'mfcc': tf.io.FixedLenSequenceFeature([14], tf.float32),
        }
        self.labeled=labeled

    @staticmethod
    def get_parser(labeled=True):
        rp = RecordParser(labeled)
        return rp.parse

    def parse(self, sample_proto):
        ctx, feats = tf.io.parse_single_sequence_example(sample_proto,
                                                         context_features=self.ctx_descr,
                                                         sequence_features=self.features_descr)
        return feats, ctx['label']


def get_splits(data_paths, debug=False, speakers_per_fold=2, val_folds=1):
    if debug:
        return [(np.array([0]),np.array([0]))], np.array([0]), np.array(['3_4']), {'file': np.array(['./data_tiny']), 'interaction_id': np.array(['3_4'])}

    index_list = []
    for data_path in data_paths:
        local_index = pd.read_csv(os.path.join(data_path, "index.csv")).astype(str)
        local_index['file'] = data_path
        index_list.append(local_index)
    
    index = pd.concat(index_list, axis=0, ignore_index=True)

    speaker_list = np.array(index.loc[:,'speaker_id'])
    speakers = np.sort(np.array(list(set(speaker_list))))
    np.random.shuffle(speakers)
    folds = speakers.reshape((len(speakers) // speakers_per_fold, speakers_per_fold))

    groups = np.array([np.where(folds==x)[0][0] for x in speaker_list])

    splitter = LeavePGroupsOut(n_groups=val_folds)
    splits = list(splitter.split(index.index, groups=groups))

    return splits, groups, speaker_list, index


def count_examples(files):
    pos = 0
    neg = 0
    for file in files:
        with open(file) as json_file:
            data = json.load(json_file)
            pos += data['positives']
            neg += data['negatives']
    return pos, neg


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-w", "--load_weights", required=False)
    arg_parser.add_argument("-t", "--tag", required=False, default="")
    arg_parser.add_argument("-l", "--loss", required=False, default='crossentropy')
    arg_parser.add_argument('-f', '--features', nargs='+', required=False, default=['f0', 'mfcc', 'head_pose'])
    arg_parser.add_argument("-b", "--batch-size", required=False, type=int, default=64)
    arg_parser.add_argument('--debug', dest='debug', action='store_true')
    arg_parser.add_argument("--shuffle-buffer", required=False, type=int, default=2048)
    arg_parser.add_argument("--data", nargs='+', required=False, default=["data"])
    arg_parser.add_argument("--epochs", required=False, type=int, default=50)
    arg_parser.add_argument('--no-train', dest='train', action='store_false')
    arg_parser.add_argument('--no-eval', dest='eval', action='store_false')
    arg_parser.add_argument('--augmented-data', nargs='+', required=False, default=[])
    arg_parser.set_defaults(eval=True)
    arg_parser.set_defaults(train=True)
    args = arg_parser.parse_args()
    tf.config.run_functions_eagerly(True)

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    splits, groups, speaker_list, index = get_splits(args.data, debug=args.debug)


    random.shuffle(splits)

    all_results = []
    for split, (train_idx, val_test_idx) in enumerate(splits):
        held_out_speakers = np.unique(speaker_list[val_test_idx])
        held_out_groups = np.unique(groups[val_test_idx])

        val_idx = val_test_idx[groups[val_test_idx] == held_out_groups[0]]
        print("held out speakers: {}".format(held_out_speakers))

        val_sessions = np.array(index['interaction_id'][val_idx])
        train_sessions = np.array(index['interaction_id'][train_idx])

        val_files = np.array(index['file'][val_idx])
        train_files = np.array(index['file'][train_idx])

        train_data = [os.path.join(data_path, "{}.tfrecord".format(x)) for x, data_path in zip(train_sessions, train_files)]
        for augmented_data_dir in args.augmented_data:
            train_data.extend([os.path.join(augmented_data_dir, "{}.tfrecord".format(x)) for x in train_sessions])

        raw_val_dataset = tf.data.TFRecordDataset([os.path.join(data_path, "{}.tfrecord".format(x)) for x, data_path in zip(val_sessions, val_files)])
        raw_train_dataset = tf.data.TFRecordDataset(train_data)

        raw_train_dataset.shuffle(args.shuffle_buffer)

        val_dataset = raw_val_dataset.map(RecordParser.get_parser()).batch(args.batch_size)
        train_dataset = raw_train_dataset.map(RecordParser.get_parser()).batch(args.batch_size)

        pos, neg = count_examples([os.path.join(data_path, "{}_metadata.json".format(x)) for x, data_path in zip(train_sessions, train_files)])
        total = neg + pos
        weight_for_0 = (1 / neg) * (total) / 2.0
        weight_for_1 = (1 / pos) * (total) / 2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        initial_bias = math.log(pos / neg)

        model = LSTM(output_bias=initial_bias)

        datestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath="best_weights." + datestamp + ".hdf5",
                                               save_weights_only=True,
                                               save_best_only=True,
                                               monitor='val_accuracy',
                                               mode='max',
                                               verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath="best_weights_f1." + datestamp + ".hdf5",
                                               save_weights_only=True,
                                               save_best_only=True,
                                               monitor='val_f1_score',
                                               mode='max',
                                               verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath="latest_weights." + datestamp + ".hdf5",
                                           save_weights_only=True,
                                           save_best_only=False,
                                           verbose=1),
            
            tf.keras.callbacks.TensorBoard(log_dir="logs/" + datestamp + args.tag + str(split), profile_batch=0)
        ]

        model.compile(
            run_eagerly=True,
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer="sgd",
            metrics=["accuracy", tfa.metrics.F1Score(num_classes=1, threshold=0.5)]
        )

        if args.load_weights:
            model.build({'f0': tf.TensorShape([64, 199, 1]), 'head_pose': tf.TensorShape([64, 199, 3]), 'mfcc': tf.TensorShape([64, 199, 14])})
            model({'f0': tf.random.uniform((64,199,1)), 'head_pose': tf.random.uniform((64, 199, 3)), 'mfcc': tf.random.uniform((64,199,14))})
            model.load_weights(args.weights)

        if args.train:
            model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset,
                                        callbacks=callbacks,
                                        class_weight=class_weight, 
                                        batch_size=args.batch_size)

        if args.eval:
            result = model.evaluate(val_dataset, batch_size=1024)
            result_dict = dict(zip(model.metrics_names, result))
            result_dict["held_out"] = str(held_out_speakers)
            result_dict["datestamp"] = datestamp

            pd.DataFrame(result_dict, index=[split]).to_csv("val_results_" + args.tag + ".csv", mode='a')
    
        
        
   
