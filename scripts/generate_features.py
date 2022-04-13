import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import argparse
import json
import os
import re
import numpy as np

import tensorflow as tf
import pandas as pd

from active_listening.annotations import load_anvil_annotation, annotation_spans_to_sequence_labels, find_session_bounds
from active_listening.feature_extraction.feature_extractor import cached_feature_extract
from active_listening.feature_extraction.augmentation import augment_mfcc, augment_1d


def stretch(data, new_length):
    interpted = np.interp(np.arange(new_length), np.linspace(0, new_length, len(data)), data.flatten())
    return interpted


def bake_feature_vectors(feature_sequence, augmentations=[]):
    baked = []
    for i, (feats, _) in enumerate(feature_sequence):

        if augmentations and 'mfcc' in augmentations:
            feats["mfcc"] = augment_mfcc(feats["mfcc"])
        if augmentations and 'f0' in augmentations:
            feats["f0"] = np.expand_dims(augment_1d(feats["f0"].squeeze()), 1)
        if augmentations and 'head_pose' in augmentations:
            feats["head_pose_euler_steady"][:,0] = augment_1d(feats["head_pose_euler_steady"][:,0])
            feats["head_pose_euler_steady"][:,1] = augment_1d(feats["head_pose_euler_steady"][:,1])
            feats["head_pose_euler_steady"][:,2] = augment_1d(feats["head_pose_euler_steady"][:,2])
        
        stretched_head_pose = np.zeros((199,3))
        angle = np.nan_to_num(feats["head_pose_euler_steady"], nan=0.0)
        stretched_head_pose[:, 0] = stretch(angle[:, 0], 199)
        stretched_head_pose[:, 1] = stretch(angle[:, 1], 199)
        stretched_head_pose[:, 2] = stretch(angle[:, 2], 199)

        stretched_f0 = np.zeros((199,1))
        stretched_f0[:, 0] = stretch(feats["f0"], 199)

        head_pose_features = [tf.train.Feature(float_list=tf.train.FloatList(value=np.nan_to_num(x, nan=0.0))) for x in stretched_head_pose.tolist()]
        f0_features = [tf.train.Feature(float_list=tf.train.FloatList(value=x)) for x in stretched_f0.tolist()]
        mfcc_features = [tf.train.Feature(float_list=tf.train.FloatList(value=x)) for x in feats["mfcc"].tolist()]

        features = {
            'head_pose': tf.train.FeatureList(feature=head_pose_features),
            'f0': tf.train.FeatureList(feature=f0_features),
            'mfcc': tf.train.FeatureList(feature=mfcc_features),
        }

        if 'vad' in feats.keys():
            stretched_vad = np.zeros((199,1), dtype=int)
            stretched_vad[:, 0] = np.around(stretch(feats['vad'], 199))
            vad_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=x)) for x in stretched_vad.tolist()]
            features['vad'] = tf.train.FeatureList(feature=vad_features)

        baked.append(features)
    return baked


def featurize_interaction(data_path, interaction_name, augmentations=[], crop_session=True):
    speaker_feats, shape, meta = cached_feature_extract(data_path, interaction_name + "_s.mp4",
                                                        components={"f0", "mfcc", "vad", "spectrogram", "head_pose"})

    baked = bake_feature_vectors(speaker_feats, augmentations)

    listener_file = interaction_name + "_l.anvil"
    listener_annotations = load_anvil_annotation(os.path.join(data_path, listener_file))
    duration_seconds = meta["video_frame_duration"] / meta["video_frame_rate"]

    labels = annotation_spans_to_sequence_labels(listener_annotations, duration_seconds, meta["video_frame_rate"])

    # Clip any labels beyond the length of the features, and for now just take the 0th (nodding) track
    labels = labels[:len(baked), 0]

    # Clip labels and features outside of the annotated session
    if crop_session:
        session_start, session_end = find_session_bounds(listener_annotations, meta["video_frame_rate"])
        baked = baked[session_start:session_end]
        labels = labels[session_start:session_end]

    return baked, labels


def write_interactions(data_path, out_path, augmentations, crop_session=True):
    interaction_ids = set([re.sub('_[sl]', '', n.split('.')[0]) for n in os.listdir(data_path)])
    index = pd.read_csv(data_path + "/index.csv", index_col="interaction_id")
    total_count = 0
    for interaction_id in interaction_ids:
        positive_count = 0
        negative_count = 0
        if not os.path.exists(os.path.join(data_path, "{}_l.anvil".format(interaction_id))):
            print("Annotations do not exist")
            continue
        print(interaction_id)
        x_n, y_n = featurize_interaction(data_path, interaction_id, augmentations, crop_session)
        with tf.io.TFRecordWriter(os.path.join(out_path, "{}.tfrecord".format(interaction_id))) as writer:
            for features, label in zip(x_n, y_n):
                if int(label) > 0:
                    positive_count += 1
                else:
                    negative_count += 1

                context = tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                    'speaker': tf.train.Feature(int64_list=tf.train.Int64List(value=[index.loc[interaction_id][0]])),
                })
                feature_lists = tf.train.FeatureLists(feature_list=features)
                example = tf.train.SequenceExample(feature_lists=feature_lists, context=context)
                writer.write(example.SerializeToString())
                total_count += 1
        with open(os.path.join(out_path, "{}_metadata.json".format(interaction_id)), 'w') as json_file:
            json.dump({'positives': positive_count, 'negatives': negative_count}, json_file)

    print("Wrote {} total".format(total_count))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-a', '--augmentations', required=False, nargs='+')
    arg_parser.add_argument("--data", required=False, default="data")
    arg_parser.add_argument("--output", required=False, default="data")
    arg_parser.add_argument('--crop-sessions', dest='crop_sessions', action='store_true')
    args = arg_parser.parse_args()

    write_interactions(args.data, args.output, args.augmentations, args.crop_sessions)

    print("Done")
    
