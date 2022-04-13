from __future__ import division

import math
import os

import numpy as np
import pysptk
import dlib
import webrtcvad as webrtcvad
from imutils import face_utils
from .head_pose_estimation import get_head_pose, get_cam_matrix
from .stabilizer import Stabilizer
from collections import deque
from numpy.lib.stride_tricks import as_strided as ast
from scipy.signal import spectrogram
from tqdm import trange

from active_listening.feature_extraction.caching import component_shelf
from active_listening.loading import load_audio_video_streams

face_landmark_weights_path = './shape_predictor_68_face_landmarks.dat'


@component_shelf('features.shelve')
def cached_feature_extract(data_path, filename, components=["f0", "mfcc", "head_pose"]):
    speaker_feats, shape, meta = feature_extract(data_path, filename, components)
    return speaker_feats, shape, meta


def feature_extract(data_path, filename, components=["f0", "mfcc", "head_pose"]):
    speaker_audio, speaker_video, meta = load_audio_video_streams(None, os.path.join(data_path, filename))
    audio_sample_rate = meta["audio_sample_rate"]
    video_frame_rate = meta["video_frame_rate"]
    av_mult = int(audio_sample_rate / video_frame_rate)
    extractor = FeatureExtractor(video_frame_rate, audio_sample_rate, components=components)
    speaker_stream = extractor.process_streams(speaker_audio, speaker_video, 2, yield_data=False)
    shape = extractor.shape(audio_sample_rate * 2, int(video_frame_rate * 2))
    speaker_feats = list(speaker_stream)
    return speaker_feats, shape, meta


def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))


def chunk_data_list(data, window_size, overlap_size=0):
    num_windows = (len(data) - window_size) // (window_size - overlap_size) + 1

    stride_size = window_size - overlap_size
    i = 0
    while True:
        start = int(i * stride_size)
        if start + window_size > len(data):
            assert i == num_windows
            break
        yield data[start:start + window_size]
        i += 1


def chunk_data_fractional_overlap(data, window_size, overlap_size=0):
    num_windows = math.ceil((data.shape[0] - window_size) / (window_size - overlap_size))

    chunked = np.empty([num_windows, window_size])
    stride_size = window_size - overlap_size
    i = 0
    while True:
        start = int(i * stride_size)
        if start + window_size >= len(data):
            assert i == num_windows
            break
        chunked[i, :] = data[start:start + window_size]
        i += 1
    return chunked


class FeatureExtractor:
    def __init__(self, video_frame_rate: float, audio_sample_rate: int, components: [str] = ("f0", "head_pose"), normalize: bool = False):
        # Nominal values; we expect these rates to be a bit variable in practice because of system noise
        self.video_frame_rate = video_frame_rate
        self.audio_sample_rate = audio_sample_rate
        # 20ms. Convenient because it divides 1000ms evenly, and because it's one of the sizes that works with the
        # VAD model
        self.hopsize = int(np.ceil(audio_sample_rate * 0.02))
        self.mfcc_order = 14
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(os.path.dirname(__file__) + "/" + face_landmark_weights_path)
        self.__vad = webrtcvad.Vad(3)
        self.__components = components
        self._pose_stabilizers =[]
        self.normalize = normalize

        # Some features require more than a video frame's worth of audio so we'll accumulate
        # an internal audio buffer.
        self.__accumulator_size = 10
        # Fill em up with zeros to start so we can plot right away
        self.__audio_accumulator = deque()
        self.reset_buffers()

    def get_components(self):
        return self.__components

    def reset_buffers(self):
        # Eventually we can keep some head pose tracking state here too.
        #  Assuming synced audio/video input this is the audio chunk size we'll get
        audio_frames_per_video_frame = int(self.audio_sample_rate / self.video_frame_rate)
        self.__audio_accumulator = deque([np.zeros(audio_frames_per_video_frame)] * self.__accumulator_size)
        # We might get fed a different resolution image
        self._camera_matrix = None

        self._pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(9)]

    def process_audio_buffer(self, audio_raw) -> dict:
        """
        Extracts  the currently enabled component features.
        :param audio_raw: an np.array of integer audio samples
        :return: dict of features indexed by their name
        """

        # Helpful guidance on windowed feature extraction, parameters:
        # https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
        # https://maelfabien.github.io/machinelearning/Speech9/#
        # https://www.ee.columbia.edu/~dpwe/pubs/Ellis10-introspeech.pdf

        if self.normalize:
            mean = np.mean(audio_raw)
            std = np.std(audio_raw)
            if std == 0:
                std = np.finfo(np.float32).eps # replace 0 by eps to avoid division by zero
            audio_raw = audio_raw - mean
            audio_raw = audio_raw / std

        audio = np.pad(audio_raw, (0, len(audio_raw) % self.hopsize), "constant")
        features = {}

        if "f0" in self.__components:
            # You won't get good (or any) results for too small an audio buffer (<50ms).
            # I've found rapt to work better than swipe but this may vary across the dataset.
            f0 = pysptk.rapt(audio.astype(np.float32), fs=self.audio_sample_rate, hopsize=self.hopsize,
                             min=60,
                             max=240,
                             otype="f0")
            features["f0"] = np.expand_dims(f0, 1)

        if "mfcc" in self.__components:
            # Run 30ms MFCCs with 50% overlapped windows.
            chunked_data = chunk_data(audio.astype(np.float64), self.hopsize, self.hopsize // 2)
            # MFCC runs FFTs inside, so can only operate on a power of 2 window. We'll zero pad to next power of 2
            pad_to_size = 2 ** math.ceil(math.log2(self.hopsize))
            amount_to_pad = pad_to_size - self.hopsize
            # Just pad the windows using default pad value (0)
            padded = np.pad(chunked_data, [(0, 0), (0, amount_to_pad)], 'constant')
            features["mfcc"] = np.array([pysptk.mfcc(x, fs=self.audio_sample_rate, use_hamming=True) for x in padded])

        if "vad" in self.__components:
            # The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
            # A frame must be either 10, 20, or 30 ms in duration
            # We'll use 30ms segments
            chunked_data = chunk_data(audio, self.hopsize)
            # Filter; don't think Google's implementation benefits from this since it's not doing signal processing
            # chunked_data = chunked_data.astype(np.float64) * np.hanning(self.hopsize)
            vad = [self.__vad.is_speech((segment).astype(np.int16).tobytes(), self.audio_sample_rate) for segment in
                   chunked_data]
            # You'll probably want to turn these to ints on the outside
            features["vad"] = np.array(vad)

        if "spectrogram" in self.__components:
            # First two return arrays have labels for frequency bins and time increments. Throw away for now.
            features["spectrogram"] = spectrogram(audio, fs=self.audio_sample_rate, nperseg=self.hopsize, noverlap=0)[2]

        return features

    def process_video_frames(self, video_frames):
        features = {}
        if "head_pose" not in self.__components:
            return features
        features["head_pose_rotation"] = np.empty((len(video_frames), 3))
        features["head_pose_translation"] = np.empty((len(video_frames), 3))
        features["head_pose_euler"] = np.empty((len(video_frames), 3))
        features["head_pose_shape"] = np.empty((len(video_frames), 68, 2), dtype=np.int64)
        features["head_pose_rotation_steady"] = np.empty((len(video_frames), 3))
        features["head_pose_translation_steady"] = np.empty((len(video_frames), 3))
        features["head_pose_euler_steady"] = np.empty((len(video_frames), 3))
        for i, video_frame in enumerate(video_frames):
            face_rects = self.face_detector(video_frame, 0)
            # TODO(nickswalker): Copying in the tracking solution would be nice.

            #  Sometimes we won't see the face
            if len(face_rects) > 0:
                if self._camera_matrix is None:
                    self._camera_matrix = get_cam_matrix(video_frame.shape[:2])
                shape = self.landmark_predictor(video_frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)
                rotation, translation, euler_angle = get_head_pose(shape, self._camera_matrix)

                steady_pose = []
                pose_np = np.array([rotation,translation, euler_angle]).flatten()
                for value, ps_stb in zip(pose_np, self._pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))

                # TODO(nickswalker): Simplified "looking at camera" feature
                features["head_pose_rotation"][i, :] = rotation.flatten()
                features["head_pose_translation"][i,:] = translation.flatten()
                features["head_pose_euler"][i,:] = euler_angle.flatten()
                features["head_pose_shape"][i,:,:] = shape
                features["head_pose_rotation_steady"][i,:] = steady_pose[0].flatten()
                features["head_pose_translation_steady"][i,:] = steady_pose[1].flatten()
                features["head_pose_euler_steady"][i,:] = steady_pose[2].flatten()
            else:
                features["head_pose_rotation"][i, :] = np.nan
                features["head_pose_translation"][i,:] = np.nan
                features["head_pose_euler"][i,:] = np.nan
                features["head_pose_shape"][i,:,:] = np.nan
                features["head_pose_rotation_steady"][i,:] = np.nan
                features["head_pose_translation_steady"][i,:] = np.nan
                features["head_pose_euler_steady"][i, :] = np.nan

        return features

    def shape(self, aud_length, vid_length) -> dict:
        """
        The shape of each component feature for a given audio buffer length
        :param buffer_length:
        :return:
        """
        buffer_length = aud_length + aud_length % self.hopsize
        shape = {}
        if "f0" in self.__components:
            shape["f0"] = (int(np.ceil(buffer_length / self.hopsize)), 1)
        if "mfcc" in self.__components:
            # 2x the number of hops because we do 50% overlap. One less because the front half frame and the tail
            # half frame don't get double covered
            num_slices = int(np.ceil(buffer_length / self.hopsize)) * 2 - 1
            shape["mfcc"] = (num_slices, self.mfcc_order)
        if "head_pose" in self.__components:
            shape["head_pose_rotation"] = (vid_length, 3)
            shape["head_pose_translation"] = (vid_length, 3)
            shape["head_pose_euler"] = (vid_length, 3)
            shape["head_pose_shape"] = (vid_length, 68, 2)
        if "spectrogram" in self.__components:
            # FIXME(nickswalker): Figure out how to calculate the bin dim
            shape["spectrogram"] = (721, int(np.ceil(buffer_length / self.hopsize)))
        if "vad" in self.__components:
            # If at all possible, use a device with a 48000hz sample rate. Otherwise we'll have to add resampling
            assert webrtcvad.valid_rate_and_frame_length(self.audio_sample_rate, self.hopsize)
            shape["vad"] = math.floor(buffer_length / self.hopsize)
        return shape

    def process_streams(self, audio_buffer, video_stream, window_length, yield_data=False):
        """
        Streams are assumed to begin and end at the same time and match the configured sample and frame rates.
        :param audio_buffer:
        :param video_stream:
        :param window_length: how many seconds of data to process together
        :param yield_data: Whether to return the audio and video
        :return: a generator that yields (audio_feats, video_feats), (audio_buffer, video_frame)
        """
        # Assume we get the whole audio buffer. We'll streamify it
        # First audio chunk needs to be all 0s (no audio yet in the first frame)
        aud_length = window_length * self.audio_sample_rate
        vid_length = round(window_length * self.video_frame_rate)
        aud_frames_per_vid_frames = self.audio_sample_rate / self.video_frame_rate
        amount_to_pad = round(aud_length - aud_frames_per_vid_frames)
        padded = np.pad(audio_buffer, (amount_to_pad, 0), "constant")
        # Get windows, advancing by a video frame's worth of samples each step
        audio_stream = chunk_data_fractional_overlap(padded, aud_length, aud_length - aud_frames_per_vid_frames)

        vid_frames = []
        while True:
            vid_ok, vid_frame = video_stream.read()
            if vid_ok is False:
                break
            vid_frames.append(vid_frame)
        nominal_vid_duration = len(vid_frames) / self.video_frame_rate

        vid_frame_pad = []
        vid_frame_size = vid_frames[0].shape
        for i in range(vid_length - 1):
            vid_frame_pad.append(np.zeros(vid_frame_size, dtype=np.uint8))
        vid_frames = vid_frame_pad + vid_frames
        vid_chunks = list(chunk_data_list(vid_frames, vid_length, vid_length - 1))

        nominal_audio_duration = len(audio_buffer) / self.audio_sample_rate
        # We'll accept 1 second of total drift across the whole track
        assert abs(nominal_audio_duration - nominal_vid_duration) < 1.0

        for i in trange(min(len(audio_stream), len(vid_chunks))):
            audio_chunk, vid_chunk = audio_stream[i], vid_chunks[i]
            frames = None
            # In case you need the unprocessed frames for a visualization or something
            if yield_data:
                frames = (audio_chunk, vid_chunk)
            audio_feats = self.process_audio_buffer(audio_chunk)
            # The video features are too expensive to do redundant extraction work.
            # Avoid redundant computation by processing each frame once and composing
            # the window of features with the values from the previous window
            if i == 0:
                vid_feats = self.process_video_frames(vid_chunk)
            else:
                next_frame_feats = self.process_video_frames([vid_chunk[-1]])
                if "head_pose_rotation" in vid_feats:
                    # You'll need to manually compose the window for any additional video features you want to use
                    vid_feats["head_pose_rotation"] = np.vstack(
                        [vid_feats["head_pose_rotation"][1:, ], next_frame_feats["head_pose_rotation"]])
                    vid_feats["head_pose_translation"] = np.vstack(
                        [vid_feats["head_pose_translation"][1:, ], next_frame_feats["head_pose_translation"]])
                    vid_feats["head_pose_euler"] = np.vstack(
                        [vid_feats["head_pose_euler"][1:, ], next_frame_feats["head_pose_euler"]])
                    vid_feats["head_pose_shape"] = np.vstack(
                        [vid_feats["head_pose_shape"][1:, ], next_frame_feats["head_pose_shape"]])
                    vid_feats["head_pose_rotation_steady"] = np.vstack(
                        [vid_feats["head_pose_rotation_steady"][1:, ], next_frame_feats["head_pose_rotation_steady"]])
                    vid_feats["head_pose_translation_steady"] = np.vstack(
                        [vid_feats["head_pose_translation_steady"][1:, ],
                         next_frame_feats["head_pose_translation_steady"]])
                    vid_feats["head_pose_euler_steady"] = np.vstack(
                        [vid_feats["head_pose_euler_steady"][1:, ], next_frame_feats["head_pose_euler_steady"]])
            yield {**audio_feats, **vid_feats}, frames


class FeatureAccumulator:
    def __init__(self, extractor, size):
        self.__size = size
        self.__buffers = {}
        self.__components = extractor.get_components()
        self.__shape = extractor.shape()
        self.reset()

    def reset(self):
        for component in self.__components:
            self.__buffers[component] = deque([np.zeros(self.__shape[component])] * self.__size, maxlen=self.__size)

    def append(self, features):
        for feature, value in features.items():
            self.__buffers[feature].append(value)

    def __getitem__(self, key):
        return self.__buffers[key]
