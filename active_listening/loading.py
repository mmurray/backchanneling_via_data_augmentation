import os
import cv2
import scipy.io.wavfile as wav


def load_audio_video_streams(audio_path, video_path):
    # Look for a wav file with the same name by default
    if not audio_path:
        path, ext = os.path.splitext(video_path)
        audio_path = path + ".wav"
    audio_sample_rate, audio_signal = wav.read(os.path.expanduser(audio_path))

    video_path = os.path.abspath(os.path.expanduser(video_path))
    vid_cap = cv2.VideoCapture(video_path)
    speaker_ok, speaker_sample_frame = vid_cap.read()
    if not speaker_ok:
        raise Exception("Unable to read speaker video: {}".format(video_path))

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        video_frame_rate = vid_cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        video_frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)

    num_vid_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Wild framerates happen when transcoding tries to turn variable frame rate
    # source (usually from phones or webcams) into constant frame rate. Change your
    # encoder settings.
    assert 23 <= video_frame_rate <= 32
    return audio_signal, vid_cap, {"audio_sample_rate": audio_sample_rate, "video_frame_rate": video_frame_rate, "video_frame_duration": num_vid_frames}
