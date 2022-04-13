import math

from lxml import etree as et
import numpy as np
from collections import defaultdict

type_to_id = {
    "nod": defaultdict(lambda: 1),
    "emotion": {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
    }
}

def load_anvil_annotation(file_path):
    root = et.parse(file_path).getroot()
    nod_nodes = root.findall("*track[@name='nod']/el")
    session_nodes = root.findall("*track[@name='session']/el")
    # Anvil stores start and end stamps in seconds
    nods = [(float(n.attrib["start"]), float(n.attrib["end"]), n.attrib.get("type", None)) for n in nod_nodes]
    emotion = []
    verbalizations = []
    sessions = [(float(n.attrib["start"]), float(n.attrib["end"])) for n in session_nodes]
    return nods, emotion, verbalizations, sessions


def second_to_sample(stamp, frequency=30):
    return math.floor(stamp * frequency)


def annotation_spans_to_sequence_labels(tracks, duration, frequency=30):
    samples = int(duration * frequency)
    labels = np.zeros([samples, 3], dtype=np.int8)
    nods, emotion, verbalizations, sessions = tracks

    for start, end, type in nods:
        i = second_to_sample(start, frequency)
        j = second_to_sample(end, frequency)
        labels[i:j, 0] = type_to_id["nod"][type]

    return labels


def find_session_bounds(tracks, frequency=30):
    if len(tracks) < 4 or len(tracks[3]) == 0:
        raise Exception("Session annotations not found")
    session_start, session_end = tracks[3][0]
    return second_to_sample(session_start, frequency), second_to_sample(session_end, frequency)

