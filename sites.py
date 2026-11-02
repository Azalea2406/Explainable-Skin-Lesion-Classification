"""Lesion-site picker -> dataset tokens.

Users choose a friendly body region; each maps to the base-location tokens the model
was trained with (multi-hot). ADJUST `SITES` / `encode_site` to match your actual
feature encoding if it differs.
"""
BASE_LOCATIONS = ['abdomen', 'acral', 'anterior', 'arm', 'back', 'chest', 'ear', 'extremity',
    'face', 'foot', 'forearm', 'genital', 'hand', 'head', 'lateral', 'lip', 'lower', 'neck',
    'nose', 'oral', 'palms', 'posterior', 'scalp', 'soles', 'thigh', 'torso', 'trunk',
    'unknown', 'upper']

SITES = {
    "Scalp": ["scalp", "head"],
    "Face": ["face", "head"],
    "Ear": ["ear", "head"],
    "Nose": ["nose", "face", "head"],
    "Lip / mouth": ["lip", "oral", "head"],
    "Neck": ["neck"],
    "Chest (upper trunk)": ["chest", "anterior", "upper", "trunk", "torso"],
    "Abdomen (lower trunk)": ["abdomen", "anterior", "lower", "trunk", "torso"],
    "Upper back": ["back", "posterior", "upper", "trunk", "torso"],
    "Lower back": ["back", "posterior", "lower", "trunk", "torso"],
    "Side of trunk": ["lateral", "trunk", "torso"],
    "Genital area": ["genital"],
    "Upper arm": ["arm", "upper", "extremity"],
    "Forearm": ["forearm", "upper", "extremity"],
    "Hand (back of hand)": ["hand", "upper", "extremity"],
    "Palm": ["palms", "acral", "hand", "upper", "extremity"],
    "Thigh (any part)": ["thigh", "lower", "extremity"],
    "Lower leg": ["lower", "extremity"],
    "Foot (top)": ["foot", "lower", "extremity"],
    "Sole": ["soles", "acral", "foot", "lower", "extremity"],
    "Unknown": ["unknown"],
}

def encode_site(label: str) -> list[float]:
    tokens = set(SITES[label])
    return [1.0 if t in tokens else 0.0 for t in BASE_LOCATIONS]
