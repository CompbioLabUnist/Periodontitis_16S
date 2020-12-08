"""
step00: for general purpose within all step
"""
import hashlib
import hmac
import os
import pickle
import tarfile
import tempfile
import typing
import numpy

key = bytes("asdf", "UTF-8")


def file_list(path: str) -> typing.List[str]:
    """
    file_list: return a list of files in path
    """
    return list(filter(lambda x: os.path.isfile(x), list(map(lambda x: os.path.join(path, x), os.listdir(path)))))


def directory_list(path: str) -> typing.List[str]:
    """
    directory_list: return a list of directories in path
    """
    return list(filter(lambda x: os.path.isdir(x), list(map(lambda x: os.path.join(path, x), os.listdir(path)))))


def make_hmac(message: bytes) -> bytes:
    """
    make_hmac: return a HMAC
    """
    return hmac.new(key, message, hashlib.sha512).digest()


def make_pickle(path: str, data: typing.Any) -> None:
    """
    make_pickle: create a pickle
    """
    if not path.endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")

    pkl = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    key = make_hmac(pkl)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "data.pkl"), "wb") as f:
            f.write(pkl)
        with open(os.path.join(tmp_dir, "key.txt"), "wb") as f:
            f.write(key)

        with tarfile.open(path, "w:gz") as tar:
            tar.add(os.path.join(tmp_dir, "data.pkl"), arcname="data.pkl")
            tar.add(os.path.join(tmp_dir, "key.txt"), arcname="key.txt")


def read_pickle(path: str) -> typing.Any:
    """
    read_pickle: read a pickle file
    """
    if not path.endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")
    if not tarfile.is_tarfile(path):
        raise ValueError("Path cannot be read as a tar file")

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        with open(os.path.join(tmp_dir, "data.pkl"), "rb") as f:
            pkl = f.read()
        with open(os.path.join(tmp_dir, "key.txt"), "rb") as f:
            key = f.read()

    if not hmac.compare_digest(make_hmac(pkl), key):
        raise ValueError("Data is not valid")

    return pickle.loads(pkl)


def consistency_taxonomy(taxonomy: str) -> str:
    """
    consistency_taxonomy: make taxonomy information with consistency
    """
    return "; ".join(list(filter(lambda x: x != "__", list(map(lambda x: x.strip(), taxonomy.split(";"))))))


def simplified_taxonomy(taxonomy: str) -> str:
    """
    simplified_taxonomy: simplified taxonomy information for file name
    """
    return "_".join(list(filter(lambda x: x, list(map(lambda x: x.strip().replace("[", "").replace("]", "")[3:], taxonomy.split(";"))))))


derivations = ("sensitivity", "specificity", "precision", "negative_predictive_value", "miss_rate", "fall_out", "false_discovery_rate", "false_ommission_rate", "accuracy", "F1_score", "odds_ratio", "balanced_accuracy", "threat_score")


def aggregate_confusion_matrix(confusion_matrix: numpy.ndarray, derivation: str = "") -> float:
    """
    aggregate_confusion_matrix: derivations from confusion matrix
    """

    assert (derivation in derivations)

    TP, FP, FN, TN = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    if derivation == derivations[0]:
        if (TP + FN) == 0:
            return numpy.nan
        else:
            return TP / (TP + FN)
    elif derivation == derivations[1]:
        if (TN + FP) == 0:
            return numpy.nan
        else:
            return TN / (TN + FP)
    elif derivation == derivations[2]:
        if (TP + FP) == 0:
            return numpy.nan
        else:
            return TP / (TP + FP)
    elif derivation == derivations[3]:
        if (TN + FN) == 0:
            return numpy.nan
        else:
            return TN / (TN + FN)
    elif derivation == derivations[4]:
        if (FN + TP) == 0:
            return numpy.nan
        else:
            return FN / (FN + TP)
    elif derivation == derivations[5]:
        if (FP + TN) == 0:
            return numpy.nan
        else:
            return FP / (FP + TN)
    elif derivation == derivations[6]:
        if (FP + TP) == 0:
            return numpy.nan
        else:
            return FP / (FP + TP)
    elif derivation == derivations[7]:
        if (FN + TN) == 0:
            return numpy.nan
        else:
            return FN / (FN + TN)
    elif derivation == derivations[8]:
        return (TP + TN) / (TP + TN + FP + FN)
    elif derivation == derivations[9]:
        if (2 * TP + FP + FN) == 0:
            return numpy.nan
        else:
            return 2 * TP / (2 * TP + FP + FN)
    elif derivation == derivations[10]:
        if (FP == 0) or (FN == 0):
            return numpy.nan
        else:
            return (TP * TN) / (FP * FN)
    elif derivation == derivations[11]:
        if (TP + FN == 0) or (TN + FP == 0):
            return numpy.nan
        else:
            return ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    elif derivation == derivations[12]:
        if (TP + FN + FP == 0):
            return numpy.nan
        else:
            return TP / (TP + FN + FP)

    raise ValueError(derivation)


long_stage_order = ["Healthy", "Early", "Moderate", "Severe"]
short_stage_order = ["H", "E", "M", "S"]
