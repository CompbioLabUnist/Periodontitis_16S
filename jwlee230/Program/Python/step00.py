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


def aggregate_confusion_matrix(confusion_matrix: typing.Union[None, numpy.ndarray], derivation: str = "") -> typing.Union[typing.Tuple[str, str, str, str, str, str, str, str, str, str, str, str], float]:
    """
    aggregate_confusion_matrix: derivations from confusion matrix
    """
    derivations = ("sensitivity", "specificity", "precision", "negative_predictive_value", "miss_rate", "fall_out", "false_discovery_rate", "false_ommission_rate", "thread_score", "accuracy", "F1_score", "odds_ratio")

    if confusion_matrix is None:
        return derivations

    assert (derivation == "") or (derivation in derivations)

    TP, FP, FN, TN = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    with numpy.errstate(divide="ignore"):
        return list(filter(lambda x: (derivation == "") or (x[0] == derivation), zip(derivations, (TP / (TP + FN), TN / (TN + FP), TP / (TP + FP), TN / (TN + FN), FN / (FN + TP), FP / (FP + TN), FP / (FP + TP), FN / (FN + TN), TP / (TP + FN + FP), (TP + TN) / (TP + TN + FP + FN), 2 * TP / (2 * TP + FP + FN), (TP / FP) / (FN / TN)))))[0][1]
