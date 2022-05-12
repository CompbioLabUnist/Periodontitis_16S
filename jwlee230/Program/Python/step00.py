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
matplotlib_parameters = {"font.size": 50, "axes.labelsize": 50, "axes.titlesize": 75, "xtick.labelsize": 50, "ytick.labelsize": 50, "font.family": "Arial", "legend.fontsize": 50, "legend.title_fontsize": 50, "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42}


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


def add_spp(taxonomy: str) -> str:
    if taxonomy.endswith(";__"):
        return taxonomy[:-2] + ";s__spp."
    elif len(taxonomy.split(";")) == 6:
        return taxonomy + "; s__spp."
    else:
        return taxonomy


def consistency_taxonomy(taxonomy: str) -> str:
    """
    consistency_taxonomy: make taxonomy information with consistency
    """
    return "; ".join(list(filter(None, list(map(lambda x: x.strip()[3:], add_spp(taxonomy).split(";"))))))


def simplified_taxonomy(taxonomy: str) -> str:
    """
    simplified_taxonomy: simplified taxonomy information for file name
    """
    return "_".join(list(filter(None, list(map(lambda x: x.strip().replace("[", "").replace("]", "")[3:], add_spp(taxonomy).split(";"))))))


derivations = ("sensitivity", "specificity", "precision", "negative_predictive_value", "miss_rate", "fall_out", "false_discovery_rate", "false_ommission_rate", "accuracy", "F1_score", "odds_ratio", "balanced_accuracy", "threat_score")
selected_derivations = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision")


def aggregate_confusion_matrix(confusion_matrix: numpy.ndarray, derivation: str = "") -> float:
    """
    aggregate_confusion_matrix: derivations from confusion matrix
    """

    assert (derivation in derivations)
    assert confusion_matrix.shape == (2, 2)

    TP, FP, FN, TN = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    if derivation == derivations[0]:
        return TP / (TP + FN)
    elif derivation == derivations[1]:
        return TN / (TN + FP)
    elif derivation == derivations[2]:
        return TP / (TP + FP)
    elif derivation == derivations[3]:
        return TN / (TN + FN)
    elif derivation == derivations[4]:
        return FN / (FN + TP)
    elif derivation == derivations[5]:
        return FP / (FP + TN)
    elif derivation == derivations[6]:
        return FP / (FP + TP)
    elif derivation == derivations[7]:
        return FN / (FN + TN)
    elif derivation == derivations[8]:
        return (TP + TN) / (TP + TN + FP + FN)
    elif derivation == derivations[9]:
        return 2 * TP / (2 * TP + FP + FN)
    elif derivation == derivations[10]:
        return (TP * TN) / (FP * FN)
    elif derivation == derivations[11]:
        return ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    elif derivation == derivations[12]:
        return TP / (TP + FN + FP)

    raise ValueError(derivation)


short_stage_dict = {"H": "H", "E": "Sli", "M": "M", "S": "S"}
short_stage_order = ["H", "Sli", "M", "S"]
number_stage_order = ["0", "1", "2", "3"]
long_stage_order = ["Healthy", "Slight", "Moderate", "Severe"]
color_stage_order = ["tab:green", "tab:cyan", "tab:olive", "tab:red"]
color_stage_dict = dict(zip(long_stage_order, color_stage_order))


def change_ID_into_short_stage(ID: str) -> str:
    """
    change_ID_into_short_stage: change ID (e.g. H19 or CPM20) into short stage
    """

    if ID.startswith("H"):
        return short_stage_dict[ID[0]]
    else:
        return short_stage_dict[ID[2]]


def change_short_into_number(given_short: str) -> str:
    """
    change_short_into_number: change short stage into number stage
    """
    for short, number in zip(short_stage_order, number_stage_order):
        if short == given_short:
            return number
    raise Exception("Something went wrong!!")


def change_short_into_long(given_short: str) -> str:
    """
    change_short_into_long: change short stage into long stage
    """
    for short, long_stage in zip(short_stage_order, long_stage_order):
        if short == given_short:
            return long_stage
    raise Exception("Something went wrong!!")


def change_ID_into_long_stage(ID: str) -> str:
    return change_short_into_long(change_ID_into_short_stage(ID))


def sorting(ID: str) -> typing.Tuple[int, str]:
    """
    sorting: sorting key by patient-type
    """
    return (long_stage_order.index(change_ID_into_long_stage(ID)), ID)
