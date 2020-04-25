# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-02 21:00
# @Author  : yingyuankai@aliyun.com
# @File    : io_utils.py

import sys
import collections
import csv
import json
import logging
import os.path
import pickle
import random
import tarfile
import zipfile
from six.moves import urllib
import requests
import h5py
import numpy as np
import pandas as pd
from pandas.errors import ParserError
import tensorflow as tf

from aispace.utils.file_utils import default_download_dir, maybe_create_dir

logger = logging.getLogger(__name__)

__all__ = [
    "load_csv",
    "read_csv",
    "save_csv",
    "save_json",
    "load_json",
    "load_hdf5",
    "save_hdf5",
    "load_object",
    "save_object",
    "load_vocab",
    "load_array",
    "load_from_file",
    "load_glove",
    "load_matrix",
    "load_pretrained_embeddings",
    "maybe_download",
    "save_array",
]


def load_csv(data_fp):
    data = []
    with open(data_fp, 'rb') as f:
        data = list(csv.reader(f))
    return data


def read_csv(data_fp, header=0):
    """
    Helper method to read a csv file. Wraps around pd.read_csv to handle some
    exceptions. Can extend to cover cases as necessary
    :param data_fp: path to the csv file
    :return: Pandas dataframe with the data
    """
    try:
        df = pd.read_csv(data_fp, header=header)
    except ParserError:
        logging.WARNING(r'Failed to parse the CSV with pandas default way, trying \ as escape character.')
        df = pd.read_csv(data_fp, header=header, escapechar='\\')

    return df


def save_csv(data_fp, data):
    writer = csv.writer(open(data_fp, 'w'))
    for row in data:
        if not isinstance(row, collections.Iterable) or isinstance(row, str):
            row = [row]
        writer.writerow(row)


def load_json(data_fp):
    data = {}
    with open(data_fp, 'r', encoding="utf-8") as input_file:
        data = json.load(input_file)
    return data


def save_json(data_fp, data, sort_keys=True, indent=4, mode='w'):
    with open(data_fp, mode, encoding='utf8') as output_file:
        json.dump(data, output_file, cls=NumpyEncoder, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)

# to be tested
# also, when loading an hdf5 file
# most of the times you don't want
# to put everything in memory
# like this function does
# it's jsut for convenience for relatively small datasets
def load_hdf5(data_fp):
    data = {}
    with h5py.File(data_fp, 'r') as h5_file:
        for key in h5_file.keys():
            data[key] = h5_file[key].value
    return data


# def save_hdf5(data_fp: str, data: Dict[str, object]):
def save_hdf5(data_fp, data, metadata=None):
    if metadata is None:
        metadata = {}
    mode = 'w'
    if os.path.isfile(data_fp):
        mode = 'r+'
    with h5py.File(data_fp, mode) as h5_file:
        for key, value in data.items():
            dataset = h5_file.create_dataset(key, data=value)
            if key in metadata:
                if 'in_memory' in metadata[key]:
                    if metadata[key]['in_memory']:
                        dataset.attrs['in_memory'] = True
                    else:
                        dataset.attrs['in_memory'] = False


def load_object(object_fp):
    with open(object_fp, 'rb') as f:
        return pickle.load(f)


def save_object(object_fp, obj):
    with open(object_fp, 'wb') as f:
        pickle.dump(obj, f)


def load_array(data_fp, dtype=float):
    list_num = []
    with open(data_fp, 'r') as input_file:
        for x in input_file:
            list_num.append(dtype(x.strip()))
    return np.array(list_num)


def load_matrix(data_fp, dtype=float):
    list_num = []
    with open(data_fp, 'r') as input_file:
        for row in input_file:
            list_num.append([dtype(elem) for elem in row.strip().split()])
    return np.squeeze(np.array(list_num))


def save_array(data_fp, array):
    with open(data_fp, 'w') as output_file:
        for x in np.nditer(array):
            output_file.write(str(x) + '\n')


def load_pretrained_embeddings(embeddings_path, vocab):
    embeddings = load_glove(embeddings_path)

    # find out the size of the embeddings
    embeddings_size = len(next(iter(embeddings.values())))

    # calculate an average embedding, to use for initializing missing words
    avg_embedding = np.zeros(embeddings_size)
    count = 0
    for word in vocab:
        if word in embeddings:
            avg_embedding += embeddings[word]
            count += 1
    if count > 0:
        avg_embedding /= count

    # create the embedding matrix
    embeddings_vectors = []
    for word in vocab:
        if word in embeddings:
            embeddings_vectors.append(embeddings[word])
        else:
            embeddings_vectors.append(
                avg_embedding + np.random.uniform(-0.01, 0.01, embeddings_size))
    embeddings_matrix = np.stack(embeddings_vectors)

    # let's help the garbage collector free some memory
    embeddings = None

    return embeddings_matrix


def load_glove(file_path):
    logging.info('  Loading Glove format file {}'.format(file_path))
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line:
                split = line.split()
                word = split[0]
                embedding = np.array([float(val) for val in split[1:]])
                embeddings[word] = embedding
    logging.info('  {0} embeddings loaded'.format(len(embeddings)))
    return embeddings


def split_data(split, data):
    # type: (float, list) -> (list, list)
    split_length = int(round(split * len(data)))
    random.shuffle(data)
    return data[:split_length], data[split_length:]


def shuffle_unison_inplace(list_of_lists, random_state=None):
    if list_of_lists:
        assert all(len(l) == len(list_of_lists[0]) for l in list_of_lists)
        if random_state is not None:
            random_state.permutation(len(list_of_lists[0]))
        else:
            p = np.random.permutation(len(list_of_lists[0]))
        return [l[p] for l in list_of_lists]
    return None


def shuffle_dict_unison_inplace(np_dict, random_state=None):
    keys = list(np_dict.keys())
    list_of_lists = list(np_dict.values())

    # shuffle up the list of lists according to previous fct
    shuffled_list = shuffle_unison_inplace(list_of_lists, random_state)

    recon = {}
    for ii in range(len(keys)):
        dkey = keys[ii]
        recon[dkey] = shuffled_list[ii]

    # we've shuffled the dictionary in place!
    return recon


def shuffle_inplace(np_dict):
    if len(np_dict) == 0:
        return
    size = np_dict[next(iter(np_dict))].shape[0]
    for k in np_dict:
        if np_dict[k].shape[0] != size:
            raise ValueError(
                'Invalid: dictionary contains variable length arrays')

    p = np.random.permutation(size)

    for k in np_dict:
        np_dict[k] = np_dict[k][p]


def split_dataset_tvt(dataset, split):
    if 'split' in dataset:
        del dataset['split']
    training_set = split_dataset(dataset, split, value_to_split=0)
    validation_set = split_dataset(dataset, split, value_to_split=1)
    test_set = split_dataset(dataset, split, value_to_split=2)
    return training_set, test_set, validation_set


def split_dataset(dataset, split, value_to_split=0):
    splitted_dataset = {}
    for key in dataset:
        splitted_dataset[key] = dataset[key][split == value_to_split]
    return splitted_dataset


def collapse_rare_labels(labels, labels_limit):
    if labels_limit > 0:
        labels[labels >= labels_limit] = labels_limit
    return labels


def class_counts(dataset, labels_field):
    return np.bincount(dataset[labels_field].flatten()).tolist()


def text_feature_data_field(text_feature):
    return text_feature['name'] + '_' + text_feature['level']


def load_from_file(file_name, field=None, dtype=int):
    if file_name.endswith('.hdf5') and field is not None:
        hdf5_data = h5py.File(file_name, 'r')
        split = hdf5_data['split'].value
        column = hdf5_data[field].value
        hdf5_data.close()
        array = column[split == 2]  # ground truth
    elif file_name.endswith('.npy'):
        array = np.load(file_name)
    elif file_name.endswith('.csv'):
        array = read_csv(file_name, header=None)[0].tolist()
    elif file_name.endswith('.json'):
        array = load_json(file_name)
    elif file_name.endswith(".model"):
        array = None
    else:
        array = load_matrix(file_name, dtype)
    return array


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


def load_vocab(file_path):
    import tensorflow as tf
    if not file_path.endswith(".model"):
        with tf.io.gfile.GFile(file_path) as vocab_file:
            # Converts to 'unicode' (Python 2) or 'str' (Python 3)
            vocab = list(tf.compat.as_text(line.strip()) for line in vocab_file)
    else:
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use AlbertTokenizer: "
                           "https://github.com/google/sentencepiece pip install sentencepiece")
        tmp_vocab = spm.SentencePieceProcessor()
        tmp_vocab.Load(file_path)
        vocab = [tmp_vocab.id_to_piece(idx) for idx in range(len(tmp_vocab))]
    return vocab


def maybe_download(urls, path, filenames=None, extract=False):
    """Downloads a set of files.

    Args:
        urls: A (list of) urls to download files.
        path (str): The destination path to save the files.
        filenames: A (list of) strings of the file names. If given,
            must have the same length with :attr:`urls`. If `None`,
            filenames are extracted from :attr:`urls`.
        extract (bool): Whether to extract compressed files.

    Returns:
        A list of paths to the downloaded files.
    """
    maybe_create_dir(path)

    if not isinstance(urls, (list, tuple)):
        urls = [urls]
    if filenames is not None:
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        if len(urls) != len(filenames):
            raise ValueError(
                '`filenames` must have the same number of elements as `urls`.')

    result = []
    for i, url in enumerate(urls):
        if filenames is not None:
            filename = filenames[i]
        elif 'drive.google.com' in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split('/')[-1]
            # If downloading from GitHub, remove suffix ?raw=True
            # from local filename
            if filename.endswith("?raw=true"):
                filename = filename[:-9]

        filepath = os.path.join(path, filename)
        result.append(filepath)

        if not tf.io.gfile.exists(filepath):
            if 'drive.google.com' in url:
                filepath = _download_from_google_drive(url, filename, path)
            else:
                filepath = _download(url, filename, path)

            if extract:
                logger.info('Extract %s', filepath)
                if tarfile.is_tarfile(filepath):
                    tarfile.open(filepath, 'r').extractall(path)
                elif zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath) as zfile:
                        zfile.extractall(path)
                else:
                    logger.info("Unknown compression type. Only .tar.gz, "
                                    ".tar.bz2, .tar, and .zip are supported")

    return result


def _download(url, filename, path):
    def _progress(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.
        # pylint: disable=cell-var-from-loop
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (filename, percent))
        sys.stdout.flush()

    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded {} {} bytes.'.format(
        filename, statinfo.st_size))

    return filepath


def _extract_google_drive_file_id(url):
    # id is between `/d/` and '/'
    url_suffix = url[url.find('/d/') + 3:]
    file_id = url_suffix[:url_suffix.find('/')]
    return file_id


def _download_from_google_drive(url, filename, path):
    """Adapted from `https://github.com/saurabhshri/gdrive-downloader`
    """
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    file_id = _extract_google_drive_file_id(url)

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    response = sess.get(gurl, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = sess.get(gurl, params=params, stream=True)

    filepath = os.path.join(path, filename)
    CHUNK_SIZE = 32768
    with tf.io.gfile.GFile(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print('Successfully downloaded {}.'.format(filename))

    return filepath