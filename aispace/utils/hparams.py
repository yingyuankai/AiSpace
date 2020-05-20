# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-04 11:12
# @Author  : yingyuankai@aliyun.com
# @File    : hparams.py

import os
import ast
import re
import yaml
import json
import logging
import collections
import argparse
import traceback
from copy import deepcopy
from ast import literal_eval
from pathlib import Path

from aispace.utils.timer import Timer
from aispace.utils.io_utils import save_json
from aispace.utils.file_utils import default_download_dir, maybe_create_dir
from aispace.utils.io_utils import maybe_download, load_from_file, load_json
from aispace.utils.logger import setup_logging

__all__ = [
    "Hparams", "client"
]

logger = logging.getLogger(__name__)


def _parse_placeholder(text):
    return re.findall(r".*{{(.*)}}.*", text)


class Hparams(collections.OrderedDict):
    def __init__(self, init_dict={}):
        self._init(init_dict)

    def _init(self, init_dict):
        super().__init__(init_dict)
        for key in self:
            if isinstance(self[key], collections.abc.Mapping):
                self[key] = Hparams(self[key])
            elif isinstance(self[key], list):
                for idx, item in enumerate(self[key]):
                    if isinstance(item, collections.abc.Mapping):
                        self[key][idx] = Hparams(item)

    def load_from_config_file(self, config_yaml_file):
        config_path = config_yaml_file
        base_config = dict()
        user_config = dict()
        if config_path is not None:
            user_config = self.load_yaml(config_path)
        config = self.dict_merge(base_config, user_config)
        self._init(config)

    def reuse_saved_json_hparam(self):
        json_obj = load_json(os.path.join(self.get_workspace_dir(), 'hparams.json'))
        self.merge_from_dict(json_obj)

    def load_yaml(self, file) -> dict:
        """Load config from yaml file

        :param file:
        :return:
        """
        with open(file, 'r', encoding="utf8") as stream:
            mapping = yaml.safe_load(stream)
            if mapping is None:
                mapping = dict()
        includes = mapping.get('includes', [])
        if not isinstance(includes, list):
            raise AttributeError(
                f'Includes must be a list, {type(includes)} provided'
            )

        dirname = os.path.dirname(file)
        include_mapping = dict()
        for include_config_file in includes:
            tmp_config_file = os.path.join(dirname, include_config_file)
            cur_include_mapping = self.load_yaml(tmp_config_file)
            include_mapping = self.dict_merge(
                include_mapping, cur_include_mapping
            )
        mapping.pop('includes', None)

        mapping = self.dict_merge(include_mapping, mapping)
        return mapping

    def dict_merge(self, merge_to, merge_from) -> dict:
        """Updates merge_from into merge_to

        :param merge_to: dict
            Dictionary to be updated
        :param merge_from: dict
            Dictionary which has to be added to merge_to
        :return: dict
            merge_to which has updated from merge_from
        """
        if merge_to is None:
            merge_to = {}

        for k, v in merge_from.items():
            if isinstance(v, collections.abc.Mapping):
                merge_to[k] = self.dict_merge(merge_to.get(k, {}), v)
            else:
                merge_to[k] = self._decode_value(v)
        return merge_to

    def merge_from_hparams(self, merge_from):
        """Updates self from merge_from

        :param merge_from: Hparams
             Hparams object which has to be merged into self
        :return:
        """
        for k, v in merge_from.items():
            if isinstance(self.get(k, None), collections.abc.Mapping) and \
                    isinstance(v, collections.abc.Mapping):
                self[k].merge_from_hparams(v)
            else:
                self[k] = v

    def merge_from_dict(self, merge_from: dict):
        """Updates self from merge_from, whose type is dict

        :param merge_from: dict
        :return:
        """
        merge_from_hparam = Hparams(merge_from)
        self.merge_from_hparams(merge_from_hparam)

    def pretty_print(self):
        logger.logging(
            json.dumps(self, indent=4, sort_keys=True)
        )

    def to_dict(self) -> collections.OrderedDict:
        """Converts to dict format

        :return:
        """
        hparam_dict = collections.OrderedDict()
        for k, v in self.items():
            if isinstance(v, Hparams):
                hparam_dict[k] = v.to_dict()
            else:
                hparam_dict[k] = v
        return hparam_dict

    def to_json(self) -> collections.OrderedDict:
        """Converts to dict format and save

        :return:
        """
        self['hparams_json_file'] = os.path.join(self.get_workspace_dir(), 'hparams.json')
        hparam_dict = self.to_dict()
        save_json(self.hparams_json_file, hparam_dict)

    def get_experiment_name(self, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        base_exp_name = f'{self.experiment_name}_{self.model_name}_{self.random_seed}'
        suffix = 0
        found_previous_result = os.path.isdir(os.path.join(save_dir, f'{base_exp_name}_{suffix}'))
        while found_previous_result:
            suffix += 1
            found_previous_result = os.path.isdir(os.path.join(save_dir, f'{base_exp_name}_{suffix}'))
        exp_dir = f'{base_exp_name}_{suffix}'
        return exp_dir

    def get_workspace_dir(self):
        if 'workspace_dir' in self and self.workspace_dir is not None:
            return self.workspace_dir
        if hasattr(self, "model_resume_path") and self.model_resume_path is not None:
            if os.path.exists(self.model_resume_path):
                workspace_dir = self.model_resume_path
            else:
                raise ValueError(f'{self.model_resume_path} does not exists!')
        else:
            save_dir = self.save_dir
            exp_name = self.get_experiment_name(save_dir)
            workspace_dir = os.path.join(save_dir, exp_name)
            if not os.path.exists(workspace_dir):
                os.mkdir(workspace_dir)
        self['workspace_dir'] = workspace_dir
        return workspace_dir

    def get_log_folder(self):
        timer = Timer()
        workspace_dir = self.get_workspace_dir()
        time_format = "%Y-%m-%dT%H:%M:%S"
        log_filename = timer.get_time_hhmmss(None, format=time_format)
        # log_filename += '.log'
        log_folder = os.path.join(workspace_dir, 'logs')
        log_folder_from_hparam = self.log_dir
        if log_folder_from_hparam is not None:
            log_folder = log_folder_from_hparam
        # log_path = os.path.join(log_folder, log_filename)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        return log_folder

    def get_model_filename(self):
        model_path = os.path.join(self.get_workspace_dir(), 'model_saved', "model")
        return model_path

    def get_deploy_dir(self):
        deploy_dir = os.path.join(self.get_workspace_dir(), "deploy")
        return deploy_dir

    def get_saved_model_dir(self):
        saved_model_dir = os.path.join(self.get_workspace_dir(), "saved_model")
        return saved_model_dir

    def notice(self, fields):
        for field in fields:
            if field not in self:
                continue
            print(f'please update {field} to {self[field]} before next stage running.')

    def _decode_value(self, value):
        if not isinstance(value, str):
            return value
        if value == 'None':
            value = None
        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def _get_base_config_path(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        base_config_path = os.path.join(
            cur_dir, '../..', 'configs', 'default', 'base.yml'
        )
        return base_config_path

    def _get_config_path(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            cur_dir, "../..", "configs"
        )
        return config_path

    def _indent(self, st, num_spaces):
        st = st.split("\n")
        first = st.pop(0)
        st = [(num_spaces * " ") + line for line in st]
        st = [first] + st
        st = "\n".join(st)
        return st

    def has_key(self, key_name):
        flag = False
        if key_name in self:
            flag = True
        return flag

    def __getattr__(self, key):
        if key not in self:
            # raise AttributeError(key)
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
        if isinstance(value, collections.abc.Mapping):
            self[key] = Hparams(value)
        elif isinstance(value, list):
            for idx, v in enumerate(value):
                if isinstance(v, collections.abc.Mapping):
                    self[key][idx] = Hparams(v)

    def __str__(self):
        strs = []
        if isinstance(self, collections.abc.Mapping):
            for key, value in sorted(self.items()):
                seperator = "\n" if isinstance(value, Hparams) else " "
                if isinstance(value, list):
                    attr_str = ["{}:".format(key)]
                    for item in value:
                        item_str = self._indent(str(item), 4)
                        attr_str.append("- {}".format(item_str))
                    attr_str = "\n".join(attr_str)
                else:
                    attr_str = "{}:{}{}".format(str(key), seperator, str(value))
                    attr_str = self._indent(attr_str, 4)
                strs.append(attr_str)
        return "\n".join(strs)

    def cascade_get(self, cascade_key):
        key_pieces = cascade_key.split(".")
        cur_hparam = self
        for key in key_pieces[:-1]:
            if key not in cur_hparam:
                return None
            if not isinstance(cur_hparam, collections.abc.Mapping):
                return None
            cur_hparam = cur_hparam[key]
        return getattr(cur_hparam, key_pieces[-1])

    def cascade_set(self, cascade_key, value):
        key_pieces = cascade_key.split(".")
        cur_hparam = self
        for key in key_pieces[:-1]:
            if key not in cur_hparam:
                cur_hparam[key] = Hparams()
            cur_hparam = cur_hparam[key]

        if key_pieces[-1] not in cur_hparam:
            cur_hparam[key_pieces[-1]] = value
        else:
            if isinstance(cur_hparam[key_pieces[-1]], Hparams):
                if isinstance(value, dict):
                    cur_hparam[key_pieces[-1]].merge_from_dict(value)
                elif isinstance(value, Hparams):
                    cur_hparam[key_pieces[-1]].merge_from_hparams(value)
                else:
                    cur_hparam[key_pieces[-1]] = value
            else:
                cur_hparam[key_pieces[-1]] = value

    def _replace_placeholder(self, work_hparams):
        def __replace(text):
            placeholders = _parse_placeholder(text)
            for ph in placeholders:
                real_value = self.cascade_get(ph)
                assert real_value is not None, f"real value of {ph} must not be None."
                assert isinstance(real_value, str), f"{ph}'s value must be a str"
                assert len(_parse_placeholder(real_value)) == 0, \
                    f'The real value {real_value} itself is a placeholder.'
                text = text.replace("{{" + ph + "}}", real_value)
            return text

        if isinstance(work_hparams, str):
            try:
                return __replace(work_hparams)
            except:
                print(traceback.format_exc())
                __replace(work_hparams)
                return work_hparams
        elif isinstance(work_hparams, Hparams):
            for k, v in work_hparams.items():
                work_hparams[k] = self._replace_placeholder(v)
            return work_hparams
        elif isinstance(work_hparams, (list, tuple)):
            return [self._replace_placeholder(item) for item in work_hparams]
        else:
            return work_hparams

    def stand_by(self):
        """Download resources and enrich the hparams.

        For example:
            1) pretrained model, vocab, and config
            2) replace placeholder, like {{workspace_dir}}
            ...

        :return:
        """

        if self.gpus is None:
            self['gpus'] = [0]

        # setup logging
        setup_logging(self.get_log_folder(), self.logging)

        # # prepare workspace
        # self.get_workspace_dir()

        # pretrained
        if "pretrained" in self:
            name = self.pretrained.name
            resources = self.pretrained.family.get(name)
            assert isinstance(resources, dict), ValueError(f"Could not get the resource with name: {name}")
            cache_dir = self.pretrained.cache_dir
            if cache_dir is None:
                cache_path = default_download_dir(name)
            else:
                cache_path = Path(cache_dir)
            cache_path = cache_path / name
            if not cache_path.exists():
                maybe_create_dir(str(cache_path))

            for k, item in resources.items():
                url = item.url
                suffix = item.suffix
                to_insert_paths = item.to_insert_paths
                to_replaces = item.to_replaces
                others = item.others
                # when specify urls of resource
                if url.startswith("http"):
                    filename = url.split("/")[-1]
                    file_path = cache_path / filename
                    if not file_path.exists():
                        try:
                            maybe_download(url, str(cache_path))
                        except Exception as e:
                            logger.error(f"Download from {url} failure!", exc_info=True)
                            raise e
                    # assigning filename to the corresponding config variable.
                    if to_insert_paths and file_path.exists():
                        for to_insert_path in to_insert_paths:
                            self.cascade_set(to_insert_path, str(file_path))
                    # replace corresponding variable with file content
                    if to_replaces and file_path.exists():
                        replace_content = load_from_file(str(file_path))
                        for to_replace in to_replaces:
                            self.cascade_set(to_replace, replace_content)
                else:  # when specify paths of resource which have downloaded.
                    maybe_filename = Path(url)
                    if maybe_filename.is_dir():
                        if suffix:
                            file_path = maybe_filename / suffix
                    else:
                        file_path = maybe_filename
                    # assigning filename to the corresponding config variable.
                    if to_insert_paths:
                        for to_insert_path in to_insert_paths:
                            self.cascade_set(to_insert_path, str(file_path))
                    # replace corresponding variable with file content
                    if to_replaces:
                        replace_content = load_from_file(str(file_path))
                        for to_replace in to_replaces:
                            self.cascade_set(to_replace, replace_content)
                logging.info(f"Prepare resource {k} from {url}, whose path is {str(file_path)}")
                # others config
                if others:
                    for _k, v in others.items():
                        value = v.get("value")
                        other_to_replaces = v.get("to_replaces")
                        for other_to_replace in other_to_replaces:
                            self.cascade_set(other_to_replace, value)

        # replace placeholder
        self._replace_placeholder(self)


def client(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script trains a model.',
        prog='aispace train',
        usage='%(prog)s [options]'
    )

    parser.add_argument(
        '-cn',
        '--config_name',
        help='YAML config file name'
    )

    parser.add_argument(
        '-cd',
        '--config_dir',
        help='YAML config dir'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='save',
        help='directory that contains the results'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        help='directory that contains the logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='experiment',
        help='experiment name'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='run',
        help='name for the model'
    )
    parser.add_argument(
        '--schedule',
        default='train_and_eval',
        help="Method of Experiment to run. "
             "[train_and_eval, train, eval, infer, debug]"
    )
    parser.add_argument(
        '--workspace_dir',
        type=str,
        help="workspace directory in which has log directory, hparams.json, etc."
    )
    parser.add_argument(
        '-mlp',
        '--model_load_path',
        help='path of a pretrained model to load as initialization'
    )
    parser.add_argument(
        '--force_rebuild_data',
        default=False,
        type=ast.literal_eval,
        help='force to rebuild tfrecord and meta data'
    )
    parser.add_argument(
        '-eub',
        '--eval_use_best',
        default=False,
        type=ast.literal_eval,
        help='use best model when evaluate or not.'
    )
    parser.add_argument(
        '-mrp',
        '--model_resume_path',
        help='path of a the model directory to resume training of'
    )
    parser.add_argument(
        '-cps',
        '--prefix_or_checkpoints',
        help="Comma-separated list of checkpoints to average, "
             "or prefix (e.g., directory) to append to each checkpoint."
    )
    parser.add_argument(
        '-ckptw',
        '--ckpt_weights',
        nargs='+',
        type=float,
        default=None,
        help='list of weights for each ckpts'
    )
    parser.add_argument(
        '-nlc',
        '--num_last_checkpoints',
        help="Averages the last N saved checkpoints."
             " If the checkpoints flag is set, this is ignored."
    )
    parser.add_argument(
        '-rs',
        '--random_seed',
        type=int,
        default=119,
        help='a random seed that is going to be used anywhere there is a call '
             'to a random number generator: data splitting, parameter '
             'initialization and training set shuffling'
    )
    parser.add_argument(
        '-g',
        '--gpus',
        nargs='+',
        type=int,
        default=None,
        help='list of gpus to use'
    )
    parser.add_argument(
        '-gf',
        '--gpu_fraction',
        type=float,
        default=1.0,
        help='fraction of gpu memory to initialize the process with'
    )
    parser.add_argument(
        '-dbg',
        '--debug',
        action='store_true',
        default=False, help='enables debugging mode'
    )
    parser.add_argument(
        '-l',
        '--logging_level',
        default='info',
        help='the level of logging to use',
        choices=['critical', 'error', 'warning', 'info', 'debug']
    )

    args = parser.parse_args(sys_argv)
    # hparams from args
    arg_hparams = Hparams(args.__dict__)
    hparams = Hparams()
    # 1. merge from config file
    if arg_hparams.config_dir is not None and arg_hparams.config_name is not None:
        hparam_file = os.path.join(arg_hparams.config_dir, f'{arg_hparams.config_name}.yml')
        # hparams from config file
        hparams.load_from_config_file(hparam_file)
    # 2. merge hparams from args
    hparams.merge_from_hparams(arg_hparams)
    # 3. merge hparams which is built in transform stage
    # transform_hparams_file = os.path.join(hparams.dataset.data_dir, 'hparams.json')
    # if os.path.exists(transform_hparams_file) and hparams.schedule != TRANSFORM_SCHEDULE:
    #     transform_hparams_dict = load_json(transform_hparams_file)
    #     # update three csv files and workspace directory
    #     hparams.data_train_csv = transform_hparams_dict.get('data_train_csv')
    #     hparams.data_validation_csv = transform_hparams_dict.get('data_validation_csv')
    #     hparams.data_test_csv = transform_hparams_dict.get('data_test_csv')
    #     hparams.workspace_dir = transform_hparams_dict.get('data_validation_csv')
    return hparams
