# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 15:38
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tnew_transformer.py


import os
from tqdm import tqdm
import json
import logging
import numpy as np
import hanlp
from random import random, randrange
from pathlib import Path
from .base_transformer import BaseTransformer
from aispace.datasets import BaseTokenizer
from aispace.utils.io_utils import json_dumps
from aispace.utils.file_utils import default_download_dir, maybe_create_dir
from aispace.utils.io_utils import maybe_download, load_from_file

__all__ = [
    "DuEETriggerTransformer",
    "DuEERoleTransformer",
    "DuEERoleAsQATransformer",
    "DuEERoleReduceLabelTransformer",
    "DuEETriggerAsClassifierTransformer"
]

logger = logging.getLogger(__name__)


@BaseTransformer.register("lstc_2020/DuEE_trigger")
class DuEETriggerTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(DuEETriggerTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

        self.hanlp_tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        self.hanlp_tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
        # self.hannlp_pipeline = hanlp.pipeline() \
        #     .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
        #     .append(tokenizer, output_key='tokens') \
        #     .append(tagger, output_key='part_of_speech_tags')

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")

        self.ner_label_to_id = {l: idx for idx, l in enumerate(self._hparams.dataset.outputs[0].labels)}
        # self.label_to_id = {l: idx for idx, l in enumerate(self._hparams.dataset.outputs[1].labels)}
        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    if len(line_json) == 0: continue
                    feature = self._build_feature_v3(line_json, split, False)
                    if not feature: continue
                    new_line = f"{json_dumps(feature)}\n"
                    ouf.write(new_line)

                    # 数据增强
                    # if split != "train":
                    #     continue
                    # visited = set()
                    # visited.add(sum(feature['input_ids']))
                    # for i in range(3):
                    #     feature = self._build_feature(line_json, split, True)
                    #     if feature:
                    #         if sum(feature['input_ids']) in visited:
                    #             continue
                    #         visited.add(sum(feature['input_ids']))
                    #         new_line = f"{json_dumps(feature)}\n"
                    #         ouf.write(new_line)
        return output_path

    def _build_feature(self, one_json, split="train", data_aug=False):
        text = one_json.get("text")
        id = one_json.get("id")
        event_list = one_json.get("event_list", [])
        event_list.sort(key=lambda s: s.get("trigger_start_index"))
        if len(event_list) == 0:
            return {}

        tokens = []
        labels = []
        pre_start = 0
        event_types = set()
        flag = True
        for event in event_list:
            event_type = event.get("event_type")
            event_types.add(event_type)
            trigger = event.get("trigger")
            trigger_start_index = event.get("trigger_start_index")
            if event_type is None or not trigger or trigger_start_index is None:
                continue

            if trigger_start_index + len(trigger) + 2 >= self.tokenizer.max_len:
                continue

            span_start = trigger_start_index
            span_end = span_start + len(trigger)

            pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
            tokens.extend(pre_tokens)
            labels.extend(["O"] * len(pre_tokens))

            cur_tokens = self.tokenizer.tokenize(trigger)
            if data_aug and split == "train" and random() < 0.3:
                flag = False
                random_len = max(2, randrange(2, len(cur_tokens) + 3))
                cur_tokens = [self.tokenizer.vocab.pad_token] * random_len
            tokens.extend(cur_tokens)
            labels.extend([self._hparams.duee_trigger_ner_labels[f"B-{event_type}"]])
            labels.extend([self._hparams.duee_trigger_ner_labels[f"I-{event_type}"]] * (len(cur_tokens) - 1))

            pre_start = span_end
        cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
        tokens.extend(cur_tokens)
        labels.extend(['O'] * len(cur_tokens))

        # bert lables
        labels = labels[:self.tokenizer.max_len - 2]
        labels = ['O'] + labels + ['O']
        labels += ['O'] * (self.tokenizer.max_len - len(labels))
        # bert base input
        tokens = tokens[:self.tokenizer.max_len - 2]
        tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
        input_ids = self.tokenizer.vocab.transformer(tokens)
        lens = len(input_ids)
        input_ids += [0] * (self.tokenizer.max_len - lens)
        token_type_ids = [0] * self.tokenizer.max_len
        attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

        feature = {
            "id": id,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "ner_labels": labels,
        }
        if data_aug and flag:
            return None

        return feature

    def _build_feature_v2(self, one_json, split="train", data_aug=False):
        """
        ner 分类 joint
        :param one_json:
        :return:
        """
        text = one_json.get("text")
        id = one_json.get("id")
        event_list = one_json.get("event_list", [])
        event_list.sort(key=lambda s: s.get("trigger_start_index"))
        if len(event_list) == 0:
            return {}

        tokens = []
        labels = []
        pre_start = 0
        event_types = set()
        for event in event_list:
            event_type = event.get("event_type")
            event_types.add(event_type)
            trigger = event.get("trigger")
            trigger_start_index = event.get("trigger_start_index")
            if event_type is None or not trigger or trigger_start_index is None:
                continue

            if trigger_start_index + len(trigger) + 2 >= self.tokenizer.max_len:
                continue

            span_start = trigger_start_index
            span_end = span_start + len(trigger)

            pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
            tokens.extend(pre_tokens)
            labels.extend(["O"] * len(pre_tokens))

            cur_tokens = self.tokenizer.tokenize(trigger)
            if data_aug and split == "train" and random() <= 0.1:
                cur_tokens = [self.tokenizer.vocab.mask_token] * len(cur_tokens)
            tokens.extend(cur_tokens)
            labels.extend([self._hparams.duee_trigger_ner_labels[f"B-{event_type}"]])
            labels.extend([self._hparams.duee_trigger_ner_labels[f"I-{event_type}"]] * (len(cur_tokens) - 1))

            pre_start = span_end
        cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
        tokens.extend(cur_tokens)
        labels.extend(['O'] * len(cur_tokens))

        # bert lables
        labels = labels[:self.tokenizer.max_len - 2]
        labels = ['O'] + labels + ['O']
        labels += ['O'] * (self.tokenizer.max_len - len(labels))
        # bert base input
        tokens = tokens[:self.tokenizer.max_len - 2]
        tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
        input_ids = self.tokenizer.vocab.transformer(tokens)
        lens = len(input_ids)
        input_ids += [0] * (self.tokenizer.max_len - lens)
        token_type_ids = [0] * self.tokenizer.max_len
        attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

        event_labels = [0] * len(self.label_to_id)
        for e in event_types:
            event_labels[self.label_to_id[self._hparams.duee_event_type_labels[e]]] = 1
        feature = {
            "id": id,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "ner_labels": labels,
            "event_labels": event_labels
        }

        return feature

    def _build_feature_v3(self, one_json, split="train", data_aug=False):
        """
        增加pos 特征
        :param one_json:
        :param split:
        :param data_aug:
        :return:
        """
        text = one_json.get("text")
        id = one_json.get("id")
        event_list = one_json.get("event_list", [])
        event_list.sort(key=lambda s: s.get("trigger_start_index"))
        if len(event_list) == 0:
            return {}

        tokens = []
        token_pos = []
        labels = []
        flag = True
        tokens_hanlp = self.hanlp_tokenizer(text)
        pos_hanlp = self.hanlp_tagger(tokens_hanlp)
        event_dict = {event.get("trigger_start_index"): event for event in event_list}

        def common_span(word_start, word_end):
            final_start = word_start
            final_end = word_end
            event_type = "O"
            for k, v in event_dict.items():
                cur_word_start = k
                cur_word_end = k + len(v["trigger"])

                tmp_start = max(final_start, cur_word_start)
                tmp_end = min(final_end, cur_word_end)
                if tmp_start >= tmp_end:
                    continue
                final_start = tmp_start
                final_end = tmp_end
                event_type = v['event_type']
            return final_start - word_start, final_end - word_start, event_type

        worker_i = 0

        for i in range(len(tokens_hanlp)):
            cur_word = tokens_hanlp[i]
            cur_pos = pos_hanlp[i]
            span_s, span_e, event_type = common_span(worker_i, worker_i + len(cur_word))
            cur_all_tokens = []

            cur_tokens = self.tokenizer.tokenize(cur_word[: span_s])
            tokens.extend(cur_tokens)
            labels.extend(["O"] * len(cur_tokens))
            cur_all_tokens += cur_tokens

            cur_tokens = self.tokenizer.tokenize(cur_word[span_s: span_e])
            tokens.extend(cur_tokens)
            if event_type != "O":
                labels.extend([self._hparams.duee_trigger_ner_labels[f"B-{event_type}"]])
                labels.extend([self._hparams.duee_trigger_ner_labels[f"I-{event_type}"]] * (len(cur_tokens) - 1))
            else:
                labels.extend(["O"] * len(cur_tokens))
            cur_all_tokens += cur_tokens

            cur_tokens = self.tokenizer.tokenize(cur_word[span_e:])
            tokens.extend(cur_tokens)
            labels.extend(["O"] * len(cur_tokens))
            cur_all_tokens += cur_tokens

            token_pos += [f"B-{cur_pos}"] + [f"I-{cur_pos}"] * (len(cur_all_tokens) - 1)
            worker_i += len(cur_word)

        # pos
        token_pos = token_pos[: self.tokenizer.max_len - 2]
        token_pos = ["O"] + token_pos + ["O"]
        token_pos += ['O'] * (self.tokenizer.max_len - len(token_pos))

        # bert lables
        labels = labels[:self.tokenizer.max_len - 2]
        labels = ['O'] + labels + ['O']
        labels += ['O'] * (self.tokenizer.max_len - len(labels))
        # bert base input
        tokens = tokens[:self.tokenizer.max_len - 2]
        tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
        input_ids = self.tokenizer.vocab.transformer(tokens)
        lens = len(input_ids)
        input_ids += [0] * (self.tokenizer.max_len - lens)
        token_type_ids = [0] * self.tokenizer.max_len
        attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

        feature = {
            "id": id,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "pos": token_pos,
            "ner_labels": labels,
        }
        if data_aug and flag:
            return None

        return feature

    # read labels from file
    def duee_trigger_ner_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        labels = OrderedDict()
        labels["O"] = "O"
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                event_type = one_json.get("event_type")
                id = one_json.get("id")
                labels[f"B-{event_type}"] = f"B-{id}"
                labels[f"I-{event_type}"] = f"I-{id}"
        return labels

    def duee_event_type_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        labels = OrderedDict()
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                event_type = one_json.get("event_type")
                id = one_json.get("id")
                labels[event_type] = id
        return labels

    def hanlp_pos_labels(self, url, name=""):
        from collections import OrderedDict
        tag_vocab = self.hanlp_tagger.tag_vocab.idx_to_token
        labels = OrderedDict()

        labels["O"] = "O"
        for k in tag_vocab:
            labels[f"B-{k}"] = f"B-{k}"
            labels[f"I-{k}"] = f"I-{k}"
        return labels


@BaseTransformer.register("lstc_2020/DuEE_trigger_as_classifier")
class DuEETriggerAsClassifierTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(DuEETriggerAsClassifierTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")

        self.label_to_id = {l: idx for idx, l in enumerate(self._hparams.dataset.outputs[0].labels)}
        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    if len(line_json) == 0: continue
                    feature = self._build_feature(line_json)
                    if len(feature) == 0: continue
                    new_line = f"{json_dumps(feature)}\n"
                    ouf.write(new_line)
        return output_path

    def _build_feature(self, one_json):
        text = one_json.get("text")
        id = one_json.get("id")
        event_list = one_json.get("event_list", [])
        if len(event_list) == 0:
            return {}
        event_types = set()
        for event in event_list:
            event_type = event.get("event_type")
            event_types.add(event_type)

        input_ids, token_type_ids, attention_mask = self.tokenizer.encode(text)

        event_labels = [0] * len(self.label_to_id)
        for e in event_types:
            event_labels[self.label_to_id[self._hparams.duee_event_type_labels[e]]] = 1
        feature = {
            "id": id,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "event_labels": event_labels
        }

        return feature

    # read labels from file
    def duee_event_type_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        labels = OrderedDict()
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                event_type = one_json.get("event_type")
                id = one_json.get("id")
                labels[event_type] = id
        return labels

@BaseTransformer.register("lstc_2020/DuEE_role")
class DuEERoleTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(DuEERoleTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")

        # read schema and build entity_type to role mask
        schema = {}
        schema_raw = {}
        with open(self.schema_file, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                s_event_type = one_json.get("event_type")
                s_roles = one_json.get("role_list")
                schema[s_event_type] = [f"B-{r['role']}" for r in s_roles] + [f"I-{r['role']}" for r in s_roles]
                schema_raw[s_event_type] = "-".join([r['role'] for r in s_roles])

        label2ids = {l: idx for idx, l in enumerate(list(self._hparams.duee_role_ner_labels.keys()))}

        self.trigger_mapping, self.role_mapping = {}, {}
        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    if len(line_json) == 0: continue
                    # features = self._build_featureV3(line_json, schema, label2ids)
                    # features = self._build_featureV4(line_json, schema, label2ids, split)
                    features = self._build_featureV5(line_json, schema, schema_raw, label2ids, split)
                    for feature in features:
                        new_line = f"{json_dumps(feature)}\n"
                        ouf.write(new_line)
        return output_path

    def _build_feature(self, one_json):
        text = one_json.get("text")
        id = one_json.get("id")
        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            trigger = event.get("trigger")
            trigger_start_index = event.get("trigger_start_index")
            arguments = event.get("arguments", [])
            arguments.sort(key=lambda s: s.get("argument_start_index"))
            if len(arguments) == 0:
                return {}

            pre_trigger_tokens = self.tokenizer.tokenize(text[0: trigger_start_index])
            trigger_tokens = self.tokenizer.tokenize(trigger)
            trigger_span = [len(pre_trigger_tokens) + 1, len(pre_trigger_tokens) + len(trigger_tokens) + 1]

            tokens = []
            labels = []
            pre_start = 0
            for one_argument in arguments:
                role = one_argument.get("role")
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if role is None or not argument or argument_start_index is None:
                    continue

                span_start = argument_start_index
                span_end = span_start + len(argument)

                pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                cur_tokens = self.tokenizer.tokenize(argument)
                tokens.extend(cur_tokens)
                labels.extend([self._hparams.duee_role_ner_labels[f"B-{role}"]])
                labels.extend([self._hparams.duee_role_ner_labels[f"I-{role}"]] * (len(cur_tokens) - 1))

                pre_start = span_end
            cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            # append event_type
            cur_tokens = self.tokenizer.tokenize(event_type)
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            # bert lables
            labels = labels[:self.tokenizer.max_len - 2]
            labels = ['O'] + labels + ['O']
            labels += ['O'] * (self.tokenizer.max_len - len(labels))
            # bert base input
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
            input_ids = self.tokenizer.vocab.transformer(tokens)
            lens = len(input_ids)
            input_ids += [0] * (self.tokenizer.max_len - lens)
            token_type_ids = [0] * self.tokenizer.max_len
            attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

            feature = {
                "id": id,
                "trigger_span": trigger_span,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

            yield feature

    def _build_featureV2(self, one_json, schema, label2ids):
        text = one_json.get("text")
        id = one_json.get("id")
        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            trigger = event.get("trigger")
            trigger_start_index = event.get("trigger_start_index")
            if trigger_start_index + len(trigger) + 2 >= self.tokenizer.max_len:
                continue
            arguments = event.get("arguments", [])
            arguments.sort(key=lambda s: s.get("argument_start_index"))
            if len(arguments) == 0:
                return {}

            pre_trigger_tokens = self.tokenizer.tokenize(text[0: trigger_start_index])
            trigger_tokens = self.tokenizer.tokenize(trigger)
            trigger_span = [len(pre_trigger_tokens) + 1, len(pre_trigger_tokens) + len(trigger_tokens)]

            tokens = []
            labels = []
            pre_start = 0
            for one_argument in arguments:
                role = one_argument.get("role")
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if role is None or not argument or argument_start_index is None:
                    continue

                span_start = argument_start_index
                span_end = span_start + len(argument)

                pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                cur_tokens = self.tokenizer.tokenize(argument)
                tokens.extend(cur_tokens)
                labels.extend([self._hparams.duee_role_ner_labels[f"B-{role}"]])
                labels.extend([self._hparams.duee_role_ner_labels[f"I-{role}"]] * (len(cur_tokens) - 1))

                pre_start = span_end

            cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            # bert lables
            labels = labels[:self.tokenizer.max_len - 2]
            labels = ['O'] + labels + ['O']
            labels += ['O'] * (self.tokenizer.max_len - len(labels))

            # trigger bert labels
            # trigger_labels = trigger_labels[:self.tokenizer.max_len - 2]
            # trigger_labels = ['O'] + trigger_labels + ['O']
            # trigger_labels += ['O'] * (self.tokenizer.max_len - len(trigger_labels))

            # bert base input
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
            input_ids = self.tokenizer.vocab.transformer(tokens)
            lens = len(input_ids)
            input_ids += [0] * (self.tokenizer.max_len - lens)
            token_type_ids = [0] * self.tokenizer.max_len
            attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

            # append event_type
            type_max_len = 15
            type_tokens = self.tokenizer.tokenize(event_type)
            type_tokens = [self.tokenizer.vocab.cls_token] + type_tokens + [self.tokenizer.vocab.sep_token]
            type_tokens = type_tokens[: type_max_len]
            type_input_ids = self.tokenizer.vocab.transformer(type_tokens)
            type_lens = len(type_input_ids)
            type_input_ids += [0] * (type_max_len - type_lens)
            type_token_type_ids = [0] * type_max_len
            type_attention_mask = [1] * type_lens + [0] * (type_max_len - type_lens)

            # label mask
            mask = [0] * len(self._hparams.duee_role_ner_labels)
            mask[0] = 1
            for r in schema[event_type]:
                mask[label2ids[r]] = 1

            # 相对trigger 的相对位置特征
            pre_rel_pos = [trigger_span[0] - i for i in range(trigger_span[0])] + \
                          [0] * (self.tokenizer.max_len - trigger_span[0])
            pos_rel_pos = [0] * (trigger_span[1] + 1) + \
                          [i - trigger_span[1] for i in range(trigger_span[1] + 1, self.tokenizer.max_len)]

            pre_rel_pos = pre_rel_pos[:self.tokenizer.max_len]
            pos_rel_pos = pos_rel_pos[:self.tokenizer.max_len]

            feature = {
                "id": id,
                "trigger_span": trigger_span,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "type_input_ids": type_input_ids,
                "type_token_type_ids": type_token_type_ids,
                "type_attention_mask": type_attention_mask,
                "pre_rel_pos": pre_rel_pos,
                "pos_rel_pos": pos_rel_pos,
                "label_mask": mask,
                # "trigger_labels": trigger_labels,
                "labels": labels,
            }

            yield feature

    def _build_featureV3(self, one_json, schema, label2ids):
        text = one_json.get("text")
        id = one_json.get("id")

        # merge same event_type
        event_list_combined = {}
        role_arg_dict = {}
        for event in one_json.get("event_list"):
            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                if role not in role_arg_dict:
                    role_arg_dict[role] = []
                role_arg_dict[role].append(one_argument)

        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            if event_type not in event_list_combined:
                event_list_combined[event_type] = []
            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                event_list_combined[event_type].extend(role_arg_dict.get(role, []))

        for event_type, old_arguments in event_list_combined.items():
            old_arguments.sort(key=lambda s: s.get("argument_start_index"))
            # 去重
            i = 0
            arguments = []
            for i, arg in enumerate(old_arguments):
                if i == 0:
                    arguments.append(arg)
                    continue
                if arguments[-1].get("argument") == old_arguments[i].get("argument") and \
                        arguments[-1].get("argument_start_index") == old_arguments[i].get("argument_start_index"):
                    continue
                arguments.append(arg)

            if len(arguments) == 0:
                return {}

            tokens = []
            labels = []
            pre_start = 0
            for one_argument in arguments:
                role = one_argument.get("role")
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if role is None or not argument or argument_start_index is None:
                    continue

                span_start = argument_start_index
                span_end = span_start + len(argument)

                pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                cur_tokens = self.tokenizer.tokenize(argument)
                tokens.extend(cur_tokens)
                labels.extend([self._hparams.duee_role_ner_labels[f"B-{role}"]])
                labels.extend([self._hparams.duee_role_ner_labels[f"I-{role}"]] * (len(cur_tokens) - 1))

                pre_start = span_end

            cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            cur_tokens = [self.tokenizer.vocab.sep_token]
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))
            first_seq_len = len(tokens) + 1

            # append event_type
            cur_tokens = self.tokenizer.tokenize(event_type)
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))
            second_seq_len = len(cur_tokens) + 1

            if first_seq_len + second_seq_len > self.tokenizer.max_len:
                continue

            # bert lables
            labels = labels[:self.tokenizer.max_len - 2]
            labels = ['O'] + labels + ['O']
            labels += ['O'] * (self.tokenizer.max_len - len(labels))

            # bert base input
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
            input_ids = self.tokenizer.vocab.transformer(tokens)
            lens = len(input_ids)
            input_ids += [0] * (self.tokenizer.max_len - lens)
            token_type_ids = [0] * first_seq_len + \
                             [1] * second_seq_len + \
                             [0] * (self.tokenizer.max_len - first_seq_len - second_seq_len)
            attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

            # label mask
            mask = [0] * len(self._hparams.duee_role_ner_labels)
            mask[0] = 1
            for r in schema[event_type]:
                mask[label2ids[r]] = 1

            feature = {
                "id": id,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_mask": mask,
                "labels": labels,
            }

            yield feature

    def _build_featureV4(self, one_json, schema, label2ids, split="train"):
        """
        随机替换role
        :param one_json:
        :param schema:
        :param label2ids:
        :param split:
        :return:
        """
        text = one_json.get("text")
        id = one_json.get("id")

        # merge same event_type
        event_list_combined = {}
        role_arg_dict = {}
        for event in one_json.get("event_list"):
            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                if role not in role_arg_dict:
                    role_arg_dict[role] = []
                role_arg_dict[role].append(one_argument)

        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            if event_type not in event_list_combined:
                event_list_combined[event_type] = []
            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                event_list_combined[event_type].extend(role_arg_dict.get(role, []))

        for event_type, old_arguments in event_list_combined.items():
            old_arguments.sort(key=lambda s: s.get("argument_start_index"))
            # 去重
            i = 0
            arguments = []
            for i, arg in enumerate(old_arguments):
                if i == 0:
                    arguments.append(arg)
                    continue
                if arguments[-1].get("argument") == old_arguments[i].get("argument") and \
                        arguments[-1].get("argument_start_index") == old_arguments[i].get("argument_start_index"):
                    continue
                arguments.append(arg)

            if len(arguments) == 0:
                return {}

            tokens = []
            labels = []
            pre_start = 0
            for one_argument in arguments:
                role = one_argument.get("role")
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if role is None or not argument or argument_start_index is None:
                    continue

                span_start = argument_start_index
                span_end = span_start + len(argument)
                pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                cur_tokens = self.tokenizer.tokenize(argument)
                if split == "train" and random() <= 0.15:
                    cur_tokens = [self.tokenizer.vocab.mask_token] * len(cur_tokens)
                tokens.extend(cur_tokens)
                labels.extend([self._hparams.duee_role_ner_labels[f"B-{role}"]])
                labels.extend([self._hparams.duee_role_ner_labels[f"I-{role}"]] * (len(cur_tokens) - 1))

                pre_start = span_end

            cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            cur_tokens = [self.tokenizer.vocab.sep_token]
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))
            first_seq_len = len(tokens) + 1

            # append event_type
            cur_tokens = self.tokenizer.tokenize(event_type)
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))
            second_seq_len = len(cur_tokens) + 1

            if first_seq_len + second_seq_len > self.tokenizer.max_len:
                continue

            # bert lables
            labels = labels[:self.tokenizer.max_len - 2]
            labels = ['O'] + labels + ['O']
            labels += ['O'] * (self.tokenizer.max_len - len(labels))

            # bert base input
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
            input_ids = self.tokenizer.vocab.transformer(tokens)
            lens = len(input_ids)
            input_ids += [0] * (self.tokenizer.max_len - lens)
            token_type_ids = [0] * first_seq_len + \
                             [1] * second_seq_len + \
                             [0] * (self.tokenizer.max_len - first_seq_len - second_seq_len)
            attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

            # label mask
            mask = [0] * len(self._hparams.duee_role_ner_labels)
            mask[0] = 1
            for r in schema[event_type]:
                mask[label2ids[r]] = 1

            feature = {
                "id": id,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_mask": mask,
                "labels": labels,
            }

            yield feature

    def _build_featureV5(self, one_json, schema, schema_raw, label2ids, split="train"):
        """
        :param one_json:
        :param schema:
        :param label2ids:
        :param split:
        :return:
        """
        text = one_json.get("text")
        id = one_json.get("id")

        # merge same event_type
        event_list_combined = {}
        role_arg_dict = {}
        for event in one_json.get("event_list"):
            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                if role not in role_arg_dict:
                    role_arg_dict[role] = []
                role_arg_dict[role].append(one_argument)

        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            if event_type not in event_list_combined:
                event_list_combined[event_type] = []
            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                event_list_combined[event_type].extend(role_arg_dict.get(role, []))

        for event_type, old_arguments in event_list_combined.items():
            old_arguments.sort(key=lambda s: s.get("argument_start_index"))
            # 去重
            i = 0
            arguments = []
            for i, arg in enumerate(old_arguments):
                if i == 0:
                    arguments.append(arg)
                    continue
                if arguments[-1].get("argument") == old_arguments[i].get("argument") and \
                        arguments[-1].get("argument_start_index") == old_arguments[i].get("argument_start_index"):
                    continue
                arguments.append(arg)

            if len(arguments) == 0:
                return {}

            tokens = []
            labels = []
            pre_start = 0
            for one_argument in arguments:
                role = one_argument.get("role")
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if role is None or not argument or argument_start_index is None:
                    continue

                span_start = argument_start_index
                span_end = span_start + len(argument)
                pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                cur_tokens = self.tokenizer.tokenize(argument)
                tokens.extend(cur_tokens)
                labels.extend([self._hparams.duee_role_ner_labels[f"B-{role}"]])
                labels.extend([self._hparams.duee_role_ner_labels[f"I-{role}"]] * (len(cur_tokens) - 1))

                pre_start = span_end

            cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))
            first_seq_len = len(tokens)

            # append event_type
            event_type_tokens = self.tokenizer.tokenize(f"{event_type}")
            second_seq_len = len(event_type_tokens)

            if first_seq_len + second_seq_len + 3 > self.tokenizer.max_len:
                cur_len = first_seq_len + second_seq_len + 3 - self.tokenizer.max_len
                tokens = tokens[0: -1 * cur_len]
                labels = labels[0: -1 * cur_len]
                first_seq_len = len(tokens)

            cur_tokens = [self.tokenizer.vocab.sep_token]
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            tokens.extend(event_type_tokens)
            labels.extend(['O'] * len(event_type_tokens))

            # bert lables
            labels = labels[:self.tokenizer.max_len - 2]
            labels = ['O'] + labels + ['O']
            labels += ['O'] * (self.tokenizer.max_len - len(labels))

            # bert base input
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
            input_ids = self.tokenizer.vocab.transformer(tokens)
            lens = len(input_ids)
            input_ids += [0] * (self.tokenizer.max_len - lens)
            token_type_ids = [0] * (first_seq_len + 2) + \
                             [1] * (second_seq_len + 1) + \
                             [0] * (self.tokenizer.max_len - first_seq_len - second_seq_len - 3)
            attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

            # label mask
            mask = [0] * len(self._hparams.duee_role_ner_labels)
            mask[0] = 1
            for r in schema[event_type]:
                mask[label2ids[r]] = 1

            feature = {
                "id": id,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_mask": mask,
                "labels": labels,
            }

            yield feature

    # read labels from file
    def duee_role_ner_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        self.schema_file = file_path
        labels = OrderedDict()
        labels["O"] = "O"
        visited = set()
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                role_list = one_json.get("role_list")
                for role in role_list:
                    tmp = role["role"]
                    if tmp in visited:
                        continue
                    cur_num = len(labels)
                    visited.add(tmp)
                    labels[f"B-{tmp}"] = f"B-{cur_num}"
                    labels[f"I-{tmp}"] = f"I-{cur_num}"
        return labels

@BaseTransformer.register("lstc_2020/DuEE_role_reduce_label")
class DuEERoleReduceLabelTransformer(BaseTransformer):
    """3.0.0"""
    def __init__(self, hparams, **kwargs):
        super(DuEERoleReduceLabelTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

        self.label_reduce_map = {}
        label_reduce_file = "/search/data1/yyk/workspace/projects/AiSpace/data/reduce_labels.txt"
        with open(label_reduce_file, "r", encoding="utf-8") as inf:
            for line in inf:
                pieces = line.split()
                norm_name = pieces[0]
                for t in pieces[1:]:
                    self.label_reduce_map[t] = norm_name

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")

        # read schema and build entity_type to role mask
        schema = {}
        schema_raw = {}
        with open(self.schema_file, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                s_event_type = one_json.get("event_type")
                s_roles = one_json.get("role_list")
                schema[s_event_type] = [f"B-{self.label_reduce_map.get(r['role'], r['role'])}" for r in s_roles] + [f"I-{self.label_reduce_map.get(r['role'], r['role'])}" for r in s_roles]
                schema_raw[s_event_type] = "-".join([self.label_reduce_map.get(r['role'], r['role']) for r in s_roles])

        label2ids = {l: idx for idx, l in enumerate(list(self._hparams.duee_role_ner_labels.keys()))}

        self.trigger_mapping, self.role_mapping = {}, {}
        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    if len(line_json) == 0: continue
                    # features = self._build_featureV3(line_json, schema, label2ids)
                    # features = self._build_featureV4(line_json, schema, label2ids, split)
                    features = self._build_feature(line_json, schema, schema_raw, label2ids, split)
                    for feature in features:
                        new_line = f"{json_dumps(feature)}\n"
                        ouf.write(new_line)
        return output_path

    def _build_feature(self, one_json, schema, schema_raw, label2ids, split="train"):
        """
        使用类别规约
        :param one_json:
        :param schema:
        :param label2ids:
        :param split:
        :return:
        """
        text = one_json.get("text")
        id = one_json.get("id")

        event_args = {}
        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            if event_type not in event_args:
                event_args[event_type] = []
            arguments = event.get("arguments", [])
            event_args[event_type].extend(arguments)

        new_event_args = {}
        for event_type, args in event_args.items():
            visited = set()
            n_a = []
            for arg in args:
                erk = f"{arg['argument']}{arg['argument_start_index']}"
                if erk in visited:
                    continue
                visited.add(erk)
                n_a.append(arg)
            new_event_args[event_type] = n_a

        for event_type, arguments in new_event_args.items():
            arguments.sort(key=lambda s: s.get("argument_start_index"))

            if len(arguments) == 0:
                return {}

            tokens = []
            labels = []
            pre_start = 0
            for one_argument in arguments:
                role = one_argument.get("role")
                role = self.label_reduce_map.get(role, role)
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if role is None or not argument or argument_start_index is None:
                    continue

                span_start = argument_start_index
                span_end = span_start + len(argument)
                pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                cur_tokens = self.tokenizer.tokenize(argument)
                tokens.extend(cur_tokens)
                labels.extend([self._hparams.duee_role_ner_labels[f"B-{role}"]])
                labels.extend([self._hparams.duee_role_ner_labels[f"I-{role}"]] * (len(cur_tokens) - 1))

                pre_start = span_end

            cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))
            first_seq_len = len(tokens)

            # append event_type
            event_type_tokens = self.tokenizer.tokenize(f"{event_type}")
            second_seq_len = len(event_type_tokens)

            if first_seq_len + second_seq_len + 3 > self.tokenizer.max_len:
                cur_len = first_seq_len + second_seq_len + 3 - self.tokenizer.max_len
                tokens = tokens[0: -1 * cur_len]
                labels = labels[0: -1 * cur_len]
                first_seq_len = len(tokens)
            if len(set(labels)) == 1:
                continue

            cur_tokens = [self.tokenizer.vocab.sep_token]
            tokens.extend(cur_tokens)
            labels.extend(['O'] * len(cur_tokens))

            tokens.extend(event_type_tokens)
            labels.extend(['O'] * len(event_type_tokens))

            # bert lables
            labels = labels[:self.tokenizer.max_len - 2]
            labels = ['O'] + labels + ['O']
            labels += ['O'] * (self.tokenizer.max_len - len(labels))

            # bert base input
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
            input_ids = self.tokenizer.vocab.transformer(tokens)
            lens = len(input_ids)
            input_ids += [0] * (self.tokenizer.max_len - lens)
            token_type_ids = [0] * (first_seq_len + 2) + \
                             [1] * (second_seq_len + 1) + \
                             [0] * (self.tokenizer.max_len - first_seq_len - second_seq_len - 3)
            attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

            # label mask
            mask = [0] * len(self._hparams.duee_role_ner_labels)
            mask[0] = 1
            for r in schema[event_type]:
                mask[label2ids[r]] = 1

            feature = {
                "id": id,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_mask": mask,
                "labels": labels,
            }

            yield feature

    # read labels from file
    def duee_role_ner_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        self.schema_file = file_path
        labels = OrderedDict()
        labels["O"] = "O"
        visited = set()
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                role_list = one_json.get("role_list")
                for role in role_list:
                    role = role['role']
                    tmp = self.label_reduce_map.get(role, role)
                    if tmp in visited:
                        continue
                    visited.add(tmp)
                    cur_num = len(visited)
                    labels[f"B-{tmp}"] = f"B-{cur_num}"
                    labels[f"I-{tmp}"] = f"I-{cur_num}"
        return labels

@BaseTransformer.register("lstc_2020/DuEE_role_as_qa")
class DuEERoleAsQATransformer(BaseTransformer):
    "2.0.0"
    def __init__(self, hparams, **kwargs):
        super(DuEERoleAsQATransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")

        # read schema and build entity_type to role mask
        schema = {}
        self.schema_file = "/root/aispace_data/duee/event_schema/event_schema.json"
        with open(self.schema_file, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                s_event_type = one_json.get("event_type")
                s_roles = one_json.get("role_list")
                schema[s_event_type] = [r['role'] for r in s_roles]

        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    if len(line_json) == 0: continue
                    features = self._build_feature_v2(line_json, schema)
                    for feature in features:
                        new_line = f"{json_dumps(feature)}\n"
                        ouf.write(new_line)
        return output_path

    # 单个答案
    def _build_feature(self, one_json, schema):
        text = one_json.get("text")
        id = one_json.get("id")
        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            trigger = event.get("trigger")
            trigger_start_index = event.get("trigger_start_index")
            arguments = event.get("arguments", [])
            arguments.sort(key=lambda s: s.get("argument_start_index"))
            if len(arguments) == 0:
                return {}

            arguments_have_answer = {one_argument.get("role"): one_argument for one_argument in arguments}
            for role_name in schema[event_type]:
                query_token = f"{event_type}-{role_name}"
                if role_name in arguments_have_answer:
                    one_argument = arguments_have_answer[role_name]
                    role = one_argument.get("role")
                    argument = one_argument.get("argument")
                    argument_start_index = one_argument.get("argument_start_index")

                    token1 = self.tokenizer.tokenize(text[0: argument_start_index])
                    token2 = self.tokenizer.tokenize(text[argument_start_index: argument_start_index + len(argument)])
                    token3 = self.tokenizer.tokenize(text[argument_start_index + len(argument):])
                    token4 = self.tokenizer.tokenize(query_token)
                    if len(token2 + token4) + 3 >= self.tokenizer.max_len:
                        continue

                    if len(token1 + token2 + token3 + token4) + 3 > self.tokenizer.max_len:
                        last_len = (self.tokenizer.max_len - len(token2 + token4) - 3) // 2
                        if len(token1) >= last_len and len(token3) >= last_len:
                            token1 = token1[-1 * last_len:]
                            token3 = token3[:last_len]
                        elif len(token1) >= last_len > len(token3):
                            token1 = token1[-1 * (last_len * 2 - (len(token3) // 2) - 1):]
                            token3 = token3[: len(token3) // 2]
                        elif len(token1) < last_len <= len(token3):
                            token3 = token3[:(last_len * 2 - (len(token1) // 2) - 1)]
                            token1 = token1[-1 * (len(token1) // 2):]
                        else:
                            continue

                    tokens = [self.tokenizer.vocab.cls_token] + token1 + token2 + token3 + [
                        self.tokenizer.vocab.sep_token] + token4 + [self.tokenizer.vocab.sep_token]
                    mask_len = len(tokens)
                    tokens += [self.tokenizer.vocab.pad_token] * (self.tokenizer.max_len - len(tokens))
                    input_ids = self.tokenizer.vocab.transformer(tokens)
                    token_type_ids = [0] * (len(token1 + token2 + token3) + 2) + [1] * (len(token4) + 1) + [0] * (
                                self.tokenizer.max_len - mask_len)
                    attention_mask = [1] * (len(token1 + token2 + token3 + token4) + 3) + [0] * (
                                self.tokenizer.max_len - mask_len)

                    start_positions = [0] * self.tokenizer.max_len
                    start_positions[len(token1) + 1] = 1
                    end_positions = [0] * self.tokenizer.max_len
                    end_positions[len(token1) + len(token2)] = 1
                    is_impossible = [0, 1]
                else:
                    input_ids, token_type_ids, attention_mask = self.tokenizer.encode(text, query_token)
                    start_positions = [0] * self.tokenizer.max_len
                    start_positions[0] = 1
                    end_positions = [0] * self.tokenizer.max_len
                    end_positions[0] = 1
                    is_impossible = [1, 0]

                feature = {
                    "id": id,
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                    "is_impossible": is_impossible
                }

                yield feature

    # 多个答案
    def _build_feature_v2(self, one_json, schema):
        text = one_json.get("text")
        id = one_json.get("id")

        event_roles = {}
        event_types = set()
        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            event_types.add(event_type)
            arguments = event.get("arguments", [])
            arguments.sort(key=lambda s: s.get("argument_start_index"))
            if len(arguments) == 0:
                return {}
            for argument in arguments:
                event_type_role_key = (event_type, argument['role'])
                if event_type_role_key not in event_roles:
                    event_roles[event_type_role_key] = []
                event_roles[event_type_role_key].append(argument)

        new_event_roles = {}
        for k, v in event_roles.items():
            n_v = []
            visited = set()
            for i, t in enumerate(v):
                tt = t['argument'] + str(t["argument_start_index"])
                if tt in visited:
                    continue
                visited.add(tt)
                n_v.append(t)
            new_event_roles[k] = n_v

        for event_type in event_types:
            for role_name in schema[event_type]:
                query_token_str = f"{event_type}-{role_name}"

                if (event_type, role_name) in new_event_roles:
                    arguments = new_event_roles[(event_type, role_name)]
                    arguments.sort(key=lambda s: s.get("argument_start_index"))
                    tokens = []
                    start_positions, end_positions = [], []
                    pre_start = 0
                    for one_argument in arguments:
                        role = one_argument.get("role")
                        argument = one_argument.get("argument")
                        argument_start_index = one_argument.get("argument_start_index")
                        if role is None or not argument or argument_start_index is None:
                            continue

                        span_start = argument_start_index
                        span_end = span_start + len(argument)
                        pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
                        tokens.extend(pre_tokens)
                        start_positions += [0] * len(pre_tokens)
                        end_positions += [0] * len(pre_tokens)

                        cur_tokens = self.tokenizer.tokenize(argument)
                        tokens.extend(cur_tokens)
                        start_positions += [1] + [0] * (len(cur_tokens) - 1)
                        end_positions += [0] * (len(cur_tokens) - 1) + [1]

                        pre_start = span_end

                    cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
                    tokens.extend(cur_tokens)
                    start_positions += [0] * len(cur_tokens)
                    end_positions += [0] * len(cur_tokens)
                    first_seq_len = len(tokens)

                    # append event_type
                    query_tokens = self.tokenizer.tokenize(query_token_str)
                    second_seq_len = len(query_tokens)

                    if first_seq_len + second_seq_len + 3 > self.tokenizer.max_len:
                        cur_len = first_seq_len + second_seq_len + 3 - self.tokenizer.max_len
                        tokens = tokens[0: -1 * cur_len]
                        start_positions = start_positions[0: -1 * cur_len]
                        end_positions = end_positions[0: -1 * cur_len]
                        first_seq_len = len(tokens)

                    start_positions = [0] + start_positions
                    start_positions += [0] * (self.tokenizer.max_len - len(start_positions))
                    end_positions = [0] + end_positions
                    end_positions += [0] * (self.tokenizer.max_len - len(end_positions))

                    tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token] + \
                             query_tokens + [self.tokenizer.vocab.sep_token]
                    t_len = len(tokens)
                    tokens += [self.tokenizer.vocab.pad_token] * (self.tokenizer.max_len - len(tokens))
                    input_ids = self.tokenizer.vocab.transformer(tokens)
                    token_type_ids = [0] * (first_seq_len + 2) + \
                                     [1] * (second_seq_len + 1) + \
                                     [0] * (self.tokenizer.max_len - first_seq_len - second_seq_len - 3)
                    attention_mask = [1] * t_len + [0] * (self.tokenizer.max_len - t_len)
                    if sum(start_positions) == sum(end_positions) == 0:
                        start_positions = [0] * self.tokenizer.max_len
                        end_positions = [0] * self.tokenizer.max_len
                        start_positions[0] = 1
                        end_positions[0] = 1
                        is_impossible = [1, 0]
                    else:
                        is_impossible = [0, 1]
                else:
                    start_positions = [0] * self.tokenizer.max_len
                    end_positions = [0] * self.tokenizer.max_len
                    input_ids, token_type_ids, attention_mask = self.tokenizer.encode(text, query_token_str)
                    start_positions[0] = 1
                    end_positions[0] = 1
                    is_impossible = [1, 0]

                feature = {
                    "id": id,
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                    "is_impossible": is_impossible
                }

                yield feature

@BaseTransformer.register("lstc_2020/DuEE_joint")
class DuEEJointTransformer(BaseTransformer):
    """
    1）trigger 抽取、事件类型确定；2）role抽取；3）trigger 和 role 将关系确定 联合训练

    """
    def __init__(self, hparams, **kwargs):
        super(DuEEJointTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")

        # read schema and build entity_type to role mask
        schema = {}
        schema_raw = {}
        with open(self.schema_file, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                s_event_type = one_json.get("event_type")
                s_roles = one_json.get("role_list")
                schema[s_event_type] = [f"B-{r['role']}" for r in s_roles] + [f"I-{r['role']}" for r in s_roles]
                schema_raw[s_event_type] = "-".join([r['role'] for r in s_roles])

        label2ids = {l: idx for idx, l in enumerate(list(self._hparams.duee_role_ner_labels.keys()))}

        self.trigger_mapping, self.role_mapping = {}, {}
        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    if len(line_json) == 0: continue
                    features = self._build_feature(line_json, schema, schema_raw, label2ids, split)
                    new_line = f"{json_dumps(features)}\n"
                    ouf.write(new_line)
        return output_path

    def _build_feature(self, one_json, schema, schema_raw, label2ids, split="train"):
        """
        :param one_json:
        :param schema:
        :param label2ids:
        :param split:
        :return:
        """
        text = one_json.get("text")
        id = one_json.get("id")

        mentions = {}
        for event in one_json.get("event_list"):
            event_type = event.get("event_type")
            trigger = event.get("trigger")
            trigger_start_index = event.get("trigger_start_index")

            if (trigger, trigger_start_index) not in mentions:
                mentions[(trigger, trigger_start_index)] = {
                    "text": trigger,
                    "span_start": trigger_start_index,
                    "span_end": trigger_start_index + len(trigger),
                    "event_type": event_type,
                    "other": set()
                }

            arguments = event.get("arguments", [])
            for one_argument in arguments:
                role = one_argument.get("role")
                argument = one_argument.get("argument")
                argument_start_index = one_argument.get("argument_start_index")
                if (argument, argument_start_index) not in mentions:
                    mentions[(argument, argument_start_index)] = {
                        "text": argument,
                        "span_start": argument_start_index,
                        "span_end": argument_start_index + len(argument),
                        "role_type": role,
                        "other": {(trigger, trigger_start_index)}
                    }
                else:
                    mentions[(argument, argument_start_index)]['other'].add((trigger, trigger_start_index))
                mentions[(trigger, trigger_start_index)]['other'].add((argument, argument_start_index))

        trigger_labels = []
        role_labels = []
        tokens = []
        pre_start = 0
        mvs = list(mentions.values())
        mvs.sort(key=lambda s: s['span_start'])
        for mention in mvs:
            span_start, span_end = mention['span_start'], mention['span_end']

            pre_tokens = self.tokenizer.tokenize(text[pre_start: span_start])
            tokens.extend(pre_tokens)
            trigger_labels.extend(["O"] * len(pre_tokens))
            role_labels.extend(["O"] * len(pre_tokens))

            cur_tokens = self.tokenizer.tokenize(mention['text'])
            if "event_type" in mention:
                trigger_labels.extend([self._hparams.duee_trigger_ner_labels[f"B-{mention['event_type']}"]])
                trigger_labels.extend([self._hparams.duee_trigger_ner_labels[f"I-{mention['event_type']}"]] * (len(cur_tokens) - 1))
                role_labels.extend(['O'] * len(cur_tokens))
            else:
                role_labels.extend([self._hparams.duee_role_ner_labels[f"B-{mention['role_type']}"]])
                role_labels.extend(
                    [self._hparams.duee_role_ner_labels[f"I-{mention['role_type']}"]] * (len(cur_tokens) - 1))
                trigger_labels.extend(['O'] * len(cur_tokens))
            mention['token_idx'] = len(tokens) + 1
            tokens.extend(cur_tokens)
            pre_start = span_end

        cur_tokens = self.tokenizer.tokenize(text[pre_start: len(text)])
        tokens.extend(cur_tokens)
        trigger_labels.extend(['O'] * len(cur_tokens))
        role_labels.extend(['O'] * len(cur_tokens))

        # bert lables
        trigger_labels = trigger_labels[:self.tokenizer.max_len - 2]
        trigger_labels = ['O'] + trigger_labels + ['O']
        trigger_labels += ['O'] * (self.tokenizer.max_len - len(trigger_labels))

        role_labels = role_labels[:self.tokenizer.max_len - 2]
        role_labels = ['O'] + role_labels + ['O']
        role_labels += ['O'] * (self.tokenizer.max_len - len(role_labels))
        # bert base input
        tokens = tokens[:self.tokenizer.max_len - 2]
        tokens = [self.tokenizer.vocab.cls_token] + tokens + [self.tokenizer.vocab.sep_token]
        input_ids = self.tokenizer.vocab.transformer(tokens)
        lens = len(input_ids)
        input_ids += [0] * (self.tokenizer.max_len - lens)
        token_type_ids = [0] * self.tokenizer.max_len
        attention_mask = [1] * lens + [0] * (self.tokenizer.max_len - lens)

        relation_labels = np.zeros(shape=(self.tokenizer.max_len, self.tokenizer.max_len), dtype=np.int)

        for mention in mentions.values():
            i = mention['token_idx']
            if i >= self.tokenizer.max_len:
                continue
            for other_k in mention['other']:
                j = mentions[other_k]['token_idx']
                if j >= self.tokenizer.max_len:
                    continue
                relation_labels[i, j] = 1
                relation_labels[j, i] = 1

        feature = {
            "id": id,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "trigger_labels": trigger_labels,
            "role_labels": role_labels,
            "relation_labels": relation_labels.flatten().tolist()
        }

        return feature


    # read labels from file
    def duee_trigger_ner_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        labels = OrderedDict()
        labels["O"] = "O"
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                event_type = one_json.get("event_type")
                id = one_json.get("id")
                labels[f"B-{event_type}"] = f"B-{id}"
                labels[f"I-{event_type}"] = f"I-{id}"
        return labels

    # read labels from file
    def duee_role_ner_labels(self, url, name=""):
        from collections import OrderedDict
        if url.startswith("http"):
            filename = "event_schema/event_schema.json"
            cache_path = default_download_dir(name)
            file_path = cache_path / filename
            if not file_path.exists():
                try:
                    maybe_download(url, str(cache_path), extract=True)
                except Exception as e:
                    logger.error(f"Download from {url} failure!", exc_info=True)
                    raise e
        else:  # when specify paths of resource which have downloaded.
            file_path = Path(url)

        self.schema_file = file_path
        labels = OrderedDict()
        labels["O"] = "O"
        visited = set()
        with open(file_path, "r", encoding="utf8") as inf:
            for line in inf:
                one_json = json.loads(line)
                role_list = one_json.get("role_list")
                for role in role_list:
                    tmp = role["role"]
                    if tmp in visited:
                        continue
                    cur_num = len(labels)
                    visited.add(tmp)
                    labels[f"B-{tmp}"] = f"B-{cur_num}"
                    labels[f"I-{tmp}"] = f"I-{cur_num}"
        return labels