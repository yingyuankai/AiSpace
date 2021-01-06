# -*- coding: utf-8 -*-
# @Time    : 2019-11-28 13:57
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tf_model_adapters.py

import re
import numpy as np
from collections import OrderedDict
import tensorflow as tf


__all__ = [
    "tf_huggingface_bert_adapter",
    "tf_huggingface_ernie_adapter",
    "tf_huggingface_xlnet_adapter",
    "tf_huggingface_albert_chinese_adapter",
    "tf_huggingface_albert_chinese_google_adapter",
    "tf_huggingface_electra_adapter",
    "tf_huggingface_gpt2_adapter"
]


def tf_huggingface_bert_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface bert names to bert_wwm variables,
    and then set values for current model.

    :param hf_model_variables:
    :return:
    """
    name_to_values = list()

    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match("^.*/(bert/.*):\\d+$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        # for bert/encoder
        encoder_matched = re.match("^bert/encoder/layer_._\\d+.*$", matched_name)
        if encoder_matched is not None:
            matched_name = matched_name.replace("_._", "_")
        # for bert/embeddings
        if matched_name == "bert/embeddings/weight":
            matched_name = "bert/embeddings/word_embeddings"
        elif matched_name == "bert/embeddings/position_embeddings/embeddings":
            matched_name = "bert/embeddings/position_embeddings"
        elif matched_name == "bert/embeddings/token_type_embeddings/embeddings":
            matched_name = "bert/embeddings/token_type_embeddings"
        elif matched_name == "bert/embeddings/task_type_embeddings/embeddings":
            matched_name = "bert/embeddings/task_type_embeddings"
        value = tf.train.load_variable(init_checkpoint, matched_name)
        name_to_values.append((item, value))

    tf.keras.backend.batch_set_value(name_to_values)


def tf_huggingface_ernie_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface bert names to bert_wwm variables,
    and then set values for current model.

    :param hf_model_variables:
    :return:
    """
    name_to_values = list()

    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match("^.*/(ernie/.*):\\d+$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        # for bert/encoder
        encoder_matched = re.match("^ernie/encoder/layer_._\\d+.*$", matched_name)
        if encoder_matched is not None:
            matched_name = matched_name.replace("_._", "_").replace("ernie", "bert")
        # for bert/embeddings
        if matched_name == "ernie/embeddings/weight":
            matched_name = "bert/embeddings/word_embeddings"
        elif matched_name == "ernie/embeddings/position_embeddings/embeddings":
            matched_name = "bert/embeddings/position_embeddings"
        elif matched_name == "ernie/embeddings/token_type_embeddings/embeddings":
            matched_name = "bert/embeddings/token_type_embeddings"
        elif matched_name == "ernie/embeddings/task_type_embeddings/embeddings":
            matched_name = "bert/embeddings/task_type_embeddings"
        matched_name = matched_name.replace("ernie", "bert")
        value = tf.train.load_variable(init_checkpoint, matched_name)
        name_to_values.append((item, value))

    tf.keras.backend.batch_set_value(name_to_values)


def tf_huggingface_xlnet_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface xlnet names to xlnet_chinese variables,
    and then set values for current model.

    :param hf_model_variables:
    :return:
    """
    name_to_values = list()

    r_r_bias_values = tf.train.load_variable(init_checkpoint, "model/transformer/r_r_bias")
    r_s_bias_values = tf.train.load_variable(init_checkpoint, "model/transformer/r_s_bias")
    r_w_bias_values = tf.train.load_variable(init_checkpoint, "model/transformer/r_w_bias")
    seg_embed_values = tf.train.load_variable(init_checkpoint, "model/transformer/seg_embed")

    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match("^.*/(xl_net/.*):\\d+$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        # for bert/encoder
        encoder_matched = re.match("^xl_net/layer_._\\d+.*$", matched_name)
        if encoder_matched is not None:
            matched_name = matched_name.replace("_._", "_").\
                replace("xl_net", "model/transformer").\
                replace("layer_norm", "LayerNorm")
            i = int(re.match("^.*/layer_(\\d+).*$", matched_name).group(1))
            # for r_r_bias
            r_r_bias_matched = re.match("^.*/r_r_bias$", matched_name)
            if r_r_bias_matched is not None:
                value = np.squeeze(r_r_bias_values[i])
                name_to_values.append((item, value))
                continue
            # for r_s_bias
            r_s_bias_matched = re.match("^.*/r_s_bias$", matched_name)
            if r_s_bias_matched is not None:
                value = np.squeeze(r_s_bias_values[i])
                name_to_values.append((item, value))
                continue
            # for r_w_bias
            r_w_bias_matched = re.match("^.*/r_w_bias$", matched_name)
            if r_w_bias_matched is not None:
                value = np.squeeze(r_w_bias_values[i])
                name_to_values.append((item, value))
                continue
            # for seq_embed
            seg_embed_matched = re.match("^.*/seg_embed$", matched_name)
            if seg_embed_matched is not None:
                value = np.squeeze(seg_embed_values[i])
                name_to_values.append((item, value))
                continue

        # for ending with kqvor
        kqvor_matched = re.match("^.*/[kqvor]$", matched_name)
        if kqvor_matched is not None:
            matched_name += "/kernel"
        # for bert/embeddings
        if matched_name == 'xl_net/word_embedding/weight':
            matched_name = "model/transformer/word_embedding/lookup_table"
        if matched_name.endswith("mask_emb"):
            matched_name = "model/transformer/mask_emb/mask_emb"

        value = tf.train.load_variable(init_checkpoint, matched_name)
        name_to_values.append((item, value))

    tf.keras.backend.batch_set_value(name_to_values)


def tf_huggingface_albert_chinese_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface albert names to albert_chinese variables,
    and then set values for current model.

    brightmart version
    ref: https://github.com/brightmart/albert_zh
    :param hf_model_variables:
    :return:
    """
    name_to_values = list()
    default_prefix = "bert/encoder/layer_shared/"
    default_var_name = "albert_brightmart"
    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match(f"^.*/({default_var_name}/.*):\\d+$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        # for pooler
        if matched_name == f"{default_var_name}/pooler/bias":
            matched_name = "bert/pooler/dense/bias"
        elif matched_name == f"{default_var_name}/pooler/kernel":
            matched_name = "bert/pooler/dense/kernel"

        # for embeddings
        elif matched_name == f"{default_var_name}/embeddings/word_embeddings/weight":
            matched_name = "bert/embeddings/word_embeddings"
        elif matched_name == f"{default_var_name}/embeddings/position_embeddings/embeddings":
            matched_name = "bert/embeddings/position_embeddings"
        elif matched_name == f"{default_var_name}/embeddings/token_type_embeddings/embeddings":
            matched_name = "bert/embeddings/token_type_embeddings"
        elif matched_name == f"{default_var_name}/embeddings/LayerNorm/gamma":
            matched_name = "bert/embeddings/LayerNorm/gamma"
        elif matched_name == f"{default_var_name}/embeddings/LayerNorm/beta":
            matched_name = "bert/embeddings/LayerNorm/beta"

        # for encoder
        elif matched_name == f"{default_var_name}/embeddings/embedding_hidden_mapping_in":
            matched_name = "bert/embeddings/word_embeddings_2"

        # for transformer layers
        elif matched_name.endswith("ffn/kernel"):
            matched_name = f"{default_prefix}intermediate/dense/kernel"
        elif matched_name.endswith("ffn/bias"):
            matched_name = f"{default_prefix}intermediate/dense/bias"

        elif matched_name.endswith("ffn_output/kernel"):
            matched_name = f"{default_prefix}output/dense/kernel"
        elif matched_name.endswith("ffn_output/bias"):
            matched_name = f"{default_prefix}output/dense/bias"

        elif matched_name.endswith("full_layer_layer_norm/gamma"):
            matched_name = f"{default_prefix}output/LayerNorm/gamma"
        elif matched_name.endswith("full_layer_layer_norm/beta"):
            matched_name = f"{default_prefix}output/LayerNorm/beta"

        elif matched_name.endswith("attention/LayerNorm/gamma"):
            matched_name = f"{default_prefix}attention/output/LayerNorm/gamma"
        elif matched_name.endswith("attention/LayerNorm/beta"):
            matched_name = f"{default_prefix}attention/output/LayerNorm/beta"

        elif matched_name.find("attention/dense") != -1:
            matched_name = re.match("^.*attention/(.*)$", matched_name).group(1)
            matched_name = f"{default_prefix}attention/output/{matched_name}"
        elif matched_name.find("attention") != -1:
            matched_name = re.match("^.*attention/(.*)$", matched_name).group(1)
            matched_name = f"{default_prefix}attention/self/{matched_name}"
        # else:
        #     continue

        value = tf.train.load_variable(init_checkpoint, matched_name)
        name_to_values.append((item, value))

    tf.keras.backend.batch_set_value(name_to_values)


def tf_huggingface_albert_chinese_google_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface albert names to albert_chinese_google variables,
    and then set values for current model.

    :param hf_model_variables:
    :return:
    """
    name_to_values = list()
    default_prefix = "bert/encoder/transformer/group_0/inner_group_0/"
    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match("^.*/(albert/.*):\\d+$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        # for pooler
        if matched_name == "albert/pooler/bias":
            matched_name = "bert/pooler/dense/bias"
        elif matched_name == "albert/pooler/kernel":
            matched_name = "bert/pooler/dense/kernel"

        # for embeddings
        elif matched_name == "albert/embeddings/word_embeddings/weight":
            matched_name = "bert/embeddings/word_embeddings"
        elif matched_name == "albert/embeddings/position_embeddings/embeddings":
            matched_name = "bert/embeddings/position_embeddings"
        elif matched_name == "albert/embeddings/token_type_embeddings/embeddings":
            matched_name = "bert/embeddings/token_type_embeddings"
        elif matched_name == "albert/embeddings/LayerNorm/gamma":
            matched_name = "bert/embeddings/LayerNorm/gamma"
        elif matched_name == "albert/embeddings/LayerNorm/beta":
            matched_name = "bert/embeddings/LayerNorm/beta"

        # for encoder
        elif matched_name == "albert/encoder/embedding_hidden_mapping_in/kernel":
            matched_name = "bert/encoder/embedding_hidden_mapping_in/kernel"
        elif matched_name == "albert/encoder/embedding_hidden_mapping_in/bias":
            matched_name = "bert/encoder/embedding_hidden_mapping_in/bias"

        # for transformer layers
        elif matched_name.endswith("ffn/kernel"):
            matched_name = f"{default_prefix}ffn_1/intermediate/dense/kernel"
        elif matched_name.endswith("ffn/bias"):
            matched_name = f"{default_prefix}ffn_1/intermediate/dense/bias"
        elif matched_name.endswith("ffn_output/kernel"):
            matched_name = f"{default_prefix}ffn_1/intermediate/output/dense/kernel"
        elif matched_name.endswith("ffn_output/bias"):
            matched_name = f"{default_prefix}ffn_1/intermediate/output/dense/bias"
        elif matched_name.endswith("full_layer_layer_norm/gamma"):
            matched_name = f"{default_prefix}LayerNorm_1/gamma"
        elif matched_name.endswith("full_layer_layer_norm/beta"):
            matched_name = f"{default_prefix}LayerNorm_1/beta"
        elif matched_name.endswith("attention/LayerNorm/gamma"):
            matched_name = f"{default_prefix}LayerNorm/gamma"
        elif matched_name.endswith("attention/LayerNorm/beta"):
            matched_name = f"{default_prefix}LayerNorm/beta"
        elif matched_name.find("attention/dense") != -1:
            matched_name = re.match("^.*attention/(.*)$", matched_name).group(1)
            matched_name = f"{default_prefix}attention_1/output/{matched_name}"
        elif matched_name.find("attention") != -1:
            matched_name = re.match("^.*attention/(.*)$", matched_name).group(1)
            matched_name = f"{default_prefix}attention_1/self/{matched_name}"

        value = tf.train.load_variable(init_checkpoint, matched_name)
        name_to_values.append((item, value))

    tf.keras.backend.batch_set_value(name_to_values)


def tf_huggingface_electra_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface electra names to electra variables,
    and then set values for current model.

    :param hf_model_variables:
    :return:
    """
    name_to_values = list()

    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match("^.*/(electra/.*):\\d+$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        # for bert/encoder
        encoder_matched = re.match("^electra/encoder/layer_._\\d+.*$", matched_name)
        if encoder_matched is not None:
            matched_name = matched_name.replace("_._", "_")
        # for bert/embeddings
        if matched_name == "electra/embeddings/weight":
            matched_name = "electra/embeddings/word_embeddings"
        elif matched_name == "electra/embeddings/position_embeddings/embeddings":
            matched_name = "electra/embeddings/position_embeddings"
        elif matched_name == "electra/embeddings/token_type_embeddings/embeddings":
            matched_name = "electra/embeddings/token_type_embeddings"
        elif matched_name == "electra/embeddings/task_type_embeddings/embeddings":
            matched_name = "electra/embeddings/task_type_embeddings"
        value = tf.train.load_variable(init_checkpoint, matched_name)
        name_to_values.append((item, value))

    tf.keras.backend.batch_set_value(name_to_values)


def tf_huggingface_gpt2_adapter(hf_model_variables: list, init_checkpoint: str):
    """Build name to variable map from huggingface gpt2 names to gpt2 variables,
    and then set values for current model.

    :param hf_model_variables:
    :return:
    """
    model_gold = tf.keras.models.load_model(init_checkpoint)
    vars_gold = model_gold.trainable_variables
    vars_gold_refinded = {}

    name_to_values = list()

    for var in vars_gold:
        name, value = var.name, var.numpy()
        name = name.replace("kernel", "weight")
        name_pieces = name.split("/")
        prefix = "/".join(name_pieces[:3] + [name_pieces[-1]])
        if name.endswith("bias:0"):
            value = np.reshape(value, [1, value.shape[0]])
        # need merge
        if name.find("query_layer") != -1 or name.find("key_layer") != -1 or name.find("value_layer") != -1:
            if prefix not in vars_gold_refinded:
                vars_gold_refinded[prefix] = value
            else:
                vars_gold_refinded[prefix] = np.concatenate((vars_gold_refinded[prefix], value), axis=1)
        else:
            vars_gold_refinded[name] = value

    for item in hf_model_variables:
        var_name = item.name
        matched_name = re.match("^.*/(gpt2/.*)$", var_name)
        if matched_name is None:
            continue
        matched_name = matched_name.group(1)
        matched_name = matched_name.replace("gpt2", "gpt")
        name_pieces = matched_name.split("/")
        if name_pieces[1] == "wte":
            matched_name = "gpt/embedding/embeddings:0"
        elif name_pieces[1] == "wpe":
            matched_name = "position_embeddings:0"
        elif name_pieces[1] == "ln_f":
            matched_name = matched_name.replace(name_pieces[1], "LayerNorm_final_norm")
        elif name_pieces[1].startswith("h_._"):
            layer_name = name_pieces[1]
            layer_idx = int(layer_name.split("_._")[-1])
            new_layer_name = f"layer{layer_idx:02}"
            matched_name = matched_name.replace(layer_name, new_layer_name)
            if len(name_pieces) >= 4:
                if name_pieces[2] == "attn":
                    if name_pieces[3] == "c_attn":
                        matched_name = matched_name.replace("/".join(name_pieces[2: 4]), "attention")
                    elif name_pieces[3] == "c_proj":
                        matched_name = matched_name.replace("/".join(name_pieces[2: 4]), "attention/context_projection_layer")
                elif name_pieces[2] == "ln_1":
                    matched_name = matched_name.replace(name_pieces[2], "LayerNorm_mlp_ln0")
                elif name_pieces[2] == "ln_2":
                    matched_name = matched_name.replace(name_pieces[2], "LayerNorm_mlp_ln1")
                elif name_pieces[2] == "mlp":
                    if name_pieces[3] == "c_fc":
                        matched_name = matched_name.replace("/".join(name_pieces[2: 4]), "intermediate")
                    elif name_pieces[3] == "c_proj":
                        matched_name = matched_name.replace("/".join(name_pieces[2: 4]), "output")
        else:
            continue

        value = vars_gold_refinded.get(matched_name)
        if value is None:
            continue
        assert value.shape == item.shape
        tf.keras.backend.set_value(item, value)
        # name_to_values.append((item, value))

    # tf.keras.backend.batch_set_value(name_to_values)