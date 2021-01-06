# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 19:35
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : gpt2.py


import logging
import inspect
import tensorflow as tf

from aispace.layers import BaseLayer
from aispace.utils.hparams import Hparams
from aispace.layers.encoders import TFConv1D
from aispace.utils.tf_utils import get_shape
from aispace.layers.activations import ACT2FN
from aispace.utils.tf_utils import get_initializer
from aispace.layers.embeddings import SharedEmbeddings
from aispace.layers.decoders import SequenceSummary


__all__ = [
    "Gpt2"
]

logger = logging.getLogger(__name__)


class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.n_ctx = n_ctx
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions

        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(get_shape(k)[-1], dtype=w.dtype)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = get_shape(w)
        b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = tf.nn.softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = get_shape(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = get_shape(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # to cope with keras serialization
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        self.act = ACT2FN["gelu"]
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = TFAttention(nx, n_ctx, config, scale, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp = TFMLP(inner_dim, config, name="mlp")

    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        a = self.ln_1(x)
        output_attn = self.attn(
            a, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=training
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


def input_processing(func, config, input_ids, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (:obj:`callable`):
            The callable function of the TensorFlow model.
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    signature.pop("kwargs", None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = (tf.Tensor, bool, int, tuple, list, dict)

    if "inputs" in kwargs["kwargs_call"]:
        logger.warning(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning)

        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    if "decoder_cached_states" in kwargs["kwargs_call"]:
        logger.warning(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    if len(kwargs["kwargs_call"]) > 0:
        raise ValueError(
            f"The following keyword arguments are not supported by this model: {list(kwargs['kwargs_call'].keys())}."
        )

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_ids, (tuple, list)):
        for i, input in enumerate(input_ids):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if type(input) == tf.Tensor:
                # Tensor names have always the pattern name:device_id then we check only the
                # name and not the device id
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for {parameter_names[i]}."
                )
    elif isinstance(input_ids, dict):
        if "inputs" in input_ids:
            logger.warning(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
                FutureWarning,
            )

            output["input_ids"] = input_ids.pop("inputs")

        if "decoder_cached_states" in input_ids:
            logger.warning(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_ids.pop("decoder_cached_states")

        for k, v in dict(input_ids).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if isinstance(input_ids, tf.Tensor) or input_ids is None:
            output[parameter_names[0]] = input_ids
        else:
            raise ValueError(
                f"Data of type {type(input_ids)} is not allowed only {allowed_types} is accepted for {parameter_names[0]}."
            )

    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and type(output["args"]) == tf.Tensor:
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_ids`
            output["input_ids"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    output.update(
        booleans_processing(
            config=config,
            **boolean_dict,
        )
    )

    return output


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model in order to be sure they are compliant with the execution mode (eager or
    graph)

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}

    if tf.executing_eagerly():
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
        final_booleans["output_hidden_states"] = (
            kwargs["output_hidden_states"]
            if kwargs["output_hidden_states"] is not None
            else config.output_hidden_states
        )

        if "return_dict" in kwargs:
            final_booleans["return_dict"] = (
                kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict
            )

        if "use_cache" in kwargs:
            final_booleans["use_cache"] = kwargs["use_cache"] if kwargs["use_cache"] is not None else config.use_cache
    else:
        if (
            kwargs["output_attentions"] is not None
            or kwargs["output_hidden_states"] is not None
            or ("use_cache" in kwargs and kwargs["use_cache"] is not None)
        ):
            logger.warning(
                "The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model."
                "They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`)."
            )

        final_booleans["output_attentions"] = config.output_attentions
        final_booleans["output_hidden_states"] = config.output_hidden_states

        if "return_dict" in kwargs:
            if kwargs["return_dict"] is not None:
                logger.warning(
                    "The parameter `return_dict` cannot be set in graph mode and will always be set to `True`."
                )
            final_booleans["return_dict"] = True

        if "use_cache" in kwargs:
            final_booleans["use_cache"] = config.use_cache

    return final_booleans


@BaseLayer.register("gpt2")
class Gpt2(BaseLayer):

    def __init__(self, config: Hparams, **kwargs):
        super().__init__(config, **kwargs)
        config = config.config
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd

        self.wte = SharedEmbeddings(
            config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name="wte"
        )
        self.wpe = tf.keras.layers.Embedding(
            config.n_positions,
            config.n_embd,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="wpe",
        )
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        if config.has_key("layers"):
            self.h = [TFBlock(config.n_ctx, config, scale=True, name="h_._{}".format(i)) for i in range(config.layers.start, config.layers.end, config.layers.step)]
        else:
            self.h = [TFBlock(config.n_ctx, config, scale=True, name="h_._{}".format(i)) for i in range(config.n_layer)]

        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        self.pooler = SequenceSummary(config, name="pooler")

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = self.wte.weight.shape[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    def call(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = get_shape(inputs["input_ids"])
            inputs["input_ids"] = tf.reshape(inputs["input_ids"], [-1, input_shape[-1]])
        elif inputs["inputs_embeds"] is not None:
            input_shape = get_shape(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["past"] is None:
            past_length = 0
            inputs["past"] = [None] * len(self.h)
        else:
            past_length = get_shape(inputs["past"][0][0])[-2]

        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]

        if inputs["attention_mask"] is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            inputs["attention_mask"] = inputs["attention_mask"][:, tf.newaxis, tf.newaxis, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.

            inputs["attention_mask"] = tf.cast(inputs["attention_mask"], tf.float32)
            inputs["attention_mask"] = (1.0 - inputs["attention_mask"]) * -10000.0
        else:
            inputs["attention_mask"] = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        inputs["position_ids"] = tf.reshape(inputs["position_ids"], [-1, get_shape(inputs["position_ids"])[-1]])

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.wte(inputs["input_ids"], mode="embedding")

        position_embeds = self.wpe(inputs["position_ids"])

        if inputs["token_type_ids"] is not None:
            inputs["token_type_ids"] = tf.reshape(
                inputs["token_type_ids"], [-1, get_shape(inputs["token_type_ids"])[-1]]
            )
            token_type_embeds = self.wte(inputs["token_type_ids"], mode="embedding")
        else:
            token_type_embeds = 0

        position_embeds = tf.cast(position_embeds, dtype=inputs["inputs_embeds"].dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs["inputs_embeds"].dtype)
        hidden_states = inputs["inputs_embeds"] + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=inputs["training"])

        output_shape = input_shape + [get_shape(hidden_states)[-1]]

        presents = () if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None
        for i, (block, layer_past) in enumerate(zip(self.h, inputs["past"])):
            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block(
                hidden_states,
                layer_past,
                inputs["attention_mask"],
                inputs["head_mask"][i],
                inputs["use_cache"],
                inputs["output_attentions"],
                training=inputs["training"],
            )

            hidden_states, present = outputs[:2]
            if inputs["use_cache"]:
                presents = presents + (present,)

            if inputs["output_attentions"]:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)
        pooled_output = self.pooler(hidden_states)

        # Add last hidden state
        if inputs["output_hidden_states"]:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if inputs["output_attentions"]:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + get_shape(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)

        # if not inputs["return_dict"]:
        #     return tuple(v for v in [hidden_states, pooled_output, presents, all_hidden_states, all_attentions] if v is not None)
        return tuple(v for v in [hidden_states, pooled_output, presents, all_hidden_states, all_attentions] if v is not None)

        # return outputs  # hidden_states, pooled_output, presents, all_hidden_states, all_attentions