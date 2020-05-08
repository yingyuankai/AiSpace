# -*- coding: utf-8 -*-
# @Time    : 2020-05-07 16:17
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_hannlp.py


import unittest

import hanlp


class TestHannlp(unittest.TestCase):
    def test_han(self):

        tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
        # recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
        # syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
        # semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
        #
        # pipeline = hanlp.pipeline() \
        #     .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
        #     .append(tokenizer, output_key='tokens') \
        #     .append(tagger, output_key='part_of_speech_tags') \
        #     .append(recognizer, input_key="tokens", output_key='ner_tag') \
        #     .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies') \
        #     .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies')

        text = "近期，蔚来美国裁员 70 人，其中有 20 人位于圣何塞的北美总部办公室和研发中心，50 人位于旧金山办公室，此外，旧金山办公室也在这次裁员中正式关闭。"

        # res = pipeline(text)
        tokens = tokenizer(text)
        res = tagger(tokens)
        print(tagger.transform.tag_vocab)

        print(res)

    def test_jieba(self):
        import jieba.posseg as psg
        text = "近期，蔚来美国裁员 70 人，其中有 20 人位于圣何塞的北美总部办公室和研发中心，50 人位于旧金山办公室，此外，旧金山办公室也在这次裁员中正式关闭。"
        res = psg.cut(text)
        for t in res:
            print(t)
