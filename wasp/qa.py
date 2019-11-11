from typing import List
import jieba
import numpy as np
from .fasttext import FastText

jieba_init = False

def initialize_jieba():
    ft = FastText()
    for w in ft.vocab.keys():
        jieba.add_word(w)
    jieba_init = True

def answer(question: str, options: List[str]):
    if not jieba_init:
        initialize_jieba()

    stem, target_idx = decode_target_position(question)
    ques_words, opt_words = segment_words(question, options)
    q_mat, opt_list = max_similarity(ques_words, opt_words)
    return ques_words, opt_words

def segment_words(question, options):
    ques_words = jieba.lcut(question)
    opt_words = [jieba.lcut(x) for x in options]
    idx = None
    return ques_words, opt_words

def max_similarity(qs, opts):
    def get_vectors(xs, ft):
        mat = np.vstack([ft.get_vector(x) for x in xs if x in ft.vocab])
    ft = FastText()
    q_mat = get_vectors(qs, ft)
    opts_list = [get_vectors(opt_x, ft) for opt_x in opts]
    return q_mat, opts_list

def decode_target_position(sentence):
    try:
        tokens = sentence.split("<")
        first = tokens[0]
        tokens = "".join(tokens[1:]).split(">")
        target = tokens[0]
        second = "".join(tokens[1:]).replace("<", "").replace(">", "")
        return first+target+second, len(first)
    except:
        return (sentence, -1)


    