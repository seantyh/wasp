from typing import List
import jieba
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field
from .fasttext import FastText
from typing import Dict, Any

jieba_init = False

@dataclass()
class MultiChoiceAnswer:
    avg_ans:int = -1
    avg_vec:np.array = None
    max_ans:int = -1
    max_vec:np.array = None
    segments:Dict[str, Any] = field(default_factory=dict)


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
    qmat, opt_list = to_vectors(ques_words, opt_words)

    ans = MultiChoiceAnswer()
    ans.avg_ans, ans.avg_vec = avg_similarity(qmat, opt_list)
    ans.max_ans, ans.max_vec = max_similarity(qmat, opt_list)
    ans.segments = {'questions': ques_words, 'options': opt_words}    

    return ans

def segment_words(question, options):
    if not jieba_init:
        initialize_jieba()
    ques_words = jieba.lcut(question)
    opt_words = [jieba.lcut(x) for x in options]
    idx = None
    return ques_words, opt_words

def to_vectors(qs, opts):
    def get_vectors(xs, ft):
        mat = np.vstack([ft.get_vector(x) for x in xs if x in ft.vocab])
        return mat 

    ft = FastText()
    qmat = get_vectors(qs, ft)    
    opt_list = [get_vectors(opt_x, ft) for opt_x in opts]
    return qmat, opt_list


def avg_similarity(qmat, opt_list):
    sim_list = []
    ft = FastText()
    qvec = qmat.mean(0)
    opt_vecs = np.vstack([opt_x.mean(0) for opt_x in opt_list])    
    sims = ft.cosine_similarities(qvec, opt_vecs)
    max_idx = np.argmax(sims)
    return max_idx, sims


def max_similarity(qmat, opt_list):       
    sim_list = []
    ft = FastText()
    for opt_x in opt_list:
        sim_mat = np.vstack([ft.cosine_similarities(qmat[i, :], opt_x) 
                for i in range(qmat.shape[0])])
        sim_list.append(sim_mat)
    
    max_sim = [np.max(x) for x in sim_list]
    max_idx = np.argmax(max_sim)
    
    return max_idx, max_sim

def decode_target_position(sentence):
    try:
        tokens = sentence.split("<")
        first = tokens[0]
        tokens = "".join(tokens[1:]).split(">")
        target = tokens[0]
        second = "".join(tokens[1:]).replace("<", "").replace(">", "")
        return first+target+second, len(first)
    except Exception as ex:
        return (sentence, -1)


    