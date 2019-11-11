from .utils import *
import pickle

class FastText:
    inst = None
    class __FastText:
        def __init__(self):
            fasttext_path = get_resource_path("", "gensim_kv_fasttext_tc.pkl")
            with fasttext_path.open("rb") as fin:
                self.kv = pickle.load(fin)
        
        def __getattr__(self, attr):
            if hasattr(self.kv, attr):
                return getattr(self.kv, attr)
    
    def __init__(self):
        if not FastText.inst:
            FastText.inst = FastText.__FastText()
        
    def __getattr__(self, attr):
        return getattr(self.inst, attr)
