import warnings
warnings.filterwarnings("ignore")
from os.path import join
from LIWC.LIWC.LIWC import LIWC_class

warnings.filterwarnings("ignore")

class Lexical_LIWC:

    def __init__(self, path=''):
        self.LIWC = LIWC_class(path=join(path, 'LIWC/'))

    def one_vector_LIWC(self, sentence):
        global_vec = []
        global_vec.extend(self.LIWC.score(sentence))
        return global_vec

if __name__ == '__main__':
    snt = Lexical_LIWC()
    s = ["I don't like the movie, it's bad I hate it", "I love you sweety kiss", 'I was so sad, he called me the bitch, he was killed']
    print(snt.one_vector_LIWC(s[0]))
