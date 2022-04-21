from os.path import join
import pandas as pd
import re
from LIWC.LIWC.liwc_readDict import readDict

#To optimise the code: insert new word as a key in a new dictionary and as values its category, then look it up first as key

class LIWC_class:

    def __init__(self, path=''):
        # LIWC1, sad, anger, neg & pos emotion
        self.liwc = readDict('./LIWC/LIWC/liwc.dic')
        self.liwc = pd.DataFrame(self.liwc, columns=['word', 'category'])
        # self.liwc['word'] = self.liwc['word'].map(lambda x: re.sub(r'[*]', '', x))
        self.liwc['value'] = 1
        categories = ['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number','swear','social','family','friend','humans','affect','posemo',',negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home','money','relig','death','assent','nonfl','filler']

        self.liwc = pd.pivot_table(self.liwc, index='word', columns=['category'],values='value', fill_value=0).reset_index().reindex(['word', *categories], axis=1)
        self.categories_sets = [(item,LIWC_class.split_words_with_asterisc(set(self.liwc[self.liwc[item] == 1]['word'].tolist()))) for item in categories]
        self.categories = categories

    @staticmethod
    def split_words_with_asterisc(words):
        list_w, list_w_ast = [],[]
        for w in words:
            if w[-1]=="*":
                list_w_ast.append(w)
            else:
                list_w.append(w)
        return list_w, list_w_ast

    def score(self, sentence):
        words = sentence.split()
        result = []
        for cat,(list_w, list_w_ast) in self.categories_sets:
            intersection = []
            for word in words:
                if word in list_w:
                    intersection.append(word)
                    continue
                for word_cat in list_w_ast:
                    if word.startswith(word_cat[:-1]):
                        intersection.append(word_cat)
                        break
            result.append((cat,len(intersection),intersection))

        #Convert to list and take the relative value
        if len(words)!=0:
            result = [e[1]/len(words) for e in result]
        else:
            result = [e[1] for e in result]
        return result

if __name__ == '__main__':
    sentence = "me encanta ser programador"
    h = LIWC_class()
    print(h.score(sentence))
