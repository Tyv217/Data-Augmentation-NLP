import nltk
from nltk.corpus import wordnet as wn, stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


class Synonym_Replacer():
    def __init__(self):
        self.stopwords = stopwords.words('english')
    
    def get_synoynm(self, word):
        synset = wn.synsets(word)
        synoym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synoym_deeplist for synonym in sublist if synonym != word]
        return synonyms

    def replace_with_synonyms(self, sentence):
        pass

    def word_bag(self):
        n = 2
        word_bag = ["1","2","3","4","5","6"]
        combined_bag = [word_bag[i * n:(i + 1) * n] for i in range((len(word_bag) + n - 1) // n)]
        return combined_bag


if __name__ == "__main__":
    s = Synonym_Replacer()
    print(s.word_bag())