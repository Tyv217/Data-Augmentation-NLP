import nltk
from nltk.corpus import wordnet as wn, stopwords
import random
import re
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


class Synonym_Replacer():
    def __init__(self, stopword_language):
        self.stopwords = stopwords.words(stopword_language)
    
    def get_synonym(self, word):
        synset = wn.synsets(word)
        synonym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synonym_deeplist for synonym in sublist if synonym != word]
        return synonyms

    def replace_with_synonyms(self, sentence):
        sentences = [sentence]
        word_list = sentence.split(" ")
        word_list = [i for i in word_list if i not in self.stopwords
                and any(map(str.isupper, i)) is False
                and i != 'NUMERIC']
        random.shuffle(word_list)
        N = 2
        words_to_replace = [word_list[i*N:(i+1)*N] for i in range(int(((len(word_list) + (N-1)) / N)))]
        for words in words_to_replace:
            curr_sentence = sentence
            for word in words:
                synonyms = self.get_synonym(word)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = re.sub(word, synonym, curr_sentence)
            sentences.append(curr_sentence)
            sentences = list(set(sentences))
        sentences.sort()
        random.shuffle(sentences)
        return sentences


    def word_bag(self):
        n = 2
        word_bag = ["1","2","3","4","5","6","7"]
        combined_bag = [word_bag[i * n:(i + 1) * n] for i in range((len(word_bag) + n - 1) // n)]
        return combined_bag

    def augment_dataset(self, data_iter):
        augmented_sentences_deeplist = [self.replace_with_synonyms(sentence) for sentence in list(data_iter)]
        augmented_sentences = [augmented_sentence for augmented_sentence_list in\
         augmented_sentences_deeplist for augmented_sentence in augmented_sentence_list]
        return augmented_sentences


if __name__ == "__main__":
    s = Synonym_Replacer('english')
    print(s.augment_dataset(["I go to school by bus", "I like apples"]))