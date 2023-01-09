import nltk
from nltk.corpus import wordnet as wn, stopwords
from torchdata.datapipes.iter import IterableWrapper
import random
import re
import googletrans as trans

class Synonym_Replacer():
    def __init__(self, stopword_language):
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        self.stopwords = stopwords.words(stopword_language)
    
    def get_synonym(self, word):
        synset = wn.synsets(word)
        synonym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synonym_deeplist for synonym in sublist if synonym != word]
        return synonyms

    def replace_with_synonyms(self, label, sentence, augmentation_percentage):
        word_list = sentence.split(" ")
        word_list = [i for i in word_list if i not in self.stopwords
                and any(map(str.isupper, i)) is False
                and not any(char.isdigit() for char in i)]
        N = min(5, len(word_list))
        if(N > 0) and (random.uniform(0,1) < augmentation_percentage):
            words_to_replace = random.sample(word_list, N)
            curr_sentence = sentence
            for word in words_to_replace:
                synonyms = self.get_synonym(word)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = re.sub(word, synonym, curr_sentence)
            sentence = curr_sentence
        return (label, sentence)

    def augment_dataset(self, data_iter, augmentation_percentage):
        augmented_sentences = [self.replace_with_synonyms(label, sentence, augmentation_percentage) for (label, sentence) in list(data_iter)]
        return IterableWrapper(augmented_sentences)

class Back_Translator():
    def __init__(self, translator, src, dest):
        self.translator = translator
        self.src = src
        self.dest = dest
    
    def bulk_translate(self, sentences, src, dest):
        translations = self.translator.translate(sentences, dest = dest, src = src)

        return [translation.text for translation in translations]

    def augment_dataset(self, data_iter, augmentation_percentage):
        data_list = list(data_iter)
        random.shuffle(data_list)
        to_translate_len = int(len(data_list) * augmentation_percentage)
        translated_data = self.bulk_translate([sentence for (label, sentence) in data_list[:to_translate_len]], self.src, self.dest)
        back_translated_data = self.bulk_translate(translated_data, self.dest, self.src)
        labels = [label for (label, sentence) in data_list[:to_translate_len]]
        data_list[:to_translate_len] = zip(labels, back_translated_data)
        return IterableWrapper(data_list)
            