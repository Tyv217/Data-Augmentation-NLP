import nltk, torch, time
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from torchdata.datapipes.iter import IterableWrapper
import random, re, spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Synonym_Replacer():
    def __init__(self, stopword_language, word_to_replace_per_sentence = 2):
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        self.stopwords = stopwords.words(stopword_language)
        self.word_to_replace_per_sentence = word_to_replace_per_sentence
        self.nlp = spacy.load("en_core_web_sm")
        self.pos_mapper = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
        self.lemmatizer = WordNetLemmatizer()
        self.augmentation_percentage = 0

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage
    
    def get_synonym(self, word, pos = None):
        if (pos):
            synset = wn.synsets(word, pos=pos)
        else:
            synset = wn.synsets(word)
        synonym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synonym_deeplist for synonym in sublist if synonym != word]
        lemma = self.lemmatizer.lemmatize(word)
        synonyms = filter(lambda x: self.lemmatizer.lemmatize(x) != lemma, synonyms)
        return synonyms
        
    def get_word_list(self, sentence):
        word_list = []
        doc = self.nlp(sentence)
        for token in doc:
            word = token.text
            if word not in self.stopwords and any(map(str.isupper, word)) is False and not any(char.isdigit() for char in word):
                if token.pos_ in self.pos_mapper:
                    word_list.append((word, self.pos_mapper[token.pos_]))
                    # word_list.append(word)
        return word_list

    def replace_with_synonyms(self, sentence):
        word_list = self.get_word_list(sentence)
        for word,pos in word_list:
            if(random.random() < self.augmentation_percentage):
                curr_sentence = sentence
                synonyms = self.get_synonym(word, pos)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))
                if self.preprocessor is not None:
                    synonyms = list(filter(lambda x: self.preprocessor.get_text_indices(x)[0] != 0, synonyms))
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = re.sub(word, synonym, curr_sentence)
                sentence = curr_sentence
        return sentence

    def augment_dataset(self, data_iter, preprocessor = None, has_label = False):
        self.preprocessor = preprocessor
        self.start_time = time.time()
        if has_label:
            label, data_iter = zip(*data_iter)
        augmented_sentences = [self.replace_with_synonyms(sentence) for sentence in list(data_iter)]
        if has_label:
            augmented_sentences = zip(label, augmented_sentences)
        return list(augmented_sentences)

# class Back_Translator():
#     def __init__(self, src, dest):
#         self.src = src
#         self.dest = dest
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def translate(self, sentence, model, tokenizer, augmentation_percentage):
#         input_ids = tokenizer(text = sentence, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
#         output_ids = model.generate(input_ids)[0]
#         sentence = tokenizer.decode(output_ids, skip_special_tokens=True)
#         return sentence

#     def back_translate(self, sentence, model, tokenizer, augmentation_percentage):
#         if(random.random() > augmentation_percentage):
#             intermediate = self.translate(sentence, model, tokenizer, augmentation_percentage)
#             return self.translate(intermediate, model, tokenizer, augmentation_percentage)
#         else:
#             return sentence

#     def augment_dataset(self, data_iter, augmentation_percentage = 0.01):
#         start_time = time.time()
#         count = 0
#         model_name = "google/bert2bert_L-24_wmt_" + self.src + "_" + self.dest
#         tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 1024, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", sep_token="</s>", cls_token="<s>")
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         model.to(self.device)
#         translated_data = [(label, self.back_translate(sentence, model, tokenizer, augmentation_percentage)) for (label, sentence) in data_iter]
#         random.shuffle(translated_data)
#         print("Time to augment: " + str(time.time() - start_time))
#         return IterableWrapper(translated_data)

class Back_Translator():
    def __init__(self, src):
        self.src = src
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation_percentage = 0

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def bulk_translate(self, sentences, model, tokenizer):
        input_encoding = tokenizer(
                text = sentences,
                padding = "longest",
                truncation = True,
                return_tensors = "pt",  
        )
        input_ids = input_encoding.input_ids.to(self.device)
        attention_mask = input_encoding.attention_mask.to(self.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, attention_mask = attention_mask, max_new_tokens = 256).to(self.device)
        result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return result

    def bulk_back_translate(self, sentences, model1, model2, tokenizer1, tokenizer2):
        intermediate = self.bulk_translate(sentences, model1, tokenizer1)
        return self.bulk_translate(intermediate, model2, tokenizer2)

    def get_translators(self):
        models = []
        target_languages = ["de", "fr", "es"]
        for language in target_languages:
            model1_name = "Helsinki-NLP/opus-mt-" + self.src + "-" + language
            model2_name = "Helsinki-NLP/opus-mt-" + language + "-" + self.src
            # tokenizer1 = AutoTokenizer.from_pretrained(model1_name, model_max_length = 1024, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", sep_token="</s>", cls_token="<s>")
            # tokenizer2 = AutoTokenizer.from_pretrained(model2_name, model_max_length = 1024, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", sep_token="</s>", cls_token="<s>")
            tokenizer1 = AutoTokenizer.from_pretrained(model1_name, model_max_length = 1024)
            tokenizer2 = AutoTokenizer.from_pretrained(model2_name, model_max_length = 1024)
            model1 = AutoModelForSeq2SeqLM.from_pretrained(model1_name).to(self.device)
            model2 = AutoModelForSeq2SeqLM.from_pretrained(model2_name).to(self.device)
            models.append((model1, model2, tokenizer1, tokenizer2))
        return models


    def augment_dataset(self, data_iter, has_label = False):
        data_list = list(data_iter)
        to_augment = []
        no_augment = []
        for data in data_list:
            if(random.random() < self.augmentation_percentage):
                to_augment.append(data)
            else:
                no_augment.append(data)
        
        if(has_label):
            label, to_augment = zip(*to_augment)
        translators = self.get_translators()
        count = 0
        translated_data = []
        BATCH_SIZE = 64
        print("Start augmenting!")
        while(count < len(to_augment)):
            # torch.cuda.empty_cache()
            (model1, model2, tokenizer1, tokenizer2) = random.choice(translators)
            translated_data.append(self.bulk_back_translate(to_augment[count:min(count + BATCH_SIZE, len(to_augment))], model1, model2, tokenizer1, tokenizer2))
            count += BATCH_SIZE
            print("64 Done!")

        if has_label:
            translated_data = zip(label, translated_data)

        data_list = translated_data + no_augment
        random.shuffle(data_list)
        
        return list(data_list)
            

class Insertor():
    def __init__(self, stopword_language):
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        self.stopwords = stopwords.words(stopword_language)
        self.augmentation_percentage = 0
        self.preprocessor = None
        self.nlp = spacy.load("en_core_web_sm")
        self.pos_mapper = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
        self.lemmatizer = WordNetLemmatizer()

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def get_synonym(self, word, pos = None):
        if (pos):
            synset = wn.synsets(word, pos=pos)
        else:
            synset = wn.synsets(word)
        synonym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synonym_deeplist for synonym in sublist if synonym != word]
        lemma = self.lemmatizer.lemmatize(word)
        synonyms = filter(lambda x: self.lemmatizer.lemmatize(x) != lemma, synonyms)
        return synonyms
        
    def get_word_list(self, sentence):
        word_list = []
        doc = self.nlp(sentence)
        for token in doc:
            word = token.text
            if word not in self.stopwords and any(map(str.isupper, word)) is False and not any(char.isdigit() for char in word):
                if token.pos_ in self.pos_mapper:
                    word_list.append((word, self.pos_mapper[token.pos_]))
                    # word_list.append(word)
        return word_list

    def insert_randomly(self, word, sentence):
        space_indices = [m.start() for m in re.finditer(' ', sentence)]
        index = random.choice(space_indices)
        sentence = sentence[:index] + " " + word +  sentence[index:]
        return sentence
    
    def insert_synonyms(self, sentence):
        word_list = self.get_word_list(sentence)
        for word,pos in word_list:
            if(random.random() < self.augmentation_percentage):
                curr_sentence = sentence
                synonyms = self.get_synonym(word, pos)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))
                if self.preprocessor is not None:
                    synonyms = list(filter(lambda x: self.preprocessor.get_text_indices(x)[0] != 0, synonyms))
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = self.insert_randomly(synonym, curr_sentence)
                sentence = curr_sentence
        return sentence


    def augment_dataset(self, data_iter, preprocessor = None, has_label = False):
        self.preprocessor = preprocessor
        if has_label:
            label, data_iter = zip(*data_iter)
        augmented_sentences = [self.insert_synonyms(sentence)for sentence in list(data_iter)]
        if has_label:
            augmented_sentences = zip(label, augmented_sentences)
        return list(augmented_sentences)

class Deletor():

    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def delete_randomly(self, sentence):
        word_list = sentence.split(" ")
        to_delete = []
        for word in word_list:
            if (random.random() < self.augmentation_percentage):
                to_delete.append(word)
        for word in to_delete:
            word_list.remove(word)
        sentence = " ".join(word_list)
        return sentence

    def augment_dataset(self, data_iter, has_label = False):
        if has_label:
            label, data_iter = zip(*data_iter)
        augmented_sentences = [self.delete_randomly(sentence) for sentence in list(data_iter)]
        if has_label:
            augmented_sentences = zip(label, augmented_sentences)
        return list(augmented_sentences)


