import nltk, torch, time
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from torchdata.datapipes.iter import IterableWrapper
import random, re, spacy
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

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
        self.require_label = False
        self.operate_on_tokens = False

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

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        augmented_lines = [self.replace_with_synonyms(sentence) for sentence in list(inputs)]
        return augmented_lines, attention_mask, labels

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
        self.require_label = False
        self.operate_on_tokens = False

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage / 10000

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
        back_translated = self.bulk_translate(intermediate, model2, tokenizer2)
        with open("translated_data.txt", 'a') as f:
            for s,b in zip(sentences, back_translated):
                f.write("Original: " + s + "\nTranslated: " + b + "\n\n")
        return back_translated

    def get_translators(self):
        models = []
        target_languages = ["de", "fr", "es"]
        for language in target_languages:
            model1_name = "Helsinki-NLP/opus-mt-" + self.src + "-" + language
            model2_name = "Helsinki-NLP/opus-mt-" + language + "-" + self.src
            # tokenizer1 = AutoTokenizer.from_pretrained(model1_name, model_max_length = 1024, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", sep_token="</s>", cls_token="<s>")
            # tokenizer2 = AutoTokenizer.from_pretrained(model2_name, model_max_length = 1024, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", sep_token="</s>", cls_token="<s>")
            tokenizer1 = MarianTokenizer.from_pretrained(model1_name, model_max_length = 1024)
            tokenizer2 = MarianTokenizer.from_pretrained(model2_name, model_max_length = 1024)
            model1 = MarianMTModel.from_pretrained(model1_name).to(self.device)
            model2 = MarianMTModel.from_pretrained(model2_name).to(self.device)
            models.append((model1, model2, tokenizer1, tokenizer2))
        return models

    def augment_dataset(self, inputs, attention_mask, labels):
        if attention_mask is not None:
            raise Exception("Back Translation on Tokens Instead of Words") 
        to_augment = []
        for input in list(inputs):
            if(random.random() < self.augmentation_percentage):
                to_augment.append(input)

        translators = self.get_translators()
        count = 0
        translated_data = []
        BATCH_SIZE = 64
        start_time = time.time()
        while(count < len(to_augment)):
            # torch.cuda.empty_cache()
            (model1, model2, tokenizer1, tokenizer2) = random.choice(translators)
            text = to_augment[count:min(count + BATCH_SIZE, len(to_augment))]
            translated_data += self.bulk_back_translate(text, model1, model2, tokenizer1, tokenizer2)
            count += BATCH_SIZE
        
        for i1 in range(len(translated_data)):
            i2 = inputs.index(to_augment[i1])
            inputs[i2] = translated_data[i1]

        print("Augmentation took :", time.time() - start_time)
        return list(inputs), attention_mask, labels

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
        self.require_label = False
        self.operate_on_tokens = False

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
        if((space_indices is not None) and len(space_indices) > 0):
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


    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        augmented_sentences = [self.insert_synonyms(sentence)for sentence in list(inputs)]
        return list(augmented_sentences), attention_mask, labels

class Deletor():

    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.require_label = False
        self.operate_on_tokens = False

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

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        augmented_sentences = [self.delete_randomly(sentence)for sentence in list(inputs)]
        return list(augmented_sentences), attention_mask, labels

class CutOut():
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.cutout_percentage = 0.1
        self.require_label = False
        self.operate_on_tokens = True

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def cutout_randomly(self, sentence):
        if(random.random() < self.augmentation_percentage):
            word_list = sentence.split(" ")
            l = len(word_list)
            words_to_delete = int(l * self.cutout_percentage)
            i = random.randrange(0 - words_to_delete, l + 1)
            start_index = max(0, i)
            end_index = min(l, i + words_to_delete)
            word_list = word_list[:start_index] + word_list[end_index:l]
            sentence = " ".join(word_list)
        return sentence

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        augmented_sentences = [self.cutout_randomly(sentence)for sentence in list(inputs)]
        return list(augmented_sentences), attention_mask, labels

    

class CutMix():
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.cutmix_percentage = 1
        self.require_label = True
        self.operate_on_tokens = True

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def approach_1(self, sentence1, sentence2, label1, label2):
        lam = np.random.beta(self.cutmix_percentage, self.cutmix_percentage)
        sentence1 = sentence1.split(" ")
        sentence2 = sentence2.split(" ")
        if(len(sentence2) < len(sentence1)):
            sentence1, sentence2 = sentence2, sentence1
            label1, label2 = label2, label1
        
        l1 = len(sentence1)
        l2 = len(sentence2)
        words_to_cutout = int(l1 * lam)
        start_index1 = random.randrange(0, l1 - words_to_cutout)
        mid_index1 = start_index1 + words_to_cutout / 2
        mid_index2 = mid_index1 * l2 / l1
        start_index2 = int(mid_index2 - words_to_cutout / 2)

        end_index1 = start_index1 + words_to_cutout
        end_index2 = start_index2 + words_to_cutout
        
        sentence = sentence1[:start_index1] + sentence2[start_index2:end_index2] + sentence1[end_index1:l1]
        label = label1 * lam + label2 * (1-lam)
        return " ".join(sentence), label

    def approach_2(self, sentence1, sentence2, label1, label2):
        # Difference to approach 1 is how it selects where in sentence 2 to take out the sentence.
        # Just takes out same index as sentence 1
        lam = np.random.beta(self.cutmix_percentage, self.cutmix_percentage)
        if(len(sentence2) < len(sentence1)):
            sentence1, sentence2 = sentence2, sentence1
            label1, label2 = label2, label1
        
        l1 = len(sentence1)
        l2 = len(sentence2)
        words_to_cutout = int(l1 * lam)
        start_index = random.randrange(0, l1 - words_to_cutout)
        end_index = start_index + words_to_cutout
        sentence = sentence1[:start_index] + sentence2[start_index:end_index] + sentence1[end_index:l1]
        label = label1 * lam + label2 * (1-lam)

        return sentence, label

    def generate_pairwise_and_augment(self, data, has_label):
        generated = []

        to_generate = int(len(data) * self.augmentation_percentage)
        
        for i in range(to_generate):
            choices = np.random.choice(len(data), 2, replace = False)
            label1, sentence1 = data[choices[0]]
            label2, sentence2 = data[choices[1]]
            generated.append(self.approach_1(sentence1, sentence2, label1, label2))

        return zip(*generated)

    def augment_dataset(self, data_iter, has_label = False):
        augmented_sentences = self.generate_pairwise_and_augment(data_iter, has_label)
        return augmented_sentences
