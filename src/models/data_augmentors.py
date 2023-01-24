import nltk, torch, time
from nltk.corpus import wordnet as wn, stopwords
from torchdata.datapipes.iter import IterableWrapper
import random
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        N = min(2, len(word_list))
        if(N > 0) and (random.random() < augmentation_percentage):
            words_to_replace = random.sample(word_list, N)
            curr_sentence = sentence
            for word in words_to_replace:
                synonyms = self.get_synonym(word)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = re.sub(word, synonym, curr_sentence)
            sentence = curr_sentence
        if label is not None:
            return (label, sentence)
        else:
            return sentence

    def augment_dataset(self, data_iter, augmentation_percentage):
        augmented_sentences = [self.replace_with_synonyms(label, sentence, augmentation_percentage) for (label, sentence) in list(data_iter)]
        return IterableWrapper(augmented_sentences)

    def augment_dataset_without_label(self, data_iter, augmentation_percentage):
        augmented_sentences = [self.replace_with_synonyms(None, sentence, augmentation_percentage) for sentence in list(data_iter)]
        return IterableWrapper(augmented_sentences)


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
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def bulk_translate(self, sentences, model, tokenizer):
        input_encoding = tokenizer(
                text = sentences,
                padding = "longest",
                truncation = True,
                return_tensors = "pt",  
        )
        input_ids = input_encoding.input_ids
        attention_mask = input_encoding.attention_mask
        with torch.no_grad():
            output_ids = model.generate(input_ids, attention_mask = attention_mask)
        result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return result

    def bulk_back_translate(self, sentences, model, tokenizer):
        intermediate = self.bulk_translate(sentences, model, tokenizer)
        return self.bulk_translate(intermediate, model, tokenizer)

    def augment_dataset(self, data_iter, augmentation_percentage = 1):
        return IterableWrapper(data_iter)
        data_list = list(data_iter)
        to_augment = []
        no_augment = []
        for data in data_list:
            if(random.random() > augmentation_percentage):
                to_augment.append(data)
            else:
                no_augment.append(data)
        to_augment_labels, to_augment_sentences = zip(*to_augment)

        model_name = "google/bert2bert_L-24_wmt_" + self.src + "_" + self.dest
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 1024, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", sep_token="</s>", cls_token="<s>")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        count = 0
        translated_data = []
        BATCH_SIZE = 8
        print("Start augmenting!")
        while(count < len(to_augment_sentences)):
            # torch.cuda.empty_cache()
            translated_data.append(self.bulk_back_translate(to_augment_sentences[count:min(count + BATCH_SIZE, len(to_augment_sentences))], model, tokenizer))
            count += BATCH_SIZE

        data_list = zip(to_augment_labels, to_augment_sentences) + no_augment
        random.shuffle(data_list)
        
        return IterableWrapper(data_list)
            

class Insertor():
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

    def insert_randomly(self, word, sentence):
        space_indices = [m.start() for m in re.finditer(' ', sentence)]
        index = random.choice(space_indices)
        sentence = sentence[:index] + " " + word +  sentence[index:]
        return sentence

    def insert_synonyms(self, label, sentence, augmentation_percentage):
        word_list = sentence.split(" ")
        word_list = [i for i in word_list if i not in self.stopwords
                and any(map(str.isupper, i)) is False
                and not any(char.isdigit() for char in i)]
        N = min(2, len(word_list))
        if(N > 0) and (random.random() < augmentation_percentage):
            words_to_insert = random.sample(word_list, N)
            curr_sentence = sentence
            for word in words_to_replace:
                synonyms = self.get_synonym(word)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = self.insert_randomly(synonym, curr_sentence)
            sentence = curr_sentence
        return (label, sentence)

    def augment_dataset(self, data_iter, augmentation_percentage):
        augmented_sentences = [self.replace_with_synonyms(label, sentence, augmentation_percentage) for (label, sentence) in list(data_iter)]
        return IterableWrapper(augmented_sentences)

class Deletor():

    def delete_randomly(self, label, sentence, augmentation_percentage):
        N = 2
        if(N > 0 and random.random() < augmentation_percentage):
            word_list = sentence.split(" ")
            for i in range(N):
                index = random.randint(0, len(word_list) - 1)
                word_list.pop(index)
            sentence = " ".join(word_list)
        return (label, sentence)

    def augment_dataset(self, data_iter, augmentation_percentage):
        augmented_sentences = [self.replace_with_synonyms(label, sentence, augmentation_percentage) for (label, sentence) in list(data_iter)]
        return IterableWrapper(augmented_sentences)


