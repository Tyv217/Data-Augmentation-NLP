import nltk, torch, time
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem.snowball import SnowballStemmer
import random, re, spacy
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import math
from abc import ABC, abstractmethod

class Augmentor(ABC):

    def __init__(self):
        self.augmentation_percentage = 0

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def augment_one_sample(self, input):
        raise Exception("Not implemented")

    def augment_dataset(self, inputs, attention_mask, labels):
        if isinstance(inputs, str):
            return self.augment_one_sample(inputs), attention_mask, labels
        else:
            augmented_lines = [self.augment_one_sample(sentence) for sentence in inputs]
            return augmented_lines, attention_mask, labels
        
    def augment_one_sample_with_saliency(self, input):
        raise Exception("Not implemented")
    
    def augment_dataset_with_saliency(self, inputs, attention_mask = None, labels = None, saliency_scores = []):
        if isinstance(inputs, str):
            return self.augment_one_sample_with_saliency(inputs, saliency_scores), attention_mask, labels
        else:
            if len(saliency_scores) != len(inputs):
                saliency_scores = [[] for _ in range(len(inputs))]
            augmented_lines = [self.augment_one_sample_with_saliency(sentence, score) for sentence, score in zip(list(inputs), saliency_scores)]
            return augmented_lines, attention_mask, labels
        

class Synonym_Replacer(Augmentor):
    def __init__(self, stopword_language, word_to_replace_per_sentence = 2):
        super().__init__()
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')

        self.stopwords = stopwords.words(stopword_language)
        self.word_to_replace_per_sentence = word_to_replace_per_sentence
        self.pos_tagger = spacy.load("en_core_web_sm")
        self.pos_mapper = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
        self.stemmer = SnowballStemmer("english")
        self.operate_on_embeddings = False
    
    def get_synonym(self, word, pos = None):
        if (pos):
            synset = wn.synsets(word, pos=pos)
        else:
            synset = wn.synsets(word)

        synonym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synonym_deeplist for synonym in sublist if synonym != word]
        lemma = self.stemmer.stem(word)
        synonyms = filter(lambda x: self.stemmer.stem(x) != lemma, synonyms)
        return synonyms
        
    def get_word_list(self, sentence):
        word_list = []
        doc = self.pos_tagger(sentence)

        for token in doc:
            word = token.text
            if word not in self.stopwords and any(map(str.isupper, word)) is False and not any(char.isdigit() for char in word):
                if token.pos_ in self.pos_mapper:
                    word_list.append((word, self.pos_mapper[token.pos_]))
        return word_list

    def augment_one_sample(self, sentence):
        word_list = self.get_word_list(sentence)
        for word,pos in word_list:
            if(random.random() < self.augmentation_percentage):
                curr_sentence = sentence
                synonyms = self.get_synonym(word, pos)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))

                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = re.sub(word, synonym, curr_sentence)

                sentence = curr_sentence
        return sentence
    
    def augment_one_sample_with_saliency(self, sentence, score):
        filtered_word_list = self.get_word_list(sentence)
        if len(filtered_word_list) == 0:
            return sentence
        word_list = sentence.split(" ")
        if len(word_list) != len(score):
            filtered_word_scores = np.full((len(filtered_word_list),), 1/len(filtered_word_list))
        else:
            filtered_word_scores = np.zeros(len(filtered_word_list))
            for i in range(len(word_list)):
                for j in range(len(filtered_word_list)):
                    if filtered_word_list[j][0] in word_list[i]:
                        filtered_word_scores[j] = score[i]
            if np.sum(filtered_word_scores) == 0:
                filtered_word_scores = np.full((len(filtered_word_list),), 1/len(filtered_word_list))
            else:
                filtered_word_scores = filtered_word_scores / np.sum(filtered_word_scores)

        num_replace = int(len(filtered_word_list) * self.augmentation_percentage)
        
        to_replace = np.random.choice(np.arange(len(filtered_word_list)), size=num_replace, replace=False, p = filtered_word_scores)

        for index in to_replace:
            word, pos = filtered_word_list[index]
            curr_sentence = sentence
            synonyms = self.get_synonym(word, pos)
            synonyms = list(filter(lambda x: '_' not in x, synonyms))

            if(synonyms):
                synonym = random.choice(synonyms)
                curr_sentence = re.sub(word, synonym, curr_sentence)

            sentence = curr_sentence
        return sentence

class Back_Translator(Augmentor):
    def __init__(self, src):
        super().__init__()
        self.src = src
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation_percentage = 0
        self.operate_on_embeddings = False

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage / 100 # DONT CHANGE

    def translate(self, sentences, model, tokenizer):
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

    def back_translate(self, sentences, model1, model2, tokenizer1, tokenizer2):
        intermediate = self.translate(sentences, model1, tokenizer1)
        back_translated = self.translate(intermediate, model2, tokenizer2)
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

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        if attention_mask is not None:
            raise Exception("Back Translation on Tokens Instead of Words") 
        
        if isinstance(inputs, str):
            if random.random() < self.augmentation_percentage:
                return self.back_translate(inputs), attention_mask, labels
            else:
                return inputs, attention_mask, labels
            
        else:
            to_augment = []
            for input in list(inputs):
                if(random.random() < self.augmentation_percentage):
                    to_augment.append(input)

            translators = self.get_translators()
            count = 0
            translated_data = []
            BATCH_SIZE = 64
            while(count < len(to_augment)):
                # torch.cuda.empty_cache()
                (model1, model2, tokenizer1, tokenizer2) = random.choice(translators)
                text = to_augment[count:min(count + BATCH_SIZE, len(to_augment))]
                translated_data += self.back_translate(text, model1, model2, tokenizer1, tokenizer2)
                count += BATCH_SIZE

            inputs = list(inputs)
            
            for i1 in range(len(translated_data)):
                i2 = inputs.index(to_augment[i1])
                inputs[i2] = translated_data[i1]
            return list(inputs), attention_mask, labels

class Insertor(Augmentor):
    def __init__(self, stopword_language):
        super().__init__()
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        self.stopwords = stopwords.words(stopword_language)
        self.augmentation_percentage = 0
        self.preprocessor = None
        self.pos_tagger = spacy.load("en_core_web_sm")
        self.pos_mapper = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
        self.stemmer = SnowballStemmer("english")
        self.operate_on_embeddings = False

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def get_synonym(self, word, pos = None):
        if (pos):
            synset = wn.synsets(word, pos=pos)
        else:
            synset = wn.synsets(word)
        synonym_deeplist = [syn.lemma_names() for syn in synset]
        synonyms = [synonym for sublist in synonym_deeplist for synonym in sublist if synonym != word]
        lemma = self.stemmer.stem(word)
        synonyms = filter(lambda x: self.stemmer.stem(x) != lemma, synonyms)
        return synonyms
        
    def get_word_list(self, sentence):
        word_list = []
        doc = self.pos_tagger(sentence)
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
    
    def augment_one_sample(self, sentence):
        word_list = self.get_word_list(sentence)
        for word,pos in word_list:
            if(random.random() < self.augmentation_percentage):
                curr_sentence = sentence
                synonyms = self.get_synonym(word, pos)
                synonyms = list(filter(lambda x: '_' not in x, synonyms))

                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = self.insert_randomly(synonym, curr_sentence)
                sentence = curr_sentence
        return sentence
    
    def augment_one_sample_with_saliency(self, sentence, score):
        filtered_word_list = self.get_word_list(sentence)
        if len(filtered_word_list) == 0:
            return sentence
        word_list = sentence.split(" ")
        if len(word_list) != len(score):
            filtered_word_scores = np.full((len(filtered_word_list),), 1/len(filtered_word_list))
        else:
            filtered_word_scores = np.zeros(len(filtered_word_list))
            for i in range(len(word_list)):
                for j in range(len(filtered_word_list)):
                    if filtered_word_list[j][0] in word_list[i]:
                        filtered_word_scores[j] = score[i]
            if np.sum(filtered_word_scores) == 0:
                filtered_word_scores = np.full((len(filtered_word_list),), 1/len(filtered_word_list))
            else:
                filtered_word_scores = filtered_word_scores / np.sum(filtered_word_scores)
                
        num_insert = int(len(filtered_word_list) * self.augmentation_percentage)

        to_insert = np.random.choice(np.arange(len(filtered_word_list)), size=num_insert, replace=False, p = filtered_word_scores)
        
        for index in to_insert:
            word, pos = filtered_word_list[index]
            curr_sentence = sentence
            synonyms = self.get_synonym(word, pos)
            synonyms = list(filter(lambda x: '_' not in x, synonyms))

            if(synonyms):
                synonym = random.choice(synonyms)
                curr_sentence = self.insert_randomly(synonym, curr_sentence)

            sentence = curr_sentence
        return sentence


class Deletor(Augmentor):

    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.operate_on_embeddings = False

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def augment_one_sample(self, sentence):
        word_list = sentence.split(" ")
        to_delete = []
        for word in word_list:
            if (random.random() < self.augmentation_percentage):
                to_delete.append(word)
        for word in to_delete:
            word_list.remove(word)
        sentence = " ".join(word_list)
        return sentence
    
    def augment_one_sample_with_saliency(self, sentence, score):
        word_list = sentence.split(" ")
        if len(word_list) != len(score):
            score = [1 / len(word_list) for _ in range(len(word_list))]
        num_delete = int(len(word_list) * self.augmentation_percentage)
        to_delete = np.random.choice(np.arange(len(word_list)), size=num_delete, replace=False, p = score)
        word_list = np.delete(np.array(word_list), to_delete)
        sentence = " ".join(word_list)
        return sentence

class CutOut(Augmentor):
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.cutout_percentage = 0.5
        self.operate_on_embeddings = True

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def augment_one_sample(self, sentence: torch.Tensor):
        if(random.random() < self.augmentation_percentage):
            h, w = sentence.shape

            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            x1 = np.clip(x - int(w * self.cutout_percentage / 2), 0, w)
            x2 = np.clip(x + int(w * self.cutout_percentage / 2), 0, w)
            y1 = np.clip(y - int(h * self.cutout_percentage / 2), 0, h)
            y2 = np.clip(y + int(h * self.cutout_percentage / 2), 0, h)

            mask[y1: y2, x1: x2] = 0.

            mask = torch.tensor(mask, requires_grad = False).to(sentence.device)
            return sentence * mask
        return [sentence]
    
class MixUp(Augmentor):
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.operate_on_embeddings = True
        self.weight_sampling_distribution = 'beta'
        self.mixup_percentage = 0.5

    def sample_weight(self):
        if self.weight_sampling_distribution == 'beta':
            return np.random.beta(self.mixup_percentage, self.mixup_percentage)
        if self.weight_sampling_distribution == 'normal':
            return np.random.normal(loc = 0.5, scale = self.mixup_percentage)
        if self.weight_sampling_distribution == 'constant':
            return self.mixup_percentage

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def mixup_randomly(self, sentence1: torch.Tensor, sentence2: torch.Tensor, attention_mask1: torch.Tensor, attention_mask2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor):
        lam = self.sample_weight()
        sentence1 = sentence1.clone()
        sentence2 = sentence2.clone()
        sentence = sentence1 * lam + sentence2 * (1- lam)
        attention_mask = torch.logical_or(attention_mask1, attention_mask2).to(attention_mask1.device)
        label = label1 * lam + label2 * (1- lam)

        return sentence, attention_mask, label
    
    def augment_one_sample(self, sentence, attention_mask, label, other_samples):
        if random.random() < self.augmentation_percentage:
            sentence2, attention_mask2, label2 = np.random.choice(other_samples, 1)
            return [(sentence, attention_mask, label), (self.mixup_randomly(sentence, sentence2, attention_mask, attention_mask2, label, label2))]
        else:
            
            return None

    def generate_pairwise_and_augment(self, sentences, attention_masks, labels):
        generated_sentences = []
        generated_attention_masks = []
        generated_labels = []

        to_generate = int(len(sentences) * self.augmentation_percentage)
        
        for i in range(to_generate):
            choices = np.random.choice(len(sentences), 2, replace = False)
            sentence1 = sentences[choices[0]]
            attention_mask1 = attention_masks[choices[0]]
            label1 = labels[choices[0]]
            sentence2 = sentences[choices[1]]
            attention_mask2 = attention_masks[choices[1]]
            label2 = labels[choices[1]]
            sentence, attention_mask, label = self.mixup_randomly(sentence1, sentence2, attention_mask1, attention_mask2, label1, label2)
            generated_sentences.append(sentence)
            generated_attention_masks.append(attention_mask)
            generated_labels.append(label)

        if(to_generate > 0):
            new_sentences = torch.cat([sentences, torch.stack(generated_sentences)])
            new_attention_masks = torch.cat([attention_masks, torch.stack(generated_attention_masks)])
            new_labels = torch.cat([labels, torch.stack(generated_labels)])

            indices = torch.randperm(new_sentences.size()[0])
            new_sentences = new_sentences[indices]
            new_attention_masks = new_attention_masks[indices]
            new_labels = new_labels[indices]

            return new_sentences, new_attention_masks, new_labels

        return sentences, attention_masks, labels

    def augment_dataset(self, inputs, attention_masks = None, labels = None):
        sentences, attention_masks, labels = self.generate_pairwise_and_augment(inputs, attention_masks, labels)
        return sentences, attention_masks, labels


class CutMix(Augmentor):
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.operate_on_embeddings = True
        self.weight_sampling_distribution = 'beta'
        self.upper_limit = 0.7
        self.lower_limit = 0.3

    def sample_weight(self):
        return np.random.uniform(self.lower_limit, self.upper_limit)

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def cutmix_randomly(self, sentence1: torch.Tensor, sentence2: torch.Tensor, attention_mask1: torch.Tensor, attention_mask2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor):
        lam = self.sample_weight()
        h, w = sentence1.shape

        sentence1 = sentence1.clone()
        sentence2 = sentence2.clone()

        y_lam = int(h * math.sqrt(1 - lam))
        x_lam = int(w * math.sqrt(1 - lam))

        y = np.random.randint(h - y_lam)
        x = np.random.randint(w - x_lam)

        mask_1 = np.ones((h, w), np.float32)

        x1 = np.clip(x, 0, w)
        x2 = np.clip(x + x_lam, 0, w)
        y1 = np.clip(y, 0, h)
        y2 = np.clip(y + y_lam, 0, h)

        mask_1[y1: y2, x1: x2] = 0.

        mask_1 = torch.tensor(mask_1, requires_grad = False).to(sentence1.device)
        mask_2 = 1 - mask_1

        sentence = sentence1 * mask_1 + sentence2 * mask_2
        
        mask_1 = np.ones(h, np.float32)

        mask_1[y1: y2] = 0.

        mask_1 = torch.tensor(mask_1, requires_grad = False).to(attention_mask1.device)
        mask_2 = 1 - mask_1

        attention_mask = attention_mask1 * mask_1 + attention_mask2 * mask_2

        label = lam * label1 + (1- lam)* label2

        return sentence, attention_mask, label
    
    def augment_one_sample(self, sentence, attention_mask, label, other_samples):
        if random.random() < self.augmentation_percentage:
            sentence2, attention_mask2, label2 = np.random.choice(other_samples, 1)
            return [(sentence, attention_mask, label), (self.mixup_randomly(sentence, sentence2, attention_mask, attention_mask2, label, label2))]
        else:
            return None

    def generate_pairwise_and_augment(self, sentences, attention_masks, labels):
        generated_sentences = []
        generated_attention_masks = []
        generated_labels = []

        to_generate = int(len(sentences) * self.augmentation_percentage)
        
        for i in range(to_generate):
            choices = np.random.choice(len(sentences), 2, replace = False)
            sentence1 = sentences[choices[0]]
            attention_mask1 = attention_masks[choices[0]]
            label1 = labels[choices[0]]
            sentence2 = sentences[choices[1]]
            attention_mask2 = attention_masks[choices[1]]
            label2 = labels[choices[1]]
            sentence, attention_mask, label = self.cutmix_randomly(sentence1, sentence2, attention_mask1, attention_mask2, label1, label2)
            generated_sentences.append(sentence)
            generated_attention_masks.append(attention_mask)
            generated_labels.append(label)

        if(to_generate > 0):

            new_sentences = torch.cat([sentences, torch.stack(generated_sentences)])
            new_attention_masks = torch.cat([attention_masks, torch.stack(generated_attention_masks)])
            new_labels = torch.cat([labels, torch.stack(generated_labels)])

            indices = torch.randperm(new_sentences.size()[0])
            new_sentences = new_sentences[indices]
            new_attention_masks = new_attention_masks[indices]
            new_labels = new_labels[indices]

            return new_sentences, new_attention_masks, new_labels

        return sentences, attention_masks, labels

    def augment_dataset(self, inputs_embeds, attention_masks = None, labels = None):
        sentences, attention_masks, labels = self.generate_pairwise_and_augment(inputs_embeds, attention_masks, labels)
        return sentences, attention_masks, labels
    
AUGMENTOR_LIST = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix(), "mu": MixUp()}
AUGMENTOR_LIST_SINGLE = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut()}