import nltk, torch, time
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem.snowball import SnowballStemmer
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
        self.stemmer = SnowballStemmer("english")
        self.augmentation_percentage = 0
        self.require_label = False
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
                if(synonyms):
                    synonym = random.choice(synonyms)
                    curr_sentence = re.sub(word, synonym, curr_sentence)
                sentence = curr_sentence
        return sentence

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        augmented_lines = [self.replace_with_synonyms(sentence) for sentence in list(inputs)]
        return augmented_lines, attention_mask, labels

class Back_Translator():
    def __init__(self, src):
        self.src = src
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation_percentage = 0
        self.require_label = False
        self.operate_on_embeddings = False

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

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
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
        self.stemmer = SnowballStemmer("english")
        self.require_label = False
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
        self.operate_on_embeddings = False

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
        self.cutout_percentage = 0.5
        self.require_label = False
        self.operate_on_embeddings = True

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def cutout_randomly(self, sentence: torch.Tensor):
        if(random.random() < self.augmentation_percentage):
            h, w = sentence.shape

            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            x1 = np.clip(x - int(x * self.cutout_percentage / 2), 0, w)
            x2 = np.clip(x1 + int(x * self.cutout_percentage), 0, w)
            y1 = np.clip(y - int(y * self.cutout_percentage / 2), 0, h)
            y2 = np.clip(y1 + int(y * self.cutout_percentage), 0, h)

            mask[y1: y2, x1: x2] = 0.

            mask = torch.tensor(mask, requires_grad = False).to(sentence.device)
            return sentence * mask
        return sentence

    def augment_dataset(self, inputs: torch.Tensor, attention_mask = None, labels = None):
        augmented_sentences = [self.cutout_randomly(sentence)for sentence in inputs]
        return torch.stack(augmented_sentences), attention_mask, labels
    
class MixUp():
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.require_label = True
        self.operate_on_embeddings = True
        self.weight_sampling_distribution = 'beta'
        self.mixup_percentage = 0.5

    def sample_weight(self):
        if self.weight_sampling_distribution == 'beta':
            return np.random.beta(self.cutmix_percentage, self.cutmix_percentage)
        if self.weight_sampling_distribution == 'normal':
            return np.random.normal(loc = 0.5, scale = self.cutmix_percentage)
        if self.weight_sampling_distribution == 'constant':
            return self.cutmix_percentage

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def mixup_randomly(self, sentence1: torch.Tensor, sentence2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor):
        lam = self.sample_weight()
        sentence1 = sentence1.clone()
        sentence2 = sentence2.clone()
        sentence = sentence1 * lam + sentence2 * (1- lam)
        label = label1 * lam + label2 * (1- lam)

        return sentence, label

    def generate_pairwise_and_augment(self, sentences, labels):
        generated_sentences = []
        generated_labels = []

        to_generate = int(len(sentences) * self.augmentation_percentage)
        
        for i in range(to_generate):
            choices = np.random.choice(len(sentences), 2, replace = False)
            sentence1 = sentences[choices[0]]
            label1 = labels[choices[0]]
            sentence2 = sentences[choices[1]]
            label2 = labels[choices[1]]
            sentence, label = self.approach_1(sentence1, sentence2, label1, label2)
            generated_sentences.append(sentence)
            generated_labels.append(label)

        new_sentences = torch.cat(sentences, torch.stack(generated_sentences))
        new_labels = torch.cat(labels, torch.stack(generated_labels))

        return new_sentences, new_labels

    def augment_dataset(self, inputs, attention_mask = None, labels = None):
        sentences, labels = self.generate_pairwise_and_augment(inputs, labels)
        return sentences, attention_mask, labels


class CutMix():
    def __init__(self):
        super().__init__()
        self.augmentation_percentage = 0
        self.require_label = True
        self.operate_on_embeddings = True
        self.weight_sampling_distribution = 'beta'
        self.upper_limit = 0.7
        self.lower_limit = 0.3

    def sample_weight(self):
        return np.random.uniform(self.lower_limit, self.upper_limit)

    def set_augmentation_percentage(self, augmentation_percentage):
        self.augmentation_percentage = augmentation_percentage

    def cutout_randomly(self, sentence1: torch.Tensor, sentence2: torch.Tensor, attention_mask1: torch.Tensor, attention_mask2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor):
        lam = self.sample_weight()
        h, w = sentence1.shape

        sentence1 = sentence1.clone()
        sentence2 = sentence2.clone()

        y_lam = int(h * lam)
        x_lam = int(w * lam)

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
        mask_2 = 1 - mask_1
        attention_mask = attention_mask1 * mask_1 + attention_mask2 * mask_2

        true_lam = x_lam * y_lam / (x * y)

        label = (1- true_lam) * label1 + true_lam * label2

        return sentence, attention_mask, label

    def generate_pairwise_and_augment(self, sentences, attention_masks, labels):
        generated_sentences = []
        generated_labels = []

        to_generate = int(len(sentences) * self.augmentation_percentage)
        
        for i in range(to_generate):
            choices = np.random.choice(len(sentences), 2, replace = False)
            sentence1 = sentences[choices[0]]
            attention_mask1 = attention_masks[choices[0]]
            label1 = labels[choices[0]]
            sentence2 = sentences[choices[1]]
            attention_mask2 = attention_masks[choices[0]]
            label2 = labels[choices[1]]
            sentence, label = self.approach_1(sentence1, sentence2, attention_mask1, attention_mask2, label1, label2)
            generated_sentences.append(sentence)
            generated_labels.append(label)

        new_sentences = torch.cat(sentences, torch.stack(generated_sentences))
        new_labels = torch.cat(labels, torch.stack(generated_labels))

        return new_sentences, new_labels

    def augment_dataset(self, inputs_embeds, attention_masks = None, labels = None):
        sentences, labels = self.generate_pairwise_and_augment(inputs_embeds, attention_masks, labels)
        return sentences, attention_masks, labels