from re import L
from nltk.stem import WordNetLemmatizer

def main():
    import torch

    from transformers import AutoTokenizer, DistilBertForSequenceClassification, T5Tokenizer

    from transformers import AutoConfig, T5ForConditionalGeneration
    MODEL_NAME = "t5-small"
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration(config)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length = 512)

    input_lines = ["I ate an apple.", "I fucked your mom."]

    input_encoding = tokenizer(
        # [self.task_prefix + sequence for sequence in input_lines],
        input_lines,
        padding = "longest",
        truncation = True,
        return_tensors = "pt",
    )
    input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask
    def print_weights_hook(model, input, output):
        print("\n\n\n\n")
        print(model)
        print(input)
        print(input[0].size())
        print(output)
        print(output.size())
        with open("file.txt", "a") as f:
            f.write("1\n")

    model.shared.register_forward_hook(print_weights_hook)
    model.encoder.embed_tokens.register_forward_hook(print_weights_hook)
    model.lm_head.register_forward_hook(print_weights_hook)
    model(input_ids = input_ids, attention_mask = attention_masks, labels = input_ids)

    print(model)
    # print(y)

    return 0

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    with torch.no_grad():

        logits = model(**inputs).logits

    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

    # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`

    num_labels = len(model.config.id2label)

    model = DistilBertForSequenceClassification.from_pretrained(

        "distilbert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification"

    )

    import pdb
    pdb.set_trace()

    labels = torch.sum(

        torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1

    ).to(torch.float)

    loss = model(**inputs, labels=labels).loss

if __name__ == "__main__":
    import torch, numpy as np
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    input_texts = ["This is the first input text.", "This is the second wow input text.", "And this is the third input text."]

    encoded_inputs = tokenizer.batch_encode_plus(input_texts, padding=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors="pt")
    tokenized_inputs = np.array([tokenizer.convert_ids_to_tokens(input_ids) for input_ids in encoded_inputs['input_ids']])
    # print(encoded_inputs)
    attention_masks = encoded_inputs['attention_mask']
    attentions = torch.rand_like(encoded_inputs['input_ids'], dtype = torch.float)
    # print(tokenized_inputs)
    # print(attention_masks)
    input_words = [input_text.split(" ") for input_text in input_texts]

    for w, tokens, a in zip(input_words, tokenized_inputs, attentions):
        t = [token.lstrip('▁') for token in tokens]
        non_special_indices = np.nonzero(~np.char.startswith(np.array(t), '<'))

        a = np.array(a)[non_special_indices]
        t = np.array(t)[non_special_indices]
        num_words = len(w)
        word_weights = np.empty(num_words, dtype=float)
        token_index = 0
        for i in range(num_words):
            curr_tokens = t[token_index]
            word_weights[i] = a[token_index]
            while(curr_tokens != w[i]):
                token_index += 1
                curr_tokens += t[token_index]
                word_weights[i] += a[token_index]
            token_index += 1

    # print(tokens)
    # Output: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing', 'Ġtoday', '?']
