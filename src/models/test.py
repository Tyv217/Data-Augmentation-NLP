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
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Example input text
    input_text = "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."

    # Tokenize input text
    tokens = tokenizer.tokenize(input_text)
    print(tokens)
    # Output: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing', 'Ġtoday', '?']

    # Convert tokens to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)
    # Output: [8777, 6, 169, 33, 24, 358, 2412, 57]

    # Convert IDs back to tokens
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(decoded_tokens)
    # Output: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing', 'Ġtoday', '?']
