from re import L
from nltk.stem import WordNetLemmatizer

def main():
    import torch

    from transformers import AutoTokenizer, DistilBertForSequenceClassification, T5Tokenizer

    from transformers import AutoConfig, T5ForConditionalGeneration
    MODEL_NAME = "t5-small"
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration(config)
    print(model)
    raise Exception
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
    main()
    # import statistics
    # def batch(iterable, n=1):
    #     """Batch elements of an iterable into lists of size n."""
    #     it = iter(iterable)
    #     while True:
    #         chunk = []
    #         for i in range(n):
    #             try:
    #                 chunk.append(next(it))
    #             except StopIteration:
    #                 break
    #         if chunk:
    #             yield chunk
    #         else:
    #             return

    # saliency_scores = {"12345": 12345, "2345": 2345, "345": 345, "45": 45, "5": 5}
    # keys = list(saliency_scores.keys())
    # batch_size = 2
    # fig_count = 0
    # for i, key_batch in enumerate(batch(keys, batch_size)):
    #     score_batch = [saliency_scores[key] for key in key_batch]
    #     for j in range(len(key_batch)):
    #         words = key_batch[j]
    #         scores = score_batch[j]
    #         import pdb
    #         pdb.set_trace()

    # saliency_scores_per_word = {"12345": [1,2,3,4,5], "2345": [2,3,4,5], "345": [3,4,5], "45": [4,5], "5": [5]}
    # batch_size = 2
    # highest = []
    # lowest = []
    # for i, items in enumerate(batch(saliency_scores_per_word.items(), batch_size)):
    #     mean_saliency = {key: statistics.mean(value) for key, value in items if len(value) > 2}
    #     highest_means = sorted(mean_saliency.items(), key=lambda x: x[1], reverse=True)[:2]
    #     lowest_means = sorted(mean_saliency.items(), key=lambda x: x[1])[:10]
    #     highest += highest_means
    #     lowest += lowest_means

    # highest_means = sorted(highest, key=lambda x: x[1], reverse=True)[:2]
    # lowest_means = sorted(lowest, key=lambda x: x[1])[:2]
    # print(f"Batch {i}: Top 10 highest means:", highest_means)
    # print(f"Batch {i}: Top 10 lowest means:", lowest_means)