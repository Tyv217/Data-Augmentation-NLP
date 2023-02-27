def main():
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    import torch

    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length = 256)

    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # the following 2 hyperparameters are task-specific

    max_source_length = 256

    max_target_length = 128

    # Suppose we have the following 2 training examples:

    input_sequence_1 = "Welcome to NYC"

    output_sequence_1 = "Bienvenue à NYC"

    input_sequence_2 = "HuggingFace is a company"

    output_sequence_2 = "HuggingFace est une entreprise"

    # encode the inputs

    task_prefix = "translate English to French: "

    input_sequences = [input_sequence_1, input_sequence_2]

    encoding = tokenizer(

        [task_prefix + sequence for sequence in input_sequences],

        padding="longest",

        truncation=True,

        return_tensors="pt",

    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets

    target_encoding = tokenizer(

        [output_sequence_1, output_sequence_2],

        padding="longest",

        truncation=True,

        return_tensors="pt",

    )

    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss

    labels[labels == tokenizer.pad_token_id] = -100

    # forward pass

    print(attention_mask)

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    loss.item()

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    a = np.array([0,1,2,3,4,5,6,7,8,9])
    q_high = np.quantile(a, 0.75)
    b = np.array([9,8,7,6,5,4,3,2,1,0])
    print(b[a > q_high])

