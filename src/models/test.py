def main():
    import torch

    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", problem_type="multi_label_classification")

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
    import numpy as np
    import pandas as pd
    from pathlib import Path
    # a = np.array([0,1,2,3,4,5,6,7,8,9])
    # df = pd.DataFrame(a)
    # import os
    # import pathlib
    # x = pathlib.Path(__file__).parent.resolve()
    # filepath = '../data/augmented_data/out.csv'
    # a = os.path.join(x, filepath)
    # df.to_csv(a, index = False) 

    # y = pd.read_csv(a)
    # print(y)
    main()
