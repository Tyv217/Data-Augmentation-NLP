from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..data import IWSLT17DataModule, AGNewsDataModule, ColaDataModule, TwitterDataModule, BabeDataModule
from sentence_transformers import SentenceTransformer
from ..helpers import parse_augmentors
import torch
import heapq
from transformers import AutoModelForSequenceClassification

def visualize_data(args):
    if args.visualize == 1:
        plot_results(args)
    elif args.visualize == 2:
        visualize_augmentor_change_data(args)
    else:
        raise Exception("Incorrect visualize argument")

def plot_and_compare_emb(embeddings1, embeddings2, fig_name):
    pca = PCA(n_components=2)
    reduced1 = pca.fit_transform(embeddings1)
    reduced2 = pca.fit_transform(embeddings2)
    reduced1 = reduced1.transpose()
    reduced2 = reduced2.transpose()
    plt.scatter(reduced1[0], reduced1[1], s=5, c='blue')
    plt.scatter(reduced2[0], reduced2[1], s=5, c='orange')

    ax = plt.gca()
    abs_max = max(abs(max(ax.get_ylim(), key=abs)), abs(max(ax.get_xlim(), key=abs)))
    ax.set_ylim(ymin=-abs_max, ymax=abs_max)
    ax.set_xlim(xmin=-abs_max, xmax=abs_max)
    
    plt.savefig(fig_name)
    plt.show()
    
def plot_emb(embeddings, fig_name, datapoints):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    reduced = reduced.transpose()
    x_mean = np.mean(reduced[0])
    y_mean = np.mean(reduced[1])

    dist = np.sqrt(np.square(reduced[0] - x_mean) + np.square(reduced[1] - y_mean))
    if datapoints > 0:
        dist1 = np.copy(dist)
        dist1 = np.sort(dist1)
        num_samples = min(datapoints, len(dist1))
        cutoff = dist1[len(dist1) - num_samples]
    elif datapoints == -1:
        cutoff = np.median(dist)
    else:
        cutoff = -1

    x_coords = np.array(reduced[0])[dist >= cutoff]
    y_coords = np.array(reduced[1])[dist >= cutoff]

    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)

    plt.scatter(x_coords, y_coords, s=5, c='green')
    plt.scatter(x_mean, y_mean, s = 10, c='red')
    plt.errorbar(x_mean, y_mean, xerr = x_std, yerr = y_std, fmt = 'o', color = 'red')
    text = "Mean: ({x_mean:.2f},{y_mean:.2f})\nstd: ({x_std:.2f},{y_std:.2f})"\
        .format(x_mean = x_mean, y_mean = y_mean, x_std = x_std, y_std = y_std)

    ax = plt.gca()
    
    abs_max = 0.75
    ax.set_ylim(ymin=-abs_max, ymax=abs_max)
    ax.set_xlim(xmin=-abs_max, xmax=abs_max)
    
    plt.text(0.49,0.49,text,
        horizontalalignment='right',
        verticalalignment='top')
    plt.savefig(fig_name)
    plt.show()

def plot_saliency_scores(words, saliency_scores, fig_name):
    plt.figure()

    plt.bar(words.split(" "), saliency_scores)
    plt.ylim([0, 1]) # Set the y-axis limit to 0 and 1
    plt.xlabel('Words')
    plt.ylabel('Saliency Scores')
    plt.title('Word Saliency Scores')


    plt.savefig(fig_name)
    plt.show()
    
def plot_results(args):
    filename = args.filename
    file = filename + ".csv"
    error_file = filename + "_error_bars.csv"
    results = pd.read_csv("reports/data_for_plotting/" + file)
    try:
        error_bars = pd.read_csv("reports/data_for_plotting/" + error_file)
    except:
        error_bars = None
    first_col = results.columns[0]
    fig = plt.figure(figsize=(8,5))
    for col in results.columns[1:]:
        plt.plot(results[first_col], results[col], label=col)
        
    for col in results.columns[1:]:
        if error_bars is not None:
                plt.errorbar(results[first_col], results[col], error_bars[col],ecolor = 'red', marker='x', mfc='red',
                mec='red', ms=2, mew=1, capsize=5, elinewidth=1, fmt='none')
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.legend()
    plt.grid()
    plt.savefig("reports/figures/results/" + filename + ".png")

def visualize_augmentor_change_data(args):

    word_augmentors, _ = parse_augmentors(args)

    data_modules = {"cola": ColaDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "iwslt": IWSLT17DataModule}

    data = data_modules[args.dataset](
        dataset_percentage = 1,
        augmentors = [],
        batch_size = args.batch_size
    )

    data.prepare_data()
    data.setup("fit")

    train_data1 = list(data.get_dataset_text())
    # random.shuffle(train_data1)
    # train_data1 = train_data1[:1000]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    AUGMENT_LOOPS = 1
    train_data2 = train_data1.copy()
    print("Start augmenting!")
    for augmentor in word_augmentors:
    # start_time = time.time()
        for _ in range(AUGMENT_LOOPS):
            train_data2, _, _ = augmentor.augment_dataset(train_data2, None, None)

    if args.task == "classify":
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = len(data.id2label), id2label = data.id2label, label2id = data.label2id, problem_type="multi_label_classification", output_attentions=True).distilbert.embeddings
    elif args.task == "translate":
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    else:
        raise Exception("Incorrect task")
    
    total_difference = []

    for batch1, batch2 in zip(train_data1, train_data2):
        with torch.no_grad():
            batch_embeddings1 = model(batch1['input_id'])
            batch_embeddings2 = model(batch2['input_id'])
        difference = batch_embeddings2 - batch_embeddings1
        distance = torch.norm(difference, dim=2)
        for sentence1, sentence2, dist in zip(batch1['input_id'], batch2['input_id'], distance):
            if len(total_difference) < 1000:
                heapq.heappush(total_difference, (dist, difference, sentence1, sentence2))
            else:
                heapq.heappushpop(total_difference, (dist, difference, sentence1, sentence2))
    
    with open("highest_diff_data_" + args.task + "_" + args.dataset + ".txt", "a") as f:
        for dist,_, sentence1, sentence2 in sorted(total_difference, reverse=True):
            f.write("Sentence 1:", sentence1, "\n")
            f.write("Sentence 2:", sentence2, "\n")
            f.write("Distance:", dist, "\n")
            f.write("\n\n\n")

    _, difference, _, _ = zip(*total_difference)

    # cosine_similarities = cosine_similarity(embeddings1, embeddings2)

    # plot_and_compare_emb(embeddings1, embeddings2, args.task + '.png')

    plot_emb(list(difference), args.dataset + '_' + args.augmentors + str(AUGMENT_LOOPS) + "_" + str(args.datapoints) + '.png', args.datapoints)