"""Classify Chinese headlines with unigram and bigram bag-of-words features."""

from collections import Counter
from itertools import product
from random import Random
import re

import jieba
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .naive_bayes import SparseMultinomialNB
from .reporting import draw_confusion, multiclass_scores, save_figure, write_metrics


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+")
ALPHAS = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0)
NGRAM_RANGES = ((1, 1), (1, 2))


def read_headlines(path):
    headlines, labels = [], []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            title, tab, label = line.strip("\r\n").rpartition("\t")
            if not tab or not title.strip() or not label.strip():
                raise ValueError(f"{path.name}, line {line_number}: expected title<TAB>category.")
            headlines.append(title.strip())
            labels.append(label.strip())
    if not headlines:
        raise ValueError(f"{path.name} contains no samples.")
    return headlines, np.array(labels)


def read_stopwords(path):
    if not path.is_file():
        return set()
    return set(path.read_text(encoding="utf-8-sig").split())


def tokenize(headline, stopwords):
    words = jieba.lcut(headline.lower())
    return [
        token
        for word in words
        for token in TOKEN_PATTERN.findall(word)
        if token not in stopwords
    ]


def demo_news(seed):
    topics = {
        "体育": (("球队", "球员", "俱乐部", "教练", "主队"),
                 ("备战联赛", "公布阵容", "举行训练", "争夺冠军"), 92),
        "科技": (("芯片", "传感器", "电池", "机器人", "卫星"),
                 ("完成测试", "通过评估", "更新方案", "启动研发"), 76),
        "财经": (("证券", "基金", "债券", "期货", "理财产品"),
                 ("公布收益", "调整费率", "迎来交易", "报告行情"), 62),
        "文化": (("剧场", "美术馆", "图书馆", "出版社", "博物馆"),
                 ("策划展览", "介绍作品", "举办讲座", "开放预约"), 49),
    }
    rng = Random(seed)
    prefixes = ("本周", "动态", "最新", "观察", "简报")
    splits = {name: [] for name in ("train", "validation", "test")}
    for label, (subjects, actions, count) in topics.items():
        examples = [(f"{prefix}｜{subject}{action}", label)
                    for prefix, subject, action in product(prefixes, subjects, actions)]
        rng.shuffle(examples)
        train_end, val_end = round(count * 0.7), round(count * 0.85)
        splits["train"].extend(examples[:train_end])
        splits["validation"].extend(examples[train_end:val_end])
        splits["test"].extend(examples[val_end:count])
    datasets = {}
    for split, rows in splits.items():
        rng.shuffle(rows)
        titles, labels = zip(*rows)
        datasets[split] = (titles, np.array(labels))
    return datasets, {"的", "了", "和", "在"}


def dataset_overview(datasets):
    overview = {}
    for split, (titles, labels) in datasets.items():
        p25, median, p75 = np.percentile([len(title) for title in titles], [25, 50, 75])
        overview[split] = {
            "samples": len(titles),
            "class_counts": dict(sorted(Counter(labels).items())),
            "title_length": {"p25": float(p25), "median": float(median), "p75": float(p75)},
        }
    return overview


def plot_overview(datasets, labels, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    counts = Counter(datasets["train"][1])
    axes[0].bar(range(len(labels)), [counts[label] for label in labels], color="#326684")
    axes[0].set(title="Training class counts", xlabel="Class code", ylabel="Headlines")
    axes[0].set_xticks(range(len(labels)))

    lengths = {split: [len(title) for title in titles] for split, (titles, _) in datasets.items()}
    max_length = max(map(max, lengths.values()))
    bins = np.linspace(0, max_length + 1, num=26)
    for split, values in lengths.items():
        axes[1].hist(values, bins=bins, density=True, histtype="step", linewidth=1.6, label=split)
    axes[1].set(title="Headline length by split", xlabel="Characters", ylabel="Density")
    axes[1].legend(frameon=False)
    save_figure(fig, out_dir / "dataset-overview.png")


def run(args):
    if args.demo:
        datasets, stopwords = demo_news(args.seed)
    else:
        datasets = {
            split: read_headlines(args.data_dir / file)
            for split, file in (("train", "train.txt"), ("validation", "dev.txt"), ("test", "test.txt"))
        }
        stopwords = read_stopwords(args.data_dir / "stopwords.txt")

    train_titles, train_y = datasets["train"]
    val_titles, val_y = datasets["validation"]
    test_titles, test_y = datasets["test"]
    labels = np.unique(train_y)
    if len(labels) < 2 or not np.isin(np.concatenate((val_y, test_y)), labels).all():
        raise ValueError("Train on at least two categories and include every validation/test category.")
    train_tokens = [" ".join(tokenize(title, stopwords)) for title in train_titles]
    val_tokens = [" ".join(tokenize(title, stopwords)) for title in val_titles]

    features_by_range = {}
    validation = []
    for ngrams in NGRAM_RANGES:
        vectorizer = CountVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False, min_df=2,
            max_features=10_000, ngram_range=ngrams,
        )
        train_x = vectorizer.fit_transform(train_tokens)
        val_x = vectorizer.transform(val_tokens)
        features_by_range[ngrams] = (vectorizer, train_x, val_x)
        for alpha in ALPHAS:
            model = MultinomialNB(alpha=alpha).fit(train_x, train_y)
            scores = multiclass_scores(val_y, model.predict(val_x), labels)
            validation.append({"ngram_range": ngrams, "alpha": alpha, "features": train_x.shape[1], **scores})

    best = max(validation, key=lambda row: row["macro_f1"])
    test_tokens = [" ".join(tokenize(title, stopwords)) for title in test_titles]
    vectorizer, train_x, _ = features_by_range[best["ngram_range"]]
    test_x = vectorizer.transform(test_tokens)
    tuned = MultinomialNB(alpha=best["alpha"]).fit(train_x, train_y)
    tuned_predictions = tuned.predict(test_x)

    unigram, train_uni, val_uni = features_by_range[(1, 1)]
    test_uni = test_x if unigram is vectorizer else unigram.transform(test_tokens)
    reference = MultinomialNB(alpha=1.0).fit(train_uni, train_y)
    scratch = SparseMultinomialNB(alpha=1.0).fit(train_uni, train_y)
    parity = {}
    for split, matrix, actual in (("validation", val_uni, val_y), ("test", test_uni, test_y)):
        reference_pred, scratch_pred = reference.predict(matrix), scratch.predict(matrix)
        parity[split] = {
            "equal_predictions": bool(np.array_equal(reference_pred, scratch_pred)),
            "sklearn": multiclass_scores(actual, reference_pred, labels),
            "from_scratch": multiclass_scores(actual, scratch_pred, labels),
        }

    tfidf = TfidfTransformer()
    tfidf_nb = MultinomialNB(alpha=best["alpha"]).fit(tfidf.fit_transform(train_x), train_y)
    svc = LinearSVC(random_state=args.seed, dual="auto").fit(train_uni, train_y)
    test_scores = {
        "BOW + MultinomialNB": multiclass_scores(test_y, tuned_predictions, labels),
        "TF-IDF + MultinomialNB": multiclass_scores(test_y, tfidf_nb.predict(tfidf.transform(test_x)), labels),
        "unigram BOW + LinearSVC": multiclass_scores(test_y, svc.predict(test_uni), labels),
    }
    confusion = confusion_matrix(test_y, tuned_predictions, labels=labels)
    errors = [
        {"actual": labels[row], "predicted": labels[col], "count": int(confusion[row, col])}
        for row, col in zip(*np.nonzero(confusion))
        if row != col
    ]
    report = {
        "dataset": "synthetic headlines" if args.demo else "local headline splits",
        "seed": args.seed,
        "overview": dataset_overview(datasets),
        "validation_grid": validation,
        "selected_nb": best,
        "unigram_alpha_1_parity": parity,
        "test": test_scores,
        "label_order": labels.tolist(),
        "test_confusion": confusion.tolist(),
        "test_classification_report": classification_report(
            test_y, tuned_predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "frequent_confusions": sorted(errors, key=lambda pair: pair["count"], reverse=True)[:10],
    }
    write_metrics(args.out, report)
    for model_name, scores in test_scores.items():
        print(f"{model_name}: accuracy={scores['accuracy']:.4f}, macro-F1={scores['macro_f1']:.4f}")

    plot_overview(datasets, labels, args.out)
    fig, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    draw_confusion(axis, confusion, range(len(labels)), "Tuned bag-of-words Naive Bayes")
    save_figure(fig, args.out / "confusion.png")
