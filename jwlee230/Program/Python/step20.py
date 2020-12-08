"""
step20: Random Forest Classifier
"""
import argparse
import itertools
import tarfile
import typing
import pandas
import matplotlib
import matplotlib.pyplot
import numpy
import scipy
import seaborn
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import statannot
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("tsne", type=str, help="t-SNE TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR file")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU to use")
    parser.add_argument("--one", action="store_true", default=False, help="Merge Healthy+Early")
    parser.add_argument("--two", action="store_true", default=False, help="Merge Moderate+Severe")

    args = parser.parse_args()

    if not args.output.endswith(".tar"):
        raise ValueError("Output file must end with .TAR!!")
    elif args.cpu < 1:
        raise ValueError("CPU must be greater than zero!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    tar_files: typing.List[str] = list()

    tsne_data = step00.read_pickle(args.tsne)
    tsne_data["ShortStage"] = list(map(lambda x: x[0] if x[0] == "H" else x[2], tsne_data["ID"]))
    tsne_data["LongStage"] = list(map(lambda x: {"H": "Healthy", "E": "Early", "M": "Moderate", "S": "Severe"}[x], tsne_data["ShortStage"]))

    data = step00.read_pickle(args.input)
    data.drop(labels="ShortStage", axis="columns", inplace=True)
    train_columns = sorted(set(data.columns) - {"LongStage"})

    if args.one:
        data["LongStage"] = list(map(lambda x: "Healthy+Early" if (x == "Healthy") or (x == "Early") else x, data["LongStage"]))
        tsne_data["LongStage"] = list(map(lambda x: "Healthy+Early" if (x == "Healthy") or (x == "Early") else x, tsne_data["LongStage"]))

    if args.two:
        data["LongStage"] = list(map(lambda x: "Moderate+Severe" if (x == "Moderate") or (x == "Severe") else x, data["LongStage"]))
        tsne_data["LongStage"] = list(map(lambda x: "Moderate+Severe" if (x == "Moderate") or (x == "Severe") else x, tsne_data["LongStage"]))

    print(tsne_data)
    print(data)

    # Get Feature Importances
    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=args.cpu, random_state=0)
    classifier.fit(data[train_columns], data["LongStage"])
    feature_importances = list(classifier.feature_importances_)
    best_features = list(map(lambda x: x[1], sorted(list(filter(lambda x: x[0] > 0, zip(feature_importances, train_columns))), reverse=True)))

    # Save Features
    tar_files.append("features.txt")
    with open(tar_files[-1], "w") as f:
        f.write("Order,Taxonomy Classification,Importances\n")
        for i, (feature, importance) in enumerate(zip(best_features, sorted(feature_importances, reverse=True))):
            f.write(str(i))
            f.write(",")
            f.write(" ".join(step00.simplified_taxonomy(feature).split("_")))
            f.write(",")
            f.write(str(importance))
            f.write("\n")

    # Draw Feature Importances
    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
    seaborn.histplot(data=feature_importances, kde=False, ax=ax)
    matplotlib.pyplot.title("Feature Importances by Feature Counts")
    matplotlib.pyplot.xlabel("Feature Importances")
    matplotlib.pyplot.ylabel("Counts")
    matplotlib.pyplot.grid(True)
    tar_files.append("importances.png")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Calculate Metrics by Feature Counts
    highest_metrics = {metric: (0, 0.0) for metric in step00.derivations}
    lowest_metrics = {metric: (0, 0.0) for metric in step00.derivations}
    scores = list()

    for i in range(1, len(best_features) + 1):
        print("With", i, "/", len(best_features), "features!!")
        used_columns = best_features[:i]
        tmp: typing.List[typing.Union[int, float]] = [i]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data[used_columns], data["LongStage"], test_size=0.1, random_state=0, stratify=data["LongStage"])

        classifier.fit(x_train, y_train)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test))

        for metric in step00.derivations:
            score = step00.aggregate_confusion_matrix(confusion_matrix, metric)
            tmp.append(score)

            if numpy.isnan(score):
                continue

            if (highest_metrics[metric][0] == 0) or (highest_metrics[metric][1] < score):
                highest_metrics[metric] = (i, score)

            if (lowest_metrics[metric][0] == 0) or (lowest_metrics[metric][1] > score):
                lowest_metrics[metric] = (i, score)

        scores.append(tmp)
    score_data = pandas.DataFrame.from_records(scores, columns=["FeatureCount"] + list(step00.derivations))
    print(score_data)

    # Draw Scores
    print("Drawing scores start!!")
    for metric in step00.derivations:
        print("--", metric)
        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
        seaborn.lineplot(data=score_data, x="FeatureCount", y=metric, ax=ax)
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.title("Higest with %s feature(s) at %.3f; Lowest with %s feature(s) at %.3f" % (highest_metrics[metric] + lowest_metrics[metric]))
        matplotlib.pyplot.ylim(0, 1)
        tar_files.append(metric + ".png")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing scores done!!")

    # Draw Trees
    print("Drawing highest trees start!!")
    for metric in step00.derivations:
        print("--", metric)

        if highest_metrics[metric][0] == 0:
            continue

        fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data[best_features[:highest_metrics[metric][0]]], data["LongStage"], test_size=0.1, random_state=0, stratify=data["LongStage"])
        sklearn.tree.plot_tree(classifier.fit(x_train, y_train).estimators_[0], ax=ax, filled=True, class_names=sorted(set(data["LongStage"])))
        matplotlib.pyplot.title("Highest %s with %s feature(s) at %.3f" % ((metric,) + highest_metrics[metric]))
        tar_files.append("highest_" + metric + ".png")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing highest trees done!!")

    print("Drawing lowest trees start!!")
    for metric in step00.derivations:
        print("--", metric)

        if lowest_metrics[metric][0] == 0:
            continue

        fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data[best_features[:lowest_metrics[metric][0]]], data["LongStage"], test_size=0.1, random_state=0, stratify=data["LongStage"])
        sklearn.tree.plot_tree(classifier.fit(x_train, y_train).estimators_[0], ax=ax, filled=True, class_names=sorted(set(data["LongStage"])))
        matplotlib.pyplot.title("Lowest %s with %s feature(s) at %.3f" % ((metric,) + lowest_metrics[metric]))
        tar_files.append("lowest_" + metric + ".png")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing lowest trees done!!")

    # Draw Violin Plots
    print("Drawing Violin plot start!!")
    for i, feature in enumerate(best_features):
        print("--", feature)

        seaborn.set(context="poster", style="whitegrid")

        fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
        seaborn.violinplot(data=data, x="LongStage", y=feature, order=sorted(set(data["LongStage"])), ax=ax)

        statannot.add_stat_annotation(ax, data=data, x="LongStage", y=feature, order=sorted(set(data["LongStage"])), test="t-test_ind", box_pairs=itertools.combinations(sorted(set(data["LongStage"])), 2), text_format="star", loc="inside", verbose=0)

        matplotlib.pyplot.title(" ".join(step00.simplified_taxonomy(feature).split("_")))
        tar_files.append("Feature_" + str(i) + ".png")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing Violin Plot done!!")

    # Draw scatter plot
    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
    seaborn.scatterplot(data=tsne_data, x="TSNE1", y="TSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order)
    tar_files.append("scatter.png")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Save data
    with tarfile.open(args.output, "w") as tar:
        for file_name in tar_files:
            tar.add(file_name, arcname=file_name)
