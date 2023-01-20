"""
step20: Random Forest Classifier with ANCOM
"""
import argparse
import itertools
import tarfile
import typing
import pandas
import matplotlib
import matplotlib.pyplot
import numpy
import scipy.stats
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

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--one", action="store_true", default=False, help="Merge Healthy+Slight")
    group.add_argument("--two", action="store_true", default=False, help="Merge Moderate+Severe")
    group.add_argument("--three", action="store_true", default=False, help="Merge Slight+Moderate+Severe")

    args = parser.parse_args()

    if not args.output.endswith(".tar"):
        raise ValueError("Output file must end with .TAR!!")
    elif args.cpu < 1:
        raise ValueError("CPU must be greater than zero!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    tar_files: typing.List[str] = list()

    tsne_data = step00.read_pickle(args.tsne)
    tsne_data["ShortStage"] = list(map(step00.change_ID_into_short_stage, tsne_data["ID"]))
    tsne_data["LongStage"] = list(map(step00.change_short_into_long, tsne_data["ShortStage"]))

    data = step00.read_pickle(args.input)
    del data["ShortStage"]
    train_columns = list(data.columns)[:-1]

    if args.one:
        data = data.loc[(data["LongStage"].isin(["Healthy", "Slight"]))]
        tsne_data = tsne_data.loc[(tsne_data["LongStage"].isin(["Healthy", "Slight"]))]
    elif args.two:
        data["LongStage"] = list(map(lambda x: "Moderate+Severe" if (x in ["Moderate", "Severe"]) else x, data["LongStage"]))
        tsne_data["LongStage"] = list(map(lambda x: "Moderate+Severe" if (x in ["Moderate", "Severe"]) else x, tsne_data["LongStage"]))
    elif args.three:
        data["LongStage"] = list(map(lambda x: "Slight+Moderate+Severe" if (x in ["Slight", "Moderate", "Severe"]) else x, data["LongStage"]))
        tsne_data["LongStage"] = list(map(lambda x: "Slight+Moderate+Severe" if (x in ["Slight", "Moderate", "Severe"]) else x, tsne_data["LongStage"]))

    print(tsne_data)
    print(data)
    print(set(data["LongStage"]))

    # Get Feature Importances
    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=args.cpu, random_state=42)
    classifier.fit(data[train_columns], data["LongStage"])
    feature_importances = list(classifier.feature_importances_)
    best_features = list(map(lambda x: x[1], sorted(zip(feature_importances, train_columns), reverse=True)))

    # Save Features
    tar_files.append("Features.csv")
    with open(tar_files[-1], "w") as f:
        f.write("Order,Taxonomy Classification,Importances\n")
        for i, (feature, importance) in enumerate(zip(best_features, sorted(feature_importances, reverse=True))):
            f.write(str(i))
            f.write(",")
            f.write(" ".join(feature.split("; ")[5:]).replace("_", " "))
            f.write(",")
            f.write(f"{importance:.3f}")
            f.write("\n")

    # Draw Feature Importances
    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
    seaborn.histplot(data=feature_importances, kde=False, ax=ax)
    matplotlib.pyplot.title("Feature Importances by Feature Counts")
    matplotlib.pyplot.xlabel("Feature Importances")
    matplotlib.pyplot.ylabel("Counts")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    tar_files.append("importances.png")
    fig.savefig(tar_files[-1])
    tar_files.append("importances.pdf")
    fig.savefig(tar_files[-1])
    tar_files.append("importances.svg")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Calculate Metrics by Feature Counts
    highest_metrics = {metric: (0, 0.0) for metric in step00.derivations}
    lowest_metrics = {metric: (0, 0.0) for metric in step00.derivations}
    scores = list()

    k_fold = sklearn.model_selection.StratifiedKFold(n_splits=10)
    for i in range(1, len(best_features) + 1):
        print("With", i, "/", len(best_features), "features!!")
        used_columns = best_features[:i]
        score_by_metric: typing.Dict[str, typing.List[float]] = dict()

        for train_index, test_index in k_fold.split(data[used_columns], data["LongStage"]):
            x_train, x_test = data.iloc[train_index][used_columns], data.iloc[test_index][used_columns]
            y_train, y_test = data.iloc[train_index]["LongStage"], data.iloc[test_index]["LongStage"]

            classifier.fit(x_train, y_train)
            if args.one or args.three:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test))
            else:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test)), axis=0)

            for metric in step00.derivations:
                score = step00.aggregate_confusion_matrix(confusion_matrix, metric)

                scores.append((i, metric, score))

                if metric in score_by_metric:
                    score_by_metric[metric].append(score)
                else:
                    score_by_metric[metric] = [score]

        for metric in step00.derivations:
            tmp = numpy.mean(score_by_metric[metric])
            if (highest_metrics[metric][0] == 0) or (highest_metrics[metric][1] < tmp):
                highest_metrics[metric] = (i, tmp)

            if (lowest_metrics[metric][0] == 0) or (lowest_metrics[metric][1] > tmp):
                lowest_metrics[metric] = (i, tmp)

    score_data = pandas.DataFrame.from_records(scores, columns=["FeatureCount", "Metrics", "Value"])
    print(score_data)

    # Export score data
    tar_files.append("metrics.csv")
    with open(tar_files[-1], "w") as f:
        f.write("Count,Metrics,Mean,STD,95% CI\n")
        for i in range(1, len(best_features) + 1):
            for metric in sorted(step00.derivations):
                selected_data = list(score_data.loc[(score_data["FeatureCount"] == i) & (score_data["Metrics"] == metric)]["Value"])
                ci = scipy.stats.t.interval(0.95, len(selected_data) - 1, loc=numpy.mean(selected_data), scale=scipy.stats.sem(selected_data))
                f.write("%d,%s,%.3f,%.3f,%.3f-%.3f\n" % (i, metric, numpy.mean(selected_data), numpy.std(selected_data), ci[0], ci[1]))

    # Draw Metrics
    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
    seaborn.lineplot(data=score_data.loc[score_data["Metrics"].isin(step00.selected_derivations)], x="FeatureCount", y="Value", hue="Metrics", style="Metrics", ax=ax, legend="full", markers=True, markersize=20, hue_order=sorted(step00.selected_derivations))
    matplotlib.pyplot.axvline(x=highest_metrics["BA"][0], color="k", linestyle="--")
    matplotlib.pyplot.text(x=highest_metrics["BA"][0], y=0.3, s=f"Highest BA {highest_metrics['BA'][1]:.3f} with {highest_metrics['BA'][0]} features", horizontalalignment="right", verticalalignment="center", rotation="vertical", fontsize="x-small", color="k")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.ylabel("Evaluations")
    matplotlib.pyplot.xlabel("Feature Counts")
    matplotlib.pyplot.xticks(sorted(set(score_data["FeatureCount"])), sorted(set(score_data["FeatureCount"])))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("Write something")
    ax.invert_xaxis()
    matplotlib.pyplot.tight_layout()
    tar_files.append("metrics.png")
    fig.savefig(tar_files[-1])
    tar_files.append("metrics.pdf")
    fig.savefig(tar_files[-1])
    tar_files.append("metrics.svg")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Draw Each Metics
    print("Drawing scores start!!")
    for metric in step00.selected_derivations:
        print("--", metric)
        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
        seaborn.lineplot(data=score_data.query("Metrics == '%s'" % metric,), x="FeatureCount", y="Value", ax=ax, markers=True, markersize=20)
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.title("Higest with %s feature(s) at %.3f" % highest_metrics[metric])
        matplotlib.pyplot.tight_layout()
        tar_files.append(metric + ".png")
        fig.savefig(tar_files[-1])
        tar_files.append(metric + ".pdf")
        fig.savefig(tar_files[-1])
        tar_files.append(metric + ".svg")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing scores done!!")

    # Draw Trees
    print("Drawing highest trees start!!")
    for metric in step00.selected_derivations:
        print("--", metric)

        if highest_metrics[metric][0] == 0:
            continue

        fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
        sklearn.tree.plot_tree(classifier.fit(data[best_features[:highest_metrics[metric][0]]], data["LongStage"]).estimators_[0], ax=ax, filled=True, class_names=sorted(set(data["LongStage"])))
        matplotlib.pyplot.title("Highest %s with %s feature(s) at %.3f" % ((metric,) + highest_metrics[metric]))
        matplotlib.pyplot.tight_layout()
        tar_files.append("highest_" + metric + ".png")
        fig.savefig(tar_files[-1])
        tar_files.append("highest_" + metric + ".pdf")
        fig.savefig(tar_files[-1])
        tar_files.append("highest_" + metric + ".svg")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing highest trees done!!")

    # Draw Violin Plots
    print("Drawing Violin plot start!!")
    for i, feature in enumerate(best_features[:9]):
        print("--", i, feature)

        order = sorted(set(data["LongStage"])) if (args.two or args.three) else step00.long_stage_order
        box_pairs = list()
        for s1, s2 in itertools.product(order, repeat=2):
            if s1 == s2:
                continue
            _, p = scipy.stats.mannwhitneyu(data.loc[(data["LongStage"] == s1), feature], data.loc[(data["LongStage"] == s2), feature])
            if (p < 0.05) and ((s2, s1) not in box_pairs):
                box_pairs.append((s1, s2))

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))
        if args.two or args.three:
            seaborn.violinplot(data=data, x="LongStage", y=feature, order=order, ax=ax, inner="box", cut=1, linewidth=5)
        else:
            seaborn.violinplot(data=data, x="LongStage", y=feature, order=order, ax=ax, inner="box", cut=1, palette=step00.color_stage_dict, linewidth=5)

        if box_pairs:
            statannot.add_stat_annotation(ax, data=data, x="LongStage", y=feature, order=order, test="Mann-Whitney", box_pairs=box_pairs, text_format="star", loc="inside", verbose=1, comparisons_correction=None)
        stat, p = scipy.stats.kruskal(*[data.loc[(data["LongStage"] == stage), feature] for stage in order])

        matplotlib.pyplot.title(" ".join(feature.split("; ")[-2:]).replace("_", " ") + f" (K.W. p={p:.2e})", fontsize=50)
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.ylabel("Abundance")
        matplotlib.pyplot.tight_layout()

        tar_files.append("Feature_" + str(i) + ".png")
        fig.savefig(tar_files[-1])
        tar_files.append("Feature_" + str(i) + ".pdf")
        fig.savefig(tar_files[-1])
        tar_files.append("Feature_" + str(i) + ".svg")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)
    print("Drawing Violin Plot done!!")

    # Draw scatter plot
    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
    seaborn.scatterplot(data=tsne_data, x="tSNE1", y="tSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", s=1000)
    matplotlib.pyplot.tight_layout()
    tar_files.append("scatter.png")
    fig.savefig(tar_files[-1])
    tar_files.append("scatter.pdf")
    fig.savefig(tar_files[-1])
    tar_files.append("scatter.svg")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Save data
    with tarfile.open(args.output, "w") as tar:
        for file_name in tar_files:
            tar.add(file_name, arcname=file_name)
