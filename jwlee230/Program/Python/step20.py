"""
step20.py: Random Forest Classifier with ANCOM
"""
import argparse
import tarfile
import typing
import pandas
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import scipy.stats
import seaborn
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("tsne", type=str, help="t-SNE TAR.gz file")
    parser.add_argument("clinical", help="Clinical TSV file", type=str)
    parser.add_argument("output", type=str, help="Output TAR file")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU to use")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--one", action="store_true", default=False, help="Merge Healthy+Slight")
    group.add_argument("--two", action="store_true", default=False, help="Merge Moderate+Severe")
    group.add_argument("--three", action="store_true", default=False, help="Merge Slight+Moderate+Severe")

    args = parser.parse_args()

    if not args.output.endswith(".tar"):
        raise ValueError("Output file must end with .TAR!!")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("Clinical file must end with .TSV!!")
    elif args.cpu < 1:
        raise ValueError("CPU must be greater than zero!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    tar_files: typing.List[str] = list()

    clinical_data = pandas.read_csv(args.clinical, sep="\t", index_col=0, skiprows=[1])
    print(clinical_data)

    tsne_data = step00.read_pickle(args.tsne)
    tsne_data["ShortStage"] = clinical_data.loc[tsne_data.index, "ShortStage"]
    tsne_data["LongStage"] = clinical_data.loc[tsne_data.index, "LongStage"]
    print(tsne_data)

    data = step00.read_pickle(args.input)
    del data["ShortStage"]
    train_columns = list(data.columns)[:-1]

    if args.one:
        data = data.loc[(data["LongStage"].isin(["Healthy", "Stage I"]))]
        tsne_data = tsne_data.loc[(tsne_data["LongStage"].isin(["Healthy", "Stage I"]))]
        stage_list = ["Healthy", "Stage I"]
    elif args.two:
        data["LongStage"] = list(map(lambda x: "Stage II/III" if (x in ["Stage II", "Stage III"]) else x, data["LongStage"]))
        tsne_data["LongStage"] = list(map(lambda x: "Stage II/III" if (x in ["Stage II", "Stage III"]) else x, tsne_data["LongStage"]))
        stage_list = ["Healthy", "Stage I", "Stage II/III"]
    elif args.three:
        data["LongStage"] = list(map(lambda x: "Stage I/II/III" if (x in ["Stage I", "Stage II", "Stage III"]) else x, data["LongStage"]))
        tsne_data["LongStage"] = list(map(lambda x: "Stage I/II/III" if (x in ["Stage I", "Stage II", "Stage III"]) else x, tsne_data["LongStage"]))
        stage_list = ["Healthy", "Stage I/II/III"]
    else:
        stage_list = ["Healthy", "Stage I", "Stage II", "Stage III"]

    print(data)
    print(stage_list, set(data["LongStage"]))

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
    highest_metrics = {metric: (0, 0.0, 0.0) for metric in step00.selected_derivations + ["AUC"]}
    scores = list()

    k_fold = sklearn.model_selection.StratifiedKFold(n_splits=10)
    for i in tqdm.trange(1, len(best_features) + 1):
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

            for metric in step00.selected_derivations:
                score = step00.aggregate_confusion_matrix(confusion_matrix, metric)

                scores.append((i, metric, score))

                if metric in score_by_metric:
                    score_by_metric[metric].append(score)
                else:
                    score_by_metric[metric] = [score]

            label_binarizer = sklearn.preprocessing.LabelBinarizer().fit(data["LongStage"])
            y_onehot_test = label_binarizer.transform(y_test)
            y_score = classifier.predict_proba(x_test)

            if len(stage_list) > 2:
                for class_id, stage in enumerate(stage_list):
                    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_onehot_test[:, class_id], y_score[:, class_id])
                    roc_auc = sklearn.metrics.auc(fpr, tpr)

                    if "AUC" in score_by_metric:
                        score_by_metric["AUC"].append(roc_auc)
                    else:
                        score_by_metric["AUC"] = [roc_auc]

                    scores.append((i, "AUC", roc_auc))
            else:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_onehot_test, y_score[:, 1])
                roc_auc = sklearn.metrics.auc(fpr, tpr)

                if "AUC" in score_by_metric:
                    score_by_metric["AUC"].append(roc_auc)
                else:
                    score_by_metric["AUC"] = [roc_auc]

                scores.append((i, "AUC", roc_auc))

        for metric in step00.selected_derivations:
            tmp, std = numpy.mean(score_by_metric[metric], dtype=float), numpy.std(score_by_metric[metric], dtype=float)
            if (highest_metrics[metric][0] == 0) or (highest_metrics[metric][1] < tmp):
                highest_metrics[metric] = (i, tmp, std)

        tmp, std = numpy.mean(score_by_metric["AUC"], dtype=float), numpy.std(score_by_metric["AUC"], dtype=float)
        if (highest_metrics["AUC"][0] == 0) or (highest_metrics["AUC"][1] < tmp):
            highest_metrics["AUC"] = (i, tmp, std)

    score_data = pandas.DataFrame(scores, columns=["FeatureCount", "Metrics", "Value"])
    print(score_data)

    # Export score data
    tar_files.append("metrics.csv")
    with open(tar_files[-1], "w") as f:
        f.write("Count,Metrics,Mean,STD,95% CI\n")
        for i in tqdm.trange(1, len(best_features) + 1):
            for metric in sorted(step00.derivations + ["AUC"]):
                selected_data = list(score_data.loc[(score_data["FeatureCount"] == i) & (score_data["Metrics"] == metric)]["Value"])
                ci = scipy.stats.t.interval(0.95, len(selected_data) - 1, loc=numpy.mean(selected_data), scale=scipy.stats.sem(selected_data))
                f.write(f"{i},{metric},{numpy.mean(selected_data):.3f},{numpy.std(selected_data):.3f},{ci[0]:.3f},{ci[1]:.3f}\n")

    # ROC curves
    for metric in tqdm.tqdm(step00.selected_derivations + ["AUC"]):
        y_score = numpy.zeros((len(data), len(stage_list)), dtype=float)
        used_columns = best_features[:highest_metrics[metric][0]]

        for train_index, test_index in k_fold.split(data[used_columns], data["LongStage"]):
            x_train, x_test = data.iloc[train_index][used_columns], data.iloc[test_index][used_columns]
            y_train, y_test = data.iloc[train_index]["LongStage"], data.iloc[test_index]["LongStage"]

            y_score[test_index, :] = classifier.fit(x_train, y_train).predict_proba(x_test)

        label_binarizer = sklearn.preprocessing.LabelBinarizer().fit(data["LongStage"])
        y_onehot_test = label_binarizer.transform(data["LongStage"])

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        matplotlib.pyplot.plot([0, 1], [0, 1], linestyle="--", linewidth=10, color="k", alpha=0.5)
        if len(stage_list) > 2:
            for class_id, (stage, color) in enumerate(zip(stage_list, matplotlib.colors.TABLEAU_COLORS.keys())):
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_onehot_test[:, class_id], y_score[:, class_id])
                roc_auc = sklearn.metrics.auc(fpr, tpr)
                sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax, name=f"ROC curve for {stage}", color=color, linewidth=5)
        else:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_onehot_test, y_score[:, 1])
            roc_auc = numpy.mean(score_by_metric["AUC"], dtype=float)
            sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax, name="ROC curve", linewidth=5)

        matplotlib.pyplot.axis("square")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xlabel("1 - SPE")
        matplotlib.pyplot.ylabel("SEN")
        matplotlib.pyplot.title(f"ROC for {metric}")
        matplotlib.pyplot.legend()

        matplotlib.pyplot.tight_layout()
        tar_files.append(f"ROC_{metric}.png")
        fig.savefig(tar_files[-1])
        tar_files.append(f"ROC_{metric}.pdf")
        fig.savefig(tar_files[-1])
        tar_files.append(f"ROC_{metric}.svg")
        fig.savefig(tar_files[-1])

    # Draw Metrics
    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
    seaborn.lineplot(data=score_data, x="FeatureCount", y="Value", hue="Metrics", style="Metrics", ax=ax, legend="full", markers=True, markersize=20, hue_order=sorted(step00.selected_derivations + ["AUC"]))
    matplotlib.pyplot.axvline(x=highest_metrics["BA"][0], color="k", linestyle="--")
    matplotlib.pyplot.text(x=highest_metrics["BA"][0], y=0.3, s=f"Best BA {highest_metrics['BA'][1]:.3f}±{highest_metrics['BA'][2]:.3f} with {highest_metrics['BA'][0]} features", horizontalalignment="right", verticalalignment="center", rotation="vertical", fontsize="x-small", color="k")
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
    for metric in tqdm.tqdm(step00.selected_derivations):
        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
        seaborn.lineplot(data=score_data.query("Metrics == '%s'" % metric,), x="FeatureCount", y="Value", ax=ax, markers=True, markersize=20)
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.title(f"Best {highest_metrics[metric][0]} feature(s) at {highest_metrics[metric][1]:.3f}±{highest_metrics[metric][2]}")
        matplotlib.pyplot.tight_layout()
        tar_files.append(metric + ".png")
        fig.savefig(tar_files[-1])
        tar_files.append(metric + ".pdf")
        fig.savefig(tar_files[-1])
        tar_files.append(metric + ".svg")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)

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
        for file_name in tqdm.tqdm(tar_files):
            tar.add(file_name, arcname=file_name)
