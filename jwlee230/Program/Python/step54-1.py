"""
step54-1.py: Random Forest Classifier
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
import statannotations.Annotator
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
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
    print(set(clinical_data["DB"]))

    data = step00.read_pickle(args.input).dropna(axis="index")
    del data["ShortStage"]
    train_columns = sorted(set(data.columns) - {"LongStage"})

    if args.one:
        data = data.loc[(data["LongStage"].isin(["Healthy", "Stage I"]))]
        stage_list = ["Healthy", "Stage I"]
    elif args.two:
        data["LongStage"] = list(map(lambda x: "Stage II/III" if (x in ["Stage II", "Stage III"]) else x, data["LongStage"]))
        stage_list = ["Healthy", "Stage I", "Stage II/III"]
    elif args.three:
        data["LongStage"] = list(map(lambda x: "Stage I/II/III" if (x in ["Stage I", "Stage II", "Stage III"]) else x, data["LongStage"]))
        stage_list = ["Healthy", "Stage I/II/III"]
    else:
        data = data.loc[(data["LongStage"].isin(["Healthy", "Stage I", "Stage II", "Stage III"]))]
        stage_list = ["Healthy", "Stage I", "Stage II", "Stage III"]

    print(data)
    print(stage_list, set(data["LongStage"]))

    Korea_data = data.loc[list(filter(lambda x: x in set(clinical_data[(clinical_data["DB"] == "Korea")].index), list(data.index))), :]
    Spain_data = data.loc[list(filter(lambda x: x in set(clinical_data[(clinical_data["DB"] == "Spain")].index), list(data.index))), :]
    Portugal_data = data.loc[list(filter(lambda x: x in set(clinical_data[(clinical_data["DB"] == "Portugal")].index), list(data.index))), :]

    # Get Feature Importances
    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=args.cpu, random_state=42)
    classifier.fit(Korea_data[train_columns], Korea_data["LongStage"])
    feature_importances = list(classifier.feature_importances_)
    best_features = list(map(lambda x: x[1], sorted(list(filter(lambda x: (x[0] > 0) and (step00.consistency_taxonomy(x[1]).count(";") == 6), zip(feature_importances, train_columns))), reverse=True)))

    # Save Features
    tar_files.append("features.csv")
    with open(tar_files[-1], "w") as f:
        f.write("Order,Taxonomy Classification,Importances\n")
        for num, (feature, importance) in enumerate(zip(best_features, sorted(feature_importances, reverse=True))):
            f.write(str(num))
            f.write(",")
            f.write(" ".join(step00.consistency_taxonomy(feature).split("; ")[5:]))
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
    tar_files.append("importances.pdf")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Calculate Metrics by Feature Counts
    scores = list()
    k_fold = sklearn.model_selection.StratifiedKFold(n_splits=10)

    for i in tqdm.trange(1, len(best_features) + 1):
        used_columns = best_features[:i]

        for train_index, test_index in k_fold.split(Korea_data[used_columns], Korea_data["LongStage"]):
            x_train, x_test = data.iloc[train_index][used_columns], data.iloc[test_index][used_columns]
            y_train, y_test = data.iloc[train_index]["LongStage"], data.iloc[test_index]["LongStage"]

            classifier.fit(x_train, y_train)

            if args.one:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I"])
            elif args.two:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I", "Stage II/III"]), axis=0)
            elif args.three:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I/II/III"])
            else:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test)), axis=0)

            for metric in step00.derivations:
                score = step00.aggregate_confusion_matrix(confusion_matrix, metric)
                scores.append((i, "Korean", metric, score))

    for i in tqdm.trange(1, len(best_features) + 1):
        used_columns = best_features[:i]

        for train_index, test_index in k_fold.split(Spain_data[used_columns], Spain_data["LongStage"]):
            x_train, x_test = data.iloc[train_index][used_columns], data.iloc[test_index][used_columns]
            y_train, y_test = data.iloc[train_index]["LongStage"], data.iloc[test_index]["LongStage"]

            classifier.fit(x_train, y_train)

            if args.one:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I"])
            elif args.two:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I", "Stage II/III"]), axis=0)
            elif args.three:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I/II/III"])
            else:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test)), axis=0)

            for metric in step00.derivations:
                score = step00.aggregate_confusion_matrix(confusion_matrix, metric)
                scores.append((i, "Spanish", metric, score))

    for i in tqdm.trange(1, len(best_features) + 1):
        used_columns = best_features[:i]

        for train_index, test_index in k_fold.split(Portugal_data[used_columns], Portugal_data["LongStage"]):
            x_train, x_test = data.iloc[train_index][used_columns], data.iloc[test_index][used_columns]
            y_train, y_test = data.iloc[train_index]["LongStage"], data.iloc[test_index]["LongStage"]

            classifier.fit(x_train, y_train)

            if args.one:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I"])
            elif args.two:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I", "Stage II/III"]), axis=0)
            elif args.three:
                confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test), labels=["Healthy", "Stage I/II/III"])
            else:
                confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test)), axis=0)

            for metric in step00.derivations:
                score = step00.aggregate_confusion_matrix(confusion_matrix, metric)
                scores.append((i, "Portuguese", metric, score))

    score_data = pandas.DataFrame(scores, columns=["FeatureCount", "DB", "Metrics", "Value"])
    print(score_data)

    # Export score data
    tar_files.append("metrics.csv")
    with open(tar_files[-1], "w") as f:
        f.write("Count,Metrics,Mean\n")
        for i in tqdm.trange(1, len(best_features) + 1):
            for metric in sorted(step00.derivations):
                selected_data = list(score_data.loc[(score_data["FeatureCount"] == i) & (score_data["Metrics"] == metric)]["Value"])
                f.write(f"{i},{metric},{numpy.mean(selected_data):.3f}\n")

    # Draw Metrics
    for i in tqdm.trange(1, len(best_features) + 1):
        drawing_data = score_data.loc[(score_data["FeatureCount"] == i)]

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))
        seaborn.barplot(data=drawing_data, x="Metrics", y="Value", hue="DB", order=sorted(step00.selected_derivations), hue_order=["Korean", "Spanish", "Portuguese"], ax=ax)
        statannotations.Annotator.Annotator(ax, [((metric, "Korean"), (metric, "Spanish")) for metric in sorted(step00.selected_derivations)] + [((metric, "Korean"), (metric, "Portuguese")) for metric in sorted(step00.selected_derivations)], data=drawing_data, x="Metrics", y="Value", hue="DB", order=sorted(step00.selected_derivations), hue_order=["Korean", "Spanish", "Portuguese"]).configure(test="Mann-Whitney", text_format="star", loc="inside", comparisons_correction=None, verbose=True).apply_and_annotate()

        matplotlib.pyplot.title("Write something")
        # matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.ylabel("Evaluations")
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.legend(loc="lower left")
        matplotlib.pyplot.tight_layout()

        tar_files.append(f"metrics_{i}.pdf")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)

    # Draw Violin Plots
    for i, feature in tqdm.tqdm(enumerate(best_features[:5])):
        order = sorted(set(Spain_data["LongStage"])) if (args.two or args.three) else ["Healthy", "Stage I", "Stage II", "Stage III"]

        box_pairs = list()
        for s1, s2 in itertools.combinations(order, 2):
            try:
                _, p = scipy.stats.mannwhitneyu(Spain_data.loc[(Spain_data["LongStage"] == s1), feature], Spain_data.loc[(Spain_data["LongStage"] == s2), feature])
            except ValueError:
                continue
            if p < 0.05:
                box_pairs.append((s1, s2))

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))
        if args.two or args.three:
            seaborn.violinplot(data=Spain_data, x="LongStage", y=feature, order=order, ax=ax, inner="box", cut=1, linewidth=10)
        else:
            seaborn.violinplot(data=Spain_data, x="LongStage", y=feature, order=order, ax=ax, inner="box", cut=1, palette=step00.color_stage_dict, linewidth=10)

        if box_pairs:
            try:
                statannot.add_stat_annotation(ax, data=Spain_data, x="LongStage", y=feature, order=order, test="Mann-Whitney", box_pairs=box_pairs, text_format="star", loc="inside", verbose=0, comparisons_correction=None)
            except ValueError:
                pass
        stat, p = scipy.stats.kruskal(*[Spain_data.loc[(Spain_data["LongStage"] == stage), feature] for stage in order])

        matplotlib.pyplot.title(" ".join(feature.split("; ")[-2:]).replace("_", " ") + f" (K.W. p={p:.2e})", fontsize=50)
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.ylabel("Proportion in Spanish")
        matplotlib.pyplot.tight_layout()

        tar_files.append(f"Spain_{i}.pdf")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)

    for i, feature in tqdm.tqdm(enumerate(best_features[:5])):
        order = sorted(set(Portugal_data["LongStage"])) if (args.two or args.three) else ["Healthy", "Stage I", "Stage II", "Stage III"]

        box_pairs = list()
        for s1, s2 in itertools.combinations(order, 2):
            try:
                _, p = scipy.stats.mannwhitneyu(Portugal_data.loc[(Portugal_data["LongStage"] == s1), feature], Portugal_data.loc[(Portugal_data["LongStage"] == s2), feature])
            except ValueError:
                continue
            if p < 0.05:
                box_pairs.append((s1, s2))

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))
        if args.two or args.three:
            seaborn.violinplot(data=Portugal_data, x="LongStage", y=feature, order=order, ax=ax, inner="box", cut=1, linewidth=10)
        else:
            seaborn.violinplot(data=Portugal_data, x="LongStage", y=feature, order=order, ax=ax, inner="box", cut=1, palette=step00.color_stage_dict, linewidth=10)

        if box_pairs:
            statannot.add_stat_annotation(ax, data=Portugal_data, x="LongStage", y=feature, order=order, test="Mann-Whitney", box_pairs=box_pairs, text_format="star", loc="inside", verbose=0, comparisons_correction=None)
        stat, p = scipy.stats.kruskal(*[Portugal_data.loc[(Portugal_data["LongStage"] == stage), feature] for stage in order])

        matplotlib.pyplot.title(" ".join(feature.split("; ")[-2:]).replace("_", " ") + f" (K.W. p={p:.2e})", fontsize=50)
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.ylabel("Proportion in Portuguese")
        matplotlib.pyplot.tight_layout()

        tar_files.append(f"Portugal_{i}.pdf")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)

    # Save data
    with tarfile.open(args.output, "w") as tar:
        for file_name in tqdm.tqdm(tar_files):
            tar.add(file_name, arcname=file_name)
