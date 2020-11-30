"""
step20: Random Forest Classifier
"""
import argparse
import tarfile
import typing
import pandas
import matplotlib
import matplotlib.pyplot
import seaborn
import sklearn.ensemble
import sklearn.model_selection
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR file")
    parser.add_argument("--cpu", type=int, help="Number of CPU to use")

    args = parser.parse_args()

    if not args.output.endswith(".tar"):
        raise ValueError("Output file must end with .TAR!!")
    elif args.cpu < 1:
        raise ValueError("CPU must be greater than zero!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    tar_files: typing.List[str] = list()

    data = step00.read_pickle(args.input)
    data.drop(labels="ShortStage", axis="columns", inplace=True)
    train_columns = sorted(set(data.columns) - {"LongStage"})

    # Get Feature Importances
    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=args.cpu, random_state=0)
    classifier.fit(data[train_columns], data["LongStage"])
    feature_importances = list(classifier.feature_importances_)
    best_features = list(map(lambda x: x[1], sorted(list(filter(lambda x: x[0] > 0, zip(feature_importances, train_columns))), reverse=True)))

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

    # Run Accurary by Feature Counts
    accuracies = list()
    best_accuracy = (0, 0)
    for i in range(1, len(best_features) + 1):
        print("With", i, "/", len(best_features), "features!!")
        used_columns = best_features[:i]
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data[used_columns], data["LongStage"], test_size=0.1, random_state=0, stratify=data["LongStage"])

        classifier.fit(x_train, y_train)
        accuracy = classifier.score(x_test, y_test)
        accuracies.append((i, accuracy))

        if best_accuracy[0] < accuracy:
            best_accuracy = (accuracy, i)

    # Draw Accuracy
    score_data = pandas.DataFrame.from_records(accuracies, columns=["FeatureCount", "Accuracy"])
    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))
    seaborn.lineplot(data=score_data, x="FeatureCount", y="Accuracy", ax=ax)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.title("Best Accuracy at %s with %s features" % best_accuracy)
    tar_files.append("accuracy.png")
    fig.savefig(tar_files[-1])
    matplotlib.pyplot.close(fig)

    # Save data
    with tarfile.open(args.output, "w") as tar:
        for file_name in tar_files:
            tar.add(file_name, arcname=file_name)
