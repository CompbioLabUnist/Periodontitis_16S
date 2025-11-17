"""
step56.py: Random Forest Classifier
"""
import argparse
import typing
import pandas
import numpy
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("clinical", help="Clinical TSV file", type=str)
    parser.add_argument("output", type=str, help="Output TSV file")
    parser.add_argument("--features", type=int, default=-1, help="Select number of features")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU to use")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--one", action="store_true", default=False, help="Merge Healthy+Slight")
    group.add_argument("--two", action="store_true", default=False, help="Merge Moderate+Severe")
    group.add_argument("--three", action="store_true", default=False, help="Merge Slight+Moderate+Severe")

    args = parser.parse_args()

    if not args.output.endswith(".tsv"):
        raise ValueError("Output file must end with .TSV!!")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("Clinical file must end with .TSV!!")
    elif args.cpu < 1:
        raise ValueError("CPU must be greater than zero!!")

    clinical_data = pandas.read_csv(args.clinical, sep="\t", index_col=0, skiprows=[1])
    print(clinical_data)

    data = step00.read_pickle(args.input).dropna(axis="index")
    del data["ShortStage"]
    train_columns = list(data.columns)[:-1]

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
        stage_list = ["Healthy", "Stage I", "Stage II", "Stage III"]

    print(data)
    print(stage_list, set(data["LongStage"]))

    # Get Feature Importances
    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=args.cpu, random_state=42)
    classifier.fit(data[train_columns], data["LongStage"])
    feature_importances = list(classifier.feature_importances_)
    best_features = list(map(lambda x: x[1], sorted(zip(feature_importances, train_columns), reverse=True)))

    if args.features == -1:
        best_features = best_features[:args.features]

    scores = list()

    k_fold = sklearn.model_selection.StratifiedKFold(n_splits=10)
    score_by_metric: typing.Dict[str, typing.List[float]] = dict()

    for train_index, test_index in k_fold.split(data[best_features], data["LongStage"]):
        x_train, x_test = data.iloc[train_index][best_features], data.iloc[test_index][best_features]
        y_train, y_test = data.iloc[train_index]["LongStage"], data.iloc[test_index]["LongStage"]

        classifier.fit(x_train, y_train)
        if args.one or args.three:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classifier.predict(x_test))
        else:
            confusion_matrix = numpy.sum(sklearn.metrics.multilabel_confusion_matrix(y_test, classifier.predict(x_test)), axis=0)

        for metric in step00.selected_derivations:
            score = step00.aggregate_confusion_matrix(confusion_matrix, metric)

            scores.append((metric, score))

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

                scores.append(("AUC", roc_auc))
        else:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_onehot_test, y_score[:, 1])
            roc_auc = sklearn.metrics.auc(fpr, tpr)

            if "AUC" in score_by_metric:
                score_by_metric["AUC"].append(roc_auc)
            else:
                score_by_metric["AUC"] = [roc_auc]

            scores.append(("AUC", roc_auc))

    score_data = pandas.DataFrame(scores, columns=["Metrics", "Value"])
    print(score_data)
    score_data.to_csv(args.output, sep="\t")
