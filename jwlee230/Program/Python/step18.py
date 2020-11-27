"""
step18.py: making t-SNE
"""
import argparse
import pandas
import sklearn.manifold
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR.gz file")
    parser.add_argument("--cpu", type=int, default=1, help="CPU to use")

    args = parser.parse_args()

    if args.cpu < 1:
        raise ValueError("CPU must be greater than zero")

    raw_data = step00.read_pickle(args.input)

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, init="pca", random_state=0, method="exact", n_jobs=args.cpu).fit_transform(raw_data), columns=["TSNE1", "TSNE2"])

    for column in tsne_data.columns:
        tsne_data[column] = sklearn.preprocessing.scale(tsne_data[column])

    tsne_data["ID"] = list(raw_data.index)
    print(tsne_data)

    step00.make_pickle(args.output, tsne_data)
