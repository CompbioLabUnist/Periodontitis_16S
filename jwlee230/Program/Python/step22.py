"""
step22.py: Draw t-SNE from Beta-diversity TSV
"""
import argparse
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import sklearn.manifold
import sklearn.preprocessing
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output PNG file", type=str)
    parser.add_argument("--cpu", type=int, default=1, help="CPU to use")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT file must end with .TSV!!")
    elif not args.output.endswith(".png"):
        raise ValueError("OUTPUT file must end with .PNG!!")
    elif args.cpu < 1:
        raise ValueError("CPU must be greater than zero")

    raw_data = pandas.read_csv(args.input, sep="\t")
    raw_data.set_index(keys="Unnamed: 0", inplace=True, verify_integrity=True)
    print(raw_data)

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, init="pca", random_state=0, method="exact", n_jobs=args.cpu).fit_transform(raw_data), columns=["TSNE1", "TSNE2"])

    for column in tsne_data.columns:
        tsne_data[column] = sklearn.preprocessing.scale(tsne_data[column])

    tsne_data["ShortStage"] = list(map(lambda x: x[0] if x[0] == "H" else x[2], list(raw_data.index)))
    tsne_data["LongStage"] = list(map(lambda x: {"H": "Healthy", "E": "Early", "M": "Moderate", "S": "Severe"}[x], tsne_data["ShortStage"]))
    print(tsne_data)

    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
    seaborn.scatterplot(data=tsne_data, x="TSNE1", y="TSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order)
    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
