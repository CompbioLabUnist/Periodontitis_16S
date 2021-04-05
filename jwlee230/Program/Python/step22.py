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

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, init="pca", random_state=0, method="exact", n_jobs=args.cpu).fit_transform(raw_data), columns=["tSNE1", "tSNE2"])

    for column in tsne_data.columns:
        tsne_data[column] = sklearn.preprocessing.scale(tsne_data[column])

    tsne_data["ShortStage"] = list(map(step00.change_ID_into_short_stage, list(raw_data.index)))
    tsne_data["LongStage"] = list(map(step00.change_short_into_long, tsne_data["ShortStage"]))
    print(tsne_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 100, "axes.labelsize": 50, "axes.titlesize": 100, "xtick.labelsize": 50, "ytick.labelsize": 50, "font.family": "serif"})
    seaborn.set(context="poster", style="whitegrid")

    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
    seaborn.scatterplot(data=tsne_data, x="tSNE1", y="tSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order, palette=step00.color_stage_order, s=1000, edgecolor="none")
    matplotlib.pyplot.legend(fontsize="50", title_fontsize="100")

    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
