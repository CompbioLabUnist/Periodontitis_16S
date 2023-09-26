"""
step22.py: Draw t-SNE from Beta-diversity TSV
"""
import argparse
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.pyplot
import matplotlib.transforms
import numpy
import pandas
import seaborn
import skbio.stats
import sklearn.manifold
import sklearn.preprocessing
import step00


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = matplotlib.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


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

    raw_data = pandas.read_csv(args.input, sep="\t", index_col="Unnamed: 0")
    print(raw_data)

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, init="pca", random_state=42, method="exact", n_jobs=args.cpu).fit_transform(raw_data), columns=["tSNE1", "tSNE2"])

    for column in tsne_data.columns:
        tsne_data[column] = sklearn.preprocessing.scale(tsne_data[column])

    tsne_data["ShortStage"] = list(map(step00.change_ID_into_short_stage, list(raw_data.index)))
    tsne_data["LongStage"] = list(map(step00.change_short_into_long, tsne_data["ShortStage"]))
    print(tsne_data)

    p_value = skbio.stats.distance.permanova(skbio.stats.distance.DistanceMatrix(raw_data, ids=list(raw_data.index)), grouping=list(tsne_data["LongStage"]))["p-value"]
    print(p_value)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.scatterplot(data=tsne_data, x="tSNE1", y="tSNE2", hue="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, palette=step00.color_stage_order, s=2000, edgecolor="none")

    for stage, color in zip(step00.long_stage_order, step00.color_stage_order):
        confidence_ellipse(tsne_data.loc[(tsne_data["LongStage"] == stage), "tSNE1"], tsne_data.loc[(tsne_data["LongStage"] == stage), "tSNE2"], ax, color=color, alpha=0.3)

    matplotlib.pyplot.title(f"PERMANOVA p={p_value:.2e}")

    legend = matplotlib.pyplot.legend()
    for handle in legend.legendHandles:
        handle.set_sizes([1000])
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))

    matplotlib.pyplot.close(fig)
