"""
step19.py: draw default t-SNE
"""
import argparse
import pandas
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.pyplot
import matplotlib.transforms
import numpy
import seaborn
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

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.output.endswith(".png"):
        raise ValueError("OUTPUT file must end with .PNG!!")

    data: pandas.DataFrame = step00.read_pickle(args.input)
    data["ShortStage"] = list(map(step00.change_ID_into_short_stage, data["ID"]))
    data["LongStage"] = list(map(step00.change_short_into_long, data["ShortStage"]))
    print(data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.scatterplot(data=data, x="tSNE1", y="tSNE2", hue="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, palette=step00.color_stage_order, s=1000, edgecolor="none")

    legend = matplotlib.pyplot.legend()
    for handle in legend.legendHandles:
        handle.set_sizes([1000])
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
