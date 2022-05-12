"""
step24.py: Make 2D plot of Pg. Act.
"""
import argparse
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

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    data = step00.read_pickle(args.input)
    for i, index in enumerate(data.index):
        data.iloc[i, :-2] = data.iloc[i, :-2] / sum(data.iloc[i, :-2])
    act_column = step00.consistency_taxonomy("k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Actinomycetales; f__Actinomycetaceae; g__Actinomyces; s__spp.")
    pg_column = step00.consistency_taxonomy("k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Porphyromonadaceae; g__Porphyromonas; s__gingivalis")
    data = data[[act_column, pg_column, "LongStage"]]
    print(data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.scatterplot(data=data, x=act_column, y=pg_column, hue="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, palette=step00.color_stage_dict, s=1000, edgecolor="none")

    for stage, color in zip(step00.long_stage_order, step00.color_stage_order):
        confidence_ellipse(data.loc[(data["LongStage"] == stage), act_column], data.loc[(data["LongStage"] == stage), pg_column], ax, color=color, alpha=0.3)

    matplotlib.pyplot.xlabel("Actinomyces spp. proportion")
    matplotlib.pyplot.ylabel("Porphyromonas gingivalis proportion")
    legend = matplotlib.pyplot.legend()
    for handle in legend.legendHandles:
        handle.set_sizes([1000])
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
