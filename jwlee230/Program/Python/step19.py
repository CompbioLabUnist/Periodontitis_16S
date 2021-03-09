"""
step19.py: draw default t-SNE
"""
import argparse
import pandas
import matplotlib
import matplotlib.pyplot
import seaborn
import step00

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

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 100, "axes.labelsize": 50, 'axes.titlesize': 100, 'xtick.labelsize': 50, 'ytick.labelsize': 50})
    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))

    seaborn.scatterplot(data=data, x="TSNE1", y="TSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order, palette=step00.color_stage_order, s=1000, edgecolor="none")

    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
