"""
step19.py: draw default t-SNE
"""
import argparse
import math
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

    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.scatterplot(data=data, x="TSNE1", y="TSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order, palette=step00.color_stage_order)

    for stage, color in zip(step00.long_stage_order, step00.color_stage_order):
        selected_data = data.loc[(data["LongStage"] == stage)]
        x, y = selected_data.mean(axis="index")["TSNE1"], selected_data.mean(axis="index")["TSNE2"]

        r_list = [math.sqrt((x - a) * (x - a) + (y - b) * (y - b)) for a, b in zip(selected_data["TSNE1"], selected_data["TSNE2"])]
        r = sum(r_list) / len(r_list)

        ax.add_artist(matplotlib.pyplot.Circle((x, y), r, color=color, alpha=0.2))

    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
