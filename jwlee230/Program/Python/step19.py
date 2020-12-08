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
    data["ShortStage"] = list(map(lambda x: x[0] if x[0] == "H" else x[2], data["ID"]))
    data["LongStage"] = list(map(lambda x: {"H": "Healthy", "E": "Early", "M": "Moderate", "S": "Severe"}[x], data["ShortStage"]))

    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.scatterplot(data=data, x="TSNE1", y="TSNE2", hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order)

    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
