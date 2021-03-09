"""
step26.py: Draw violin plot by alpha-diversity
"""
import argparse
import itertools
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import statannot
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TSV file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT must end with .tsv!!")
    elif not args.output.endswith(".png"):
        raise ValueError("OUTPUT must end with .png!!")

    input_data = pandas.read_csv(args.input, sep="\t", names=["ID", "Index"], skiprows=1)
    input_data["ShortStage"] = list(map(step00.change_ID_into_short_stage, input_data["ID"]))
    input_data["LongStage"] = list(map(step00.change_short_into_long, input_data["ShortStage"]))
    print(input_data)

    seaborn.set(context="poster", style="whitegrid")
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 100, "axes.labelsize": 50, 'axes.titlesize': 100, 'xtick.labelsize': 50, 'ytick.labelsize': 50})

    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
    seaborn.violinplot(data=input_data, x="LongStage", y="Index", order=step00.long_stage_order, ax=ax, inner="box")
    statannot.add_stat_annotation(ax, data=input_data, x="LongStage", y="Index", order=step00.long_stage_order, test="t-test_ind", box_pairs=itertools.combinations(step00.long_stage_order, 2), text_format="star", loc="inside", verbose=0)
    matplotlib.pyplot.legend(fontsize="50", title_fontsize="100")
    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
