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

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 36))
    seaborn.violinplot(data=input_data, x="LongStage", y="Index", order=step00.long_stage_order, ax=ax, inner="box")
    statannot.add_stat_annotation(ax, data=input_data, x="LongStage", y="Index", order=step00.long_stage_order, test="t-test_ind", box_pairs=itertools.combinations(step00.long_stage_order, 2), text_format="star", loc="inside", verbose=1)
    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
