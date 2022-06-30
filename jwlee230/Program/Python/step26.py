"""
step26.py: Draw violin plot by alpha-diversity
"""
import argparse
import itertools
import matplotlib
import matplotlib.pyplot
import pandas
import scipy.stats
import seaborn
import statannotations.Annotator
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

    box_pairs = list()
    for s1, s2 in itertools.product(step00.long_stage_order, repeat=2):
        if s1 == s2:
            continue
        _, p = scipy.stats.mannwhitneyu(input_data.loc[(input_data["LongStage"] == s1), "Index"], input_data.loc[(input_data["LongStage"] == s2), "Index"])
        print(s1, s2, p)
        if (p < 0.05) and ((s2, s1) not in box_pairs):
            box_pairs.append((s1, s2))
    print(box_pairs)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

    seaborn.violinplot(data=input_data, x="LongStage", y="Index", order=step00.long_stage_order, ax=ax, inner="box", palette=step00.color_stage_dict, cut=1, linewidth=10)
    statannotations.Annotator.Annotator(ax, box_pairs, data=input_data, x="LongStage", y="Index", order=step00.long_stage_order).configure(test="Mann-Whitney", text_format="star", loc="inside").apply_and_annotate()

    stat, p = scipy.stats.kruskal(*[input_data.loc[(input_data["LongStage"] == stage), "Index"] for stage in step00.long_stage_order])

    matplotlib.pyplot.xlabel("")
    matplotlib.pyplot.title(f"Kruskal-Wallis p={p:.2e}")
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
