"""
step35.py: Pathway analysis
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import pandas
import scipy.stats
import seaborn
import statannotations.Annotator
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input XLSX file")
    parser.add_argument("output", type=str, help="Output TAR file")

    args = parser.parse_args()

    if not args.input.endswith(".xlsx"):
        raise ValueError("Input file must be XLSX file")
    elif not args.output.endswith(".tar"):
        raise ValueError("Output file must be TAR file")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = pandas.read_excel(args.input, index_col="description")
    del input_data["pathway"]
    input_data = input_data.T
    print(input_data)

    for index in list(input_data.index):
        input_data.loc[index, :] = input_data.loc[index, :] / sum(input_data.loc[index, :])

    pathways = sorted(input_data.columns)
    input_data["LongStage"] = list(map(step00.change_ID_into_long_stage, list(input_data.index)))
    print(input_data)

    figures = list()
    for pathway in pathways:
        stat, p_val = scipy.stats.kruskal(*[input_data.loc[(input_data["LongStage"] == stage), pathway] for stage in step00.long_stage_order])
        if p_val >= 0.05:
            continue

        stat, p = scipy.stats.mannwhitneyu(input_data.loc[(input_data["LongStage"] == "Healthy"), pathway], input_data.loc[(input_data["LongStage"] == "Stage I"), pathway])
        if p >= 0.01:
            continue

        box_pairs = list()
        for s1, s2 in itertools.combinations(step00.long_stage_order, r=2):
            stat, p = scipy.stats.mannwhitneyu(input_data.loc[(input_data["LongStage"] == s1), pathway], input_data.loc[(input_data["LongStage"] == s2), pathway])
            if (p < 0.01):
                box_pairs.append((s1, s2))

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        seaborn.violinplot(data=input_data, x="LongStage", y=pathway, order=step00.long_stage_order, palette=step00.color_stage_dict, inner="box", cut=1, linewidth=10, ax=ax)
        statannotations.Annotator.Annotator(ax, box_pairs, data=input_data, x="LongStage", y=pathway, order=step00.long_stage_order).configure(test="Mann-Whitney", text_format="star", comparisons_correction=None, loc="inside", verbose=0).apply_and_annotate()

        matplotlib.pyplot.title(f"Kruskal-Wallis p={p_val:.2e}")
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.tight_layout()

        figures.append(pathway.replace("/", "_") + ".pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for f in figures:
            tar.add(f, arcname=f)
