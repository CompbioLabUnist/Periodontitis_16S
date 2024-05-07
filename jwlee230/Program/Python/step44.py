"""
step44.py: read count comparison
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import statannot
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TAR file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT file must end with .tsv!!")
    elif not args.output.endswith(".tar"):
        raise ValueError("OUTPUT file must end with .tar!!")

    input_data = pandas.read_csv(args.input, sep="\t", index_col=0, skiprows=[1])
    input_data["Stage"] = list(map(step00.change_ID_into_long_stage, list(input_data.index)))
    print(input_data)

    columns = ["input", "filtered", "denoised", "merged"]

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    figures = list()
    order = ["Healthy", "Stage I", "Stage II", "Stage III"]

    for column in tqdm.tqdm(columns):
        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.violinplot(data=input_data, x="Stage", y=column, order=order, ax=ax, inner="box", palette=step00.color_stage_dict, cut=1, linewidth=10)
        statannot.add_stat_annotation(ax, data=input_data, x="Stage", y=column, order=order, test="t-test_ind", box_pairs=list(itertools.combinations(order, r=2)), text_format="star", loc="inside", verbose=0, comparisons_correction=None)

        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.ylabel("Read count")
        matplotlib.pyplot.title(column)
        matplotlib.pyplot.tight_layout()

        figures.append(f"{column}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for file_name in tqdm.tqdm(figures):
            tar.add(file_name, arcname=file_name)
