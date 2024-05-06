"""
step39.py: Draw rarefaction violin plot
"""
import argparse
import itertools
import pandas
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import seaborn
import statannot
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input CSV file")
    parser.add_argument("clinical", help="Clinical TSV file", type=str)
    parser.add_argument("output", type=str, help="Output PNG file")
    parser.add_argument("--depth", help="Depth to plot", type=int, default=1)

    args = parser.parse_args()

    if not args.input.endswith(".csv"):
        raise ValueError("Input file must be a CSV file")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("Clinical file must end with .TSV!!")
    elif not args.output.endswith(".png"):
        raise ValueError("Output file must end with .PNG!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = pandas.read_csv(args.input)
    print(input_data)

    clinical_data = pandas.read_csv(args.clinical, sep="\t", index_col=0, skiprows=[1])
    print(clinical_data)

    indices = sorted(input_data.index)
    columns = sorted(list(input_data.columns)[1:])

    depth_list = sorted(map(lambda x: int(x[x.find("-") + 1:x.find("_")]), columns))
    depth_cutoff = -1
    for depth in tqdm.tqdm(depth_list):
        if depth > args.depth:
            break
        else:
            depth_cutoff = depth
    print("Cutoff:", depth_cutoff)

    column_set = sorted(filter(lambda x: f"-{depth_cutoff}_" in x, columns))
    print(column_set)

    value_name = "Depth"
    output_data = pandas.melt(input_data, id_vars=["sample-id"], value_vars=column_set, value_name=value_name)
    output_data["Stage"] = list(map(step00.change_ID_into_long_stage, output_data["sample-id"]))
    print(output_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

    seaborn.violinplot(data=output_data, x="Stage", y=value_name, order=step00.long_stage_order[1:], ax=ax, inner="box", cut=1, linewidth=10)
    statannot.add_stat_annotation(ax, data=output_data, x="Stage", y=value_name, order=step00.long_stage_order[1:], test="t-test_ind", box_pairs=list(itertools.combinations(step00.long_stage_order[1:], r=2)), text_format="star", loc="inside", verbose=0, comparisons_correction=None)

    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    matplotlib.pyplot.close(fig)
