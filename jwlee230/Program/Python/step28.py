"""
step28.py: ANCOM cluster map
"""
import argparse
import matplotlib
import matplotlib.pyplot
import numpy
import seaborn
import step00


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
    print(data)
    data = data.iloc[:, :-2]
    data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]).replace("_", " "), list(data.columns)))
    data = data.T
    data = numpy.log10(data + 1)
    print(data)

    minimum = numpy.min(data.to_numpy())
    maximum = numpy.max(data.to_numpy())
    center_ratio = 0.3

    g = seaborn.clustermap(data=data, xticklabels=False, yticklabels=True, row_cluster=True, col_cluster=False, col_colors=list(map(lambda x: step00.color_stage_dict[step00.change_ID_into_long_stage(x)], list(data.columns))), figsize=(32, 32), cmap="YlOrRd", cbar_pos=(0.90, 0.8, 0.05, 0.18), dendrogram_ratio=(0.2, 0.01))

    g.savefig(args.output)
    g.savefig(args.output.replace(".png", ".pdf"))
    g.savefig(args.output.replace(".png", ".svg"))
