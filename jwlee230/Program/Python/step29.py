"""
step29.py: ANCOM violin plot
"""
import argparse
import matplotlib
import matplotlib.pyplot
import pandas
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
    for i, index in enumerate(data.index):
        data.iloc[i, :-2] = data.iloc[i, :-2] / sum(data.iloc[i, :-2])
    data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]).replace("_", " "), list(data.columns)[:-2])) + list(data.columns)[-2:]
    print(data)

    taxa = list(data.columns)[:-2]
    raw_output_data = list()
    for index, row in data.iterrows():
        for taxo in taxa:
            raw_output_data.append((taxo, row["LongStage"], row[taxo]))
    output_data = pandas.DataFrame(raw_output_data, columns=["Taxonomy", "LongStage", "Value"])
    print(output_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(2 * len(taxa), 36))

    seaborn.boxplot(data=output_data, x="Taxonomy", y="Value", hue="LongStage", palette=step00.color_stage_dict, order=taxa, hue_order=step00.long_stage_order, ax=ax)

    matplotlib.pyplot.xlabel("")
    matplotlib.pyplot.ylabel("Proprotion")
    matplotlib.pyplot.xticks(rotation="vertical")
    matplotlib.pyplot.legend(loc="upper right")
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
