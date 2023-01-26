"""
step32.py: draw pairplot
"""
import argparse
import matplotlib
import matplotlib.colors
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

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = step00.read_pickle(args.input)
    print(input_data)

    input_data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]), list(input_data.columns)[:-2])) + list(input_data.columns)[-2:]
    print(list(input_data.columns))

    taxa = list(input_data.columns)[:-2]
    for index in list(input_data.index):
        input_data.loc[index, taxa] = input_data.loc[index, taxa] / sum(input_data.loc[index, taxa])
    print(input_data)

    g = seaborn.pairplot(data=input_data, hue="LongStage", hue_order=step00.long_stage_order, palette=step00.color_stage_dict, kind="reg", diag_kind="kde", height=4, aspect=1)
    g.tight_layout()
    g.fig.savefig(args.output)
    g.fig.savefig(args.output.replace(".png", ".pdf"))
