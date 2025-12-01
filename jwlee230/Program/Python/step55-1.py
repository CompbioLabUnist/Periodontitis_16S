"""
step55-1.py: Venn diagram comparison
"""
import argparse
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import tqdm
import venn
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file", nargs=3)
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    venn_data = dict()

    korea_data = step00.read_pickle(args.input[0])
    venn_data["Korea"] = set(list(korea_data.columns)[:-2])
    print(korea_data)

    spain_data = step00.read_pickle(args.input[1])
    venn_data["Spain"] = set(list(spain_data.columns)[:-2])
    print(spain_data)

    portugal_data = step00.read_pickle(args.input[2])
    venn_data["Portugal"] = set(list(portugal_data.columns)[:-2])
    print(portugal_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    ax = venn.venn(venn_data, fmt="{size:d} ({percentage:.1f}%)", fontsize=12, ax=ax)
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)

    taxa_list = sorted(venn_data["Korea"] | venn_data["Spain"] | venn_data["Portugal"])
    output_data = pandas.DataFrame(index=taxa_list, columns=["Korea", "Spain", "Portugal"], dtype=str)
    for taxon in tqdm.tqdm(taxa_list):
        for column in ["Korea", "Spain", "Portugal"]:
            if taxon in venn_data[column]:
                output_data.loc[taxon, column] = "O"
            else:
                output_data.loc[taxon, column] = "X"
    output_data.index = list(map(step00.simplified_taxonomy, taxa_list))
    output_data.to_csv(args.output.replace(".png", ".tsv"), sep="\t")
    print(output_data)
