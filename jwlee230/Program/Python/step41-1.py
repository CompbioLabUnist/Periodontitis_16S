"""
step41-1.py: Taxonomy TSV output by group
"""
import argparse
import itertools
import typing
import pandas
import numpy
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TSV file")

    args = parser.parse_args()

    if not args.output.endswith(".tsv"):
        raise ValueError("OUTPUT file must end with .TSV!!")

    data: pandas.DataFrame = step00.read_pickle(args.input)
    print(data)

    taxa_list = sorted(list(data.columns)[:-2])

    output_data = pandas.DataFrame(index=step00.long_stage_order)
    for taxon in tqdm.tqdm(taxa_list):
        output_data[taxon] = [f"{numpy.mean(data.loc[(data['LongStage'] == stage), taxon]):.3f}+-{numpy.std(data.loc[(data['LongStage'] == stage), taxon]):.3f}" for stage in step00.long_stage_order]
    output_data = output_data.T
    output_data.index.name = "Taxa"
    print(output_data)
    output_data.to_csv(args.output, sep="\t")
