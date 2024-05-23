"""
step42.py: Normality test for beta-diversity
"""
import argparse
import itertools
import typing
import numpy
import pandas
import scipy.stats
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TSV file")
    parser.add_argument("output", type=str, help="Output TSV file")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT must end with .TSV!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("OUTPUT must end with .TSV!!")

    input_data = pandas.read_csv(args.input, sep="\t", index_col=0)
    input_data = input_data.loc[list(filter(lambda x: not x.startswith("SRR"), list(input_data.index)))]
    print(input_data)

    sample_by_stage_dict: typing.Dict[str, typing.List[str]] = {x: list() for x in step00.long_stage_order[1:]}
    for sample in tqdm.tqdm(list(input_data.index)):
        sample_by_stage_dict[step00.change_ID_into_long_stage(sample)].append(sample)
    print(sample_by_stage_dict)

    output_data = pandas.DataFrame(index=step00.long_stage_order[1:], columns=step00.long_stage_order[1:])
    for index_stage, column_stage in tqdm.tqdm(list(itertools.product(step00.long_stage_order[1:], repeat=2))):
        output_data.loc[index_stage, column_stage] = scipy.stats.normaltest(numpy.ravel(input_data.loc[sample_by_stage_dict[index_stage], sample_by_stage_dict[column_stage]].to_numpy()))[1]
    print(output_data)
    output_data.to_csv(args.output, sep="\t", float_format="%.3f")
