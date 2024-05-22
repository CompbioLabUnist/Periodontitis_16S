"""
step42.py: Normality test for alpha-diversity
"""
import argparse
import itertools
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
    alpha_list = list(input_data.columns)
    print(input_data)

    input_data = input_data.loc[list(filter(lambda x: not x.startswith("SRR"), list(input_data.index)))]
    input_data["Stage"] = list(map(step00.change_ID_into_long_stage, list(input_data.index)))
    print(input_data)

    output_data = pandas.DataFrame(index=["All"] + step00.long_stage_order[1:], columns=alpha_list)
    for alpha in tqdm.tqdm(alpha_list):
        output_data.loc["All", alpha] = scipy.stats.normaltest(input_data[alpha])[1]

    for alpha, stage in tqdm.tqdm(list(itertools.product(alpha_list, step00.long_stage_order[1:]))):
        output_data.loc[stage, alpha] = scipy.stats.normaltest(input_data.loc[(input_data["Stage"] == stage), alpha])[1]
    print(output_data)

    output_data.to_csv(args.output, sep="\t", float_format="%.3f")
