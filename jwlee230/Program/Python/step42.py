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
    parser.add_argument("output", type=str, help="Output TXT file")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT must end with .TSV!!")
    elif not args.output.endswith(".txt"):
        raise ValueError("OUTPUT must end with .TXT!!")

    input_data = pandas.read_csv(args.input, sep="\t", names=["ID", "Index"], skiprows=1)
    input_data["ShortStage"] = list(map(step00.change_ID_into_short_stage, input_data["ID"]))
    print(input_data)

    whole_stat, whole_p = scipy.stats.normaltest(input_data["Index"])
    print("Whole", ":", whole_p)

    for stage in tqdm.tqdm(step00.short_stage_order[1:]):
        stage_data = input_data.loc[(input_data["ShortStage"] == stage)]
        stage_stat, stage_p = scipy.stats.normaltest(stage_data["Index"])
        print(stage, ":", stage_p)
