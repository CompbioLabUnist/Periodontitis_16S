"""
step16.py: Clearify raw TSV into pandas
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TAR.gz file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")

    raw_data = pandas.read_csv(args.input, sep="\t", skiprows=1)
    raw_data.set_index(inplace=True, keys=["taxonomy", "#OTU ID"], verify_integrity=True)
    raw_data = raw_data.T

    data = pandas.DataFrame()
    taxonomy_list = sorted(set(map(lambda x: x[0], raw_data.columns)))
    for taxonomy in taxonomy_list:
        data[taxonomy] = raw_data[list(filter(lambda x: x[0] == taxonomy, raw_data.columns))].sum(axis=1)
    data.columns = list(map(lambda x: step00.consistency_taxonomy(x), data.columns))

    data["ShortStage"] = list(map(step00.change_ID_into_short_stage, list(data.index)))
    data["LongStage"] = list(map(step00.change_short_into_long, data["ShortStage"]))

    print(data)
    step00.make_pickle(args.output, data)
