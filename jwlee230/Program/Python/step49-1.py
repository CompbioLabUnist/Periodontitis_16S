"""
step49.py: merge TSV
"""
import argparse
import collections
import pandas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("clinical", help="Clinical TSV file", type=str)
    parser.add_argument("output", help="Output TAR file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("Clinical file must end with .TSV!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("Output file must end with .TSV!!")

    raw_data = pandas.read_csv(args.input, sep="\t", skiprows=1)
    print(raw_data)

    input_data = raw_data.groupby(by="taxonomy").sum()
    print(input_data)

    clinical_data = pandas.read_csv(args.clinical, skiprows=[1], sep="\t", index_col=0)
    clinical_data = clinical_data.loc[~(clinical_data["LongStage"].isna())]
    print(clinical_data)

    counter = collections.Counter(clinical_data["LongStage"])
    print(counter.most_common())

    if counter["Stage III"] < 3:
        clinical_data = clinical_data.loc[(clinical_data["LongStage"] != "Stage III")]
        print(clinical_data)

    selected_samples = sorted(set(input_data.columns) & set(clinical_data.index))
    input_data = input_data[selected_samples]
    print(input_data)
    input_data.to_csv(args.output, sep="\t")
