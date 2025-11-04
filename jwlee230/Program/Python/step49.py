"""
step49.py: merge TSV
"""
import argparse
import pandas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TAR file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("Output file must end with .TSV!!")

    raw_data = pandas.read_csv(args.input, sep="\t", skiprows=1)
    print(raw_data)

    data = raw_data.groupby(by="taxonomy").sum()
    data = data.loc[sorted(list(data.index)), :]
    print(data)

    data.to_csv(args.output, sep="\t")
