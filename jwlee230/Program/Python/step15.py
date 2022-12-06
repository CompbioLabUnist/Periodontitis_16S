"""
step15.py: change abundance to proportion
"""
import argparse
import pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TSV file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("Output file must end with .TSV!!")

    input_data = pandas.read_csv(args.input, sep="\t", skiprows=1, index_col=0)
    print(input_data)

    for column in list(input_data.columns)[:-1]:
        input_data.loc[:, column] = input_data.loc[:, column] / sum(input_data.loc[:, column])

    with open(args.output, "w") as f:
        f.write("# Constructed from TSV file\n")

    input_data.to_csv(args.output, sep="\t", mode="a")
