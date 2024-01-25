"""
step07-1.py: select metadata for Korea data
"""
import argparse
import pandas
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TSV file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT file must end with .tsv!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("OUTPUT file must end with .tsv!!")

    input_data = pandas.read_csv(args.input, sep="\t", skiprows=[1], keep_default_na=False, na_values={})
    print(input_data)

    output_data = input_data.loc[(input_data["DB"] == "Korea")]
    print(output_data)

    with open(args.output, "w") as f:
        f.write("\t".join(list(output_data.columns)))
        f.write("\n")
        f.write("#q2:types\t")
        f.write("\t".join(["categorical"] * (len(output_data.columns) - 1)))
        f.write("\n")

        for index, row in tqdm.tqdm(output_data.iterrows(), total=len(output_data)):
            f.write("\t".join(row))
            f.write("\n")
