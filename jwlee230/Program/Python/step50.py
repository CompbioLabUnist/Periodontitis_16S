"""
step50.py: Select by ANCOM-BC2
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TAR.gz file", type=str)
    parser.add_argument("ancom", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TAR.gz file", type=str)

    args = parser.parse_args()

    if not args.ancom.endswith(".tsv"):
        raise ValueError("ANCOM file must end with .TSV!!")

    data = step00.read_pickle(args.input)
    print(data)

    ancom_data = pandas.read_csv(args.ancom, sep="\t", index_col=0)
    ancom_data = ancom_data.loc[list(map(step00.filter_taxonomy, ancom_data["taxon"]))]
    ancom_data = ancom_data.loc[(ancom_data["q_(Intercept)"] < 0.05)]
    ancom_data["taxon"] = list(map(step00.consistency_taxonomy, ancom_data["taxon"]))
    print(ancom_data)
    print(len(ancom_data["taxon"]), ":", sorted(ancom_data["taxon"]))

    data = data[list(ancom_data["taxon"]) + ["ShortStage", "LongStage"]]
    print(data)

    step00.make_pickle(args.output, data)
