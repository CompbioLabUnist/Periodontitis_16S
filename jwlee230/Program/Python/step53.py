"""
step53.py: Transform into centered log-ratio
"""
import argparse
import pandas
import skbio.stats.composition
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TAR.gz file", type=str)
    parser.add_argument("output", help="Output TAR.gz file", type=str)

    args = parser.parse_args()

    input_data = step00.read_pickle(args.input)
    print(input_data)

    stage_data = input_data["ShortStage"], input_data["LongStage"]
    del input_data["ShortStage"], input_data["LongStage"]
    print(input_data)

    for index in tqdm.tqdm(list(input_data.index)):
        input_data.loc[index, :] = input_data.loc[index, :] / sum(input_data.loc[index, :]) + 1e-6
    print(input_data)

    transfomed_data = pandas.DataFrame(skbio.stats.composition.clr(input_data.to_numpy()), index=input_data.index, columns=input_data.columns)
    print(transfomed_data)

    transfomed_data["ShortStage"] = stage_data[0]
    transfomed_data["LongStage"] = stage_data[1]
    print(transfomed_data)

    step00.make_pickle(args.output, transfomed_data)
