"""
step48.py: Aitchson distance
"""
import argparse
import multiprocessing
import numpy
import pandas
import tqdm

input_data = pandas.DataFrame()


def aitchson_distance(x, y):
    u = input_data[x]
    v = input_data[y]

    log_u_v = numpy.log(u / v)
    return numpy.linalg.norm(log_u_v - numpy.mean(log_u_v))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", type=str, help="Output TSV file")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU to use")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("Output file must end with .TSV!!")

    input_data = pandas.read_csv(args.input, sep="\t", index_col=0, skiprows=1)
    del input_data["taxonomy"]
    input_data += 1.0
    print(input_data)

    patient_list = list(input_data.columns)
    output_data = pandas.DataFrame(numpy.zeros((len(patient_list), len(patient_list))), index=patient_list, columns=patient_list, dtype=float)

    with multiprocessing.Pool(args.cpu) as pool:
        for x in tqdm.tqdm(patient_list):
            output_data.loc[x, :] = pool.starmap(aitchson_distance, [(x, y) for y in patient_list])
    print(output_data)

    output_data.to_csv(args.output, sep="\t")
