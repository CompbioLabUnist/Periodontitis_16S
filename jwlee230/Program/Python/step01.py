"""
step01: make manifest file
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input FASTQ.gz files", type=str, nargs="+")

    args = parser.parse_args()

    if list(filter(lambda x: x.endswith(".fastq.gz"), args.input)):
        raise ValueError("Input files must end with .fastq.gz!!")

    args.input.sort()

    f1 = sorted(list(filter(lambda x: x.endswith("1.fastq.gz"), args.input)))
    f2 = sorted(list(filter(lambda x: x.endswith("2.fastq.gz"), args.input)))

    print("sample-id\tforward-absolute-filepath\treverse-absolute-filepath")
    for fa, fb in zip(f1, f2):
        name = fa[:-11]
        if fb.startswith(name):
            print(name[name.rfind("/") + 1:], fa, fb, sep="\t")
        else:
            raise ValueError(fa, fb)
