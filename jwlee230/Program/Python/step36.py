"""
step36.py: Calculate extended alpha-diversity
"""
import argparse
import io
import pandas
import skbio.diversity
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TSV file")
    parser.add_argument("tree", type=str, help="Tree NWk file")
    parser.add_argument("output", type=str, help="Output TSV file")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT must end with .tsv!!")
    elif not args.tree.endswith(".nwk"):
        raise ValueError("TREE must end with .nwk!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("OUTPUT must end with .tsv!!")

    input_data = pandas.read_csv(args.input, sep="\t", skiprows=1, index_col=0).iloc[:, :-1].T
    print(input_data)

    ids = list(input_data.index)
    otu_ids = list(input_data.columns)
    counts = input_data.to_numpy().astype(int)
    with open(args.tree, "r") as f:
        tree = skbio.TreeNode.read(io.StringIO(f.readline()))

    output_data = pandas.DataFrame(index=ids)
    for alpha in tqdm.tqdm(step00.alphas):
        try:
            output_data[alpha] = skbio.diversity.alpha_diversity(alpha, counts, ids)
        except ValueError:
            output_data[alpha] = skbio.diversity.alpha_diversity(alpha, counts, ids, otu_ids=otu_ids, tree=tree)
    output_data.to_csv(args.output, sep="\t")
    print(output_data)
