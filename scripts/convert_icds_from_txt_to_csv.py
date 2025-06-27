"""The script takes a txt file with ICD codes and converts it to a csv file."""

import argparse
from pathlib import Path

import pandas as pd


def main(filename):
    file = Path(filename)
    with open(file) as f:
        lines = f.readlines()

    records = []
    for line in lines:
        code = line[6:13].strip()
        billable = line[14:15].strip()
        short = line[16:77].strip()
        long = line[77:].strip()
        records.append((code, billable, short, long))

    df = pd.DataFrame.from_records(records, columns=["code", "billable", "short", "long"])
    df.to_csv(file.with_suffix(".csv"), index=False)
    print(f"Converted file saved to '{file.with_suffix('.csv')}'")
    print(f"Found {len(df)} ICDs codes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="The path of the txt file with ICD codes with their descriptions.",
    )
    args = parser.parse_args()
    main(args.filename)
