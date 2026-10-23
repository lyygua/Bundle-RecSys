#!/usr/bin/env python3
"""Stream Amazon Reviews 2023 JSONL reviews into a RecBole .inter file."""

import argparse
from contextlib import nullcontext
import gzip
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile


DOMAINS = (
    "Amazon_All_Beauty",
    "Amazon_Appliances",
    "Amazon_Digital_Music",
    "Amazon_Health_and_Personal_Care",
)
HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def interaction_line(review):
    user = review["user_id"]
    item = review["parent_asin"]
    for value in (user, item):
        if not isinstance(value, str) or not value or any(c.isspace() for c in value):
            raise ValueError("user_id and parent_asin must be nonempty tokens")
    rating = float(review["rating"])
    timestamp = review["timestamp"]
    if not math.isfinite(rating):
        raise ValueError("rating must be finite")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp must be numeric")
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return f"{user}\t{item}\t{rating}\t{timestamp}\n"


def convert_reviews(input_path, output_path, dry_run=False):
    input_path, output_path = Path(input_path), Path(output_path)
    if not dry_run and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    opener = gzip.open if input_path.suffix == ".gz" else open
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use disk-backed deduplication for conversion; dry runs create no files.
    workspace = nullcontext(None) if dry_run else tempfile.TemporaryDirectory(
        dir=output_path.parent
    )
    with workspace as directory:
        database = ":memory:" if dry_run else str(Path(directory) / "reviews.sqlite")
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "CREATE TABLE interactions (user_id TEXT, item_id TEXT, row TEXT, "
                "position INTEGER, PRIMARY KEY (user_id, item_id))"
            )
            with opener(input_path, "rt", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    try:
                        row = interaction_line(json.loads(line))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
                    user, item = row.split("\t", 2)[:2]
                    connection.execute(
                        "INSERT INTO interactions VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(user_id, item_id) DO UPDATE SET "
                        "row=excluded.row, position=excluded.position",
                        (user, item, row, line_number),
                    )
            count = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            if count == 0:
                raise ValueError(f"No reviews found in {input_path}")
            if not dry_run:
                temporary_path = Path(directory) / "interactions.inter"
                with temporary_path.open("w", encoding="utf-8") as stream:
                    stream.write(HEADER)
                    for (row,) in connection.execute(
                        "SELECT row FROM interactions ORDER BY position"
                    ):
                        stream.write(row)
                # Publish the complete file without replacing an existing file.
                os.link(temporary_path, output_path)
            return count
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DOMAINS)
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and count without writing files"
    )
    args = parser.parse_args()
    output = args.output_root / args.dataset / f"{args.dataset}.inter"
    count = convert_reviews(args.input, output, args.dry_run)
    print(f"{'Validated' if args.dry_run else 'Converted'} {count} unique interactions: {output}")


if __name__ == "__main__":
    main()
