from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = ["Title", "Abstract", "Author Keywords"]
OUTPUT_COLUMNS = ["paper_id", "title", "abstract", "keywords"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Scopus CSV into pipeline-ready CSV format."
    )
    parser.add_argument("--input-csv", required=True, help="Path to Scopus CSV file.")
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file (paper_id,title,abstract,keywords).",
    )
    parser.add_argument(
        "--id-prefix",
        default="P",
        help="Prefix used for generated paper IDs. Default: P",
    )
    parser.add_argument(
        "--id-width",
        type=int,
        default=5,
        help="Zero-padding width for generated IDs. Default: 5",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting integer for generated IDs. Default: 1",
    )
    return parser.parse_args()


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validate_columns(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError("Input CSV does not have a header row.")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            "Missing required Scopus columns: " + ", ".join(missing)
        )


def convert_scopus_csv(
    input_csv: Path,
    output_csv: Path,
    id_prefix: str,
    id_width: int,
    start_id: int,
) -> tuple[int, int]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0

    with input_csv.open("r", newline="", encoding="utf-8") as infile, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        _validate_columns(reader.fieldnames)

        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in reader:
            rows_read += 1
            paper_id_number = start_id + rows_written
            paper_id = f"{id_prefix}{paper_id_number:0{id_width}d}"

            writer.writerow(
                {
                    "paper_id": paper_id,
                    "title": _normalize(row.get("Title")),
                    "abstract": _normalize(row.get("Abstract")),
                    "keywords": _normalize(row.get("Author Keywords")),
                }
            )
            rows_written += 1

    return rows_read, rows_written


def main() -> None:
    args = parse_args()

    rows_read, rows_written = convert_scopus_csv(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        id_prefix=args.id_prefix,
        id_width=args.id_width,
        start_id=args.start_id,
    )

    print("Scopus conversion completed.")
    print(f"- Rows read: {rows_read}")
    print(f"- Rows written: {rows_written}")
    print(f"- Output: {args.output_csv}")


if __name__ == "__main__":
    main()
