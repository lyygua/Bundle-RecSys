#!/usr/bin/env python3
"""Create review and metadata subsets for target users and recommended items."""

import argparse
import gzip
import json
from pathlib import Path


DOMAINS = {
    "Amazon_All_Beauty": {
        "reviews": "All_Beauty.jsonl.gz",
        "metadata": "meta_All_Beauty.jsonl.gz",
        "results": (
            "sasrec_all_beauty_top20_top50.md",
            "sasreccpr_all_beauty_top20_top50.md",
        ),
    },
    "Amazon_Appliances": {
        "reviews": "Appliances.jsonl.gz",
        "metadata": "meta_Appliances.jsonl.gz",
        "results": (
            "sasrec_appliances_top20_top50.md",
            "sasreccpr_appliances_top20_top50.md",
        ),
    },
    "Amazon_Digital_Music": {
        "reviews": "Digital_Music.jsonl.gz",
        "metadata": "meta_Digital_Music.jsonl.gz",
        "results": (
            "sasrec_digital_music_top20_top50.md",
            "sasreccpr_digital_music_top20_top50.md",
        ),
    },
    "Amazon_Health_and_Personal_Care": {
        "reviews": "Health_and_Personal_Care.jsonl.gz",
        "metadata": "meta_Health_and_Personal_Care.jsonl.gz",
        "results": (
            "sasrec_health_and_personal_care_top20_top50.md",
            "sasreccpr_health_and_personal_care_top20_top50.md",
        ),
    },
}


def parse_args():
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", "--repo-dir",
        dest="repo_dir",
        type=Path,
        default=repo_dir / "outputs" / "recommendations",
        help="Directory containing compact recommendation Markdown files",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo_dir / "dataset",
        help="Directory containing the raw Amazon .jsonl.gz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_dir / "dataset" / "amazon_target_user_content",
        help="Directory for the separate review and metadata JSONL subsets",
    )
    return parser.parse_args()


def users_from_result(path):
    users = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| A"):
            continue
        columns = line.split("|")
        if len(columns) >= 3:
            users.append(columns[1].strip())
    return users


def items_from_result(path):
    items = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| A"):
            continue
        columns = line.split("|")
        if len(columns) >= 3:
            items.update(
                item.strip() for item in columns[2].split(",") if item.strip()
            )
    return items


def load_target_users(repo_dir, result_names):
    result_users = []
    for name in result_names:
        path = repo_dir / name
        users = users_from_result(path)
        if not users:
            raise ValueError(f"No recommendation users found in {path}")
        result_users.append(users)

    first = result_users[0]
    for name, users in zip(result_names[1:], result_users[1:]):
        if users != first:
            raise ValueError(
                f"Target users differ between {result_names[0]} and {name}"
            )
    return first


def load_recommended_items(repo_dir, result_names):
    recommended_items = set()
    for name in result_names:
        path = repo_dir / name
        items = items_from_result(path)
        if not items:
            raise ValueError(f"No recommended items found in {path}")
        recommended_items.update(items)
    return recommended_items


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line), line
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def extract_domain(repo_dir, dataset_dir, domain, config):
    target_users = load_target_users(repo_dir, config["results"])
    recommended_parent_asins = load_recommended_items(repo_dir, config["results"])
    target_user_set = set(target_users)
    selected_reviews = []
    selected_parent_asins = set()
    found_users = set()

    review_path = dataset_dir / config["reviews"]
    for review, original_line in read_jsonl_gz(review_path):
        user_id = review.get("user_id")
        if user_id not in target_user_set:
            continue
        selected_reviews.append(original_line)
        found_users.add(user_id)
        parent_asin = review.get("parent_asin")
        if parent_asin:
            selected_parent_asins.add(parent_asin)

    missing_users = [user_id for user_id in target_users if user_id not in found_users]
    if missing_users:
        raise ValueError(
            f"{domain}: target users missing from raw reviews: {missing_users}"
        )

    metadata_parent_asins = selected_parent_asins | recommended_parent_asins
    metadata_path = dataset_dir / config["metadata"]
    product_metadata = []
    found_parent_asins = set()
    for metadata, original_line in read_jsonl_gz(metadata_path):
        parent_asin = metadata.get("parent_asin")
        if parent_asin in metadata_parent_asins:
            product_metadata.append(original_line)
            found_parent_asins.add(parent_asin)

    return {
        "domain": domain,
        "target_user_count": len(target_users),
        "review_count": len(selected_reviews),
        "reviewed_parent_asin_count": len(selected_parent_asins),
        "recommended_parent_asin_count": len(recommended_parent_asins),
        "metadata_parent_asin_count": len(metadata_parent_asins),
        "product_metadata_record_count": len(product_metadata),
        "missing_review_metadata_parent_asins": sorted(
            selected_parent_asins - found_parent_asins
        ),
        "missing_recommended_metadata_parent_asins": sorted(
            recommended_parent_asins - found_parent_asins
        ),
        "reviews": selected_reviews,
        "product_metadata": product_metadata,
    }


def write_original_lines(path, lines):
    with path.open("w", encoding="utf-8") as stream:
        stream.writelines(lines)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for domain, config in DOMAINS.items():
        output = extract_domain(args.repo_dir, args.dataset_dir, domain, config)
        review_output_path = args.output_dir / config["reviews"].removesuffix(".gz")
        metadata_output_path = args.output_dir / config["metadata"].removesuffix(".gz")
        write_original_lines(review_output_path, output["reviews"])
        write_original_lines(metadata_output_path, output["product_metadata"])
        print(
            f"{domain}: {output['target_user_count']} users, "
            f"{review_output_path.name}={output['review_count']} reviews, "
            f"recommended items={output['recommended_parent_asin_count']}, "
            f"{metadata_output_path.name}="
            f"{output['product_metadata_record_count']} metadata records, "
            f"{len(output['missing_review_metadata_parent_asins'])} missing review "
            f"metadata keys, "
            f"{len(output['missing_recommended_metadata_parent_asins'])} missing "
            f"recommended metadata keys"
        )


if __name__ == "__main__":
    main()
