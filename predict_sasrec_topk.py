import argparse
import contextlib
import io
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate top-k SASRec recommendations from a trained RecBole checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pth checkpoint")
    parser.add_argument(
        "--user",
        action="append",
        default=[],
        help="External user id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--users_file",
        help="Optional file with one external user id per line. Blank lines are ignored.",
    )
    parser.add_argument(
        "--top_users",
        type=int,
        default=0,
        help="Also predict for the N users with the longest interaction histories in the test split.",
    )
    parser.add_argument("--topk", type=int, default=50, help="Number of items per user")
    parser.add_argument(
        "--format",
        choices=["markdown", "markdown_compact", "csv"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--title",
        help="Optional title used in markdown output, e.g. 'Simple X, amazon-all-beauty-23'.",
    )
    parser.add_argument(
        "--output",
        help="Optional output file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--show_recbole_logs",
        action="store_true",
        help="Show RecBole config/log output while loading the checkpoint.",
    )
    return parser.parse_args()


def load_requested_users(args):
    users = list(args.user)
    if args.users_file:
        users.extend(
            line.strip()
            for line in Path(args.users_file).read_text().splitlines()
            if line.strip()
        )
    return users


def select_top_users(test_data, dataset, top_users):
    if top_users <= 0:
        return []

    uid_field = dataset.uid_field
    item_length_field = dataset.config["ITEM_LIST_LENGTH_FIELD"]
    inter_feat = test_data._dataset.inter_feat
    lengths = inter_feat[item_length_field]
    order = torch.argsort(lengths, descending=True)

    selected = []
    seen = set()
    for idx in order.tolist():
        internal_uid = int(inter_feat[uid_field][idx])
        if internal_uid in seen:
            continue
        seen.add(internal_uid)
        selected.append(str(dataset.id2token(uid_field, internal_uid)))
        if len(selected) >= top_users:
            break
    return selected


def dedupe_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def get_user_interaction(test_data, dataset, external_user_id):
    uid_field = dataset.uid_field
    internal_uid = dataset.token2id(uid_field, external_user_id)
    matched = (test_data._dataset.inter_feat[uid_field] == internal_uid).nonzero(
        as_tuple=False
    )
    if len(matched) == 0:
        raise ValueError(
            f"user [{external_user_id}] exists in the filtered dataset, "
            "but has no row in the test split"
        )

    interaction = test_data._dataset[matched[-1:].view(-1).cpu().numpy()]
    return test_data.transform(test_data._dataset, interaction)


@torch.no_grad()
def recommend_for_user(model, test_data, dataset, external_user_id, topk):
    interaction = get_user_interaction(test_data, dataset, external_user_id)
    interaction = interaction.to(model.device)
    model.eval()

    scores = model.full_sort_predict(interaction).view(-1)
    scores[0] = -float("inf")

    item_seq_field = dataset.iid_field + dataset.config["LIST_SUFFIX"]
    if item_seq_field in interaction:
        seen_items = interaction[item_seq_field].view(-1)
        seen_items = seen_items[seen_items > 0]
        scores[seen_items] = -float("inf")

    k = min(topk, scores.numel())
    top_scores, top_item_ids = torch.topk(scores, k)
    item_tokens = dataset.id2token(dataset.iid_field, top_item_ids.cpu())
    return [
        (rank, str(item), float(score))
        for rank, (item, score) in enumerate(zip(item_tokens, top_scores.cpu()), start=1)
    ]


def render_markdown(model_name, dataset_name, recommendations):
    lines = [f"# Recommendation Results - Top-{len(next(iter(recommendations.values()), []))}"]
    for user_id, rows in recommendations.items():
        lines.extend(
            [
                "",
                f"## {model_name}, {dataset_name}",
                f"### User: `{user_id}`",
                "| Rank | Item | Score |",
                "|---:|---|---:|",
            ]
        )
        for rank, item, score in rows:
            lines.append(f"| {rank} | {item} | {score:.6f} |")
    return "\n".join(lines) + "\n"


def render_markdown_compact(title, topk, recommendations):
    lines = [
        "## Recommendation Results",
        f"### {title}",
        f"| User | Top-{topk} items | Scores |",
        "| ---- | ------------ | ------ |",
    ]
    for user_id, rows in recommendations.items():
        items = ", ".join(item for _rank, item, _score in rows)
        scores = ", ".join(str(score) for _rank, _item, score in rows)
        lines.append(f"| {user_id} | {items} | {scores} |")
    return "\n".join(lines) + "\n"


def render_csv(recommendations):
    lines = ["user_id,rank,item_id,score"]
    for user_id, rows in recommendations.items():
        for rank, item, score in rows:
            lines.append(f"{user_id},{rank},{item},{score:.8f}")
    return "\n".join(lines) + "\n"


def load_checkpoint_data(checkpoint, show_recbole_logs):
    if show_recbole_logs:
        from recbole.quick_start import load_data_and_model

        return load_data_and_model(checkpoint)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        from recbole.quick_start import load_data_and_model

        return load_data_and_model(checkpoint)


def main():
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.top_users < 0:
        raise ValueError("--top_users must be nonnegative")
    users = load_requested_users(args)

    config, model, dataset, _train_data, _valid_data, test_data = load_checkpoint_data(
        args.checkpoint, args.show_recbole_logs
    )
    model.eval()
    users = dedupe_preserve_order(users + select_top_users(test_data, dataset, args.top_users))
    if not users:
        raise ValueError("Pass --top_users, at least one --user, or --users_file.")

    recommendations = {}
    for user_id in users:
        try:
            recommendations[user_id] = recommend_for_user(
                model, test_data, dataset, user_id, args.topk
            )
        except ValueError as exc:
            recommendations[user_id] = [(1, f"ERROR: {exc}", float("nan"))]

    title = args.title or f"{config['model']}, {config['dataset']}"
    if args.format == "markdown":
        output = render_markdown(config["model"], config["dataset"], recommendations)
    elif args.format == "markdown_compact":
        output = render_markdown_compact(title, args.topk, recommendations)
    else:
        output = render_csv(recommendations)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
