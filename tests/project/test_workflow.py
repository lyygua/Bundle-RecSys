"""Regression checks for the portable Amazon workflow and local compatibility fixes."""

import gzip
import json
from pathlib import Path

import pytest
import torch

from prepare_amazon import HEADER, convert_reviews
from extract_amazon_target_user_content import extract_domain
from recbole.config import Config
from recbole.trainer.trainer import _checkpoint_prefix
from recbole.utils import torch_load_checkpoint


def write_reviews(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_review_conversion_preserves_product_group_and_timestamp(tmp_path):
    raw = tmp_path / "reviews.jsonl.gz"
    write_reviews(raw, [{"user_id": "user1", "asin": "variant1", "parent_asin": "group1",
                        "rating": 5, "timestamp": 1417479293000}])
    output = tmp_path / "atomic" / "example.inter"
    assert convert_reviews(raw, output, dry_run=True) == 1
    assert not output.parent.exists()
    assert convert_reviews(raw, output) == 1
    assert output.read_text() == HEADER + "user1\tgroup1\t5.0\t1417479293000\n"
    with pytest.raises(FileExistsError):
        convert_reviews(raw, output)


def test_invalid_review_does_not_publish_partial_interactions(tmp_path):
    raw = tmp_path / "reviews.jsonl.gz"
    write_reviews(raw, [{"user_id": "user1", "parent_asin": "group1", "rating": 5,
                        "timestamp": 1}, {"user_id": "missing_fields"}])
    output = tmp_path / "atomic" / "example.inter"
    with pytest.raises(ValueError, match=":2:"):
        convert_reviews(raw, output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_conversion_keeps_last_review_per_user_product_pair(tmp_path):
    raw = tmp_path / "reviews.jsonl.gz"
    write_reviews(raw, [
        {"user_id": "user1", "parent_asin": "group1", "rating": 1, "timestamp": 1},
        {"user_id": "user2", "parent_asin": "group1", "rating": 3, "timestamp": 2},
        {"user_id": "user1", "parent_asin": "group1", "rating": 5, "timestamp": 3},
    ])
    output = tmp_path / "atomic" / "example.inter"
    assert convert_reviews(raw, output, dry_run=True) == 2
    assert not output.parent.exists()
    assert convert_reviews(raw, output) == 2
    assert output.read_text() == HEADER + (
        "user2\tgroup1\t3.0\t2\nuser1\tgroup1\t5.0\t3\n"
    )


def test_subset_includes_metadata_for_unreviewed_recommendations(tmp_path):
    results = tmp_path / "recommendations"
    results.mkdir()
    (results / "model.md").write_text(
        "| User | Top-2 items | Scores |\n| ---- | ---- | ---- |\n"
        "| A_SYNTHETIC | recommended | 0.9 |\n"
    )
    write_reviews(tmp_path / "reviews.jsonl.gz", [
        {"user_id": "A_SYNTHETIC", "parent_asin": "reviewed"},
        {"user_id": "OTHER", "parent_asin": "unrelated"},
    ])
    write_reviews(tmp_path / "metadata.jsonl.gz", [
        {"parent_asin": "reviewed"}, {"parent_asin": "recommended"},
        {"parent_asin": "unrelated"},
    ])
    output = extract_domain(results, tmp_path, "synthetic", {
        "reviews": "reviews.jsonl.gz", "metadata": "metadata.jsonl.gz",
        "results": ["model.md"],
    })
    assert output["review_count"] == 1
    assert output["product_metadata_record_count"] == 2
    assert output["missing_recommended_metadata_parent_asins"] == []


def test_checkpoint_loader_restores_recbole_objects(tmp_path):
    config = Config(model="BPR", dataset="external", config_dict={"use_gpu": False})
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"config": config, "state_dict": {"weight": torch.tensor([1.0])}}, checkpoint)
    restored = torch_load_checkpoint(checkpoint, map_location="cpu")
    assert restored["config"]["dataset"] == "external"
    assert torch.equal(restored["state_dict"]["weight"], torch.tensor([1.0]))
    assert _checkpoint_prefix({"model": "SASRec", "dataset": "Amazon_All_Beauty"}) == (
        "SASRec-Amazon-All-Beauty"
    )


def test_external_data_path_applies_to_movielens_too(tmp_path):
    config = Config(model="BPR", dataset="ml-100k", config_dict={
        "data_path": str(tmp_path), "use_gpu": False,
    })
    assert Path(config["data_path"]) == tmp_path / "ml-100k"


def test_cpu_option_is_honored_on_a_cuda_host(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = Config(model="BPR", dataset="external", config_dict={"use_gpu": False})
    assert config["device"] == torch.device("cpu")
