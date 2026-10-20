import pytest

from conftest import load_tool

vr = load_tool("vlm_review")

HEADER = "path\tlabel\tpred\tscore\tdist\tdist_ns\n"


def test_joins_parts_over_both_tables_with_verdicts(tmp_path):
    review = tmp_path / "review.tsv"
    review.write_text(HEADER + "a.webp\tкот\tкит\t0.9\t1\t1\n"
                               'b.webp\tон "да"\tон "до"\t0.8\t1\t1\n', encoding="utf-8")
    spaces = tmp_path / "review_spaces.tsv"
    spaces.write_text(HEADER + "c.webp\tк о т\tкот\t0.7\t2\t0\n", encoding="utf-8")
    part0, part1 = tmp_path / "vlm_part0.tsv", tmp_path / "vlm_part1.tsv"
    part0.write_text("0\tкот\t0.95\n2\tкот\t0.9\n", encoding="utf-8")
    part1.write_text('1\tон "ду"\t0.5\n3\tобрез', encoding="utf-8")  # a cut last line is skipped
    out = tmp_path / "review_vlm.tsv"

    assert vr.main(["--reviews", str(review), str(spaces), "--parts", str(part0), str(part1), "--out", str(out)]) == 0

    lines = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == HEADER.strip().split("\t") + ["review", "vlm", "vlm_score", "verdict"]
    assert [(r[0], r[6], r[7], r[9]) for r in lines[1:]] == [
        ("a.webp", "review.tsv", "кот", "label_ok"),
        ("b.webp", "review.tsv", 'он "ду"', "disputed"),
        ("c.webp", "review_spaces.tsv", "кот", "pred_ok"),
    ]


def test_a_review_line_without_a_reading_is_an_error(tmp_path, capsys):
    review = tmp_path / "review.tsv"
    review.write_text(HEADER + "a.webp\tкот\tкит\t0.9\t1\t1\n", encoding="utf-8")
    part = tmp_path / "vlm_part0.tsv"
    part.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit):
        vr.main(["--reviews", str(review), "--parts", str(part), "--out", str(tmp_path / "o.tsv")])
    assert "no VLM reading" in capsys.readouterr().err


def test_a_review_row_with_a_wrong_field_count_is_an_error(tmp_path):
    review = tmp_path / "review.tsv"
    review.write_text(HEADER + "a.webp\tкот\tкит\t0.9\t1\n", encoding="utf-8")  # one field short
    part = tmp_path / "vlm_part0.tsv"
    part.write_text("0\tкот\t0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 6 fields"):
        vr.main(["--reviews", str(review), "--parts", str(part), "--out", str(tmp_path / "o.tsv")])
