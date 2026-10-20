import gzip

import pytest

from conftest import load_tool

br = load_tool("build_review")


def _labels(path, rows):
    path.write_text("".join(f"{p}\t{t}\t10\t5\n" for p, t in rows), encoding="utf-8")


def _table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        lines = [line.rstrip("\n").split("\t") for line in f]
    return lines[0], lines[1:]


def test_splits_char_and_space_disagreements_in_review_order(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    _labels(train, [("a.webp", "дом"), ("b.webp", "кот"), ("c.webp", "к о т")])
    _labels(val, [("d.webp", 'он "сказал"'), ("e.webp", "река")])
    preds = tmp_path / "preds_partial.tsv"
    preds.write_text("0\ta.webp\tдом\t0.99\n"        # exact
                     "1\tb.webp\tкит\t0.90\n"        # 1 char edit
                     "2\tc.webp\tкот\t0.80\n"        # spaces only
                     "3\td.webp\tон \"скзал\"\t0.70\n"  # quotes survive, 1 edit
                     "4\te.webp\tрока\t0.95\n",      # 1 edit, more confident
                     encoding="utf-8")
    out = tmp_path / "out"

    assert br.main(["--preds", str(preds), "--labels", str(train), str(val), "--out", str(out)]) == 0

    header, rows = _table(out / "preds_all.tsv.gz")
    assert header == br.COLUMNS and len(rows) == 5
    _, review = _table(out / "review.tsv")
    assert [r[0] for r in review] == ["e.webp", "b.webp", "d.webp"]  # same dist_ns: higher score first
    assert review[2][1:3] == ['он "сказал"', 'он "скзал"']
    _, spaces = _table(out / "review_spaces.tsv")
    assert [(r[0], r[4], r[5]) for r in spaces] == [("c.webp", "2", "0")]


def test_merges_sessions_and_skips_a_cut_line(tmp_path):
    labels = tmp_path / "train.txt"
    _labels(labels, [("a.webp", "да"), ("b.webp", "нет")])
    s1, s2 = tmp_path / "s1.tsv", tmp_path / "s2.tsv"
    s1.write_text("0\ta.webp\tда\t0.9\n1\tb.webp\tнет\t", encoding="utf-8")  # last line cut after a tab
    s2.write_text("1\tb.webp\tнет\t0.8\n", encoding="utf-8")

    assert br.main(["--preds", str(s1), str(s2), "--labels", str(labels), "--out", str(tmp_path / "o")]) == 0

    _, rows = _table(tmp_path / "o" / "preds_all.tsv.gz")
    assert [(r[0], r[2]) for r in rows] == [("a.webp", "да"), ("b.webp", "нет")]


def test_a_prediction_for_another_label_line_is_an_error(tmp_path):
    labels = tmp_path / "train.txt"
    _labels(labels, [("a.webp", "да")])
    preds = tmp_path / "p.tsv"
    preds.write_text("0\tother.webp\tда\t0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="other.webp"):
        br.main(["--preds", str(preds), "--labels", str(labels), "--out", str(tmp_path / "o")])
