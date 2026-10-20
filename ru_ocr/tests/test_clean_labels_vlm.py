import gzip

import pytest

from conftest import load_tool

cl = load_tool("clean_labels_vlm")


@pytest.mark.parametrize(
    "label, pred, vlm, expected",
    [
        ("кот", "кит", "кот", "label_ok"),
        ("кот", "кит", "  кит ", "pred_ok"),  # only the VLM text is normalized
        ("кот", "кит", "кат", None),
        # spaces only: the VLM's space positions decide, its letters may be misread
        ("ГОСТ 2 .4 1 1 -7 2", "ГОСТ 2.411-72", "FOCT 2.411—72", "pred_ok"),
        ("где", "где ", "rde", "label_ok"),  # an end space of pred is ignored when comparing
        ("1 0", "10 ", "10", None),  # ... but such a pred is never chosen
        ("2 От 10 д о 100 к г", "2 От 10 до 100 кг", "2 Otr 10 до 100 кг", None),  # VLM length differs
        # look-alikes only: the alphabet of the line's letters without a twin decides, the VLM is not used
        ("согласно OK (МК", "согласно ОК (МК", "", "pred_ok"),
        ("1000/FRUCS 1000 А» необходимо", "1000/FRUCS 1000 A» необходимо", "", "label_ok"),
        ("Е", "E", "E", "label_ok"),  # a lone letter keeps the label
        ("2а", "2a", "2a", "label_ok"),
    ],
)
def test_vlm_verdict(label, pred, vlm, expected):
    assert cl.vlm_verdict(label, pred, vlm) == expected


def test_unify_words_follows_the_word_majority():
    assert cl.unify_words("больные XГВ в") == "больные ХГВ в"
    assert cl.unify_words("Cтроительство and Miсrоsoft") == "Строительство and Microsoft"
    assert cl.unify_words("Mиcrosoft") == "Mиcrosoft"  # и has no Latin twin: left as it is
    assert cl.unify_words("ГОСТ 2.411-72 and ABC") == "ГОСТ 2.411-72 and ABC"


def test_clean_path():
    assert cl.clean_path("data/all_train_w_h.txt").replace("\\", "/") == "data/all_train_clean_w_h.txt"
    assert cl.clean_path("x/labels.txt").replace("\\", "/") == "x/labels_clean.txt"


def _tsv(path, header, rows, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as f:
        f.writelines("\t".join(r) + "\n" for r in [header] + rows)


def test_main_writes_cleaned_files_and_never_overwrites(tmp_path, capsys):
    labels = tmp_path / "train_w_h.txt"
    labels.write_text(
        "a.webp\tдом\t10\t5\n"      # model == label
        "b.webp\tкот\t10\t5\n"      # VLM confirms the label
        "c.webp\tкат\t10\t5\n"      # VLM confirms pred: label := pred
        "d.webp\tлес\t10\t5\n"      # disputed: dropped
        "e.webp\tсинтетика\t10\t5\n",  # no prediction (another dataset): kept
        encoding="utf-8")
    preds = tmp_path / "preds_all.tsv.gz"
    _tsv(preds, ["path", "pred"], [["a.webp", "дом"], ["b.webp", "кит"], ["c.webp", "кит"], ["d.webp", "лис"]], gz=True)
    review = tmp_path / "review_vlm.tsv"
    _tsv(review, ["path", "label", "pred", "vlm"],
         [["b.webp", "кот", "кит", "кот"], ["c.webp", "кат", "кит", "кит"], ["d.webp", "лес", "лис", "лас"]])
    argv = ["--preds", str(preds), "--vlm", str(review), "--labels", str(labels)]

    assert cl.main(argv) == 0

    out = (tmp_path / "train_clean_w_h.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[:2] for line in out] == [
        ["a.webp", "дом"], ["b.webp", "кот"], ["c.webp", "кит"], ["e.webp", "синтетика"]]
    assert "4 lines" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not overwritten"):
        cl.main(argv)
