import pytest

from conftest import load_tool

ml = load_tool("merge_labels")


def _folder(root, name, lines):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "good.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def data(tmp_path):
    a = _folder(
        tmp_path,
        "gen_a",
        [
            "crops/0/image_00000.webp\tпривет\t100\t25",
            "crops/0/image_00001.webp\tГОСТ 1\t20\t200",
        ],
    )
    b = _folder(
        tmp_path,
        "part/gen_b",
        ["crops/0/image_00000.webp\tмир\t60\t25", "broken line", ""],
    )
    return tmp_path, a, b


def test_paths_prefixed_by_folder_relative_to_root(data):
    root, a, b = data
    train, val, stats = ml.merge(root, [a, b], set())
    assert train == [
        "gen_a/crops/0/image_00000.webp\tпривет\t100\t25",
        "gen_a/crops/0/image_00001.webp\tГОСТ 1\t20\t200",
        "part/gen_b/crops/0/image_00000.webp\tмир\t60\t25",
    ]
    assert val == [] and stats["bad"] == 1


def test_drop_vertical_and_val_by_folder(data):
    root, a, b = data
    train, val, stats = ml.merge(root, [a, b], {"gen_b"}, drop_vertical=True)
    assert train == ["gen_a/crops/0/image_00000.webp\tпривет\t100\t25"]
    assert val == ["part/gen_b/crops/0/image_00000.webp\tмир\t60\t25"]
    assert stats["vertical"] == 1 and stats["lines"] == 2


@pytest.mark.parametrize(
    "text,w,h,expected",
    [("ab", 20, 200, True), ("a", 20, 200, False), ("ab", 200, 20, False)],
)
def test_is_vertical(text, w, h, expected):
    assert ml.is_vertical(text, w, h) is expected


def test_main_writes_files(data, tmp_path):
    root, a, b = data
    prefix = tmp_path / "out" / "all"
    code = ml.main(
        [
            "--root",
            str(root),
            "--inputs",
            str(a),
            str(b),
            "--val",
            "gen_b",
            "--drop-vertical",
            "--out-prefix",
            str(prefix),
        ]
    )
    assert code == 0
    assert (tmp_path / "out" / "all_train_w_h.txt").read_text(encoding="utf-8").count(
        "\n"
    ) == 1
    assert (tmp_path / "out" / "all_val_w_h.txt").read_text(encoding="utf-8").count(
        "\n"
    ) == 1


def test_main_rejects_unknown_val_folder(data, tmp_path):
    root, a, b = data
    with pytest.raises(SystemExit):
        ml.main(
            [
                "--root",
                str(root),
                "--inputs",
                str(a),
                "--val",
                "nope",
                "--out-prefix",
                str(tmp_path / "x"),
            ]
        )
