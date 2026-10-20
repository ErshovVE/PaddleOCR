import tarfile
import zipfile

from conftest import load_tool

pk = load_tool("pack_dataset")


def _crop(root, path, size):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * size)


def test_packs_label_paths_grouped_and_merges_small_groups(tmp_path):
    root = tmp_path / "ru_root"
    big = ["synth/part1/crops/0/a.webp", "synth/part2/crops/0/a.webp"]
    small = ["stroyinf/f1/crops/0/a.webp", "stroyinf/f2/crops/0/a.webp"]
    for p in big:
        _crop(root, p, 3000)
    for p in small:
        _crop(root, p, 10)
    _crop(root, "synth/part1/crops/0/unused.webp", 10)
    labels = tmp_path / "train.txt"
    labels.write_text("".join("{}\tтекст\t10\t5\n".format(p) for p in big + small), encoding="utf-8")
    out = tmp_path / "out"

    assert pk.main(["--root", str(root), "--labels", str(labels), "--min-group-mb", "0.001", "--out", str(out)]) == 0

    names = sorted(p.name for p in out.iterdir())
    assert names == ["labels.zip", "stroyinf.zip", "synth_part1.zip", "synth_part2.zip"]
    with zipfile.ZipFile(out / "synth_part1.zip") as z:
        assert z.namelist() == ["synth/part1/crops/0/a.webp"]  # the unreferenced crop is not packed
    with zipfile.ZipFile(out / "stroyinf.zip") as z:
        assert sorted(z.namelist()) == sorted(small)
    with zipfile.ZipFile(out / "labels.zip") as z:
        assert z.namelist() == ["labels/train.txt"]


def test_tar_format_writes_tarpacks_with_the_label_paths(tmp_path):
    root = tmp_path / "ru_root"
    paths = ["stroyinf/f1/crops/0/a.webp", "stroyinf/f1/crops/1/b.webp"]
    for p in paths:
        _crop(root, p, 10)
    labels = tmp_path / "val.txt"
    labels.write_text("".join("{}\tтекст\t10\t5\n".format(p) for p in paths), encoding="utf-8")
    out = tmp_path / "out"

    assert pk.main(["--root", str(root), "--labels", str(labels), "--format", "tar", "--out", str(out)]) == 0

    assert sorted(p.name for p in out.iterdir()) == ["labels.zip", "stroyinf.tarpack"]
    with tarfile.open(out / "stroyinf.tarpack") as t:
        assert sorted(t.getnames()) == paths
        assert t.extractfile(paths[0]).read() == b"x" * 10


def test_from_zip_repacks_every_file_with_its_name_and_bytes(tmp_path):
    source = tmp_path / "synth_part1.zip"
    files = {"synth/part1/crops/0/a.webp": b"abc", "synth/part1/crops/1/b.webp": b"defg"}
    with zipfile.ZipFile(source, "w", zipfile.ZIP_STORED) as z:
        z.writestr("synth/part1/crops/", "")  # a directory entry is skipped
        for name, data in files.items():
            z.writestr(name, data)
    out = tmp_path / "out"

    assert pk.main(["--from-zip", str(source), "--out", str(out)]) == 0

    with tarfile.open(out / "synth_part1.tarpack") as t:
        assert sorted(t.getnames()) == sorted(files)
        assert {n: t.extractfile(n).read() for n in files} == files


def test_without_from_zip_root_and_labels_are_required(tmp_path, capsys):
    try:
        pk.main(["--out", str(tmp_path / "out")])
    except SystemExit as exc:
        assert exc.code == 2
    assert "--root and --labels are required" in capsys.readouterr().err


def test_missing_crop_is_an_error(tmp_path, capsys):
    root = tmp_path / "ru_root"
    root.mkdir()
    labels = tmp_path / "train.txt"
    labels.write_text("synth/part1/crops/0/a.webp\tтекст\t10\t5\n", encoding="utf-8")
    try:
        pk.main(["--root", str(root), "--labels", str(labels), "--out", str(tmp_path / "out")])
    except SystemExit as exc:
        assert exc.code == 2
    assert "do not exist" in capsys.readouterr().err
