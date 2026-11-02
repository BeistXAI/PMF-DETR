"""Behavioral tests for dataset preparation and patient-level separation."""
import csv
import tempfile
import unittest
from pathlib import Path

from tools.convert_voc import convert_voc
from tools.make_smoke_data import make_smoke_data
from tools.split_dataset import split_manifest


class PreparationTests(unittest.TestCase):
    def test_patient_groups_remain_whole_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_smoke_data(Path(directory) / "data")
            manifest = root / "patients.csv"
            # Pair image records into groups to exercise patient-level assignment.
            with manifest.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            for index, row in enumerate(records):
                row["patient_id"] = f"patient_{index // 2}"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image", "patient_id"])
                writer.writeheader()
                writer.writerows(records)
            split_manifest(manifest, root, root / "split_a", seed=17)
            split_manifest(manifest, root, root / "split_b", seed=17)
            with (root / "split_a/assignments.csv").open(newline="") as stream:
                assignments = list(csv.DictReader(stream))
            seen = {}
            for row in assignments:
                self.assertEqual(seen.setdefault(row["patient_id"], row["split"]), row["split"])
            self.assertEqual(len(assignments), len(records))
            for split in ("train", "val", "test"):
                self.assertEqual((root / "split_a" / f"{split}.txt").read_text(), (root / "split_b" / f"{split}.txt").read_text())
            with self.assertRaises(FileExistsError):
                split_manifest(manifest, root, root / "split_a")

    def test_voc_boundary_and_empty_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "xml"
            annotations.mkdir()
            prefix = "<annotation><size><width>100</width><height>50</height></size>"
            (annotations / "positive.xml").write_text(prefix + "<object><name>fracture</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object></annotation>")
            (annotations / "negative.xml").write_text(prefix + "</annotation>")
            self.assertEqual(convert_voc(annotations, root / "labels"), 2)
            values = (root / "labels/positive.txt").read_text().split()
            self.assertEqual([float(v) for v in values], [0, .5, .5, 1, 1])
            self.assertEqual((root / "labels/negative.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
