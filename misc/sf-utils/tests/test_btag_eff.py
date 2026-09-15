import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conv = load("btag_converter_test", "bEff-convert-to-correction.py")
plots = load("btag_plots_test", "plot-btag-eff-families.py")
global_plots = load("btag_global_test", "plot-btag-eff-global.py")
families = load("btag_families_test", "btag_eff_families.py")
slurm_submit = load("btag_slurm_submit_test", "../../preselection/slurm/submit.py")
run_rdf = load("btag_run_rdf_test", "../../preselection/run_rdf.py")


def counts(den=10., tight=2., loose=5., lt=3., untagged=5.):
    values = {}
    medium = loose - lt
    extra_tight = 0.5 * tight
    extra_extra_tight = 0.5 * extra_tight
    states = {
        "den": den, "L": loose, "M": medium, "T": tight,
        "XT": extra_tight, "XXT": extra_extra_tight,
        "LnotM": loose - medium, "MnotT": medium - tight,
        "TnotXT": tight - extra_tight, "XTnotXXT": extra_tight - extra_extra_tight,
        "N": untagged,
    }
    for flavor in conv.FLAVORS:
        for state, value in states.items():
            values[flavor, state] = np.array([[value]])
    return values


class BTagEfficiencyTests(unittest.TestCase):
    def test_yaml_controls_strict_final_groups(self):
        sample = "VBSWZH_c2v1p0"
        self.assertEqual(families.sample_family(sample, year="2024Prompt"), "VBS_VVH")
        self.assertEqual(families.final_sample_family(sample, year="2024Prompt"), "VBS_VVH")
        self.assertEqual(families.final_channel("1lep_1FJ", year="2024Prompt"), "leptonic")
        self.assertNotIn("0lep_1FJ_met", families.retained_source_channels(year="2024Prompt"))
        self.assertIn("0lep_1FJ_met", families.excluded_source_channels(year="2024Prompt"))
        with self.assertRaises(ValueError):
            families.final_channel("all_events", year="2024Prompt")

    def test_yaml_rejects_incomplete_or_duplicate_final_groups(self):
        config = families.load_config("2024Prompt")
        bad = json.loads(json.dumps(config))
        bad["final_merges"]["samples"]["VBS_VVH"] = []
        with self.assertRaises(ValueError):
            families.validate_config(bad)
        bad = json.loads(json.dumps(config))
        bad["final_merges"]["channels"]["0lep_0FJ"].append("1lep_1FJ")
        with self.assertRaises(ValueError):
            families.validate_config(bad)

    def test_runtime_lookup_uses_final_keys_without_exact_fallback(self):
        sample = "WplusH-Wto2Q-Hto2B_Par-M-125"
        self.assertEqual(families.final_runtime_keys("2024Prompt", "1lep_1FJ", sample),
                         ("btag_2024Prompt_leptonic", "VH"))
        with self.assertRaises(ValueError):
            families.final_runtime_keys("2024Prompt", "1lep_1FJ", "not_a_configured_sample")

    def test_unweighted_category_uncertainty(self):
        uncertainty = conv.mcstat_efficiency_uncertainty(
            np.array([[2.]]), np.array([[10.]]), np.array([[2.]]), np.array([[10.]]))
        self.assertAlmostEqual(uncertainty[0, 0] ** 2, .2 * .8 / 10.)

    def test_weighted_category_uncertainty(self):
        uncertainty = conv.mcstat_efficiency_uncertainty(
            np.array([[3.]]), np.array([[5.]]), np.array([[5.]]), np.array([[13.]]))
        self.assertAlmostEqual(uncertainty[0, 0] ** 2, ((1 - 2 * .6) * 5 + .6 ** 2 * 13) / 25.)
        invalid = conv.mcstat_efficiency_uncertainty(
            np.array([[3.]]), np.array([[5.]]), np.array([[100.]]), np.array([[1.]]))
        self.assertTrue(np.isnan(invalid[0, 0]))

    def test_exclusive_category_identity_and_mask(self):
        valid = counts()
        for flavor in conv.FLAVORS:
            self.assertTrue(np.allclose(valid[flavor, "L"], valid[flavor, "LnotM"] + valid[flavor, "M"]))
            self.assertTrue(np.allclose(valid[flavor, "M"], valid[flavor, "MnotT"] + valid[flavor, "T"]))
            self.assertTrue(np.allclose(valid[flavor, "T"], valid[flavor, "TnotXT"] + valid[flavor, "XT"]))
            self.assertTrue(np.allclose(valid[flavor, "XT"], valid[flavor, "XTnotXXT"] + valid[flavor, "XXT"]))
            self.assertTrue(np.allclose(valid[flavor, "den"], sum(valid[flavor, state] for state in conv.EXCLUSIVE_CATEGORIES)))
        bad = counts(loose=4.)
        self.assertTrue(conv.invalid_count_bins(bad)["b"][0, 0])
        result = plots.efficiencies(conv, bad, {key: value.copy() for key, value in bad.items()})
        self.assertFalse(result[2]["b", "T"][0, 0])
        self.assertNotIn("L", conv.EXCLUSIVE_CATEGORIES)

    def test_manifest_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": {
                "sample": {"status": "prepared", "reason": None, "n_files": 2, "job_indices": [0, 1]},
            }, "jobs": {
                "sample_0": {"sample": "sample", "job_idx": 0},
                "sample_1": {"sample": "sample", "job_idx": 1},
            }}))
            sample_dir = root / "outputs" / "sample"
            sample_dir.mkdir(parents=True)
            (sample_dir / "output_0.root").touch()
            with self.assertRaises(ValueError):
                conv.validate_job_manifest(root / "outputs", manifest)
            (sample_dir / "output_1.root").touch()
            self.assertEqual(conv.validate_job_manifest(root / "outputs", manifest)["sample"]["discovered_jobs"], 2)
            (sample_dir / "output_01.root").touch()
            with self.assertRaises(ValueError):
                conv.validate_job_manifest(root / "outputs", manifest)

    def test_skipped_sample_is_manifested_and_rejects_final_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            output = root / "outputs"
            task.mkdir()
            output.mkdir()
            manifest = slurm_submit.JobManifest(task, output / "manifest.json")
            manifest.add_sample("empty", 0, "skipped_no_files", "no_files_found")
            manifest.save()
            published = json.loads((output / "manifest.json").read_text())
            self.assertEqual(published["samples"]["empty"]["status"], "skipped_no_files")
            with self.assertRaises(ValueError):
                conv.validate_job_manifest(output, output / "manifest.json")
            manifest = {
                "samples": {"unsubmitted": {
                    "status": "prepared", "reason": None, "n_files": 1, "job_indices": [],
                }},
                "jobs": {},
            }
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                conv.validate_job_manifest(output, output / "manifest.json")

    def test_btag_slurm_channel_layout_is_documented_and_colocated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outdir = root / "btag_outputs" / "2024Prompt" / "0lep_0FJ"
            self.assertEqual(run_rdf.output_dir_for_channel(
                str(root / "btag_outputs"), "ignored_tag", "0lep_0FJ", "slurm", True, "2024Prompt"), str(outdir))
            published = slurm_submit.prepare_output_directory(outdir, True, "2024Prompt")
            self.assertEqual(published, outdir / "manifest.json")
            task = root / "task"
            task.mkdir()
            manifest = slurm_submit.JobManifest(task, published)
            manifest.add_sample("sample", 1)
            manifest.add_job("sample", "sample", 0, ["input.root"], str(task / "sample"))
            manifest.save()
            sample_dir = outdir / "sample"
            sample_dir.mkdir()
            (sample_dir / "output_0.root").touch()
            self.assertTrue((outdir / "manifest.json").is_file())
            self.assertEqual(conv.validate_job_manifest(outdir, outdir / "manifest.json")["sample"]["discovered_jobs"], 1)

    def test_final_input_discovery_requires_all_channels_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                conv.discover_final_inputs(root, "2024Prompt")
            for channel in families.retained_source_channels(year="2024Prompt"):
                directory = root / channel
                directory.mkdir()
                (directory / "manifest.json").write_text(json.dumps({"jobs": {}}))
            (root / "0lep_1FJ_met").mkdir()
            (root / "0lep_2FJ_met").mkdir()
            (root / "all_events").mkdir()
            inputs, manifests, ignored = conv.discover_final_inputs(root, "2024Prompt")
            self.assertEqual(set(inputs), set(families.retained_source_channels(year="2024Prompt")))
            self.assertEqual(set(manifests), set(inputs))
            self.assertEqual(ignored, ["0lep_1FJ_met", "0lep_2FJ_met", "all_events"])
            (root / "1lep_1FJ" / "manifest.json").unlink()
            with self.assertRaises(ValueError):
                conv.discover_final_inputs(root, "2024Prompt")

    def test_failed_final_build_does_not_touch_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "btag_eff_2024Prompt.json"
            output.write_text("original payload\n")
            previous_argv = sys.argv
            try:
                sys.argv = ["bEff-convert-to-correction.py", "--final", "--input-root", str(root),
                            "--year", "2024Prompt", "--output", str(output)]
                with self.assertRaises(ValueError):
                    conv.main()
            finally:
                sys.argv = previous_argv
            self.assertEqual(output.read_text(), "original payload\n")

    def test_sparse_summary_is_unavailable(self):
        states = (*conv.INCLUSIVE_WPS, *conv.EXCLUSIVE_CATEGORIES)
        values = {(flavor, wp): np.array([[.2]]) for flavor in conv.FLAVORS for wp in states}
        errors = {(flavor, wp): np.array([[.1]]) for flavor in conv.FLAVORS for wp in states}
        valid = {(flavor, wp): np.array([[flavor == "b" and wp == "T"]])
                 for flavor in conv.FLAVORS for wp in states}
        pathological = {flavor: np.array([[False]]) for flavor in conv.FLAVORS}
        result = values, errors, valid, pathological
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {"plot_dir": Path(tmp)})()
            plots.summary_json({"a": None, "b": None}, {"a": result, "b": result}, args)
            summary = json.loads((Path(tmp) / "compatibility_summary.json").read_text())
            entry = summary["a__vs__b/all_flavours_all_categories"]
            self.assertFalse(entry["available"])
            self.assertIsNone(entry["fraction_gt2"])

    def test_global_diagnostics_accept_source_and_final_input_root_cli(self):
        parser = global_plots.parse_args
        previous_argv = sys.argv
        try:
            sys.argv = ["plot-btag-eff-global.py", "--input-root", "/tmp/inputs", "--year", "2024Prompt",
                        "--mode", "families"]
            args = parser()
        finally:
            sys.argv = previous_argv
        self.assertEqual(args.input_root, Path("/tmp/inputs"))

    def test_source_and_final_global_grouping_are_distinct(self):
        raw = counts()
        variances = {key: value.copy() for key, value in raw.items()}
        source = {
            ("0lep_2FJ", "DY"): raw,
            ("0lep_3FJ", "DY"): raw,
        }
        final = {("0lep_2FJ+", "EWK"): raw}
        original_source = conv.discover_source_histograms
        original_final = conv.discover_final_histograms
        try:
            conv.discover_source_histograms = lambda *args: (source, {key: variances for key in source}, {}, {}, {"x": {}}, [])
            conv.discover_final_histograms = lambda *args: (final, {key: variances for key in final}, {}, {}, {"x": {}}, [])
            grouped_source, _, _ = global_plots.grouped_inputs(conv, Path("unused"), "2024Prompt", "channels", False, {})
            grouped_final, _, _ = global_plots.grouped_inputs(conv, Path("unused"), "2024Prompt", "channels", True, {})
        finally:
            conv.discover_source_histograms = original_source
            conv.discover_final_histograms = original_final
        self.assertEqual(set(grouped_source), {"0lep_2FJ", "0lep_3FJ"})
        self.assertEqual(set(grouped_final), {"0lep_2FJ+"})

    def test_matrix_plots_use_module_categories(self):
        raw = counts()
        result = plots.efficiencies(conv, raw, {key: value.copy() for key, value in raw.items()})
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {"plot_dir": Path(tmp)})()
            plots.matrix_plots({"a": None}, {"a": result}, args, 1., 13.)
            self.assertTrue((Path(tmp) / "compatibility_b_N.pdf").exists())
            self.assertTrue((Path(tmp) / "compatibility_b_XXT.pdf").exists())


if __name__ == "__main__":
    unittest.main()
