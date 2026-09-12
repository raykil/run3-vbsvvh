#!/usr/bin/env python3
"""Convert merged weighted b-tag efficiency histograms into correctionlib JSON.

Exact-sample mode is preliminary/diagnostic-only. Family mode discovers sample
directories, merges raw weighted yields into physics families, and writes one
correction entry per family.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import uproot
import correctionlib.schemav2 as cs

from btag_eff_families import (excluded_source_channels, final_channel,
                               final_group, load_config,
                               retained_source_channels, sample_family)


FLAVORS = ("b", "c", "light")
INCLUSIVE_WPS = ("L", "M", "T", "XT", "XXT")
EXCLUSIVE_CATEGORIES = ("N", "LnotM", "MnotT", "TnotXT", "XTnotXXT", "XXT")
HISTOGRAM_STATES = ("den", *INCLUSIVE_WPS, "LnotM", "MnotT", "TnotXT", "XTnotXXT", "N")

# Starting points for adaptive binning.  These are deliberately conservative
# defaults: the converter still uses the producer's rectangular grid, but
# replaces low-statistical-quality bins with the yield of an adjacent merged
# region.  The values are configuration knobs rather than physics constants.
MIN_EFFECTIVE_DENOMINATOR = {"b": 100.0, "c": 100.0, "light": 200.0}
MIN_EFFECTIVE_CATEGORY = {"b": 20.0, "c": 20.0, "light": 50.0}
MAX_EFFICIENCY_UNCERTAINTY = {"L": 0.03, "M": 0.03, "T": 0.03,
                              "XT": 0.05, "XXT": 0.05}
MAX_CATEGORY_UNCERTAINTY = 0.05
MIN_DENOMINATOR_SIGNIFICANCE = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", nargs="+",
                        help="All raw ROOT files for one sample (local paths or root:// URLs)")
    inputs.add_argument("--input-dir", type=Path,
                        help="Directory containing one raw-output directory per exact sample")
    inputs.add_argument("--input-root", type=Path,
                        help="Final mode: root containing CHANNEL/manifest.json and CHANNEL/SAMPLE/output_N.root")
    parser.add_argument("--year", required=True, help="Metadata year, e.g. 2024Prompt")
    parser.add_argument("--channel", help="Preliminary conversion channel, e.g. 0lep_0FJ")
    parser.add_argument("--sample", help="Exact RDataFrame sample name from the input JSON (exact-sample mode)")
    parser.add_argument("--output", type=Path,
                        help="Optional output path (defaults to btag_eff_<year>.json)")
    parser.add_argument("--manifest", type=Path,
                        help="Family membership manifest (defaults beside --output in family mode)")
    parser.add_argument("--job-manifest", type=Path,
                        help="Optional Slurm manifest.json: require exactly its expected sample/job outputs")
    parser.add_argument("--final", action="store_true",
                        help="Build final grouped efficiencies from every retained YAML channel under --input-root")
    return parser.parse_args()


def default_output_path(year, preliminary=False, channel=None):
    directory = (Path(__file__).parents[2] / "preselection" / "corrections" /
                 "scalefactors" / "btagging")
    token = year
    if preliminary and channel:
        return directory / f"btag_eff_{token}_{channel}_prelim.json"
    return directory / f"btag_eff_{token}{'_prelim' if preliminary else ''}.json"


def fold_pt_flow(values, path, hist_name):
    """Fold pT flow into edge bins, matching correctionlib's pT clamp flow."""
    # The producer rejects |eta| >= 2.5, so eta flow would indicate a
    # producer/application mismatch rather than information to clamp.
    if np.any(values[:, 0] != 0) or np.any(values[:, -1] != 0):
        raise ValueError(f"{path}:{hist_name} has unexpected eta under/overflow entries")
    central = values[1:-1, 1:-1].astype(float, copy=True)
    central[0, :] += values[0, 1:-1]
    central[-1, :] += values[-1, 1:-1]
    return central


def btag_max_abs_eta(year):
    """Eta acceptance used by both production and the correction payload."""
    if year in ("2016preVFP", "2016postVFP"):
        return 2.4
    if year in ("2017", "2018", "2024Prompt"):
        return 2.5
    raise ValueError(f"Unsupported b-tag efficiency year: {year}")


def histogram_arrays_in_application_range(hist, path, hist_name, year):
    """Return weighted yields and their Sumw2 variances in application bins."""
    values, pt_edges, eta_edges = hist.to_numpy(flow=True)
    variances = hist.variances(flow=True)
    if variances is None:
        raise ValueError(f"{path}:{hist_name} has no Sumw2 information")
    values = fold_pt_flow(values, path, hist_name)
    variances = fold_pt_flow(variances, path, hist_name)
    # The payload and compatibility diagnostics use one inclusive central-jet
    # eta bin; retain the pT dependence but sum the mutually exclusive eta bins.
    values = np.sum(values, axis=1, keepdims=True)
    variances = np.sum(variances, axis=1, keepdims=True)
    # Eta is merged into one bin.  Its correction-domain boundary must still
    # match the year-dependent jet acceptance used during production.
    eta_max = btag_max_abs_eta(year)
    return values, variances, pt_edges[1:-1], np.asarray([-eta_max, eta_max])


def read_merged_histograms(paths, expected_year, expected_channel, expected_sample):
    """Return weighted yields, Sumw2 variances, and common pT/eta edges."""
    merged = {}
    merged_variances = {}
    edges = None
    for path in paths:
        with uproot.open(path) as root_file:
            if "btag_eff_format" not in root_file:
                raise ValueError(f"{path} has no b-tag efficiency format metadata")
            output_format = root_file["btag_eff_format"].member("fTitle")
            if not output_format.startswith("signed baseweight"):
                raise ValueError(
                    f"{path} contains obsolete unweighted b-tag histograms; regenerate it with the current --btag_eff workflow"
                )
            expected_metadata = {
                "btag_eff_year": expected_year,
                "btag_eff_channel": expected_channel,
                "btag_eff_sample": expected_sample,
            }
            for key, expected_value in expected_metadata.items():
                if key not in root_file:
                    raise ValueError(f"{path} has no {key} metadata")
                actual_value = root_file[key].member("fTitle")
                if actual_value != expected_value:
                    raise ValueError(f"{path} has {key}={actual_value!r}, expected {expected_value!r}")
            if "btag_eff_schema_version" not in root_file or root_file["btag_eff_schema_version"].member("fTitle") != "2":
                raise ValueError(f"{path} is not a schema-v2 five-working-point b-tag efficiency file")
            if "btag_eff_working_points" not in root_file or root_file["btag_eff_working_points"].member("fTitle") != ",".join(INCLUSIVE_WPS):
                raise ValueError(f"{path} does not declare the canonical working points {','.join(INCLUSIVE_WPS)}")
            for flavor in FLAVORS:
                for state in HISTOGRAM_STATES:
                    hist_name = f"btag_{flavor}_{state}"
                    if hist_name not in root_file:
                        raise ValueError(f"{path} does not contain {hist_name}")
                    values, variances, pt_edges, eta_edges = histogram_arrays_in_application_range(
                        root_file[hist_name], path, hist_name, expected_year)
                    if edges is None:
                        edges = (pt_edges, eta_edges)
                    elif not (np.array_equal(pt_edges, edges[0]) and np.array_equal(eta_edges, edges[1])):
                        raise ValueError(f"Histogram binning in {path} differs from the other inputs")
                    key = (flavor, state)
                    merged[key] = merged.get(key, np.zeros_like(values, dtype=float)) + values
                    merged_variances[key] = merged_variances.get(key, np.zeros_like(variances, dtype=float)) + variances
    return merged, merged_variances, edges


def add_counts(destination, source):
    for key, values in source.items():
        destination[key] = destination.get(key, np.zeros_like(values, dtype=float)) + values


def output_job_index(path):
    match = re.fullmatch(r"output_(\d+)\.root", path.name)
    if not match:
        raise ValueError(f"Unexpected b-tag output filename {path}; expected output_<job index>.root")
    return int(match.group(1))


def validate_job_manifest(input_dir, manifest_path):
    """Validate that every Slurm task in *manifest_path* produced one ROOT file."""
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"{manifest_path} has no configured-sample completeness metadata")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError(f"{manifest_path} has no jobs")
    expected = {}
    for sample, metadata in samples.items():
        if not isinstance(sample, str) or not isinstance(metadata, dict):
            raise ValueError(f"{manifest_path} has invalid configured-sample metadata")
        indices = metadata.get("job_indices")
        if (metadata.get("status") == "skipped_no_files" or not isinstance(indices, list) or
                not indices):
            raise ValueError(
                f"{manifest_path} is incomplete: configured sample {sample!r} has "
                f"status={metadata.get('status')!r}, jobs={indices!r}, reason={metadata.get('reason')!r}")
        if any(not isinstance(index, int) for index in indices) or len(indices) != len(set(indices)):
            raise ValueError(f"{manifest_path} has invalid expected job indices for {sample}")
        expected[sample] = set(indices)
    seen = {}
    for job in jobs.values():
        sample, index = job.get("sample"), job.get("job_idx")
        if not isinstance(sample, str) or not isinstance(index, int):
            raise ValueError(f"{manifest_path} has a job without sample/job_idx metadata")
        if sample not in expected:
            raise ValueError(f"{manifest_path} has a job for unconfigured sample {sample!r}")
        if index not in expected[sample]:
            raise ValueError(f"{manifest_path} has unexpected job index {sample}:{index}")
        # Build independently from the declared sample index list, so each
        # Slurm task is still required exactly once.
        if index in seen.setdefault(sample, set()):
            raise ValueError(f"{manifest_path} has duplicate expected job index {sample}:{index}")
        seen[sample].add(index)
    if set(seen) != set(expected):
        raise ValueError(f"{manifest_path} jobs do not cover every configured sample")
    for sample, indices in expected.items():
        if seen[sample] != indices:
            raise ValueError(f"{manifest_path} jobs do not match declared indices for {sample}")

    discovered = {}
    directories = {path.name for path in input_dir.iterdir() if path.is_dir()}
    unexpected_dirs = directories - set(expected)
    if unexpected_dirs:
        raise ValueError(f"Unexpected sample directories not in {manifest_path}: {sorted(unexpected_dirs)}")
    for sample, indices in expected.items():
        sample_dir = input_dir / sample
        roots = sorted(sample_dir.glob("*.root")) if sample_dir.is_dir() else []
        found = [output_job_index(path) for path in roots]
        if len(found) != len(set(found)):
            raise ValueError(f"Duplicate ROOT outputs for {sample}: {found}")
        found_set = set(found)
        if found_set != indices:
            raise ValueError(f"Incomplete outputs for {sample}: expected {sorted(indices)}, found {sorted(found_set)}")
        discovered[sample] = {"expected_jobs": len(indices), "discovered_jobs": len(found_set)}
    return discovered


def aggregate_sample_records(records, quality_key):
    """Merge the complete configured exact-sample set in all flavours/states."""
    if not records:
        raise ValueError(f"No samples available for {quality_key}")
    edges = records[0][3]
    counts, variances = {}, {}
    for sample, sample_counts, sample_variances, sample_edges in records:
        if not (np.array_equal(edges[0], sample_edges[0]) and np.array_equal(edges[1], sample_edges[1])):
            raise ValueError(f"Histogram binning differs within {quality_key}")
    for flavor in FLAVORS:
        for state in HISTOGRAM_STATES:
            values = [record[1][flavor, state] for record in records]
            errors = [record[2][flavor, state] for record in records]
            counts[flavor, state] = np.sum(values, axis=0)
            variances[flavor, state] = np.sum(errors, axis=0)
    return counts, variances, edges, [record[0] for record in records]


def discover_sample_histograms(input_dir, year, channel, job_manifest=None, config=None):
    """Read exact sample outputs without combining them."""
    if not input_dir.is_dir():
        raise ValueError(f"--input-dir is not a directory: {input_dir}")
    completeness = (validate_job_manifest(input_dir, job_manifest) if job_manifest else None)
    if completeness is None:
        print("WARNING: b-tag input completeness was not verified (no --job-manifest supplied)")
    samples = []
    for sample_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        roots = sorted(sample_dir.glob("*.root"))
        if not roots:
            # Allow diagnostics/manifests to live beside the sample directories.
            continue
        sample = sample_dir.name
        counts, variances, edges = read_merged_histograms(roots, year, channel, sample)
        samples.append((sample, counts, variances, edges))
    if not samples:
        raise ValueError(f"No sample directories found in {input_dir}")
    return samples, completeness


def discover_family_histograms(input_dir, year, channel, job_manifest=None, config=None):
    """Read exact samples and return canonical family aggregates."""
    samples, completeness = discover_sample_histograms(input_dir, year, channel, job_manifest, config)
    grouped_counts, grouped_variances, grouped_edges, members = {}, {}, {}, {}
    for family in sorted({sample_family(sample, config) for sample, _, _, _ in samples}):
        records = [(sample, counts, variances, edges) for sample, counts, variances, edges in samples
                   if sample_family(sample, config) == family]
        counts, variances, edges, family_members = aggregate_sample_records(
            records, f"{channel}/{family}")
        grouped_counts[family], grouped_variances[family] = counts, variances
        grouped_edges[family], members[family] = edges, family_members
    return grouped_counts, grouped_variances, grouped_edges, members, completeness


def invalid_count_bins(counts, variances=None):
    """Return per-flavor masks for bins that cannot form physical efficiencies."""
    masks = {}
    for flavor in FLAVORS:
        denominator = counts[(flavor, "den")]
        inclusive = {wp: counts[(flavor, wp)] for wp in INCLUSIVE_WPS}
        exclusive = {category: counts[(flavor, category)] for category in EXCLUSIVE_CATEGORIES}
        identity_scale = np.maximum.reduce([
            np.ones_like(denominator), np.abs(denominator),
            *(np.abs(values) for values in inclusive.values()),
            *(np.abs(values) for values in exclusive.values()),
        ])
        identity_tolerance = 1e-10 * identity_scale
        identity_invalid = (
            (np.abs(inclusive["L"] - (exclusive["LnotM"] + inclusive["M"])) > identity_tolerance) |
            (np.abs(inclusive["M"] - (exclusive["MnotT"] + inclusive["T"])) > identity_tolerance) |
            (np.abs(inclusive["T"] - (exclusive["TnotXT"] + inclusive["XT"])) > identity_tolerance) |
            (np.abs(inclusive["XT"] - (exclusive["XTnotXXT"] + inclusive["XXT"])) > identity_tolerance) |
            (np.abs(denominator - sum(exclusive.values())) > identity_tolerance)
        )
        tolerance = 1e-10 * identity_scale
        nonempty = np.abs(denominator) > tolerance
        ordered_invalid = (
            (inclusive["XXT"] < -tolerance) |
            (inclusive["XT"] < inclusive["XXT"] - tolerance) |
            (inclusive["T"] < inclusive["XT"] - tolerance) |
            (inclusive["M"] < inclusive["T"] - tolerance) |
            (inclusive["L"] < inclusive["M"] - tolerance) |
            (inclusive["L"] > denominator + tolerance)
        )
        exclusive_invalid = np.logical_or.reduce(
            [values < -tolerance for values in exclusive.values()]
        )
        cancellation_invalid = np.zeros_like(denominator, dtype=bool)
        if variances is not None:
            cancellation_scale = np.maximum(1., identity_scale) ** 2
            for category in EXCLUSIVE_CATEGORIES:
                cancellation_invalid |= (
                    (np.abs(exclusive[category]) <= tolerance) &
                    (variances[(flavor, category)] > 1.e-20 * cancellation_scale)
                )
        invalid = identity_invalid | (
            (nonempty & ((denominator <= 0) | ordered_invalid | exclusive_invalid)) |
            (~nonempty & np.logical_or.reduce([np.abs(values) > tolerance
                                                for values in exclusive.values()])) |
            cancellation_invalid
        )
        masks[flavor] = invalid
    return masks


def validate_counts(counts, variances=None):
    """Validate signed weighted yields before constructing an efficiency map."""
    for flavor, invalid in invalid_count_bins(counts, variances).items():
        if np.any(invalid):
            bad_bins = np.argwhere(invalid).tolist()
            raise ValueError(
                f"Signed-weight b-tag yields are unphysical for flavor {flavor} in bins {bad_bins}; "
                "merge those bins or provide more MC before conversion"
            )


def region_quality(region, flavor, neta):
    """Shared complete quality criterion for adaptive and final regions."""
    c, v, d = region["counts"], region["variances"], region["counts"]["den"]
    scale = max(1., *(float(np.max(np.abs(c[state]))) for state in HISTOGRAM_STATES))
    bad, score = False, 0.; affected = np.zeros(len(INCLUSIVE_WPS), dtype=bool)
    bounds = {"N": (0,), "LnotM": (0, 1), "MnotT": (1, 2), "TnotXT": (2, 3), "XTnotXXT": (3, 4), "XXT": (4,)}
    def violation(amount, indices=()):
        nonlocal bad, score
        if amount > 0.: bad, score = True, score + amount; affected[list(indices)] = True
    for eta in range(neta):
        den, var_den = float(d[eta]), float(v["den"][eta])
        if not np.isfinite(den) or not np.isfinite(var_den) or den <= 1.e-10 * scale or var_den < 0:
            violation(10., range(5)); continue
        significance = den / np.sqrt(var_den) if var_den > 0 else np.inf
        neff = den * den / var_den if var_den > 0 else np.inf
        violation(MIN_DENOMINATOR_SIGNIFICANCE / max(significance, 1.e-12) - 1., range(5))
        violation(MIN_EFFECTIVE_DENOMINATOR[flavor] / max(neff, 1.e-12) - 1., range(5))
        inclusive = {wp: float(c[wp][eta]) for wp in INCLUSIVE_WPS}
        exclusive = {cat: float(c[cat][eta]) for cat in EXCLUSIVE_CATEGORIES}
        violation(max(0., -exclusive["N"]) / max(scale, 1.), bounds["N"])
        for left, right in zip(INCLUSIVE_WPS[:-1], INCLUSIVE_WPS[1:]):
            violation(max(0., inclusive[right] - inclusive[left]) / max(scale, 1.),
                      (INCLUSIVE_WPS.index(left), INCLUSIVE_WPS.index(right)))
        violation(max(0., inclusive["L"] - den) / max(scale, 1.), bounds["N"])
        for parent, category, child in (("L","LnotM","M"),("M","MnotT","T"),("T","TnotXT","XT"),("XT","XTnotXXT","XXT")):
            violation(abs(inclusive[parent] - exclusive[category] - inclusive[child]) / max(scale, 1.) - 1.e-10, bounds[category])
        violation(abs(den - sum(exclusive.values())) / max(scale, 1.) - 1.e-10, range(5))
        for wp in INCLUSIVE_WPS:
            unc = float(mcstat_efficiency_uncertainty(np.asarray([inclusive[wp]]), np.asarray([den]), np.asarray([float(v[wp][eta])]), np.asarray([var_den]))[0])
            violation(10. if not np.isfinite(unc) else unc / MAX_EFFICIENCY_UNCERTAINTY[wp] - 1., (INCLUSIVE_WPS.index(wp),))
        for category in EXCLUSIVE_CATEGORIES:
            value, variance = exclusive[category], float(v[category][eta])
            violation(10., bounds[category]) if value < 0. else None
            if value > 1.e-10 * scale and variance > 0:
                violation(MIN_EFFECTIVE_CATEGORY[flavor] / max(value * value / variance, 1.e-12) - 1., bounds[category])
                unc = float(mcstat_efficiency_uncertainty(np.asarray([value]), np.asarray([den]), np.asarray([variance]), np.asarray([var_den]))[0])
                violation(10. if not np.isfinite(unc) else unc / MAX_CATEGORY_UNCERTAINTY - 1., bounds[category])
    if bad and not np.any(affected): raise AssertionError(f"Quality failure without affected WP for {flavor}")
    return bad, score, affected


def efficiency_maps(counts, variances):
    eff = {(flavor, wp): efficiency(counts[flavor, wp], counts[flavor, "den"])
           for flavor in FLAVORS for wp in INCLUSIVE_WPS}
    unc = {(flavor, wp): mcstat_efficiency_uncertainty(
        counts[flavor, wp], counts[flavor, "den"], variances[flavor, wp], variances[flavor, "den"])
           for flavor in FLAVORS for wp in INCLUSIVE_WPS}
    return eff, unc


def terminal_sequence_valid(eff, unc, counts, flavor, pt, eta):
    denominator = float(counts[flavor, "den"][pt, eta])
    if not np.isfinite(denominator) or denominator <= 0.: return False
    values = [float(eff[flavor, wp][pt, eta]) for wp in INCLUSIVE_WPS]
    errors = [float(unc[flavor, wp][pt, eta]) for wp in INCLUSIVE_WPS]
    if any(not np.isfinite(x) or x < 0. or x > 1. for x in values):
        return False
    if any(not np.isfinite(x) or x < 0. for x in errors):
        return False
    if any(values[i] > values[i - 1] + 1.e-12 for i in range(1, len(values))):
        return False
    return True


def terminal_component_valid(eff, unc, counts, flavor, wp, pt, eta):
    den = float(counts[flavor, "den"][pt, eta]); num = float(counts[flavor, wp][pt, eta])
    value, error = float(eff[flavor, wp][pt, eta]), float(unc[flavor, wp][pt, eta])
    return (np.isfinite(den) and den > 0. and np.isfinite(num) and
            np.isfinite(value) and 0. <= value <= 1. and np.isfinite(error) and error >= 0.)


def denominator_quality_failure(counts, variances, flavor, pt, eta):
    den = float(counts[flavor, "den"][pt, eta])
    var = float(variances[flavor, "den"][pt, eta])
    scale = max(1., *(abs(float(counts[flavor, state][pt, eta]))
                      for state in HISTOGRAM_STATES))
    if not np.isfinite(den) or not np.isfinite(var) or den <= 1.e-10 * scale or var < 0.:
        return True
    significance = den / np.sqrt(var) if var > 0. else np.inf
    neff = den * den / var if var > 0. else np.inf
    return (significance < MIN_DENOMINATOR_SIGNIFICANCE or
            neff < MIN_EFFECTIVE_DENOMINATOR[flavor])


def fallback_efficiencies(target, consensus_candidates):
    target_eff, target_unc = efficiency_maps(*target)
    affected_masks = {flavor: {wp: np.zeros_like(target_eff[flavor, wp], dtype=bool)
                               for wp in INCLUSIVE_WPS} for flavor in FLAVORS}
    for flavor in FLAVORS:
        # Regions that still fail the complete post-merge quality criterion.
        shape = target_eff[flavor, INCLUSIVE_WPS[0]].shape
        for pt in range(shape[0]):
            for eta in range(shape[1]):
                region = {"counts": {state: target[0][flavor, state][pt:pt+1, eta:eta+1]
                                      for state in HISTOGRAM_STATES},
                          "variances": {state: target[1][flavor, state][pt:pt+1, eta:eta+1]
                                        for state in HISTOGRAM_STATES}}
                bad_region, _, wp_mask = region_quality(region, flavor, 1)
                if bad_region:
                    for i, wp in enumerate(INCLUSIVE_WPS): affected_masks[flavor][wp][pt, eta] = wp_mask[i]
    replacements = []
    final_eff, final_unc = dict(target_eff), dict(target_unc)
    for flavor in FLAVORS:
        bad_bins = np.logical_or.reduce(list(affected_masks[flavor].values()))
        for pt, eta in np.argwhere(bad_bins):
            initial = [i for i, wp in enumerate(INCLUSIVE_WPS) if affected_masks[flavor][wp][pt, eta] or
                       not terminal_component_valid(target_eff, target_unc, target[0], flavor, wp, pt, eta)]
            if not initial: initial = list(range(len(INCLUSIVE_WPS)))
            initial_span = (min(initial), max(initial))
            lo, hi = min(initial), max(initial)
            source = None
            for name, candidate, cc, cv, strict in consensus_candidates:
                ce, cu = candidate
                candidate_lo, candidate_hi = initial_span
                def candidate_quality(lo_i, hi_i):
                    required = set(range(lo_i, hi_i + 1))
                    if lo_i > 0: required.add(lo_i - 1)
                    if hi_i + 1 < len(INCLUSIVE_WPS): required.add(hi_i + 1)
                    if not strict:
                        component_ok = all(terminal_component_valid(ce, cu, cc, flavor,
                                                                     INCLUSIVE_WPS[i], pt, eta)
                                            for i in required)
                        if not component_ok: return False
                        mixed = [float(final_eff[flavor, wp][pt, eta])
                                 for wp in INCLUSIVE_WPS]
                        for i in range(lo_i, hi_i + 1):
                            mixed[i] = float(ce[flavor, INCLUSIVE_WPS[i]][pt, eta])
                        if any(not np.isfinite(mixed[i]) or not 0. <= mixed[i] <= 1.
                               for i in required):
                            return False
                        return True
                    region = {"counts": {state: cc[flavor, state][pt:pt+1, eta:eta+1]
                                          for state in HISTOGRAM_STATES},
                              "variances": {state: cv[flavor, state][pt:pt+1, eta:eta+1]
                                            for state in HISTOGRAM_STATES}}
                    _, _, candidate_mask = region_quality(region, flavor, 1)
                    if any(candidate_mask[i] for i in required): return False
                    if any(not terminal_component_valid(ce, cu, cc, flavor,
                                                         INCLUSIVE_WPS[i], pt, eta)
                           for i in required):
                        return False
                    return True
                if not candidate_quality(candidate_lo, candidate_hi): continue
                while True:
                    values = [final_eff[flavor, wp][pt, eta] for wp in INCLUSIVE_WPS]
                    for i in range(candidate_lo, candidate_hi + 1):
                        values[i] = ce[flavor, INCLUSIVE_WPS[i]][pt, eta]
                    violations = [i for i in range(1, len(values))
                                  if values[i] > values[i-1] + 1.e-12]
                    if not violations: break
                    expanded = False
                    if candidate_lo > 0 and any(i == candidate_lo for i in violations):
                        candidate_lo -= 1; expanded = True
                    if candidate_hi + 1 < len(INCLUSIVE_WPS) and any(i == candidate_hi + 1 for i in violations):
                        candidate_hi += 1; expanded = True
                    if not expanded or not candidate_quality(candidate_lo, candidate_hi):
                        break
                if all(values[i] <= values[i-1] for i in range(1, len(values))):
                    lo, hi = candidate_lo, candidate_hi
                    for i in range(lo, hi + 1):
                        wp = INCLUSIVE_WPS[i]; final_eff[flavor, wp][pt, eta] = ce[flavor, wp][pt, eta]; final_unc[flavor, wp][pt, eta] = cu[flavor, wp][pt, eta]
                    source = name; break
            if source is None: raise ValueError(f"No valid efficiency fallback for {flavor} at bin {(int(pt), int(eta))}")
            reason = ("denominator-level failure" if denominator_quality_failure(
                          target[0], target[1], flavor, pt, eta)
                      else ("unavoidable nesting expansion" if (lo, hi) != initial_span or
                            initial_span == (0, len(INCLUSIVE_WPS) - 1)
                            else "component quality failure"))
            replacements.append((flavor, INCLUSIVE_WPS[lo], INCLUSIVE_WPS[hi], [int(pt), int(eta)], source, reason))
    if any(not np.all(np.isfinite(final_eff[key])) or not np.all(np.isfinite(final_unc[key]))
           or np.any(final_eff[key] < 0.) or np.any(final_eff[key] > 1.)
           for key in final_eff):
        raise ValueError("Final efficiency map contains invalid values")
    for flavor in FLAVORS:
        if any(np.any(final_eff[flavor, right] > final_eff[flavor, left] + 1.e-12)
               for left, right in zip(INCLUSIVE_WPS[:-1], INCLUSIVE_WPS[1:])):
            raise ValueError(f"Final efficiency map is not nested for {flavor}")
    return final_eff, final_unc, replacements


def merge_invalid_bins_downward(counts, variances):
    """Adaptively merge low-quality pT bins into adjacent regions.

    The correction schema intentionally remains rectangular and keeps the
    producer's pT edges.  Every member of a merged region receives the same
    summed yields, which is equivalent to publishing a coarser bin while
    avoiding a schema migration.  All histogram states and Sumw2 values are
    merged together, so the nesting identities remain exact.
    """
    merged = {key: values.copy() for key, values in counts.items()}
    merged_variances = {key: values.copy() for key, values in variances.items()}
    merged_bins = {}
    for flavor in FLAVORS:
        npt, neta = merged[flavor, "den"].shape
        # Each region carries one vector per state and eta bin.  Regions are
        # merged as units, then expanded back onto the original grid.
        regions = []
        for index in range(npt):
            regions.append({
                "indices": [index],
                "counts": {state: merged[flavor, state][index, :].copy()
                           for state in HISTOGRAM_STATES},
                "variances": {state: merged_variances[flavor, state][index, :].copy()
                              for state in HISTOGRAM_STATES},
            })

        while True:
            bad_regions = [i for i, region in enumerate(regions) if region_quality(region, flavor, neta)[0]]
            if not bad_regions or len(regions) == 1:
                break
            # Merge the most problematic region first.  Evaluate both
            # neighbours and choose the candidate with the best resulting
            # quality; ties deterministically prefer lower pT.
            index = max(bad_regions, key=lambda i: region_quality(regions[i], flavor, neta)[1])
            candidates = []
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(regions):
                    combined = {
                        "indices": regions[index]["indices"] + regions[neighbour]["indices"],
                        "counts": {state: regions[index]["counts"][state] + regions[neighbour]["counts"][state]
                                   for state in HISTOGRAM_STATES},
                        "variances": {state: regions[index]["variances"][state] + regions[neighbour]["variances"][state]
                                      for state in HISTOGRAM_STATES},
                    }
                    candidates.append((region_quality(combined, flavor, neta), neighbour, combined))
            if not candidates:
                break
            _, neighbour, combined = min(candidates, key=lambda item: (item[0][0], item[0][1], item[1]))
            first, second = sorted((index, neighbour))
            regions[first] = combined
            del regions[second]
            merged_bins.setdefault(flavor, []).append(sorted(combined["indices"]))

        # Expand the adaptive regions back to the fixed correction grid.
        for region in regions:
            for index in region["indices"]:
                for state in HISTOGRAM_STATES:
                    merged[flavor, state][index, :] = region["counts"][state]
                    merged_variances[flavor, state][index, :] = region["variances"][state]
    return merged, merged_variances, merged_bins


def efficiency(values, denominator):
    output = np.zeros_like(values, dtype=float)
    np.divide(values, denominator, out=output, where=denominator > 0)
    return output


def mcstat_efficiency_uncertainty(numerator, denominator, numerator_variance, denominator_variance):
    """Weighted-binomial/delta-method uncertainty for a category fraction X/D."""
    output = np.full_like(denominator, np.nan, dtype=float)
    valid = denominator > 0
    empty = ((denominator == 0) & (numerator == 0) &
             (numerator_variance == 0) & (denominator_variance == 0))
    output[empty] = 0.
    epsilon = np.divide(numerator, denominator, out=np.zeros_like(denominator, dtype=float), where=valid)
    raw_variance = np.divide((1. - 2. * epsilon) * numerator_variance + epsilon ** 2 * denominator_variance,
                             denominator ** 2, out=np.full_like(denominator, np.nan, dtype=float), where=valid)
    scale = np.divide(np.abs((1. - 2. * epsilon) * numerator_variance) +
                      np.abs(epsilon ** 2 * denominator_variance), denominator ** 2,
                      out=np.zeros_like(denominator, dtype=float), where=valid)
    tiny_negative = (raw_variance < 0.) & (raw_variance >= -1.e-12 * np.maximum(1., scale))
    raw_variance[tiny_negative] = 0.
    valid &= np.isfinite(raw_variance) & (raw_variance >= 0.)
    output[valid] = np.sqrt(raw_variance[valid])
    return output


def compute_mcstat_uncertainties(counts, variances):
    output = {(flavor, wp): mcstat_efficiency_uncertainty(
        counts[(flavor, wp)], counts[(flavor, "den")], variances[(flavor, wp)], variances[(flavor, "den")])
              for flavor in FLAVORS for wp in INCLUSIVE_WPS}
    for (flavor, wp), values in output.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid weighted-binomial MC-statistical variance for {flavor}/{wp}")
    return output


def multibinning(values, pt_edges, eta_edges):
    return {
        "nodetype": "multibinning",
        "inputs": ["pt", "eta"],
        "edges": [pt_edges.tolist(), eta_edges.tolist()],
        "content": values.reshape(-1).tolist(),
        "flow": "clamp",
    }


def sample_category(sample, values, edges):
    pt_edges, eta_edges = edges
    flavor_entries = []
    for flavor in FLAVORS:
        wp_entries = []
        for wp in INCLUSIVE_WPS:
            wp_entries.append({
                "key": wp,
                "value": multibinning(values[(flavor, wp)], pt_edges, eta_edges),
            })
        flavor_entries.append({
            "key": {"b": "B", "c": "C", "light": "L"}[flavor],
            "value": {"nodetype": "category", "input": "WP", "content": wp_entries},
        })
    return {
        "key": sample,
        "value": {"nodetype": "category", "input": "flavor", "content": flavor_entries},
    }


def make_correction(name, sample_entry, description, output_name, output_description):
    return {
        "name": name,
        "description": description,
        "version": 1,
        "inputs": [
            {"name": "sample", "type": "string", "description": "sample category key (final YAML family in production payload)"},
            {"name": "flavor", "type": "string", "description": "B/C/L"},
            {"name": "WP", "type": "string", "description": "L/M/T/XT/XXT"},
            {"name": "pt", "type": "real", "description": "selected AK4 jet pT"},
            {"name": "eta", "type": "real", "description": "selected AK4 jet eta"},
        ],
        "output": {"name": output_name, "type": "real", "description": output_description},
        "data": {"nodetype": "category", "input": "sample", "content": [sample_entry]},
    }


def update_output(path, correction_specs, replace_entries=False, replace_correction_prefix=None):
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = {"schema_version": 2, "description": "VBS VVH b-tag efficiencies", "corrections": [], "compound_corrections": []}

    expected_inputs = ["sample", "flavor", "WP", "pt", "eta"]
    if replace_correction_prefix is not None:
        expected_names = {spec[0] for spec in correction_specs}
        payload["corrections"] = [
            item for item in payload["corrections"]
            if not (item["name"].startswith(replace_correction_prefix) and item["name"] not in expected_names)
        ]
    replaced = set()
    for correction_name, sample_entry, description, output_name, output_description in correction_specs:
        correction = next((item for item in payload["corrections"] if item["name"] == correction_name), None)
        if correction is None:
            payload["corrections"].append(
                make_correction(correction_name, sample_entry, description, output_name, output_description))
            correction = payload["corrections"][-1]
        if ([item["name"] for item in correction["inputs"]] != expected_inputs or
                correction["output"]["name"] != output_name):
            raise ValueError(f"Existing {correction_name} has an incompatible schema; write a new output file")
        correction["description"] = description
        correction["output"]["description"] = output_description
        entries = correction["data"]["content"]
        if replace_entries and correction_name not in replaced:
            correction["data"]["content"] = []
            replaced.add(correction_name)
        if replace_entries:
            correction["data"]["content"].append(sample_entry)
        else:
            correction["data"]["content"] = [entry for entry in entries if entry["key"] != sample_entry["key"]]
            correction["data"]["content"].append(sample_entry)

    # Validate with correctionlib before touching the output file.
    cs.CorrectionSet.model_validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def grouped_specs(prefix, grouped_counts, grouped_variances, grouped_edges, members,
                  year_inclusive_candidate=None):
    inclusive_counts, inclusive_variances = {}, {}
    for group in members:
        add_counts(inclusive_counts, grouped_counts[group])
        add_counts(inclusive_variances, grouped_variances[group])
    inclusive_counts, inclusive_variances, inclusive_merged_bins = merge_invalid_bins_downward(
        inclusive_counts, inclusive_variances)
    if inclusive_merged_bins:
        print(f"inclusive: adaptive pT merges: {inclusive_merged_bins}")
    specs, fallbacks, adaptive_merges = [], {}, {"inclusive": inclusive_merged_bins}
    for group in sorted(members):
        counts, variances, merged_bins = merge_invalid_bins_downward(
            grouped_counts[group], grouped_variances[group])
        if merged_bins:
            print(f"{group}: adaptive pT merges: {merged_bins}")
        adaptive_merges[group] = merged_bins
        consensus_counts, consensus_variances = {}, {}
        other_groups = [name for name in members if name != group]
        if other_groups:
            for name in other_groups:
                add_counts(consensus_counts, grouped_counts[name])
                add_counts(consensus_variances, grouped_variances[name])
            consensus_counts, consensus_variances, _ = merge_invalid_bins_downward(
                consensus_counts, consensus_variances)
        candidates = []
        if other_groups:
            candidates.append(("channel-family consensus", efficiency_maps(consensus_counts, consensus_variances),
                              consensus_counts, consensus_variances, True))
        candidates.append(("all-MC inclusive consensus", efficiency_maps(inclusive_counts, inclusive_variances),
                          inclusive_counts, inclusive_variances, False))
        if year_inclusive_candidate is not None:
            candidates.append(("same-year all-channel inclusive", *year_inclusive_candidate, False))
        target_maps = (counts, variances)
        efficiency_values, mcstat_uncertainties, fallback_bins = fallback_efficiencies(target_maps, candidates)
        if fallback_bins:
            fallbacks[group] = fallback_bins
            for flavor, lo, hi, bins, source, reason in fallback_bins:
                print(f"fallback {prefix} {group} {flavor} {lo}-{hi} {bins} {source} {reason}")
        specs.extend([
            (prefix, sample_category(group, efficiency_values, grouped_edges[group]),
             "Selected-AK4 UParTAK4 b-tag efficiency", "efficiency", "MC tagging efficiency"),
            (f"{prefix}_mcstat_unc", sample_category(group, mcstat_uncertainties, grouped_edges[group]),
             "Weighted-binomial MC-statistical uncertainty of selected-AK4 UParTAK4 b-tag efficiency", "uncertainty",
             "Weighted-binomial MC tagging-efficiency uncertainty"),
        ])
    return specs, fallbacks, adaptive_merges


def discover_final_inputs(input_root, year, config=None):
    """Return every required final source channel and its colocated manifest."""
    config = load_config(year) if config is None else config
    if not input_root.is_dir():
        raise ValueError(f"--input-root is not a directory: {input_root}")
    retained = retained_source_channels(config)
    excluded = excluded_source_channels(config)
    children = {path.name for path in input_root.iterdir() if path.is_dir()}
    missing = set(retained) - children
    unexpected = children - set(retained) - set(excluded)
    if missing or unexpected:
        raise ValueError(f"Final input root channel mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    ignored = sorted(set(excluded) & children)
    if ignored:
        print(f"Ignoring excluded b-tag source channels: {ignored}")
    inputs = {}
    manifests = {}
    for channel in retained:
        directory = input_root / channel
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"Final input channel {channel} is missing required manifest {manifest}")
        inputs[channel] = directory
        manifests[channel] = manifest
    return inputs, manifests, ignored


def discover_source_histograms(input_root, year, config=None):
    """Load complete retained source channels before final YAML merging."""
    config = load_config(year) if config is None else config
    channel_inputs, channel_manifests, ignored = discover_final_inputs(input_root, year, config)
    source_counts, source_variances, source_edges, members, completeness = {}, {}, {}, {}, {}
    for channel, input_dir in channel_inputs.items():
        counts, variances, edges, preliminary_members, checked = discover_family_histograms(
            input_dir, year, channel, channel_manifests[channel], config)
        completeness[channel] = checked
        for family, samples in preliminary_members.items():
            key = (channel, family)
            source_counts[key] = counts[family]
            source_variances[key] = variances[family]
            source_edges[key] = edges[family]
            members[key] = samples
    return source_counts, source_variances, source_edges, members, completeness, ignored


def discover_final_histograms(input_root, year, config=None):
    """Merge all YAML-retained raw outputs into final channel/sample groups."""
    config = load_config(year) if config is None else config
    channel_inputs, channel_manifests, ignored = discover_final_inputs(input_root, year, config)
    records_by_target, completeness, members = {}, {}, {}
    for channel, input_dir in channel_inputs.items():
        samples, checked = discover_sample_histograms(
            input_dir, year, channel, channel_manifests[channel], config)
        completeness[channel] = checked
        by_family = {}
        for sample, counts, variances, edges in samples:
            by_family.setdefault(sample_family(sample, config), []).append(
                (sample, counts, variances, edges))
        for preliminary_family, family_records in by_family.items():
            target = (final_channel(channel, config),
                      final_group("samples", preliminary_family, config))
            for sample, counts, variances, edges in family_records:
                key = f"{channel}:{sample}"
                records_by_target.setdefault(target, []).append((key, counts, variances, edges))
                members.setdefault(target[0], {}).setdefault(target[1], []).append(key)
    grouped_counts, grouped_variances, grouped_edges = {}, {}, {}
    for target, records in records_by_target.items():
        counts, variances, edges, _ = aggregate_sample_records(records, f"{target[0]}/{target[1]}")
        grouped_counts[target], grouped_variances[target], grouped_edges[target] = counts, variances, edges
    return grouped_counts, grouped_variances, grouped_edges, members, completeness, ignored


def main():
    args = parse_args()
    if args.output is None:
        args.output = default_output_path(args.year, preliminary=not args.final, channel=args.channel)
    expected_final_name = f"btag_eff_{args.year}.json"
    if args.final and args.output.name != expected_final_name:
        raise ValueError(f"Final output must be named {expected_final_name}; year-scoped payloads are required")
    if args.final:
        if not args.input_root:
            raise ValueError("--final requires --input-root")
        if args.input or args.input_dir or args.sample or args.channel or args.job_manifest:
            raise ValueError("--final accepts only --input-root, --year, --output, and optional --manifest")
        config = load_config(year=args.year)
        counts, variances, edges, members, completeness, ignored = discover_final_histograms(
            args.input_root, args.year, config)
        year_counts, year_variances = {}, {}
        for value in counts.values():
            add_counts(year_counts, value)
        for value in variances.values():
            add_counts(year_variances, value)
        year_counts, year_variances, _ = merge_invalid_bins_downward(year_counts, year_variances)
        year_inclusive_candidate = (efficiency_maps(year_counts, year_variances),
                                    year_counts, year_variances)
        specs, fallbacks, adaptive_merges = [], {}, {}
        for channel, final_members in sorted(members.items()):
            keys = {(channel, sample) for sample in final_members}
            channel_counts = {sample: counts[channel, sample] for _, sample in keys}
            channel_variances = {sample: variances[channel, sample] for _, sample in keys}
            channel_edges = {sample: edges[channel, sample] for _, sample in keys}
            channel_specs, channel_fallbacks, channel_merges = grouped_specs(
                f"btag_{args.year}_{channel}", channel_counts, channel_variances, channel_edges,
                final_members, year_inclusive_candidate)
            specs.extend(channel_specs)
            fallbacks[channel] = channel_fallbacks
            adaptive_merges[channel] = channel_merges
        update_output(args.output, specs, replace_entries=True,
                      replace_correction_prefix=f"btag_{args.year}_")
        manifest = args.manifest or args.output.with_name(
            f"btag_eff_{args.year}_final_families.json")
        manifest.write_text(json.dumps({
            "mode": "final", "year": args.year, "input_root": str(args.input_root),
            "retained_source_channels": list(retained_source_channels(config)),
            "ignored_source_channels": ignored,
            "final_members": members, "inclusive_fallback_bins": fallbacks,
            "adaptive_pT_merges": adaptive_merges,
            "input_completeness_verified": True,
            "job_counts": completeness,
        }, indent=2) + "\n")
        print(f"Wrote final efficiencies for {len(members)} channel groups and manifest {manifest}")
        return

    if not args.channel:
        raise ValueError("--channel is required unless --final is used")
    if "_prelim" not in args.output.stem:
        raise ValueError("Preliminary conversion output must be named *_prelim.json; reserve btag_eff_<year>.json for --final")
    prefix = f"btag_{args.year}_{args.channel}"
    if args.input_dir:
        grouped_counts, grouped_variances, grouped_edges, members, completeness = discover_family_histograms(
            args.input_dir, args.year, args.channel, args.job_manifest, load_config(year=args.year))
        specs, fallbacks, adaptive_merges = grouped_specs(prefix, grouped_counts, grouped_variances, grouped_edges, members)
        update_output(args.output, specs, replace_entries=True)
        manifest = args.manifest or args.output.with_name(
            f"btag_eff_{args.year}_{args.channel}_families.json")
        manifest.write_text(json.dumps({
            "year": args.year, "channel": args.channel,
            "mode": "preliminary",
            "input_dir": str(args.input_dir), "families": members,
            "inclusive_fallback_bins": fallbacks,
            "adaptive_pT_merges": adaptive_merges,
            "input_completeness_verified": completeness is not None,
            "job_counts": completeness or {},
        }, indent=2) + "\n")
        print(f"Wrote {len(members)} family entries and manifest {manifest}")
        return

    if not args.sample:
        raise ValueError("--sample is required with --input")
    # Exact-sample payloads are useful for local diagnostics only.  Runtime
    # lookup intentionally accepts final family keys only.
    counts, variances, edges = read_merged_histograms(args.input, args.year, args.channel, args.sample)
    validate_counts(counts, variances)
    efficiency_values = {(flavor, wp): efficiency(counts[(flavor, wp)], counts[(flavor, "den")])
                         for flavor in FLAVORS for wp in INCLUSIVE_WPS}
    mcstat_uncertainties = compute_mcstat_uncertainties(counts, variances)
    update_output(args.output, [
        (prefix, sample_category(args.sample, efficiency_values, edges),
         "Selected-AK4 UParTAK4 b-tag efficiency", "efficiency", "MC tagging efficiency"),
        (f"{prefix}_mcstat_unc", sample_category(args.sample, mcstat_uncertainties, edges),
         "Weighted-binomial MC-statistical uncertainty of selected-AK4 UParTAK4 b-tag efficiency", "uncertainty",
         "Weighted-binomial MC tagging-efficiency uncertainty"),
    ])


if __name__ == "__main__":
    main()
