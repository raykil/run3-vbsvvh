#!/usr/bin/env python3
"""Compare year-aggregated b-tag efficiencies from complete Slurm outputs."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import uproot

from btag_eff_families import load_config, retained_source_channels


def load_module(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-base", type=Path, required=True,
                        help="Directory containing run*_btag_eff_* output roots")
    parser.add_argument("--plot-dir", type=Path,
                        help="Defaults to btagging/diagnostics/diagnostic_all_years")
    parser.add_argument("--years", nargs="+", help="Optional canonical years to include")
    return parser.parse_args()


def output_year(root):
    """Read the canonical framework year from one b-tag ROOT output."""
    roots = sorted(root.glob("*/*/output_*.root"))
    if not roots:
        raise ValueError("no b-tag ROOT outputs")
    with uproot.open(roots[0]) as source:
        return source["btag_eff_year"].member("fTitle")


def discover_year_roots(input_base, requested_years=None):
    """Discover only complete-layout production roots, keyed by ROOT metadata year."""
    # A production root is identified by its content, not a user-chosen tag.
    # Cover ROOT/YEAR, legacy run*_btag_eff_TAG, and TAG/YEAR without walking
    # an unbounded tree.
    candidates = [input_base]
    children = sorted(path for path in input_base.iterdir() if path.is_dir())
    candidates += children
    candidates += [path for child in children for path in sorted(child.iterdir()) if path.is_dir()]
    found, skipped = {}, []
    for root in candidates:
        try:
            year = output_year(root)
            config = load_config(year=year)
            missing = [channel for channel in retained_source_channels(config)
                       if not (root / channel / "manifest.json").is_file()]
            if missing:
                raise ValueError(f"missing complete-layout channels {missing}")
        except Exception as error:
            skipped.append((root, str(error)))
            continue
        if requested_years and year not in requested_years:
            continue
        if year in found:
            raise ValueError(f"Multiple b-tag output roots were found for {year}: {found[year]} and {root}")
        found[year] = root
    if not found:
        raise ValueError(f"No usable b-tag output roots found under {input_base}")
    return found, skipped


def aggregate_year(conv, root, year):
    """Sum all retained source channels and preliminary sample families for one year."""
    counts, variances, _, _, completeness, ignored = conv.discover_source_histograms(
        root, year, load_config(year=year))
    aggregate_counts, aggregate_variances = {}, {}
    for key, values in counts.items():
        conv.add_counts(aggregate_counts, values)
        conv.add_counts(aggregate_variances, variances[key])
    return aggregate_counts, aggregate_variances, completeness, ignored


def main():
    args = parse_args()
    conv = load_module("bEff-convert-to-correction.py", "btag_converter")
    plots = load_module("plot-btag-eff-families.py", "btag_plots")
    year_roots, skipped = discover_year_roots(args.input_base, args.years)
    for root, reason in skipped:
        print(f"Skipping incompatible b-tag output root {root}: {reason}")

    groups, results, completeness, ignored = {}, {}, {}, {}
    for year, root in year_roots.items():
        counts, variances, checked, excluded = aggregate_year(conv, root, year)
        groups[year] = {"counts": counts, "variances": variances}
        results[year] = plots.efficiencies(conv, counts, variances)
        completeness[year] = checked
        ignored[year] = excluded

    if args.plot_dir is None:
        args.plot_dir = (Path(__file__).parents[2] / "preselection" / "corrections" /
                         "scalefactors" / "btagging" / "diagnostics" /
                         "diagnostic_all_years")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.input_completeness_verified = all(bool(value) for value in completeness.values())
    args.ignored_source_channels = ignored
    args.channel = "all channels and samples"
    plt.style.use(hep.style.CMS)
    plots.money_plot(groups, results, args, None, None, "year")
    plots.summary_json(groups, results, args)


if __name__ == "__main__":
    main()
