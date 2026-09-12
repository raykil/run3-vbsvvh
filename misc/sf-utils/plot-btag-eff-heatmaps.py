#!/usr/bin/env python3
"""Draw per-sample 2D b-tag efficiency maps from raw --btag-eff ROOT output."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from btag_eff_families import load_config, retained_source_channels, sample_family


FLAVORS = ("b", "c", "light")
PLOT_STATES = {
    "FailLoose": "N", "Loose": "L", "LooseNotMedium": "LnotM", "Medium": "M",
    "MediumNotTight": "MnotT", "Tight": "T", "TightNotExtraTight": "TnotXT",
    "ExtraTight": "XT", "ExtraTightNotExtraExtraTight": "XTnotXXT", "ExtraExtraTight": "XXT",
}
COLORMAPS = {"b": "Blues", "c": "Oranges", "light": "Greens"}
LUMI_FB = {
    "2016preVFP": 19.5, "2016postVFP": 16.8, "2017": 41.5, "2018": 59.8,
    "2022Re-recoBCD": 8.1, "2022Re-recoE+PromptFG": 27.0,
    "2023PromptC": 17.8, "2023PromptD": 9.5, "2024Prompt": 109.95,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Directory containing CHANNEL/SAMPLE/output_N.root")
    parser.add_argument("--year", required=True, help="B-tag efficiency metadata year")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to diagnostics/efficiency_heatmaps_<year>")
    parser.add_argument("--channel", action="append", help="Optional source channel filter (repeatable)")
    parser.add_argument("--family", action="append",
                        help="Optional preliminary-family filter (repeatable)")
    return parser.parse_args()


def output_default(year):
    return (Path(__file__).parents[2] / "preselection" / "corrections" / "scalefactors" /
            "btagging" / "diagnostics" / f"efficiency_heatmaps_{year}")


def root_title(root_file, name, path):
    if name not in root_file:
        raise ValueError(f"{path} has no {name} metadata")
    return root_file[name].member("fTitle")


def read_sample_histograms(paths, year, channel, sample):
    """Merge weighted raw 2D yields and validate their job metadata."""
    totals, variances, edges = {}, {}, None
    for path in paths:
        with uproot.open(path) as root_file:
            actual = {key: root_title(root_file, f"btag_eff_{key}", path)
                      for key in ("year", "channel", "sample")}
            expected = {"year": year, "channel": channel, "sample": sample}
            if actual != expected:
                raise ValueError(f"Metadata mismatch in {path}: expected {expected}, found {actual}")
            if "btag_eff_schema_version" not in root_file or root_file["btag_eff_schema_version"].member("fTitle") != "2":
                raise ValueError(f"{path} is not a schema-v2 five-working-point b-tag efficiency file")
            for flavor in FLAVORS:
                for kind in ("den", *PLOT_STATES.values()):
                    histogram = root_file[f"btag_{flavor}_{kind}"]
                    values, pt_edges, eta_edges = histogram.to_numpy(flow=True)
                    histogram_variances = histogram.variances(flow=True)
                    if histogram_variances is None:
                        raise ValueError(f"{path}: btag_{flavor}_{kind} has no Sumw2 information")
                    key = flavor, kind
                    if edges is None:
                        edges = pt_edges, eta_edges
                    elif not (np.array_equal(pt_edges, edges[0]) and np.array_equal(eta_edges, edges[1])):
                        raise ValueError(f"Inconsistent histogram binning in {path}")
                    totals[key] = totals.get(key, np.zeros_like(values, dtype=float)) + values
                    variances[key] = variances.get(key, np.zeros_like(values, dtype=float)) + histogram_variances
    return totals, variances, edges


def physical_bins(values):
    """Fold pT overflow to its endpoint, as the converter does, and remove eta flow."""
    result = values[1:-1, 1:-1].astype(float, copy=True)
    result[0, :] += values[0, 1:-1]
    result[-1, :] += values[-1, 1:-1]
    return result


def merged_eta_bins(values):
    """Collapse all central eta bins into one bin, preserving weighted sums."""
    return np.sum(physical_bins(values), axis=1, keepdims=True)


def max_abs_eta(year):
    return 2.4 if year in ("2016preVFP", "2016postVFP") else 2.5


def cms_label(fig, year):
    """Figure-level CMS label; avoids crowding the first of three panels."""
    lumi = LUMI_FB.get(year)
    energy = 13.6 if year == "2024Prompt" else 13
    fig.text(0.065, 0.955, "CMS", fontsize=24, fontweight="bold", va="top")
    fig.text(0.155, 0.955, "Simulation Preliminary", fontsize=20, fontstyle="italic", va="top")
    fig.text(0.985, 0.955, rf"({energy:g} TeV)", ha="right", fontsize=18, va="top")


def truncated_efficiency_label(value):
    """Format to three decimal places without rounding upward."""
    return f"{math.trunc(value * 1000.) / 1000.:.3f}"


def plot_sample(totals, variances, edges, year, destination):
    pt_edges, eta_edges = edges
    pt_edges = pt_edges[1:-1]
    eta_edges = np.asarray([-max_abs_eta(year), max_abs_eta(year)])
    denoms = {flavor: merged_eta_bins(totals[flavor, "den"]) for flavor in FLAVORS}
    denom_variances = {flavor: merged_eta_bins(variances[flavor, "den"]) for flavor in FLAVORS}
    fig, axes = plt.subplots(len(PLOT_STATES), 3, figsize=(16.5, 3.4 * len(PLOT_STATES)), sharex=True, sharey=True,
                             squeeze=False)
    pt_centers = np.sqrt(pt_edges[:-1] * pt_edges[1:])
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    for row, (wp_label, kind) in enumerate(PLOT_STATES.items()):
        for column, flavor in enumerate(FLAVORS):
            axis = axes[row, column]
            numerator = merged_eta_bins(totals[flavor, kind])
            numerator_variances = merged_eta_bins(variances[flavor, kind])
            efficiency = np.divide(numerator, denoms[flavor], out=np.full_like(numerator, np.nan),
                                   where=denoms[flavor] > 0)
            variance = np.divide(
                numerator_variances + efficiency**2 * denom_variances[flavor]
                - 2.0 * efficiency * numerator_variances,
                denoms[flavor]**2,
                out=np.full_like(efficiency, np.nan),
                where=denoms[flavor] > 0,
            )
            uncertainty = np.where(variance >= 0., np.sqrt(variance), np.nan)
            image = axis.pcolormesh(pt_edges, eta_edges, efficiency.T, shading="flat",
                                    cmap=COLORMAPS[flavor], vmin=0., vmax=1.)
            for pt_index, eta_index in np.argwhere(np.isfinite(efficiency)):
                axis.text(pt_centers[pt_index], eta_centers[eta_index],
                          f"{truncated_efficiency_label(efficiency[pt_index, eta_index])}±"
                          f"{truncated_efficiency_label(uncertainty[pt_index, eta_index])}",
                          color="black", fontsize=7, ha="center", va="center", rotation=90)
            axis.set_xscale("log")
            if row == len(PLOT_STATES) - 1:
                axis.set_xlabel(r"Jet $p_{T}$ [GeV]")
            if column == 0:
                axis.set_ylabel(r"Jet $\eta$")
            colorbar = fig.colorbar(image, ax=axis, pad=0.015)
            colorbar.set_label(f"{flavor}-jet efficiency ({wp_label})", fontsize=13)
    cms_label(fig, year)
    fig.tight_layout(rect=(0.015, 0.015, 0.985, 0.91), pad=0.25, w_pad=0.35, h_pad=0.55)
    fig.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def source_channels(input_root, year, requested):
    expected = set(retained_source_channels(year=year))
    available = {path.name: path for path in input_root.iterdir() if path.is_dir()}
    if requested:
        unknown = set(requested) - expected
        if unknown:
            raise ValueError(f"Requested channels are not retained b-tag source channels: {sorted(unknown)}")
        return [(name, available[name]) for name in requested if name in available]
    return [(name, available[name]) for name in sorted(expected & set(available))]


def add_histograms(total, addition):
    if total is None:
        return {key: value.copy() for key, value in addition.items()}
    for key, value in addition.items():
        total[key] += value
    return total


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise ValueError(f"Input root does not exist: {args.input_root}")
    hep.style.use("CMS")
    output_dir = args.output_dir or output_default(args.year)
    channels = source_channels(args.input_root, args.year, args.channel)
    if not channels:
        raise ValueError(f"No retained source channels found below {args.input_root}")
    config = load_config(args.year)
    wanted_families = set(args.family or [])
    written = 0
    for channel, channel_dir in channels:
        channel_output = output_dir / channel
        families = {}
        for sample_dir in sorted(path for path in channel_dir.iterdir() if path.is_dir()):
            roots = sorted(sample_dir.glob("output_*.root"))
            if not roots:
                continue
            family = sample_family(sample_dir.name, config)
            if wanted_families and family not in wanted_families:
                continue
            totals, variances, edges = read_sample_histograms(roots, args.year, channel, sample_dir.name)
            if family in families and not (np.array_equal(edges[0], families[family][2][0]) and
                                           np.array_equal(edges[1], families[family][2][1])):
                raise ValueError(f"Binning mismatch while merging {family} in {channel}")
            family_totals, family_variances, _ = families.get(family, (None, None, edges))
            families[family] = (add_histograms(family_totals, totals),
                                add_histograms(family_variances, variances), edges)
        for family, (totals, variances, edges) in sorted(families.items()):
            channel_output.mkdir(parents=True, exist_ok=True)
            plot_sample(totals, variances, edges, args.year, channel_output / f"{family}.png")
            written += 1
    print(f"Wrote {written} efficiency heatmaps under {output_dir}")


if __name__ == "__main__":
    main()
