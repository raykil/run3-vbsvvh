#!/usr/bin/env python3
"""Plot inter-family b-tag efficiency pulls from raw --btag_eff ROOT outputs."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
import numpy as np

from btag_eff_families import load_config, sample_family


EXCLUSIVE_CATEGORIES = ("N", "LnotM", "MnotT", "TnotXT", "XTnotXXT", "XXT")


LUMI_FB = {
    "2016preVFP": 19.5, "2016postVFP": 16.8, "2017": 41.5, "2018": 59.8,
    "2022Re-recoBCD": 8.1, "2022Re-recoE+PromptFG": 27.0,
    "2023PromptC": 17.8, "2023PromptD": 9.5, "2024Prompt": 109.95,
}


def converter_module():
    path = Path(__file__).with_name("bEff-convert-to-correction.py")
    spec = importlib.util.spec_from_file_location("btag_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--plot-dir", type=Path,
                        help="Persistent diagnostics directory (defaults to diagnostic_<year>)")
    parser.add_argument("--lumi", type=float, help="Integrated luminosity in fb^-1")
    parser.add_argument("--sources", nargs="+", help="Only write detail PDFs for these source families")
    parser.add_argument("--skip-matrices", action="store_true")
    parser.add_argument("--skip-pulls", action="store_true")
    parser.add_argument("--job-manifest", type=Path,
                        help="Optional Slurm manifest.json: reject incomplete batch output")
    return parser.parse_args()


def collect_samples(conv, input_dir, year, channel, job_manifest=None, config=None):
    completeness = (conv.validate_job_manifest(input_dir, job_manifest) if job_manifest else None)
    if completeness is None:
        print("WARNING: b-tag input completeness was not verified (no --job-manifest supplied)")
    samples, family_counts, family_vars, family_edges, members = {}, {}, {}, {}, {}
    for sample_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        roots = sorted(sample_dir.glob("*.root"))
        if not roots:
            continue
        sample = sample_dir.name
        family = sample_family(sample, config)
        counts, variances, edges = conv.read_merged_histograms(roots, year, channel, sample)
        samples[sample] = (family, counts, variances, edges)
        family_counts.setdefault(family, {})
        family_vars.setdefault(family, {})
        conv.add_counts(family_counts[family], counts)
        conv.add_counts(family_vars[family], variances)
        if family in family_edges and not (np.array_equal(edges[0], family_edges[family][0]) and
                                           np.array_equal(edges[1], family_edges[family][1])):
            raise ValueError(f"Binning mismatch in family {family}")
        family_edges[family] = edges
        members.setdefault(family, []).append(sample)
    return samples, family_counts, family_vars, family_edges, members, completeness


def efficiencies(conv, counts, variances):
    values, uncertainties, valid = {}, {}, {}
    pathological = conv.invalid_count_bins(counts, variances)
    for flavor in conv.FLAVORS:
        den = counts[(flavor, "den")]
        for state in (*conv.INCLUSIVE_WPS, *conv.EXCLUSIVE_CATEGORIES):
            num = counts[(flavor, state)]
            ok = ~pathological[flavor] & (den > 0) & (num >= 0) & (num <= den)
            values[flavor, state] = np.divide(num, den, out=np.full_like(den, np.nan), where=ok)
            uncertainties[flavor, state] = conv.mcstat_efficiency_uncertainty(
                num, den, variances[(flavor, state)], variances[(flavor, "den")])
            valid[flavor, state] = ok & np.isfinite(uncertainties[flavor, state])
    return values, uncertainties, valid, pathological


def pulls(reference, other, flavor, wp):
    values_a, errors_a, valid_a, _ = reference
    values_b, errors_b, valid_b, _ = other
    variance = errors_a[flavor, wp] ** 2 + errors_b[flavor, wp] ** 2
    valid = valid_a[flavor, wp] & valid_b[flavor, wp] & (variance > 0)
    result = np.full(values_a[flavor, wp].shape, np.nan)
    result[valid] = (values_a[flavor, wp][valid] - values_b[flavor, wp][valid]) / np.sqrt(variance[valid])
    return result


def cms_label(ax, lumi, energy):
    kwargs = {"data": True, "ax": ax}
    if lumi:
        kwargs["lumi"] = round(lumi, 2 - int(math.floor(math.log10(abs(lumi)))))
        kwargs["lumi_format"] = "{0:g}"
    if energy:
        kwargs["com"] = energy
    hep.cms.label("Sim. Prelim.", **kwargs)


def matrix_plots(groups, results, args, lumi, energy, group_label="family"):
    names = sorted(groups)
    for flavor in ("b", "c", "light"):
        for wp in EXCLUSIVE_CATEGORIES:
            matrix = np.full((len(names), len(names)), np.nan)
            for row, left in enumerate(names):
                for col, right in enumerate(names):
                    if row == col:
                        matrix[row, col] = 0.
                        continue
                    pull = pulls(results[left], results[right], flavor, wp)
                    finite = np.isfinite(pull)
                    if np.count_nonzero(finite) >= 3:
                        matrix[row, col] = np.mean(np.abs(pull[finite]) > 2.)
            fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(names)), max(8, 0.55 * len(names))))
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", interpolation="nearest")
            ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(names)), names, fontsize=8)
            ax.set_xlabel(f"Comparison {group_label}")
            ax.set_ylabel(f"Comparison {group_label}")
            fig.colorbar(image, ax=ax, label=r"Fraction of valid bins with $|$pull$|>2$")
            cms_label(ax, lumi, energy)
            fig.tight_layout()
            stem = args.plot_dir / f"compatibility_{flavor}_{wp}"
            fig.savefig(stem.with_suffix(".pdf"))
            fig.savefig(stem.with_suffix(".png"), dpi=160)
            plt.close(fig)


def money_plot(groups, results, args, lumi, energy, group_label="family"):
    """One diagnostic matrix pooled over independent category pulls only."""
    names = sorted(groups)
    matrix = np.full((len(names), len(names)), np.nan)
    for row, left in enumerate(names):
        for col, right in enumerate(names):
            if row == col:
                matrix[row, col] = 0.
                continue
            values = []
            for flavor in ("b", "c", "light"):
                for wp in EXCLUSIVE_CATEGORIES:
                    pull = pulls(results[left], results[right], flavor, wp)
                    values.append(np.abs(pull[np.isfinite(pull)]))
            pooled = np.concatenate(values) if values else np.array([])
            if pooled.size >= 3:
                matrix[row, col] = np.mean(pooled > 2.)
    # Keep the authoritative money plot legible and comparable across modes.
    fig, ax = plt.subplots(figsize=(14.3, 12.1))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=11)
    ax.set_yticks(range(len(names)), names, fontsize=11)
    ax.set_xlabel(f"Comparison {group_label}")
    ax.set_ylabel(f"Comparison {group_label}")
    for row in range(len(names)):
        for col in range(len(names)):
            value = matrix[row, col]
            if np.isfinite(value) and value < 0.2:
                ax.text(col, row, f"{value:.2f}", color="white", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label=r"Diagnostic fraction of valid bins with $|$pull$|>2$ (N, LnotM, MnotT, TnotXT, XTnotXXT, XXT)")
    cms_label(ax, lumi, energy)
    fig.tight_layout()
    stem = args.plot_dir / "compatibility_all_flavours_all_wps"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)


def family_pull_pdfs(groups, results, args, lumi, energy, group_label="family"):
    names = sorted(groups)
    sources = args.sources or names
    unknown = sorted(set(sources) - set(names))
    if unknown:
        raise ValueError(f"Unknown requested source {group_label}s: {unknown}")
    for source in sources:
        others = [name for name in names if name != source]
        with PdfPages(args.plot_dir / f"pulls_{source}.pdf") as pdf:
            for flavor in ("b", "c", "light"):
                fig, axes_grid = plt.subplots(2, 3, figsize=(18, max(9, 0.42 * len(others) + 4)), sharey=True)
                axes = list(axes_grid.flat)
                for ax, wp in zip(axes, EXCLUSIVE_CATEGORIES):
                    rows = [np.clip(pulls(results[source], results[other], flavor, wp).reshape(-1), -5, 5)
                            for other in others]
                    image = ax.imshow(rows, aspect="auto", cmap="coolwarm", vmin=-5, vmax=5,
                                      interpolation="nearest")
                    ax.set_xlabel(rf"{wp}: flattened $(p_T, |\eta|)$ bin")
                    ax.set_xticks(np.arange(0, rows[0].size, 8))
                axes_grid[0, 0].set_yticks(range(len(others)), others, fontsize=8)
                axes_grid[0, 0].set_ylabel(f"Comparison {group_label}\n({source} minus other)")
                fig.colorbar(image, ax=axes, label="Pull (clipped at $\pm5$)")
                cms_label(axes_grid[0, 0], lumi, energy)
                fig.subplots_adjust(left=0.28, right=0.90, bottom=0.16, top=0.94, wspace=0.10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def summary_json(groups, results, args):
    summary = {"_metadata": {
        "categories": list(EXCLUSIVE_CATEGORIES),
        "minimum_valid_bins": 3,
        "note": "Compatibility is a diagnostic, not a formal hypothesis test.",
        "input_completeness_verified": getattr(args, "input_completeness_verified", False),
        "ignored_source_channels": getattr(args, "ignored_source_channels", []),
        "converter_fallback_bins": {
            name: {flavor: np.argwhere(mask).tolist() for flavor, mask in result[3].items() if np.any(mask)}
            for name, result in results.items()
        },
    }}
    for left in sorted(groups):
        for right in sorted(groups):
            if left == right:
                continue
            pooled = []
            candidates = excluded_total = 0
            for flavor in ("b", "c", "light"):
                for wp in EXCLUSIVE_CATEGORIES:
                    pull = pulls(results[left], results[right], flavor, wp)
                    finite = np.abs(pull[np.isfinite(pull)])
                    excluded = results[left][3][flavor] | results[right][3][flavor]
                    pooled.append(finite)
                    candidates += pull.size
                    excluded_total += int(np.count_nonzero(excluded))
                    key = f"{left}__vs__{right}/{flavor}/{wp}"
                    available = finite.size >= 3
                    summary[key] = {"candidate_bins": int(pull.size),
                                    "valid_bins": int(finite.size),
                                    "excluded_pathological_bins": int(np.count_nonzero(excluded)),
                                    "available": available,
                                    "fraction_gt2": float(np.mean(finite > 2.)) if available else None,
                                    "fraction_gt3": float(np.mean(finite > 3.)) if available else None,
                                    "max_abs_pull": float(np.max(finite)) if available else None}
            pooled = np.concatenate(pooled) if pooled else np.array([])
            available = pooled.size >= 3
            summary[f"{left}__vs__{right}/all_flavours_all_categories"] = {
                "candidate_bins": int(candidates), "valid_bins": int(pooled.size),
                "excluded_pathological_bins": excluded_total, "available": available,
                "fraction_gt2": float(np.mean(pooled > 2.)) if available else None,
                "fraction_gt3": float(np.mean(pooled > 3.)) if available else None,
                "max_abs_pull": float(np.max(pooled)) if available else None,
            }
    (args.plot_dir / "compatibility_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    args = parse_args()
    conv = converter_module()
    if args.plot_dir is None:
        args.plot_dir = (Path(__file__).parents[2] / "preselection" / "corrections" /
                          "scalefactors" / "btagging" / "diagnostics" /
                          f"diagnostic_{args.year}_{args.channel}")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    _, counts, variances, _, families, completeness = collect_samples(
        conv, args.input_dir, args.year, args.channel, args.job_manifest, load_config(year=args.year))
    args.input_completeness_verified = completeness is not None
    results = {}
    for family in families:
        # Sparse signed-weight bins are intentionally masked in comparisons;
        # the converter records any all-MC fallback used for application.
        results[family] = efficiencies(conv, counts[family], variances[family])
    lumi = args.lumi if args.lumi is not None else LUMI_FB.get(args.year)
    energy = 13.6 if args.year.startswith("202") else 13
    plt.style.use(hep.style.CMS)
    if not args.skip_matrices:
        matrix_plots(families, results, args, lumi, energy)
    # Individual pull PDFs are disabled for now; retain the helper for optional future use.
    # if not args.skip_pulls:
    #     family_pull_pdfs(families, results, args, lumi, energy)
    summary_json(families, results, args)


if __name__ == "__main__":
    main()
