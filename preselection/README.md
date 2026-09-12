# Preselection Framework

C++ framework for VBS VVH preselection using RDataFrame. The paths provided in this README assume you are running on the UAF cluster.

## Directory Structure

```
preselection/
├── src/           # C++ source files
├── include/       # Header files
├── etc/           # Config generation scripts and JSON files
├── condor/        # HTCondor batch submission scripts
├── corrections/   # Scale factors and corrections
├── bin/           # Compiled binaries (generated)
└── build/         # Object files (generated)
```

## Quick Start

```bash
# Set up environment
source setup.sh

# Compile
cd preselection/
make -j8

# Run examples in `run_wrapper.sh`, e.g., for running over locally over one signal sample on UAF:
python run_rdf.py -i etc/input_sample_jsons/run2/sig_c2v1p0_c3_1p0/all_events/2017_VBSWZH_c2v1p0_c3_1p0.json --prefix /ceph/cms/ -o some_dir -n test_small -c 1lep_1FJ -m local -r 2
```

## Prerequisites

Included in CMSSW 15_0_4:
- ROOT with RDataFrame
- Python 3.x
- Uproot
- Boost (for GoldenJSON)
- Correctionlib

Environment setup script: `setup.sh`

## Compilation
To set up and compile the preselection framework, navigate to the top level directory and run:

```bash
source setup.sh
cd preselection/
make -j8
```
This will compile the source files and place the binary in the bin/ directory.

## Configuration Files
The inputs for the preselection framework are defined using configuration JSON files. The config files define input samples, cross-sections, and metadata in JSON format for RDataFrame's `FromSpec`. The json files for all of the skim sets are provided in the `preselection/etc/input_sample_jsons` directory.  

**See [`etc/README.md`](etc/README.md) for detailed documentation.**

## Notes on running 
To run the preselection, use the compiled binary and provide the input specification and output file. The `run_rdf.py` serves as a wrapper around the `bin/runAnalysis`. This script will:
* **Prepare the json files**: The scrip will prepend a given prefix to the paths in all of the json files (to accommodate running either with direct file paths on UAF, or with xrd). 
* **Merge the given json files**: The underlying analysis code expects to receive a single json, so this script will merge all jsons into one single json. Note that `bin/runAnalysis` expects all samples to be the same kind (either signal, background, or data), so if running locally do not mix multiple kinds of jsons into a single run. The final json is saved locally to the `merged_jsons` dir for reference. 
* **Run the analysis**: It will either directly run the `bin/runAnalysis` (if the mode is local) or will wrap around the condor submission (if the mode is condor).

There are examples in `run_wrapper.sh` for running `run_rdf.py` either locally over single jsons for tests, or for running the full analysis at scale over all skim selections. 

## B-tag efficiency correction

B-tag efficiencies have been derived for all retained analysis channels and all five working points (`L`, `M`, `T`, `XT`, `XXT`) for:

- `2016preVFP`
- `2016postVFP`
- `2017`
- `2018`
- `2024Prompt`

The final year-scoped payloads are `corrections/scalefactors/btagging/btag_eff_<year>.json`. The `_met` trigger subsets and `all_events` are intentionally excluded from the payload construction. The stored `*_mcstat_unc` values are diagnostic only and are not consumed by the analysis.
For normal b-tag SF application, 2025 resolves its efficiency family, channel, and correction from the 2024Prompt payload (`btag_eff_2024Prompt.json`), while using the official 2025 BTV SF payload. Branches are emitted only for sources present in at least one input year; sources unavailable for a particular event year receive the central value in that branch, while sources absent from the input are not written. The 2025 payload maps logical `topmass` to `mass`. Its 11 regrouped BTV JES responses are emitted only as companions named for the matching analysis JES source (for example, `weightsyst_jesAbsolute`) when `--systs` is enabled; they are not independent nuisances. The 2024 payload keeps one separate inclusive BTV JES-total response (`weightsyst_jes`). Run-2 legacy granular BTV JES sources are not exposed, and no BTV JES branch is written unless a supported inclusive/total source exists.

<details>
<summary>Derivation instructions and tutorial</summary>

The canonical grouping and final channel/sample merges are defined in `corrections/scalefactors/btagging/btag_eff_families.yaml`. The MC-only `--btag-eff` mode writes signed weighted ROOT histograms; use a new output root for each submission.

```bash
# 1. Produce raw weighted yields on MC with Slurm. This creates
#    $OUT_DIR/2024Prompt/<channel>/{manifest.json,<exact-sample>/output_<job-index>.root}.
python3 run_rdf.py -p "$PREFIX" -o "$OUT_DIR" -n run3_btag_eff \
  -c all -m slurm -r 3 -f 1 --btag-eff --year 2024Prompt

INPUT_ROOT="$OUT_DIR/2024Prompt"

# 2. Build preliminary payloads and inspect family/channel compatibility.
python3 ../misc/sf-utils/bEff-convert-to-correction.py \
  --input-dir "$INPUT_ROOT/1lep_1FJ" --job-manifest "$INPUT_ROOT/1lep_1FJ/manifest.json" \
  --year 2024Prompt --channel 1lep_1FJ
python3 ../misc/sf-utils/plot-btag-eff-families.py \
  --input-dir "$INPUT_ROOT/1lep_1FJ" --job-manifest "$INPUT_ROOT/1lep_1FJ/manifest.json" \
  --year 2024Prompt --channel 1lep_1FJ
# Review the plots, then update final_merges in the YAML if needed.
python3 ../misc/sf-utils/plot-btag-eff-global.py --skip-matrices \
  --mode families --input-root "$INPUT_ROOT" --year 2024Prompt
python3 ../misc/sf-utils/plot-btag-eff-global.py --skip-matrices \
  --mode channels --input-root "$INPUT_ROOT" --year 2024Prompt

# 3. Build the final payload; channels and manifests are discovered from YAML.
python3 ../misc/sf-utils/bEff-convert-to-correction.py --final --year 2024Prompt \
  --input-root "$INPUT_ROOT"

# 4. Recheck the two final compatibility plots.
python3 ../misc/sf-utils/plot-btag-eff-global.py --final --skip-matrices \
  --mode families --input-root "$INPUT_ROOT" --year 2024Prompt
python3 ../misc/sf-utils/plot-btag-eff-global.py --final --skip-matrices \
  --mode channels --input-root "$INPUT_ROOT" --year 2024Prompt
```

The converter requires complete manifests and schema-v2 raw outputs. It uses the adjacent-WP event-reweighting formula and rejects old L/T-only ROOT files. During conversion, a pathological signed-weight pT bin is merged with its immediately lower neighbor and the merged efficiency is assigned to both bins; an irreparable first-bin pathology still uses the validated all-MC fallback. The converter sums the four producer eta bins into one central-jet payload bin; its edges are `[-2.4, 2.4]` for 2016 pre/post-VFP and `[-2.5, 2.5]` otherwise. pT binning is unchanged. The 2016 pre/post-VFP UParTAK4 fixed-WP payloads use |eta| < 2.4; 2017, 2018, and 2024Prompt use |eta| < 2.5. Efficiency production and SF application use the same year-dependent boundary.

</details>

## B-tagging SF application

Edit `applybtag.yaml` to choose the ordered WPs needed by each channel, for example `4lep: [L, T]`. The runtime evaluates only the configured WPs and automatically excludes the others. Use `[]` or `--skip-btag-sf` to disable SF application for a channel/invocation. When SFs are disabled, the `weightsyst_*_withbSF`, `weightsyst_jes`, and `weightsyst_jer` branches are not written.

The nominal `weight` already contains the central HF and LF b-tag factors, available as the static `weightsyst_btag_HF_correlated[0]` and `weightsyst_btag_LF_correlated[0]` branches. To remove only the nominal b-tag correction for a diagnostic or post-processing comparison, use

```text
weight_without_btag = weight /
  (weightsyst_btag_HF_correlated[0] *
   weightsyst_btag_LF_correlated[0])
```

This retains the other corrections in `weight`. `baseweight` is instead before all lepton, b-tag, and other corrections, while the internal `_weight_*_raw` columns are not snapshotted.

Systematic branches are `{central, up, down}` vectors. Recommended branches to use in your postprocessor:

- **Simplified (where AK4 b-tag systematics are expected to be sub-dominant):**
  - HF: `weightsyst_btag_HF_correlated`, `weightsyst_btag_HF_uncorrelated_<year>`
  - LF: `weightsyst_btag_LF_correlated`, `weightsyst_btag_LF_uncorrelated_<year>`
  - Other SFs: `weight_pileup`, `weight_PSISR`, `weight_PSFSR`, `weight_muF`, and `weight_muR`
  - With JES variation responses: simply use nominal event weight `weight`

- **Full breakdown (where AK4 b-tag systematics are found to be dominant):**
  - HF: `weightsyst_btag_HF_pdf`, `weightsyst_btag_HF_as`, `weightsyst_btag_HF_pdfas`, `weightsyst_btag_HF_ttbar`, `weightsyst_btag_HF_type3`, `weightsyst_btag_HF_bfragmentation`, `weightsyst_btag_HF_topmass`, `weightsyst_btag_HF_hdamp`, and `weightsyst_btag_HF_statistic_<year>`
  - LF: `weightsyst_btag_LF_correlated`, `weightsyst_btag_LF_uncorrelated_<year>`
  - Other SFs: `weightsyst_pileup_withbSF`, `weightsyst_PSISR_withbSF`, `weightsyst_PSFSR_withbSF`, `weightsyst_muF_withbSF`, and `weightsyst_muR_withbSF`
  - JES responses:
    - 2025: `weightsyst_jesAbsolute`, `weightsyst_jesBBEC1`, `weightsyst_jesEC2`, `weightsyst_jesHF`, `weightsyst_jesRelativeBal`, `weightsyst_jesFlavorQCD`, `weightsyst_jesAbsoluteYear`, `weightsyst_jesBBEC1Year`, `weightsyst_jesEC2Year`, `weightsyst_jesHFYear`, and `weightsyst_jesRelativeSampleYear` are companions of the corresponding analysis JES nuisances, not independent BTV nuisances.
    - 2024: `weightsyst_jes` is one separate inclusive BTV JES-total nuisance; do not duplicate it across the 11 analysis JES sources.
    - Run 2: no BTV JES branch is written for payloads without a supported inclusive/total source; legacy granular sources are intentionally not exposed.
    - `weightsyst_jer` remains the BTV JER response where available.


---
## Details on the condor batch submission

For large-scale processing, the HTCondor submission should be used. Some overview information is provided here, with more details in the condor README. 

```bash
# Submit jobs
python condor/submit.py -c <config.json> -a <channel> -r <run_number>

# Check status
python condor/status.py --task <task_name>

# Resubmit failed jobs
python condor/resubmit.py --task <task_name> --failed
```

For automatic monitoring and resubmission:
```bash
screen -S condor_monitor
python condor/submit.py -c <config.json> -a <channel> -r <run_number> --monitor --timeout 12
# Ctrl+A, D to detach
```

**See [`condor/README.md`](condor/README.md) for detailed documentation.**
