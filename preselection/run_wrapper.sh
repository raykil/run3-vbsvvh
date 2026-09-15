# Wrapper script for running the full analysis.
# Auto-picks slurm (if sbatch is available) or condor for batch submission.

#set -e

# ---- Site paths: uncomment the set that matches your machine ----

# HPG local filesystem (T2 user area — plenty of space)
# The T2 path uses your lxplus username, which can differ from
# the HPG login ($USER). Export CERN_USER in your shell or set
# it on the line below; the expansion aborts the script if unset.
CERN_USER="${CERN_USER:?Please set CERN_USER to your T2 username}"
PREFIX="/cmsuf/data/store/user/phchang/skim/"
OUT_DIR="/cmsuf/data/store/user/$CERN_USER/vbs_vvh_rdf"

# HPG blue (recurring space issues — uncomment only when needed)
#OUT_DIR="/blue/avery/$USER/samples/run3-vbsvvh"

# HPG via xrd
#PREFIX="root://cmsio3.rc.ufl.edu//store/user/phchang/skim/"
#OUT_DIR="/blue/avery/$USER/samples/run3-vbsvvh"

# UAF local filesystem
#PREFIX="/ceph/cms/"
#OUT_DIR="/data/userdata/$USER/vbs_vvh_rdf"

# UAF via xrd
#PREFIX="root://redirector.t2.ucsd.edu:1095//"
#OUT_DIR="/data/userdata/$USER/vbs_vvh_rdf"

mkdir -p "$OUT_DIR"

# ---- Pick batch mode based on what's available on the system ----
if command -v sbatch &>/dev/null; then
    MODE="slurm"
else
    MODE="condor"
fi
echo "Batch mode: $MODE"

# ---- Examples of local runs for testing (uncomment one; skip the batch block below) ----
#python3 run_rdf.py -i etc/input_sample_jsons/run2/sig/all_events/2017_VBSWZH_c2v1p0_c3_1p0.json -p $PREFIX -o $OUT_DIR -n test -c 3lep      -m local -r 2 -j 16
#python3 run_rdf.py -i etc/input_sample_jsons/run2/sig/all_events/                              -p $PREFIX -o $OUT_DIR -n r2sigSM -c 2lep_1FJ -m local -r 2 -j 16
#exit 0

# ---- Batch submission: signal + per-channel bkg/data, for Run 2 and Run 3 ----

# Channels to run. run_rdf.py auto-resolves the inputs from -c; --kinds picks
# which tier of those inputs to submit. Use `all` to run every physics channel
# in run_rdf.py's ANA_CHANNELS dict.
CHANNELS=(0lep_1FJ 0lep_2FJ)
#CHANNELS=(all)

# Run-3 MC-only b-tag efficiency production for every channel and sample:
# python3 run_rdf.py -p "$PREFIX" -o "$OUT_DIR" -n run3_btag_eff -c all -m "$MODE" -r 3 -f 1 --btag-eff --year 2024Prompt
# Compute and store JES/JER variation branches. No-op on data.
# Only passed to --kind signal, while background is left nominal (the analysis is data driven). If you need to pass it to backgorund MC, just add $SYSTS to the "--kind bkg" call below.
# Set SYSTS="" for a fully nominal pass.
SYSTS="${SYSTS---systs}"

for RUN in 2 3; do
    RUN_BASE="etc/input_sample_jsons/run${RUN}"

    # Signal through the all_events pass-through channel, i.e. no analysis selection
    # at all. Not part of the standard production -- uncomment only for studies that
    # need the signal before any channel selection (e.g. acceptance/efficiency).
    #python3 run_rdf.py -i ${RUN_BASE}/sig/all_events/  -p $PREFIX -o $OUT_DIR -n r${RUN}_sig_sm  -c all_events -m $MODE -r $RUN -f 1 $SYSTS

    # One submission per tier, so --systs reaches signal without reaching background.
    # All three share -n, so the output layout is unchanged.
    python3 run_rdf.py -p $PREFIX -o $OUT_DIR -n r${RUN} -c "${CHANNELS[@]}" -m $MODE -r $RUN -f 1 --kinds sig $SYSTS
    python3 run_rdf.py -p $PREFIX -o $OUT_DIR -n r${RUN} -c "${CHANNELS[@]}" -m $MODE -r $RUN -f 1 --kinds data
    python3 run_rdf.py -p $PREFIX -o $OUT_DIR -n r${RUN} -c "${CHANNELS[@]}" -m $MODE -r $RUN -f 1 --kinds bkg
done
