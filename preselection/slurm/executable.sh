#!/bin/bash

# SLURM job executable for run3-vbsvvh preselection framework
#
# Runs from local scratch to keep core dumps, temp files, etc. off the
# shared filesystem.  The analysis code is shipped as a tarball and
# compiled on the node (same approach as the condor executable).
#
# Arguments:
#   $1 - TASK_DIR (absolute path to task dir containing package.tar.gz)
#   $2 - N_CPUS
#   $3 - CONFIG_FILE (absolute path to job config.json on shared filesystem)
#   $4 - OUTPUT_DIR (absolute path to final output directory)
#   $5 - ANALYSIS
#   $6 - RUN_NUMBER
#   $7 - SAMPLE_NAME
#   $8 - JOB_IDX
#   $9+ - EXTRA_FLAGS (optional, e.g., --spanet_training, --spanet_infer)

TASK_DIR=$1
N_CPUS=$2
CONFIG=$3
OUTPUT_DIR=$4
ANALYSIS=$5
RUN_NUMBER=$6
SAMPLE_NAME=$7
JOB_IDX=$8
shift 8
EXTRA_FLAGS=("$@")
EXTRA_FLAGS_TEXT="${EXTRA_FLAGS[*]}"

# Constants
OUTPUTDIR="output"
OUTPUTFILE="output"
CMSSW_VERSION='CMSSW_16_1_0_pre2'

# Disable core dumps to prevent filling shared filesystem on crashes
ulimit -c 0

echo "=========================================="
echo "Job started at $(date)"
echo "=========================================="
echo "Arguments:"
echo "  TASK_DIR=$TASK_DIR"
echo "  N_CPUS=$N_CPUS"
echo "  CONFIG=$CONFIG"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  ANALYSIS=$ANALYSIS"
echo "  RUN_NUMBER=$RUN_NUMBER"
echo "  SAMPLE_NAME=$SAMPLE_NAME"
echo "  JOB_IDX=$JOB_IDX"
echo "  EXTRA_FLAGS=$EXTRA_FLAGS_TEXT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "  HOSTNAME=$(hostname)"
echo "=========================================="

# ============================================================================
# Functions
# ============================================================================

function validate_root_file {
    # Validate ROOT file integrity by checking all branches for all events
    # Returns 0 if file is valid, 1 if corrupted
    local FILEPATH=$1
    local TREENAME=${2:-"Events"}

    echo "Validating ROOT file: $FILEPATH (tree: $TREENAME)"

    python3 << EOF
import sys
try:
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError  # Suppress warnings

    filepath = "${FILEPATH}"
    treename = "${TREENAME}"

    # Check file can be opened
    f = ROOT.TFile.Open(filepath)
    if not f or f.IsZombie():
        print("[VALIDATE] ERROR: Cannot open file or file is zombie")
        sys.exit(1)

    # Check tree exists
    t = f.Get(treename)
    if not t:
        print(f"[VALIDATE] ERROR: Tree '{treename}' not found in file")
        f.Close()
        sys.exit(1)

    nevts = t.GetEntries()
    print(f"[VALIDATE] Found {nevts} entries in tree '{treename}'")

    if nevts == 0:
        print("[VALIDATE] WARNING: Tree has 0 entries (may be expected for some samples)")
        f.Close()
        sys.exit(0)

    # Check all events - GetEntry returns -1 on I/O error
    nbad = 0
    for i in range(nevts):
        if t.GetEntry(i) < 0:
            print(f"[VALIDATE] ERROR: Bad event at index {i}")
            nbad += 1
            if nbad >= 10:
                print("[VALIDATE] ERROR: Too many bad events, aborting check")
                break

    f.Close()

    if nbad > 0:
        print(f"[VALIDATE] FAILED: Found {nbad} corrupted events")
        sys.exit(1)
    else:
        print("[VALIDATE] PASSED: All events readable")
        sys.exit(0)

except Exception as e:
    print(f"[VALIDATE] ERROR: Exception during validation: {e}")
    sys.exit(1)
EOF
    return $?
}

# ============================================================================
# Setup CMS environment
# ============================================================================
echo ""
echo "=== Setting up CMS environment ==="

if [ -r /cvmfs/cms.cern.ch/cmsset_default.sh ]; then
    source /cvmfs/cms.cern.ch/cmsset_default.sh
else
    echo "ERROR: Cannot find /cvmfs/cms.cern.ch/cmsset_default.sh"
    exit 1
fi

cd /cvmfs/cms.cern.ch/el9_amd64_gcc13/cms/cmssw/$CMSSW_VERSION
eval $(scramv1 runtime -sh)
cd - > /dev/null

echo "CMSSW_BASE=$CMSSW_BASE"
echo "SCRAM_ARCH=$SCRAM_ARCH"

# ============================================================================
# Set up local scratch working directory
# ============================================================================
echo ""
echo "=== Setting up local scratch directory ==="

WORK_DIR=$(mktemp -d -p ${SLURM_TMPDIR:-/tmp} slurm_job_XXXXXX)
echo "Working directory: $WORK_DIR"

echo ""
echo "=== Extracting package ==="
tar -xzf $TASK_DIR/package.tar.gz -C $WORK_DIR
cp $CONFIG $WORK_DIR/config.json
cd $WORK_DIR
ls -la

echo ""
echo "=== Building analysis code ==="
make -j2

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    rm -rf $WORK_DIR
    exit 1
fi

echo ""
echo "=== Directory contents before analysis ==="
ls -la
ls -la bin/

# ============================================================================
# Run analysis
# ============================================================================
echo ""
echo "=== Running analysis ==="

if [[ "$EXTRA_FLAGS_TEXT" == *"--spanet_infer"* ]]; then
    echo "./bin/runAnalysis -b 518 -i config.json -n $OUTPUTFILE --outdir $OUTPUTDIR --ana $ANALYSIS --run_number $RUN_NUMBER $EXTRA_FLAGS_TEXT"
    ./bin/runAnalysis -b 518 -i config.json -n $OUTPUTFILE --outdir $OUTPUTDIR --ana $ANALYSIS --run_number $RUN_NUMBER "${EXTRA_FLAGS[@]}"
else
    echo "./bin/runAnalysis -j $N_CPUS -i config.json -n $OUTPUTFILE --outdir $OUTPUTDIR --ana $ANALYSIS --run_number $RUN_NUMBER $EXTRA_FLAGS_TEXT"
    ./bin/runAnalysis -j $N_CPUS -i config.json -n $OUTPUTFILE --outdir $OUTPUTDIR --ana $ANALYSIS --run_number $RUN_NUMBER "${EXTRA_FLAGS[@]}"
fi

ANALYSIS_STATUS=$?
if [ $ANALYSIS_STATUS -ne 0 ]; then
    echo "ERROR: Analysis failed with status $ANALYSIS_STATUS"
    rm -rf $WORK_DIR
    exit 1
fi

echo ""
echo "=== Directory contents after analysis ==="
ls -la
ls -la $OUTPUTDIR/ 2>/dev/null || echo "Output directory not found"

# ============================================================================
# Validate and copy output
# ============================================================================
echo ""
echo "=== Validating output ==="

# Determine output file path
if [[ "$EXTRA_FLAGS_TEXT" == *"--btag_eff"* ]]; then
    OUTPUT_ROOT_FILE="$OUTPUTDIR/${OUTPUTFILE}_btag_eff.root"
elif [[ "$EXTRA_FLAGS_TEXT" == *"--spanet_training"* ]]; then
    OUTPUT_ROOT_FILE="$OUTPUTDIR/${OUTPUTFILE}_spanet_training_data.root"
else
    OUTPUT_ROOT_FILE="$OUTPUTDIR/$OUTPUTFILE.root"
fi

if [ ! -f "$OUTPUT_ROOT_FILE" ]; then
    echo "ERROR: Output file not found: $OUTPUT_ROOT_FILE"
    echo "=== Searching for any .root files ==="
    find . -name "*.root" -ls 2>/dev/null
    rm -rf $WORK_DIR
    exit 1
fi

echo "Output file found: $OUTPUT_ROOT_FILE ($(du -h $OUTPUT_ROOT_FILE | cut -f1))"

echo ""
echo "=== Validating output ROOT file ==="
if [[ "$EXTRA_FLAGS_TEXT" == *"--btag_eff"* ]]; then
    python3 - << EOF
import sys
import ROOT
f = ROOT.TFile.Open("$OUTPUT_ROOT_FILE")
if not f or f.IsZombie() or not f.Get("btag_b_den"):
    sys.exit(1)
f.Close()
EOF
else
    validate_root_file "$OUTPUT_ROOT_FILE" "Events"
fi
if [ $? -ne 0 ]; then
    echo "ERROR: Output ROOT file validation failed"
    echo "Removing corrupted file: $OUTPUT_ROOT_FILE"
    rm -f "$OUTPUT_ROOT_FILE"
    rm -rf $WORK_DIR
    exit 1
fi

echo ""
echo "=== Copying output to final destination ==="

FINAL_DIR="${OUTPUT_DIR}/${SAMPLE_NAME}"
mkdir -p $FINAL_DIR

FINAL_FILE="${FINAL_DIR}/output_${JOB_IDX}.root"
cp $OUTPUT_ROOT_FILE $FINAL_FILE

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy output to $FINAL_FILE"
    rm -rf $WORK_DIR
    exit 1
fi

echo "Output copied to: $FINAL_FILE"

# Copy cutflow if it exists
if [ -f "$OUTPUTDIR/${OUTPUTFILE}_cutflow.txt" ]; then
    cp $OUTPUTDIR/${OUTPUTFILE}_cutflow.txt ${FINAL_DIR}/cutflow_${JOB_IDX}.txt
    echo "Cutflow copied to: ${FINAL_DIR}/cutflow_${JOB_IDX}.txt"
fi

# Cleanup
rm -rf $WORK_DIR

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
