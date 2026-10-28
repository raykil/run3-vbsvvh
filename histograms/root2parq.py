"""
Skimmed ROOT files in /groups/cjessop/users/jkil/HVV_2L_RDF/run*_*FJ/{longname}/output_*.root takes long time to load.
This script takes multiple ROOT file per dataset into one parquet file, and save in /groups/cjessop/users/jkil/HVV_2L_RDF_parq/run*_*FJ/{shortname}.parq.

To run on condorfe:
ssh condorfe
cd ~/run3-vbsvvh/histograms && conda activate hvv
for ch in run2_1FJ run2_2FJ run3_1FJ run3_2FJ; do python root2parq.py -m condor -c $ch -f; [ $ch != run3_2FJ ] && sleep 5; done
"""
import os, sys, glob
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from subprocess import run
import awkward as ak
import uproot
from PlotterTools import SHORT_TO_FULL

RDF_PATH  = "/groups/cjessop/users/jkil/HVV_2L_RDF"
PARQ_PATH = "/groups/cjessop/users/jkil/HVV_2L_RDF_parq"
FULL_TO_SHORT = {full: short for short, full in SHORT_TO_FULL.items()}
BRANCHES = [
    'run', 'luminosityBlock', 'event', # data de-duplication across primary datasets (removeDuplicates is off in preselection/src/main.cpp)
    'weight', 'passes_2lep_?FJ_nom', 'njet', 'nLep_Sel',
    'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_mass',
    'electron_pt', 'electron_cutBased', 'muon_pt', # makeDataset.LooseLeptons
    'met_pt', 'met_phi',
    'fatjet_pt', 'fatjet_eta', 'fatjet_phi', 'fatjet_massGloParT3', 'fatjet_msoftdrop', 'fatjet_tau1', 'fatjet_tau2',
    'fatjet_Hbb', 'ht_fatjets',
    'vbs_score', 'vbs_mjj', 'vbs_detajj', 'vbs_jet1_idx', 'vbs_jet2_idx', 'jet_pt', 'jet_eta', 'jet_jetId', # makeDataset.VBFCut
]

def MergedChunks(files, max_bytes=5e8):
    # uproot.iterate never merges across files, so buffer the many small per-file chunks into ~max_bytes row groups
    buffer = [uproot.open(f"{files[0]}:Events").arrays(filter_name=BRANCHES, entry_stop=0)] # empty seed keeps the schema when no file has events
    for chunk in uproot.iterate([f"{f}:Events" for f in files], filter_name=BRANCHES, step_size="500 MB"):
        buffer.append(chunk)
        if sum(c.nbytes for c in buffer) > max_bytes:
            yield ak.concatenate(buffer)
            buffer = []
    if buffer: yield ak.concatenate(buffer)

def Convert(sampleDir, outPath):
    files = sorted(glob.glob(f"{sampleDir}/*.root"))
    ak.to_parquet_row_groups(MergedChunks(files), f"{outPath}.tmp")
    os.replace(f"{outPath}.tmp", outPath) # an interrupted write never leaves a complete-looking parquet
    src, dst = Path(sampleDir), Path(outPath)
    print(f"\033[1;32m{src.parent.parent.name}/{src.parent.name}/{dst.stem} ({len(files)} files) -> {'/'.join(dst.parts[-3:])} done!\033[0m", flush=True)

def SamplesToConvert(channels, outdir, force):
    assert len(FULL_TO_SHORT) == len(SHORT_TO_FULL), "SHORT_TO_FULL is not bijective!"
    for ch in channels: os.makedirs(f"{outdir}/{ch}", exist_ok=True)
    todo = {f"{outdir}/{ch}/{FULL_TO_SHORT[full]}.parq": f"{RDF_PATH}/{ch}/{full}" for ch in channels for full in sorted(os.listdir(f"{RDF_PATH}/{ch}"))}
    return {out: src for out, src in todo.items() if force or not os.path.exists(out)}

def SubmitCondor(todo, memory, dryRun):
    # One job per sample on the CRC opportunistic pool; jobs run this python and read/write /groups directly, so nothing is transferred
    if not dryRun and not os.uname().nodename.startswith('condorfe'):
        sys.exit("Submit from condorfe.crc.nd.edu; condor_submit on glados goes to the ndcms pool.")
    logDir = f"{os.path.dirname(os.path.abspath(__file__))}/{datetime.now().strftime('%y%m%d')}_root2parq_condor"
    os.makedirs(logDir, exist_ok=True)
    with open(f"{logDir}/samples.txt", 'w') as f: f.writelines(f"{src} {out} {Path(out).parent.name}_{Path(out).stem}\n" for out, src in todo.items())
    with open(f"{logDir}/submit.jdl", 'w') as f: f.write(
        f"""
        universe              = vanilla
        executable            = {sys.executable}
        arguments             = {os.path.abspath(__file__)} --sample $(src) $(out)
        transfer_executable   = False
        should_transfer_files = NO
        request_cpus          = 1
        request_memory        = {memory}
        max_retries           = 3
        log                   = {logDir}/$(name).log
        output                = {logDir}/$(name).out
        error                 = {logDir}/$(name).err
        +JobBatchName         = "root2parq"
        queue src, out, name from {logDir}/samples.txt
        """
        )
    print(f"submit file: {logDir}/submit.jdl")
    if not dryRun: run(["condor_submit", f"{logDir}/submit.jdl"], check=True)

if __name__ == "__main__":
    parser = ArgumentParser(description='Merge the RDF output ROOT files of each sample into one parquet.')
    parser.add_argument('-m', '--mode'    , type=str, default='local', choices=['local', 'condor'], help='condor: one job per sample on condorfe opportunistic resources.')
    parser.add_argument('-c', '--channels', type=str, nargs='+', default=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'])
    parser.add_argument('-o', '--outdir'  , type=str, default=PARQ_PATH)
    parser.add_argument('-l', '--limit'   , type=int, default=None, help='Convert only the first N samples (for testing).')
    parser.add_argument('-f', '--force'   , action='store_true', help='If True, overwrites existing parquets. Default: skip them.')
    parser.add_argument('-j', '--jobs'    , type=int, default=1, help='local: number of samples converted in parallel.')
    parser.add_argument('--memory'        , type=str, default='4 GB', help='condor: request_memory per job.')
    parser.add_argument('-n', '--dry-run' , action='store_true', help='condor: writes the submit file without submitting.')
    parser.add_argument('--sample'        , type=str, nargs=2, metavar=('SRC', 'OUT'), help='Converts one sample; run by each condor job.')
    args = parser.parse_args()

    if args.sample:
        Convert(*args.sample)
        sys.exit()

    todo = dict(list(SamplesToConvert(args.channels, args.outdir, args.force).items())[:args.limit])
    print(f"{len(todo)} samples to convert")
    if args.mode == 'condor':
        SubmitCondor(todo, args.memory, args.dry_run)
    else:
        with ProcessPoolExecutor(args.jobs) as pool: list(pool.map(Convert, todo.values(), todo.keys()))
