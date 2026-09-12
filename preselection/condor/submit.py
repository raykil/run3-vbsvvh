#!/usr/bin/env python3
"""
Condor job submission script for run3-vbsvvh preselection framework.

Splits input files across multiple jobs and organizes outputs by sample.
Tracks job status in a manifest.json file for monitoring and resubmission.

Usage:
    python condor/submit.py --config etc/run2/0Lep2FJ_run2-bkg_QCD.json \
                            --analysis 0Lep2FJ --run_number 2 --tag test_v1
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

import uproot


# Constants
CONDOR_OUTPUT_DIR = "jobs"
OUTPUT_XRD = "root://redirector.t2.ucsd.edu:1095//store/user/{user}/vbsvvh/preselection/run3-vbsvvh"
DEFAULT_FILES_PER_JOB = 10
DEFAULT_NCPUS = 4
DEFAULT_MEMORY = "2G"
SITES = "T2_US_UCSD"
JOB_FLAVOUR = "espresso"
SINGULARITY_IMAGE = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el8:x86_64"


# ============================================================================
# Job Manifest for Tracking
# ============================================================================

class JobManifest:
    """
    Tracks job submissions and their status.

    Stored as manifest.json in the task directory.
    """

    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.manifest_path = task_dir / "manifest.json"
        self.data = {
            "task_name": task_dir.name,
            "created": datetime.now().isoformat(),
            "config": "",
            "analysis": "",
            "run_number": 0,
            "output_base": "",
            "jobs": {}
        }

    def set_metadata(self, config: str, analysis: str, run_number: int, user: str):
        """Set task-level metadata."""
        self.data["config"] = config
        self.data["analysis"] = analysis
        self.data["run_number"] = run_number
        self.data["output_base"] = OUTPUT_XRD.format(user=user) + "/" + self.data["task_name"]

    def add_job(
        self,
        job_id: str,
        sample_name: str,
        job_idx: int,
        input_files: List[str],
        job_dir: str
    ):
        """Add a job to the manifest."""
        self.data["jobs"][job_id] = {
            "sample": sample_name,
            "job_idx": job_idx,
            "input_files": input_files,
            "n_files": len(input_files),
            "job_dir": job_dir,
            "status": "prepared",
            "cluster_id": None,
            "submit_time": None,
            "resubmit_count": 0
        }

    def mark_submitted(self, job_id: str, cluster_id: int, process_id: int = 0):
        """Mark a job as submitted with its condor IDs."""
        if job_id in self.data["jobs"]:
            job = self.data["jobs"][job_id]
            job["status"] = "submitted"
            job["cluster_id"] = cluster_id
            job["process_id"] = process_id
            job["submit_time"] = datetime.now().isoformat()

    def save(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    @classmethod
    def load(cls, task_dir: Path) -> 'JobManifest':
        """Load existing manifest from disk."""
        manifest = cls(task_dir)
        if manifest.manifest_path.exists():
            with open(manifest.manifest_path) as f:
                manifest.data = json.load(f)
        return manifest

    def get_jobs_by_status(self, status: str) -> Dict[str, dict]:
        """Get all jobs with a given status."""
        return {
            job_id: job for job_id, job in self.data["jobs"].items()
            if job["status"] == status
        }

    def get_summary(self) -> Dict[str, int]:
        """Get count of jobs by status."""
        summary = {}
        for job in self.data["jobs"].values():
            status = job["status"]
            summary[status] = summary.get(status, 0) + 1
        return summary


# ============================================================================
# Utility Functions
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit condor jobs for run3-vbsvvh preselection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Submit jobs for QCD background
    python condor/submit.py -c etc/run2/0Lep2FJ_run2-bkg_QCD.json -a 0Lep2FJ -r 2

    # Submit with custom tag and files per job
    python condor/submit.py -c etc/run2/0Lep2FJ_run2-sig.json -a 0Lep2FJ -r 2 \\
                            --tag v1 --files-per-job 5

    # Check job status
    python condor/status.py --task 0Lep2FJ_run2-sig_0Lep2FJ_v1
        """
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Path to config JSON file (relative to preselection/)")
    parser.add_argument("-a", "--analysis", required=True,
                        help="Analysis channel (e.g., 0Lep2FJ, 1Lep2FJ)")
    parser.add_argument("-r", "--run_number", required=True, type=int, choices=[2, 3],
                        help="Run number (2 or 3)")
    parser.add_argument("-t", "--tag", default="",
                        help="Optional tag for output directory naming")
    parser.add_argument("-j", "--ncpus", type=int, default=DEFAULT_NCPUS,
                        help=f"Number of CPUs per job (default: {DEFAULT_NCPUS})")
    parser.add_argument("-m", "--memory", default=DEFAULT_MEMORY,
                        help=f"Memory request per job (default: {DEFAULT_MEMORY})")
    parser.add_argument("--files-per-job", type=int, default=DEFAULT_FILES_PER_JOB,
                        help=f"Number of input files per job (default: {DEFAULT_FILES_PER_JOB})")
    parser.add_argument("--events-per-job", type=int, default=None,
                        help="Maximum events per job (if set, takes priority over --files-per-job)")
    parser.add_argument("--events-per-job-override", default=None,
                        help="JSON file mapping sample regex patterns to per-sample events-per-job values")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare jobs but don't submit")
    parser.add_argument("--sample", default=None,
                        help="Only submit jobs for this specific sample (regex match)")
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor jobs and automatically resubmit failed ones")
    parser.add_argument("--max-resubmits", type=int, default=3,
                        help="Maximum resubmit attempts per job (default: 3)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Timeout in hours for monitoring (default: no timeout)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Check interval in minutes for monitoring (default: 5)")
    parser.add_argument("--spanet-training", action="store_true",
                        help="Only make training data for SPANet (--spanet_training flag)")
    parser.add_argument("--spanet-infer", action="store_true",
                        help="Run SPANet inference (--spanet_infer flag)")
    parser.add_argument("--btag-eff", action="store_true",
                        help="Write raw selected-AK4 b-tag efficiency histograms (--btag_eff flag)")
    parser.add_argument("--store-hlt", "--store_hlt", dest="store_hlt", action="store_true",
                        help="Store HLT trigger branches in output (--store_hlt flag)")
    parser.add_argument("--skip-btag-sf", action="store_true",
                        help="Skip b-tag SF application (normally enabled)")
    parser.add_argument("--systs", action="store_true",
                        help="Store JES/JER variation branches (default: nominal only)")
    parser.add_argument("--no_jetveto", "--no-jetveto", dest="no_jetveto", action="store_true",
                        help="DEBUG ONLY: compute Jet_vetoMap but do not apply it")
    return parser.parse_args()


def get_user():
    """Get current username."""
    return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


def is_xrootd_path(path: str) -> bool:
    """Check if a path is an xrootd URL."""
    return path.startswith("root://")


def parse_xrootd_url(url: str) -> tuple:
    """
    Parse an xrootd URL into (host, path) components.

    Example: root://redirector.t2.ucsd.edu:1095//store/user/foo/bar
    Returns: ("redirector.t2.ucsd.edu:1095", "/store/user/foo/bar")
    """
    remainder = url[len("root://"):]
    slash_idx = remainder.index("/")
    host = remainder[:slash_idx]
    path = "/" + remainder[slash_idx:].lstrip("/")
    return host, path


def expand_xrootd_glob(file_pattern: str) -> List[str]:
    """
    Expand a glob pattern for xrootd paths using xrdfs ls.

    Example pattern: root://redirector.t2.ucsd.edu:1095//store/user/foo/bar/*.root
    """
    host, full_path = parse_xrootd_url(file_pattern)
    directory = os.path.dirname(full_path)
    pattern = os.path.basename(full_path)

    cmd = ["xrdfs", host, "ls", directory]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  WARNING: xrdfs ls failed: {result.stderr.strip()}")
            return []
    except subprocess.TimeoutExpired:
        print(f"  WARNING: xrdfs ls timed out for {directory}")
        return []
    except FileNotFoundError:
        print(f"  ERROR: xrdfs command not found. Is XRootD client installed?")
        return []

    all_entries = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

    matched = []
    for entry in all_entries:
        filename = os.path.basename(entry)
        if fnmatch.fnmatch(filename, pattern):
            matched.append(f"root://{host}/{entry}")

    return sorted(matched)


def has_glob_wildcards(pattern: str) -> bool:
    """Check if a string contains glob wildcard characters."""
    return any(c in pattern for c in ('*', '?', '['))


def expand_file_glob(file_pattern: str) -> List[str]:
    """Expand a glob pattern to list of actual files. Supports both local and xrootd paths."""
    if not has_glob_wildcards(file_pattern):
        return [file_pattern]
    if is_xrootd_path(file_pattern):
        return expand_xrootd_glob(file_pattern)
    files = sorted(glob(file_pattern))
    return files


def split_into_chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_event_count(file_path: str) -> int:
    """
    Get the number of events in a ROOT file from the "Events" tree.

    Args:
        file_path: Path to the ROOT file

    Returns:
        Number of events in the file, or 0 if file cannot be read
    """
    try:
        with uproot.open(file_path) as f:
            return f["Events"].num_entries
    except Exception as e:
        print(f"  WARNING: Could not read {file_path}: {e}")
        return 0


def estimate_events_per_file(files: List[str]) -> int:
    """
    Estimate events per file by checking the first two files and taking the max.

    For a given sample, files typically have the same number of events,
    except possibly the last file which may have fewer. By checking two
    files and taking the max, we get a safe estimate.

    Args:
        files: List of file paths

    Returns:
        Estimated events per file
    """
    if not files:
        return 0

    files_to_check = files[:2]
    counts = [get_event_count(f) for f in files_to_check]
    return max(counts) if counts else 0


def split_by_events(
    files: List[str],
    events_per_file: int,
    max_events_per_job: int
) -> List[List[str]]:
    """
    Split files into chunks based on estimated event count.

    Args:
        files: List of file paths
        events_per_file: Estimated events per file
        max_events_per_job: Maximum events per job

    Returns:
        List of file lists, one per job
    """
    if events_per_file <= 0:
        # Fallback: one file per job if we can't estimate
        return [[f] for f in files]

    # Calculate how many files fit in one job
    files_per_job = max(1, max_events_per_job // events_per_file)

    return split_into_chunks(files, files_per_job)


def load_events_per_job_overrides(path: str) -> List[tuple]:
    """
    Load per-sample events-per-job overrides from a JSON file.

    The file maps regex patterns to events-per-job values, e.g.:
        {"QCD_Bin-PT-600to800": 100000, "TTtoLNu2Q": 200000}

    Returns a list of (compiled_regex, events_per_job) tuples.
    """
    with open(path) as f:
        raw = json.load(f)
    return [(re.compile(pattern), epj) for pattern, epj in raw.items()]


def get_events_per_job(sample_name: str, default: Optional[int],
                       overrides: Optional[List[tuple]]) -> Optional[int]:
    """Return the events-per-job for a sample, checking overrides first."""
    if overrides:
        for pattern, epj in overrides:
            if pattern.search(sample_name):
                return epj
    return default


def create_job_config(sample_name: str, sample_config: Dict, file_list: List[str],
                      job_idx: int) -> Dict:
    """Create a config dict for a single job with specific files."""
    job_config = {
        "samples": {
            sample_name: {
                "trees": sample_config["trees"],
                "files": file_list,
                "metadata": sample_config["metadata"].copy()
            }
        }
    }
    return job_config


def create_tarball(preselection_dir: Path) -> Path:
    """Create tarball of the analysis code."""
    tarball_path = preselection_dir / "condor" / "package.tar.gz"

    spanet_run2_dir = "spanet/run2/v31/"
    spanet_run3_dir = "spanet/v2/"
    bdt_dir = "bdt/"
    # Items to include in tarball (from preselection dir)
    preselection_items = [
        "Makefile", "src", "include", "corrections", "applybtag.yaml",
        "etc/goldenJson", spanet_run2_dir, spanet_run3_dir,
        bdt_dir
    ]

    # Build tar command
    tar_items = []
    for item in preselection_items:
        if (preselection_dir / item).exists():
            tar_items.append(item)

    cmd = ["tar", "-czf", str(tarball_path)] + tar_items
    print(f"Creating tarball: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=preselection_dir, check=True)

    return tarball_path


def generate_submit_file(task_dir: Path, job_dir: Path, job_name: str,
                         args: argparse.Namespace, sample_name: str, job_idx: int) -> Path:
    """Generate a condor submit file for a job."""
    user = get_user()
    submit_path = job_dir / "submit.cmd"

    # Build optional flags string
    extra_flags = ""
    if args.spanet_training:
        extra_flags += " --spanet_training"
    if args.spanet_infer:
        extra_flags += " --spanet_infer"
    if args.systs:
        extra_flags += " --systs"
    if args.no_jetveto:
        extra_flags += " --no_jetveto"
    if args.btag_eff:
        extra_flags += " --btag_eff"
    if args.store_hlt:
        extra_flags += " --store_hlt"
    if args.skip_btag_sf:
        extra_flags += " --skip-btag-sf"

    # Arguments passed to executable:
    # USER N_CPUS CONFIG_FILE OUTPUT_NAME ANALYSIS RUN_NUMBER SAMPLE_NAME JOB_IDX [EXTRA_FLAGS]
    job_args = f"{user} {args.ncpus} config.json {job_name} {args.analysis} {args.run_number} {sample_name} {job_idx} {extra_flags}"

    submit_content = f"""universe                = Vanilla
Requirements            = ((HAS_SINGULARITY=?=True))
RequestMemory           = {args.memory}
RequestCpus             = {args.ncpus}
executable              = {task_dir}/executable.sh
arguments               = {job_args}
transfer_executable     = True
transfer_input_files    = {task_dir}/package.tar.gz,{job_dir}/config.json
transfer_output_files   = ""
log                     = {job_dir}/job.$(Cluster).$(Process).log
output                  = {job_dir}/job.$(Cluster).$(Process).stdout
error                   = {job_dir}/job.$(Cluster).$(Process).stderr
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
x509userproxy           = /tmp/x509up_u{os.getuid()}
use_x509userproxy       = True
MY.SingularityImage     = "{SINGULARITY_IMAGE}"
+DESIRED_Sites          = "{SITES}"
+JobBatchName           = "{job_name}"
+project_Name           = "cmssurfandturf"
+JobFlavour             = "{JOB_FLAVOUR}"

queue
"""

    with open(submit_path, "w") as f:
        f.write(submit_content)

    return submit_path


def extract_cluster_id(condor_output: str) -> Optional[int]:
    """Extract cluster ID from condor_submit output."""
    # Example: "1 job(s) submitted to cluster 12345."
    match = re.search(r'submitted to cluster (\d+)', condor_output)
    if match:
        return int(match.group(1))
    return None


# ============================================================================
# Monitoring
# ============================================================================

def monitor_and_resubmit(
    task_dir: Path,
    manifest: 'JobManifest',
    max_resubmits: int = 3,
    timeout_hours: Optional[float] = None,
    interval_minutes: int = 5
):
    """
    Monitor jobs and automatically resubmit failed ones.

    Args:
        task_dir: Path to task directory
        manifest: JobManifest instance
        max_resubmits: Maximum resubmit attempts per job
        timeout_hours: Stop after this many hours (None = no timeout)
        interval_minutes: Check interval in minutes
    """
    # Import status functions here to avoid circular imports at module level
    from status import get_condor_jobs, get_condor_history, determine_job_status

    start_time = datetime.now()
    timeout_delta = timedelta(hours=timeout_hours) if timeout_hours else None

    print(f"\n{'='*60}")
    print("Starting job monitor")
    print(f"{'='*60}")
    print(f"Max resubmits per job: {max_resubmits}")
    print(f"Check interval: {interval_minutes} minutes")
    print(f"Timeout: {timeout_hours} hours" if timeout_hours else "Timeout: none")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    while True:
        # Check timeout
        elapsed = datetime.now() - start_time
        if timeout_delta and elapsed > timeout_delta:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Timeout reached after {timeout_hours} hours. Stopping monitor.")
            break

        # Query condor status
        jobs = manifest.data.get("jobs", {})
        condor_jobs, err = get_condor_jobs()
        if err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: {err}")

        cluster_ids = [
            j["cluster_id"] for j in jobs.values()
            if j.get("cluster_id") is not None
        ]
        missing_ids = [cid for cid in cluster_ids if cid not in condor_jobs]

        condor_history = {}
        if missing_ids:
            condor_history, err = get_condor_history(missing_ids)
            if err:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: {err}")

        # Determine status for each job
        status_counts = {"completed": 0, "running": 0, "queued": 0, "failed": 0, "other": 0}
        jobs_to_resubmit = []
        permanently_failed = []

        for job_id, job in jobs.items():
            status = determine_job_status(job, condor_jobs, condor_history)

            if status == "completed":
                status_counts["completed"] += 1
            elif status == "running":
                status_counts["running"] += 1
            elif status in ("queued", "submitted"):
                status_counts["queued"] += 1
            elif status == "failed":
                status_counts["failed"] += 1
                resubmit_count = job.get("resubmit_count", 0)
                if resubmit_count < max_resubmits:
                    jobs_to_resubmit.append((job_id, job))
                else:
                    permanently_failed.append(job_id)
            else:
                status_counts["other"] += 1

        total_jobs = len(jobs)

        # Check if all done
        if status_counts["completed"] == total_jobs:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All {total_jobs} jobs completed successfully!")
            break

        # Check if all remaining are permanently failed
        active_jobs = status_counts["running"] + status_counts["queued"] + len(jobs_to_resubmit)
        if active_jobs == 0 and permanently_failed:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All remaining jobs have permanently failed (exceeded max resubmits).")
            print(f"Permanently failed jobs: {len(permanently_failed)}")
            break

        # Resubmit failed jobs
        if jobs_to_resubmit:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Resubmitting {len(jobs_to_resubmit)} failed job(s)...")

            for job_id, job in jobs_to_resubmit:
                job_dir = Path(job.get("job_dir", ""))
                submit_file = job_dir / "submit.cmd"

                if not submit_file.exists():
                    print(f"  ERROR: Submit file not found for {job_id}")
                    continue

                result = subprocess.run(
                    ["condor_submit", str(submit_file)],
                    capture_output=True, text=True
                )

                if result.returncode != 0:
                    print(f"  ERROR submitting {job_id}: {result.stderr}")
                    continue

                # Extract cluster ID
                cluster_id = extract_cluster_id(result.stdout)
                resubmit_count = job.get("resubmit_count", 0) + 1

                # Update manifest
                manifest.data["jobs"][job_id]["cluster_id"] = cluster_id
                manifest.data["jobs"][job_id]["status"] = "submitted"
                manifest.data["jobs"][job_id]["submit_time"] = datetime.now().isoformat()
                manifest.data["jobs"][job_id]["resubmit_count"] = resubmit_count

                print(f"  {job_id}: resubmitted (attempt {resubmit_count}/{max_resubmits}, cluster {cluster_id})")

            # Save manifest after resubmissions
            manifest.save()

        # Sleep until next check
        time.sleep(interval_minutes * 60)

    # Save final status to manifest
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving final job statuses to manifest...")
    jobs = manifest.data.get("jobs", {})
    condor_jobs, _ = get_condor_jobs()
    cluster_ids = [j["cluster_id"] for j in jobs.values() if j.get("cluster_id") is not None]
    missing_ids = [cid for cid in cluster_ids if cid not in condor_jobs]
    condor_history, _ = get_condor_history(missing_ids) if missing_ids else ({}, None)

    for job_id, job in jobs.items():
        final_status = determine_job_status(job, condor_jobs, condor_history)
        manifest.data["jobs"][job_id]["status"] = final_status

    manifest.data["monitor_ended"] = datetime.now().isoformat()
    manifest.save()
    print(f"Manifest updated: {task_dir}/manifest.json")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    if args.btag_eff:
        if args.sample:
            print("ERROR: --sample cannot be used with --btag-eff because the final payload requires every configured sample.")
        else:
            print("ERROR: Direct Condor --btag-eff production is unsupported; use the year-pure Slurm workflow.")
        sys.exit(1)
    user = get_user()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    preselection_dir = script_dir.parent
    config_path = preselection_dir / args.config

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    samples = config.get("samples", {})
    if not samples:
        print("ERROR: No samples found in config")
        sys.exit(1)

    # Filter samples if --sample specified
    if args.sample:
        pattern = re.compile(args.sample)
        samples = {k: v for k, v in samples.items() if pattern.search(k)}
        if not samples:
            print(f"ERROR: No samples match pattern '{args.sample}'")
            sys.exit(1)

    # Generate job output name
    config_basename = Path(args.config).stem
    job_name = f"{config_basename}_{args.analysis}"
    if args.tag:
        job_name = f"{job_name}_{args.tag}"

    # Create task directory
    task_dir = preselection_dir / "condor" / CONDOR_OUTPUT_DIR / job_name
    if task_dir.exists():
        print(f"ERROR: Task directory already exists: {task_dir}")
        print("Please remove it or use a different --tag")
        sys.exit(1)

    task_dir.mkdir(parents=True)

    # Initialize job manifest
    manifest = JobManifest(task_dir)
    manifest.set_metadata(args.config, args.analysis, args.run_number, user)

    print(f"\n{'='*60}")
    print(f"Condor Job Submission for run3-vbsvvh")
    print(f"{'='*60}")
    print(f"Config:       {args.config}")
    print(f"Analysis:     {args.analysis}")
    print(f"Run number:   {args.run_number}")
    print(f"Tag:          {args.tag or '(none)'}")
    # Load per-sample events-per-job overrides
    epj_overrides = None
    if args.events_per_job_override:
        override_path = preselection_dir / args.events_per_job_override
        if not override_path.exists():
            print(f"ERROR: Override file not found: {override_path}")
            sys.exit(1)
        epj_overrides = load_events_per_job_overrides(str(override_path))
        print(f"Loaded {len(epj_overrides)} events-per-job override(s) from {args.events_per_job_override}")

    if args.events_per_job:
        print(f"Events/job:   {args.events_per_job} (max, default)")
    else:
        print(f"Files/job:    {args.files_per_job}")
    print(f"CPUs/job:     {args.ncpus}")
    print(f"Memory/job:   {args.memory}")
    print(f"Task dir:     {task_dir}")
    print(f"{'='*60}\n")

    # Create tarball
    print("Creating tarball...")
    create_tarball(preselection_dir)

    # Copy tarball and executable to task dir
    subprocess.run(["cp", str(preselection_dir / "condor" / "package.tar.gz"),
                    str(task_dir / "package.tar.gz")], check=True)
    subprocess.run(["cp", str(script_dir / "executable.sh"),
                    str(task_dir / "executable.sh")], check=True)

    # Process each sample
    submit_files = []
    total_jobs = 0

    for sample_name, sample_config in samples.items():
        # Expand file globs to get actual files
        all_files = []
        for file_pattern in sample_config["files"]:
            expanded = expand_file_glob(file_pattern)
            if not expanded:
                print(f"  WARNING: No files found for pattern: {file_pattern}")
            all_files.extend(expanded)

        if not all_files:
            print(f"  Skipping {sample_name}: no files found")
            continue

        # Split files into chunks - by events or by file count
        sample_epj = get_events_per_job(sample_name, args.events_per_job, epj_overrides)
        if sample_epj:
            events_per_file = estimate_events_per_file(all_files)
            total_events_est = events_per_file * len(all_files)
            file_chunks = split_by_events(all_files, events_per_file, sample_epj)
            n_jobs = len(file_chunks)
            files_per_job = len(all_files) // n_jobs if n_jobs > 0 else 0
            override_tag = " (override)" if sample_epj != args.events_per_job else ""
            print(f"  {sample_name}: {len(all_files)} files, ~{total_events_est} events ({events_per_file}/file) -> {n_jobs} job(s) ({files_per_job} files/job, {sample_epj} evts/job{override_tag})")
        else:
            file_chunks = split_into_chunks(all_files, args.files_per_job)
            n_jobs = len(file_chunks)
            print(f"  {sample_name}: {len(all_files)} files -> {n_jobs} job(s)")

        # Create job configs and submit files
        for job_idx, file_chunk in enumerate(file_chunks):
            # Create per-job directory
            job_dir_name = f"{sample_name}_{job_idx}" if n_jobs > 1 else sample_name
            job_dir = task_dir / job_dir_name
            job_dir.mkdir(parents=True)

            # Create job-specific config
            job_config = create_job_config(sample_name, sample_config, file_chunk, job_idx)
            config_file = job_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(job_config, f, indent=2)

            # Generate submit file
            submit_file = generate_submit_file(
                task_dir, job_dir, job_name, args,
                sample_name, job_idx
            )
            submit_files.append((submit_file, job_dir_name))

            # Add job to manifest
            manifest.add_job(
                job_id=job_dir_name,
                sample_name=sample_name,
                job_idx=job_idx,
                input_files=file_chunk,
                job_dir=str(job_dir)
            )
            total_jobs += 1

    # Save manifest
    manifest.save()

    print(f"\n{'='*60}")
    print(f"Total jobs to submit: {total_jobs}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN: Jobs prepared but not submitted")
        print(f"Job directories are in: {task_dir}")
        print(f"Manifest saved to: {task_dir}/manifest.json")
        print("\nTo submit manually, run:")
        print(f"  for f in {task_dir}/*/submit.cmd; do condor_submit $f; done")
        print("\nTo submit a single job:")
        print(f"  condor_submit {task_dir}/<sample_name>/submit.cmd")
        print("\nTo check status:")
        print(f"  python condor/status.py --task {job_name}")
        return

    # Submit all jobs
    print("Submitting jobs...")
    submitted_count = 0
    failed_count = 0

    for submit_file, job_id in submit_files:
        result = subprocess.run(["condor_submit", str(submit_file)],
                               capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR submitting {job_id}: {result.stderr}")
            failed_count += 1
        else:
            cluster_id = extract_cluster_id(result.stdout)
            if cluster_id:
                manifest.mark_submitted(job_id, cluster_id)
                print(f"  {job_id}: submitted to cluster {cluster_id}")
                submitted_count += 1
            else:
                print(f"  {job_id}: submitted (couldn't parse cluster ID)")
                submitted_count += 1

    # Save updated manifest with cluster IDs
    manifest.save()

    print(f"\n{'='*60}")
    print(f"Submitted: {submitted_count}, Failed: {failed_count}")
    print(f"{'='*60}")
    print(f"\nManifest saved to: {task_dir}/manifest.json")
    print(f"Job logs will be in each job's directory under: {task_dir}/")
    print(f"Output will be staged to: {OUTPUT_XRD.format(user=user)}/{job_name}/")
    print(f"\nTo check status:")
    print(f"  python condor/status.py --task {job_name}")

    # Start monitoring if requested
    if args.monitor:
        monitor_and_resubmit(
            task_dir=task_dir,
            manifest=manifest,
            max_resubmits=args.max_resubmits,
            timeout_hours=args.timeout,
            interval_minutes=args.interval
        )


if __name__ == "__main__":
    main()
