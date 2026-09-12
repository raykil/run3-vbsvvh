import os
import uproot
import json
import numpy as np

import xsec_ref
import dataset_names_ref


# Dictionary of the paths to the skim sets
SKIM_PATH_DICT = {
    "all_events" : {
        ("run2", "sig")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Sig_v15_v30_Sig",
        ("run3", "sig")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Sig_v15_v30_Sig",
    },
    "0lep_0FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_0Lep0FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_0Lep0FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_0Lep0FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_0Lep0FJ",
    },
    "0lep_1FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_0Lep1FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_0Lep1FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_0Lep1FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_0Lep1FJ",
    },
    "0lep_2FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_0Lep2FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_0Lep2FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_0Lep2FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_0Lep2FJ",
    },
    "0lep_3FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_0Lep3FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_0Lep3FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_0Lep3FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_0Lep3FJ",
    },
    "1lep_1FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_1Lep1FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_1Lep1FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_1Lep1FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_1Lep1FJ",
    },
    "2lep_1FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_2Lep1FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_2Lep1FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_2Lep1FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_2Lep1FJ",
    },
    "2lep_2FJ" : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_2Lep2FJ",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_2Lep2FJ",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_2Lep2FJ",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_2Lep2FJ",
    },
    "2lep_SS" : {
        #("run2", "bkg")  : "",
        #("run2", "data") : "",
        #("run3", "bkg")  : "",
        #("run3", "data") : "",
    },
    "3lep"    : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_3Lep",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_3Lep",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_3Lep",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_3Lep",
    },
    "4lep"    : {
        ("run2", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Bkg_v15_v30_4Lep",
        ("run2", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run2_Data_v15_v30_4Lep",
        ("run3", "bkg")  : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Bkg_v15_v30_4Lep",
        ("run3", "data") : "/cmsuf/data/store/user/phchang/skim/VBSVVH_skim_v30/Run3_Data_v15_v30_4Lep",
    },
}

LUMI_DICT = {
    "2016preVFP": 19.52,
    "2016postVFP": 16.81,
    "2017": 41.48,
    "2018": 59.83,

    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
    "2022": 34.76,
    "2023": 28.28,
    "2024Prompt": 109.95,
    "2025": 110.84
}


################# Helper functions #################

# Get the list of files in a given dir
#     - Input should be a full path to the given dir
#     - Return the list of files joined with the full path
def get_file_lst(path_to_dir,ftype="root"):
    out_lst = []
    file_name_lst = os.listdir(path_to_dir)
    for fname in file_name_lst:
        if not fname.endswith("."+ftype): continue
        fname_fullpath = os.path.join(path_to_dir,fname)
        out_lst.append(fname_fullpath)
    out_lst.sort()
    return out_lst


# Match a sample name to an xsec in the given list
# Return the name of the xsec and the value
def match_xsec(dataset_name,xsec_dict):
    n_matches = 0
    for xsec_name in xsec_dict:
        if dataset_name.startswith(xsec_name):
            dataset_name_short = xsec_name
            dataset_xsec = xsec_dict[xsec_name]
            n_matches = n_matches + 1

    # Throw and error if we did not find a match
    if n_matches == 0:
        raise Exception(f"Failed to find xsec match for sample: \"{dataset_name}\"")
    if n_matches > 1:
        raise Exception(f"Found more than one matching xsec for sample: \"{dataset_name}\"")

    return [dataset_name_short,dataset_xsec]


# Strip everything before the a given string in all paths in a list
# Assumes all paths are the same up to the given string
def strip_prefixes(fullpaths_lst,split_on="VBSVVH_skim_"):
    out_lst =[]
    for fullpath in fullpaths_lst:
        before,after = fullpath.split(split_on)
        after = split_on+after
        out_lst.append(after)
    return [before, out_lst]


# Rebuild the runs_summary dict for one skim file from its Runs tree. The skimmer's
# runs_summary_N.json is only a cache of the Runs tree of output_N.root, so the two agree
# exactly; this is the fallback when the json cannot be read.
def runs_summary_from_root(root_fpath):
    with uproot.open(root_fpath) as f:
        runs = f["Runs"]
        out_dict = {"eventCount": float(f["Events"].num_entries)}
        for branch in ["genEventCount", "genEventSumw", "genEventSumw2"]:
            out_dict[branch] = float(runs[branch].array(library="np").sum())
        for branch in ["LHEScaleSumw", "LHEPdfSumw", "PSSumw"]:
            if branch in runs:
                out_dict[branch] = [float(x) for x in np.sum(runs[branch].array(library="np"), axis=0)]
    return out_dict


# Sum the runs_summary jsons (produced by the skimmer) for all the files in this dataset
#     - If a json cannot be read, fall back to the Runs tree of the matching root file.
#     - If that fails too, abort: a missing file would silently shrink sumw while its
#       events stay in the file list, i.e. the whole sample would be overweighted.
def sum_runs_summaries(lst_of_file_metadata_jsons,dataset_fullpath):
    out_dict = {}
    for file_metadata_json in lst_of_file_metadata_jsons:
        fpath = os.path.join(dataset_fullpath,file_metadata_json)
        if os.access(fpath, os.R_OK):
            with open(fpath) as jf:
                file_metadata_dict = json.load(jf)
        else:
            root_fpath = fpath.replace("runs_summary_", "output_").replace(".json", ".root")
            print(f"  WARNING: cannot read {fpath}, reading the Runs tree of {os.path.basename(root_fpath)} instead")
            try:
                file_metadata_dict = runs_summary_from_root(root_fpath)
            except Exception as e:
                raise Exception(f"Cannot get the sum of weights for {fpath}: the json is unreadable "
                                f"and reading {root_fpath} failed ({e}). Refusing to write a json with "
                                f"an incomplete sumw.") from e
        # Sum the values in the dict
        # Assumes all keys are the same
        # Assumes all vals are either float or list of floats
        for k,v in file_metadata_dict.items():
            if k not in out_dict:
                out_dict[k] = v
            else:
                # If this is a list, sum with the list we have
                if isinstance(v, list):
                    summed_arr = np.array(out_dict[k]) + np.array(v)
                    out_dict[k] = list(summed_arr)
                # Otherwise assume this is just a number
                else:
                    out_dict[k] += float(v)
    return out_dict


# Given a path to dir of jsons, pull all the ds names from all jsons in the dir and put them into a list
def get_ds_names_from_jsons(path_to_jsons):
    ds_names_lst = []
    json_names = os.listdir(path_to_jsons)
    for json_name in json_names:
        with open(os.path.join(path_to_jsons,json_name)) as jf:
            file_metadata_dict = json.load(jf)
            ds_names_lst.append(list(file_metadata_dict["samples"].keys())[0])
    return(ds_names_lst)



################# Main wrapper for producing json #################

# Create the dict to dump to the json for a given sample
#     - dataset_info should be from the dataset ref file (something like {'year': '2016preVFP', 'dataset_name': 'VBSWWH_OS_c2v1p0_c3_1p0_UL16APV'})
#     - path should be absolute
#     - kind should be e.g. "bkg"
#     - xsec_dict should be from the xsec ref file for this kind
def make_json_for_dataset(dataset_info, path, kind, xsec_dict, skim_set_name, run_tag):
    out_dict = {}

    # Get year for this dataset
    year = dataset_info["year"]

    # Get the lumi for this year
    lumi = LUMI_DICT[year]

    # Get name of this dataset
    dataset_name = dataset_info["dataset_name"]

    # Get the xsec name and value for this dataset
    if kind == "data":
        dataset_name_short, xsec_val = dataset_name, 1.0
    else:
        dataset_name_short, xsec_val = match_xsec(dataset_name,xsec_dict)

    # Get full paths to all root and json files for this dataset
    dataset_fullpath = os.path.join(path,dataset_name)
    file_fullpath_lst = get_file_lst(dataset_fullpath,ftype="root")
    json_fullpath_lst = get_file_lst(dataset_fullpath,ftype="json")

    # Get rid of the local prefix
    local_prefix, file_fullpath_lst = strip_prefixes(file_fullpath_lst)

    # Check if dataset needs and ewk correction (note RDF metadata cannot be bool so pass int)
    do_ewk_corr = 0
    if dataset_name in dataset_names_ref.datasets_for_ewk_corr: do_ewk_corr = 1

    # What name to include for the metadata
    if kind == "data":
        name_for_metadata = dataset_name_short.split("_")[0] # For data just keep e.g. "DoubleEG"
        if name_for_metadata.endswith("0") or name_for_metadata.endswith("1") or name_for_metadata.endswith("2") or name_for_metadata.endswith("3"):
            # Run 3 dataset names include e.g. Muon, Muon0, Muon1
            name_for_metadata = name_for_metadata[:-1]
    else: name_for_metadata = dataset_name_short

    # Get the sum of weights for all of the files in this dataset and build metadata dict
    #metadata_dict = sum_runs_summaries(json_fullpath_lst, dataset_fullpath) # No good for now, since RDF cannot read lists
    metadata_dict = {}
    metadata_dict["kind"] = kind
    metadata_dict["year"] = year
    metadata_dict["xsec"] = xsec_val
    metadata_dict["lumi"] = lumi
    metadata_dict["shortname"] = name_for_metadata
    metadata_dict["do_ewk_corr"] = do_ewk_corr
    metadata_dict["local_prefix"] = local_prefix
    if kind == "data":
        # RDF expects to read something for sumw, even if not used
        metadata_dict["sumw"] = float(0)
    else:
        # Otherwise grab the sumw from the runs summary dict
        metadata_dict["sumw"] = float(sum_runs_summaries(json_fullpath_lst, dataset_fullpath)["genEventSumw"])

    # Fill the out dict
    out_dict["trees"] = ["Events"]
    out_dict["files"] = file_fullpath_lst
    out_dict["metadata"] = metadata_dict

    # Return the output path and content; main() writes them once the whole skim set has
    # been processed, so an abort part-way through never leaves a skim set half updated.
    # (Split by run so Run 2 and Run 3 JSONs never co-locate.)
    out_path = f"input_sample_jsons/{run_tag}/{kind}/{skim_set_name}/{year}_{dataset_name_short}.json"
    return out_path, {"samples": {dataset_name: out_dict}}




################# Main function #################

def main():

    # Loop over the skim sets (e.g., 3lep)
    for skim_set_name in SKIM_PATH_DICT:
        print(f"\n################## Skim set: {skim_set_name} ##################")
        #if skim_set_name not in ["0lep_1FJ", "0lep_2FJ"]: continue

        # Jsons for this skim set, written only once every dataset in it has succeeded
        jsons_to_write = []

        # Loop over the kinds of samples for each skim (e.g., bkg)
        for run_tag,kind in SKIM_PATH_DICT[skim_set_name]:

            # Get the known xsecs for this kind (sig, data, bkg) of sample
            known_datasets_lst = dataset_names_ref.datasets[(run_tag,kind)]
            xsec_dict = xsec_ref.xsec_dict[run_tag][kind]

            # Get the list of datasets we actaully have at this skim path, and list we expect
            path_to_skims = SKIM_PATH_DICT[skim_set_name][(run_tag,kind)]
            paths_to_current_jsons = f"input_sample_jsons/{run_tag}/{kind}/{skim_set_name}"
            havejson_ds_set = set(get_ds_names_from_jsons(paths_to_current_jsons))
            haveskim_ds_set = set(os.listdir(path_to_skims))
            haveref_ds_set  = set(d["dataset_name"] for d in known_datasets_lst)

            # Do some checks of the list of skims we have against what we expect
            print(f"\nChecking datasets for {run_tag} {kind}...")
            have_skim_but_not_ref  = haveskim_ds_set.difference(haveref_ds_set)
            have_json_but_not_skim = havejson_ds_set.difference(haveskim_ds_set)
            if len(have_skim_but_not_ref)>0:
                print("WARNING: Skim includes these samples, but reference info is not in the dataset_names_ref.py, so cannot make a json.")
                for ds in have_skim_but_not_ref: print(f"\t{ds}")
            if len(have_json_but_not_skim)>0:
                print("WARNING: We already have jsons for these samples, but this skim does not seem to include them. Is that expected?")
                for ds in have_json_but_not_skim: print(f"\t{ds}")


            # Form the list of datasets we will actually make jsons for
            ds_names_to_make_jsons_for = list(haveskim_ds_set.intersection(haveref_ds_set))
            datasets_lst = []
            for ds_dict in known_datasets_lst:
                if ds_dict["dataset_name"] in ds_names_to_make_jsons_for:
                    # Hardcoded skipping of double lepton datasets for single lep channel (since we make skims for them but don't want to use them)
                    if skim_set_name == "1lep_1FJ" and ds_dict["dataset_name"].startswith("MuonEG"): continue
                    if skim_set_name == "1lep_1FJ" and ds_dict["dataset_name"].startswith("DoubleMuon"): continue
                    if skim_set_name == "1lep_1FJ" and ds_dict["dataset_name"].startswith("DoubleEG"): continue
                    # Otherwise append to the list we want to make jsons for
                    datasets_lst.append(ds_dict)

            # Loop over all of the datasets and build up a dict we will dump to json
            print(f"\n{run_tag} {kind}: {len(datasets_lst)} total datasets.")
            for i,dataset_info in enumerate(datasets_lst):
                print(f"{i+1}/{len(datasets_lst)}: {dataset_info['dataset_name']}")

                # Build the output json
                jsons_to_write.append(make_json_for_dataset(dataset_info, path_to_skims, kind, xsec_dict, skim_set_name, run_tag))

        # This skim set is complete: write all of its jsons
        print(f"\nWriting {len(jsons_to_write)} jsons for skim set {skim_set_name}")
        for out_path, content in jsons_to_write:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fp:
                json.dump(content, fp, indent=4)




main()
