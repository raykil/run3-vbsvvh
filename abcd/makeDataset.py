"""
Creates abcd/dataset/{signal}/{tag}_{train/valid}.pt, as well as .parqs in same dir.
"""

from TrainingTools import *
import re, time, yaml, json, glob, warnings, uproot
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore", message=".*reduce_op.*")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_expression_variables(expression):
    tokens = re.findall(r"(?<!\.)\b[A-Za-z_][A-Za-z0-9_]*\b", expression)
    excluded = {"np", "abs", "and", "or", "not", "True", "False"}
    return [tok for tok in tokens if tok not in excluded]

def parse_training_features(cfg, flavor):
    # ——————— Fetching features/transforms ————————————————————————————
    feature_transform_dict = cfg['training_features'] # list of dicts
    split_features = tuple(cfg['split_features'])
    features, transforms   = [], {}
    for f in feature_transform_dict:
        for feature, transform in f.items():
            features.append(feature)
            is_split = feature.startswith(split_features)
            if is_split: suffixes = ("_1", "_2")
            else       : suffixes = ("",)
            transforms.update({feature+s: transform for s in suffixes})

    # ——————— Fetching constraint —————————————————————————————————————
    constraint = cfg['constraint_var']
    if flavor == "double" and constraint not in features:
        features.append(constraint)
        transforms[constraint] = cfg.get("constraint_as_feature_transform", "none")

    return features, transforms, constraint, split_features

def parse_derived_vars(cfg, features, transforms):
    # ——————— Fetching derived vars ———————————————————————————————————
    dv_dict = cfg['derived_vars']
    derivedVars = set(dv_dict.keys())
    input_for_derivedVars = []
    for variable in dv_dict.values():
        input_for_derivedVars.extend(var for var in extract_expression_variables(variable) if var not in derivedVars)
    input_for_derivedVars = list(dict.fromkeys(input_for_derivedVars))

    # ——————— Adding derived vars to features (optional) ——————————————
    include_dv = cfg.get("auto_include_derived_vars", False)
    dvt = cfg.get("auto_include_derived_vars_transform", "none")
    if include_dv:
        to_add = [name for name in derivedVars if name not in features]
        features.extend(to_add)
        transforms.update({name: dvt for name in to_add})

    return dv_dict, derivedVars, input_for_derivedVars, features, transforms

def auto_include_derived_vars(cfg, features, transforms, derived_vars_cfg):
    if not cfg.get("auto_include_derived_vars", False):
        return
    auto_tf = cfg.get("auto_include_derived_vars_transform", "none")
    added = [name for name in derived_vars_cfg if name not in features]
    features.extend(added)
    transforms.update({name: auto_tf for name in added})
    print(f"Auto-include derived vars enabled: added {len(added)} derived features with transform='{auto_tf}'")

def apply_derived_vars(events, DerivedVars):
    for name, expression in DerivedVars.items():
        events[name] = eval(expression, locals={f: events[f] for f in events.fields})
    return events

def split_features(features, split_prefixes):
    split_features = []
    for feature in features:
        if feature.startswith(split_prefixes):
            split_features.extend([f"{feature}_1", f"{feature}_2"])
        else:
            split_features.append(feature)
    return split_features

def evaluate_preselection(events, expression):
    cut = eval(expression, {"__builtins__": {}, "np": np, "ak": ak}, {f: events[f] for f in events.fields})
    return events[cut]

def local_paths_from_json(jsonpath, base_path, kind='sig', signal="c2v1p5_c3_1p0"):
    with open(jsonpath, 'r') as f: samples = json.load(f)["samples"]
    names = [name for name, s in samples.items() if s["metadata"]["kind"] == kind and (kind != "sig" or s["metadata"]["shortname"].endswith(signal))]
    files = [f for name in names for f in sorted(glob.glob(os.path.join(base_path, name, "*.root"))) if os.path.getsize(f) > 0]
    return files

def LoadEvents(paths, features, split_prefixes, label):
    """Trying to migrate to awkward array"""
    # Adding idxs
    rejected = chooseOverlaps()
    paths = sorted((p for p in paths if Path(p).parent.name not in rejected), key=lambda p: Path(p).parent.name)
    arrays, samples = [], {}
    for file_idx, path in enumerate(paths):
        arr = uproot.open(f"{path}:Events").arrays(features, library="ak")
        arr["file_idx"] = file_idx
        arr["proc_idx"] = samples.setdefault(Path(path).parent.name, len(samples)) # increments per sample (e.g. DY=0, ttbar=1)
        arrays.append(arr)
    events = ak.concatenate(arrays)

    # Additional cutflow (before flatten/split, which assume the channel's fatjet count)
    events = events[events.weight>0]
    events = events[ak.num(events.fatjet_pt) == (2 if "fatjet_" in split_prefixes else 1)] # TEMP: drop JES/JER-variation leaks from preselection (nominal fatjet count != channel's)
    events = events[events.vbs_score != -999] # TEMP: -999 means nominal njet < 2; mirrors the njet >= 2 cut now in selections.cpp

    # Handling split_prefixes
    for field in events.fields:
        if field.startswith(split_prefixes):
            events[f"{field}_1"] = events[field][:,0]
            events[f"{field}_2"] = events[field][:,1]
            events = ak.without_field(events, field)
        elif events[field].ndim > 1:
            events[field] = ak.flatten(events[field])

    # Giving sig/bkg label for training
    events["label"] = int(label)

    return events, list(samples) # samples indexed by proc_idx

def log_transform(quantity):
    # NOTE: Negatives are rounded-m² artifacts, physically ~0. Clipping at 0 instead of -0.999
    # keeps log1p(-0.999)=-6.9 from dominating the min-max range that follows.
    return np.log1p(np.clip(quantity, a_min=0, a_max=None))

def minmax_transform(quantity):
    scaler = MinMaxScaler(feature_range=(0,1))
    quantity = np.asarray(quantity).reshape(-1,1)
    quantity = scaler.fit_transform(quantity).ravel()
    return quantity

def preprocess(events, features, transforms, constraint):
    features = [f for f in features if f in events.fields]
    if constraint not in features: transforms[constraint] = 'minmax'
    features += [constraint]

    for feature in features:
        quantity = events[feature]
        if transforms[feature] == "log":
            quantity = log_transform(quantity)
            quantity = minmax_transform(quantity)
        elif transforms[feature] == "minmax":
            quantity = minmax_transform(quantity)
        else:
            # NOTE: Transform: None also does minmax
            quantity = minmax_transform(quantity)
        events[feature] = quantity

    return events

def normalize_weights(events):
    events["weight"] = events.weight / sum(events.weight)
    return events

def makeDataLoaders(events, features, constraint, batch_size):
    labl_str = np.asarray(events.label).astype(str) # ['1' '1' '1' ... '0' '0' '0']
    proc_str = np.asarray(events.proc_idx).astype(str) # ['0' '0' '0' ... '1' '1' '1']
    stratify_key = np.char.add(np.char.add(labl_str, "_"), proc_str) # '{label}_{process}'
    unique_keys, unique_counts = np.unique(stratify_key, return_counts=True) # ['0_0'(bkg, DY) '0_1'(bkg, tt) '1_0'(sig, c2v1p5_c3_1p0)], [463 88812 410]
    top_key = {k[0]: k for k in unique_keys[np.argsort(unique_counts)]} # last wins: most populous key per label
    rare = np.isin(stratify_key, unique_keys[unique_counts < 2]) # single-event processes can't be split
    stratify_key[rare] = [top_key[l] for l in labl_str[rare]]
    all_indices = np.arange(len(events)) # [    0     1     2 ... 89682 89683 89684] len(events)=89685

    train_idx, valid_idx = train_test_split(
        all_indices, 
        test_size = 0.2,   # nValid / (nTrain + nValid)
        random_state = 42, # make it deterministic
        stratify = stratify_key
    )

    print(f"Feature column order: {dict(enumerate(features))}")
    feature_matrix = np.column_stack([ak.to_numpy(events[f]) for f in features]).astype(np.float32, copy=False) # row:events, col:features. Col order matches features order.
    constraint_values = np.asarray(events[constraint], dtype=np.float32).reshape(-1, 1)
    labels  = np.asarray(events.label , dtype=np.float32)
    weights = np.asarray(events.weight, dtype=np.float32)

    train_loader = get_dataloader(
        dnn_input_data  = torch.from_numpy(feature_matrix[train_idx]),
        constraint_data = torch.from_numpy(constraint_values[train_idx]),
        labels          = torch.from_numpy(labels[train_idx]),
        weights         = torch.from_numpy(weights[train_idx]),
        batch_size      = batch_size,
        use_sampler     = False
    )
    valid_loader = get_dataloader(
        dnn_input_data  = torch.from_numpy(feature_matrix[valid_idx]),
        constraint_data = torch.from_numpy(constraint_values[valid_idx]),
        labels          = torch.from_numpy(labels[valid_idx]),
        weights         = torch.from_numpy(weights[valid_idx]),
        batch_size      = batch_size,
        use_sampler     = False,
        is_validation   = True
    )
    return train_loader, valid_loader, train_idx, valid_idx

def saveParquet(events, procNames, train_idx, valid_idx, constraint, path):
    """ Rows ordered [train, valid] so they align with ConcatDataset(train.pt, valid.pt), letting the scan just assign ABCDscore. """
    order = np.concatenate([train_idx, valid_idx])
    labels = np.asarray(events.label)[order]
    procs  = np.asarray(events.proc_idx)[order]
    df = pd.DataFrame({
        "VBSscore"     : np.asarray(events.rawVBS   , dtype=np.float32)[order],
        "weight"       : np.asarray(events.rawWeight, dtype=np.float64)[order],
        "norm_VBSscore": np.asarray(events[constraint], dtype=np.float32)[order], # min-max scaled; matches disco in the .pt
        "norm_weight"  : np.asarray(events.weight   , dtype=np.float64)[order],   # class-normalized; sums to 1 per label
        "label"   : labels.astype(np.int8),
        "process" : [procNames[int(l)][p] for l, p in zip(labels, procs)], # proc_idx restarts at 0 per sig/bkg
        "split"   : np.where(np.arange(len(order)) < len(train_idx), "train", "valid"),
    })
    df.to_parquet(path, index=False)
    print(f"\033[1;32mSaved abcd/{path}!\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(epilog="Ex) python makeDataset.py --config single/run2_2L_1FJ.yaml --flavor single")
    parser.add_argument('-c', "--config"    , required=True      , help="Path to YAML config")
    parser.add_argument('-f', "--flavor"    , default="single"   , choices=["single", "double"], help="Training flavor: single (one output) or double (two outputs). Later prob should put this in config.")
    parser.add_argument('-s', "--signal"    , default="c2v1p5_c3_1p0", choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"], help="Signal coupling point (c2v1p0_c3_1p0 is SM)")
    args = parser.parse_args()

    # —————————— Load config ————————————————————————————————————————————————————————————
    print(f"\n—————————— Processing {Path(args.config).stem} ——————————")
    cfg = load_config(args.config)
    TrainingFeatures, FeatureTransforms, constraint, split_prefixes = parse_training_features(cfg, args.flavor)
    derived_vars_cfg, DerivedVars, input_for_derivedVars, TrainingFeatures, FeatureTransforms = parse_derived_vars(cfg, TrainingFeatures, FeatureTransforms)

    # —————————— Loading samples ———————————————————————————————————————————
    VarsToLoad = TrainingFeatures + [constraint] + input_for_derivedVars + cfg.get("extra_vars", [])
    unsplit = lambda f: f[:-2] if f.startswith(split_prefixes) and f.endswith(("_1", "_2")) else f
    derived_bases = {unsplit(d) for d in DerivedVars}
    VarsToLoad = list(dict.fromkeys(unsplit(f) for f in VarsToLoad if unsplit(f) not in derived_bases))
    new_load_start = time.time()
    sig_paths = local_paths_from_json(cfg["sample_json"], cfg["local_base_path"], "sig", args.signal)
    bkg_paths = local_paths_from_json(cfg["sample_json"], cfg["local_base_path"], "bkg")
    sig_events, sig_procs = LoadEvents(sig_paths, VarsToLoad, split_prefixes, 1)
    bkg_events, bkg_procs = LoadEvents(bkg_paths, VarsToLoad, split_prefixes, 0)
    print(f"Loaded sig+bkg in {time.time()-new_load_start:.1f} s")

    # —————————— Preselection cut ———————————————————————————————————————————
    i_sigW = sum(sig_events.weight) ; i_bkgW = sum(bkg_events.weight)
    print(f"\033[1mBefore preselection\033[0m: sig:{len(sig_events.weight)} (weighted: {i_sigW:.3f})  bkg:{len(bkg_events.weight)} (weighted: {i_bkgW:.3f})")
    if cfg.get("preselection"):
        sig_events = evaluate_preselection(sig_events, cfg["preselection"])
        bkg_events = evaluate_preselection(bkg_events, cfg["preselection"])
    sig_percent = round(sum(sig_events.weight)/i_sigW*100, 1) ; bkg_percent = round(sum(bkg_events.weight)/i_bkgW*100, 1)
    print(f"\033[1mAfter preselection\033[0m : sig:{len(sig_events)} (weighted: {sum(sig_events.weight):.3f}, {sig_percent}% survived)  bkg:{len(bkg_events)} (weighted: {sum(bkg_events.weight):.3f}, {bkg_percent}% survived)")

    for events in (sig_events, bkg_events): # kept for the ABCD scan, which needs real yields and uncscaled cut values
        events["rawWeight"] = events.weight
        events["rawVBS"]    = events[constraint]
    sig_events = normalize_weights(sig_events)
    bkg_events = normalize_weights(bkg_events)

    # —————————— Preprocess data ———————————————————————————————————————————
    data = ak.concatenate([sig_events, bkg_events])
    data = apply_derived_vars(data, derived_vars_cfg)
    TrainingFeatures = split_features(TrainingFeatures, split_prefixes)
    data = preprocess(data, TrainingFeatures, FeatureTransforms, constraint)

    # —————————— Make datasets ———————————————————————————————————————————
    train_loader, valid_loader, train_idx, valid_idx = makeDataLoaders(
        events     = data,
        features   = TrainingFeatures,
        constraint = constraint,
        batch_size = cfg.get("batch_size", 4096),
    )
    dataset_dir = f"{time.strftime('%y%m%d')}_dataset/{args.signal}"
    os.makedirs(dataset_dir, exist_ok=True)
    for split, loader in (("train", train_loader), ("valid", valid_loader)):
        datasetPath = f"{dataset_dir}/{Path(args.config).stem}_{split}.pt"
        torch.save(loader.dataset, datasetPath)
        print(f"\033[1;32mSaved abcd/{datasetPath}!\033[0m")
    saveParquet(data, [bkg_procs, sig_procs], train_idx, valid_idx, constraint, f"{dataset_dir}/{Path(args.config).stem}.parq")