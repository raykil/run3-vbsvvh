import os, sys, re, csv, time, yaml, json, glob, logging, warnings
import torch, uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
# from XRootD import client
# from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from pathlib import Path

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from TrainingTools import get_dataloader
from model import ABCDLightningModule
warnings.filterwarnings("ignore", message=".*reduce_op.*")

MISSING_VALUE = -999.0

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# def paths_from_json(jsonpath, base_path, kind="sig"):
#     with open(jsonpath, "r") as f:
#         data = json.load(f)
#         paths = []
#         for entry in data["samples"]:
#             if(str(data["samples"][entry]["metadata"]["kind"]) == kind):
#                 if kind == "sig" and "c2v1p5" in str(data["samples"][entry]["metadata"]["shortname"]):
#                     # only using c2v1p5, and not c2v1p0? Okay.
#                     paths.append(base_path+entry+'/')
#                 elif kind != "sig":
#                     paths.append(base_path+entry+'/')
#         files = []
#         for path in paths:
#             prefix, server, remote_path = path.split("//", 2)
#             server = prefix+'//'+server
#             xrdfs = client.FileSystem(server)
#             status, listing = xrdfs.dirlist("/" + remote_path, client.flags.DirListFlags.STAT)
#             if status.ok:
#                 for entry in listing:
#                     if not entry.name.endswith(".root"):
#                         continue
#                     if entry.statinfo is not None and entry.statinfo.size == 0:
#                         logging.warning("Skipping empty file %s", path + entry.name)
#                         continue
#                     files.append(os.path.join(path, entry.name))
#         if len(files)>0 and files[0].endswith('.root'): logging.info(f"{kind} files successfully loaded!")
#     return files

def resolve_paths(base, paths):
    if isinstance(paths, str):
        paths = [paths]
    resolved = []
    for p in paths:
        full = os.path.expanduser(os.path.join(p, base) if base else p)
        matches = sorted(glob.glob(full))
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(full)
    return resolved

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
    logging.info(f"Auto-include derived vars enabled: added {len(added)} derived features with transform='{auto_tf}'")


# def _normalize_derived_vars_cfg(derived_vars_cfg):
#     # just ensuring derived vars is a dict.
#     if derived_vars_cfg is None:
#         return {}
#     if isinstance(derived_vars_cfg, dict):
#         return dict(derived_vars_cfg)
#     if isinstance(derived_vars_cfg, list):
#         normalized = {}
#         for item in derived_vars_cfg:
#             if not isinstance(item, dict):
#                 raise ValueError("Each item in 'derived_vars' list must be a mapping of 'new_var: expression'.")
#             for key, value in item.items():
#                 normalized[key] = value
#         return normalized
#     raise ValueError("'derived_vars' must be either a mapping or a list of single-item mappings.")


# def _collect_derived_input_vars(derived_vars_cfg):
#     derived_names = set(derived_vars_cfg.keys())
#     needed = []
#     for expr in derived_vars_cfg.values():
#         if not isinstance(expr, str):
#             continue
#         for var in extract_expression_variables(expr):
#             if var not in derived_names:
#                 needed.append(var)
#     return derived_names, list(dict.fromkeys(needed))


# def apply_derived_vars(data, derived_vars_cfg):
#     # derived_vars_cfg = _normalize_derived_vars_cfg(derived_vars_cfg)
#     # if not derived_vars_cfg:
#     #     return data

#     out = {k: np.asarray(v).copy() for k, v in data.items()}
#     n_events = _data_length(out)

#     for new_var, expression in derived_vars_cfg.items():
#         if not isinstance(expression, str): raise ValueError(f"derived_vars['{new_var}'] must be a string expression")

#         local_dict = {k: np.asarray(v) for k, v in out.items()}
#         print(local_dict)
#         values = np.asarray(eval(expression, {"__builtins__": {}, "np": np, "abs": np.abs}, local_dict))
#         # try:
#         #     values = np.asarray(eval(expression, {"__builtins__": {}, "np": np, "abs": np.abs}, local_dict))
#         # except NameError as exc: raise ValueError(f"Failed to evaluate derived var '{new_var}': missing variable in expression '{expression}'") from exc
#         # except Exception as exc: raise ValueError(f"Failed to evaluate derived var '{new_var}' with expression '{expression}': {exc}") from exc

#         if values.ndim == 0:
#             values = np.full(n_events, values.item())

#         if len(values) != n_events:
#             raise ValueError(f"Derived var '{new_var}' has length {len(values)} but expected {n_events}")

#         out[new_var] = values
#         print("Computed derived var '%s' from config expression", new_var)

#     return out

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

# def _read_root_frame(path, branches, file_idx, proc_idx, split_branches):
#     with uproot.open(path) as root_file:
#         arrays = root_file["Events"].arrays(branches, library="np")

#     columns = {}
#     n_events = None
#     for branch in branches:
#         if branch not in arrays:
#             continue

#         values = np.asarray(arrays[branch])
#         if branch in split_branches:
#             values = np.asarray(values, dtype=object)
#             columns[branch + "_1"] = np.array([v[0] if len(v) > 0 else MISSING_VALUE for v in values])
#             columns[branch + "_2"] = np.array([v[1] if len(v) > 1 else MISSING_VALUE for v in values])
#             n_events = len(columns[branch + "_1"])
#         else:
#             values = np.asarray([_flatten_awkward_cell(v) for v in values])
#             columns[branch] = values
#             n_events = len(columns[branch])
#  #       if values.ndim == 1 and values.dtype != object:
#  #           clean_values = values
#  #       else:
#  #           flat_values = [_flatten_awkward_cell(v) for v in values]
#  #           clean_values = np.asarray(values, dtype=object)
#  #           clean_values = np.array([np.asarray(v, dtype=np.float64) for v in clean_values],
#  #                           dtype=object)

# #        if n_events is None:
# #            n_events = len(columns[branch])
# #        else:
# #            n_events = min(n_events, len(columns[branch]))

#  #       columns[branch] = clean_values

#     if n_events is None or n_events == 0:
#         return {
#             "file_idx": np.array([], dtype=np.int32),
#             "proc_idx": np.array([], dtype=np.int32),
#         }

#     trimmed = {name: np.asarray(vals)[:n_events] for name, vals in columns.items()}
#     trimmed["file_idx"] = np.full(n_events, file_idx, dtype=np.int32)
#     trimmed["proc_idx"] = np.full(n_events, proc_idx, dtype=np.int32)
#     return trimmed


def _flatten_awkward_cell(value):
    if np.isscalar(value):
        return value
    return value[0] if len(value) > 0 else MISSING_VALUE

def _data_length(data):
    if not data:
        return 0
    first_key = next(iter(data))
    return len(data[first_key])

def evaluate_preselection(events, expression):
    cut = eval(expression, {"__builtins__": {}, "np": np, "ak": ak}, {f: events[f] for f in events.fields})
    return events[cut]

def _safe_minmax_scale(values, valid_mask):
    scaled = np.zeros_like(values, dtype=np.float64)
    scaler = MinMaxScaler()
    scaled[valid_mask] = scaler.fit_transform(values[valid_mask].reshape(-1, 1)).ravel()
    return scaled

def local_paths_from_json(jsonpath, base_path, kind='sig'):
    with open(jsonpath, 'r') as f: samples = json.load(f)["samples"]
    names = [name for name, s in samples.items() if s["metadata"]["kind"] == kind and (kind != "sig" or "c2v1p5" in s["metadata"]["shortname"])]
    files = [f for name in names for f in sorted(glob.glob(os.path.join(base_path, name, "*.root"))) if os.path.getsize(f) > 0]
    return files

# def new_load_data(paths, features, split_prefixes) -> dict[str, np.ndarray]:
#     Paths = [f"{path}:Events" for path in paths]
#     data = uproot.concatenate(Paths, features, library="np")

#     for branch in [b for b in features if data[b].dtype == object]:
#         values = data.pop(branch)
#         suffixes = ("_1", "_2") if branch.startswith(split_prefixes) else ("",)
#         for i, s in enumerate(suffixes): # non-split branches keep only the leading object
#             data[branch + s] = np.array([v[i] if len(v) > i else MISSING_VALUE for v in values])

#     counts = [uproot.open(p).num_entries for p in Paths]
#     samples = sorted({Path(p).parent.name for p in paths})
#     data["file_idx"] = np.repeat(np.arange(len(paths)), counts).astype(np.int32) # increments per root file
#     data["proc_idx"] = np.repeat([samples.index(Path(p).parent.name) for p in paths], counts).astype(np.int32) # increments per sample (e.g. DY=0, ttbar=1)
#     return data

def LoadEvents(paths, features, split_prefixes, label):
    """Trying to migrate to awkward array"""
    # Adding idxs
    paths = sorted(paths, key=lambda p: Path(p).parent.name)
    arrays, samples = [], {}
    for file_idx, path in enumerate(paths):
        arr = uproot.open(f"{path}:Events").arrays(features, library="ak")
        arr["file_idx"] = file_idx
        arr["proc_idx"] = samples.setdefault(Path(path).parent.name, len(samples)) # increments per sample (e.g. DY=0, ttbar=1)
        arrays.append(arr)
    events = ak.concatenate(arrays)

    # Handling split_prefixes
    for field in events.fields:
        if field.startswith(split_prefixes):
            events[f"{field}_1"] = events[field][:,0]
            events[f"{field}_2"] = events[field][:,1]
            events = ak.without_field(events, field)
        elif events[field].ndim > 1:
            events[field] = ak.flatten(events[field]) # events[field] = ak.fill_none(ak.firsts(events[field]), MISSING_VALUE)

    # saving only positive weights
    events = events[events.weight>0]

    # Giving sig/bkg label for training
    events["label"] = int(label)

    # print(np.round(events['weight'], 5).tolist()[:5], ak.count_nonzero(events['weight']<0), len(events['weight'])) # no negative weights!!
    return events

# def load_data(paths, features, num_workers, split_prefixes):
#     branches = list(dict.fromkeys(features))
#     split_branches = {b for b in branches if b.startswith(split_prefixes)}

#     sample_names = []
#     for path in paths:
#         p = Path(path)
#         parent = p.parent.name
#         grandparent = p.parent.parent.name if p.parent.parent is not None else ""
#         sample_names.append(grandparent if parent.isdigit() and grandparent else parent)
#     sample_name_to_idx = {name: idx for idx, name in enumerate(sorted(set(sample_names)))}
#     indexed_paths = [
#         (file_idx, path, sample_name_to_idx[sample_name])
#         for file_idx, (path, sample_name) in enumerate(zip(paths, sample_names))
#     ]
#     chunks = [None] * len(indexed_paths)

#     if num_workers > 1 and len(indexed_paths) > 1:
#         max_workers = min(num_workers, len(indexed_paths))
#         with ThreadPoolExecutor(max_workers=max_workers) as pool:
#             futures = {
#                 pool.submit(_read_root_frame, path, branches, file_idx, proc_idx, split_branches): file_idx
#                 for file_idx, path, proc_idx in indexed_paths
#             }
#             for future in tqdm(as_completed(futures), total=len(futures), desc="Loading ROOT files"):
#                 file_idx = futures[future]
#                 chunks[file_idx] = future.result()
#     else:
#         for file_idx, path, proc_idx in tqdm(indexed_paths, total=len(indexed_paths), desc="Loading ROOT files"):
#             chunks[file_idx] = _read_root_frame(path, branches, file_idx, proc_idx, split_branches)

#     chunks = [chunk for chunk in chunks if chunk is not None]

#     data = {key: np.concatenate([c[key] for c in chunks if key in c]) for key in {key for c in chunks for key in c}}
#     print(f"Loaded {_data_length(data)} events from {len(paths)} files.")
#     if "weight" in data:
#         data = _apply_mask(data, np.asarray(data["weight"]) > 0)
#     return data

# def feature_length(data, feature):
#     arr = np.asarray(data[feature])

#     if arr.ndim == 1:
#         if arr.dtype == object:
#             lengths = [len(np.asarray(x).ravel()) for x in arr]
#             return lengths
#         return 1

#     return arr.shape[1]

def preprocess_data(data, training_features, feature_transforms, Constraint):
    out = {k: np.asarray(v).copy() for k, v in data.items()}
    cols_to_clean = list(dict.fromkeys(training_features + [Constraint]))
    present_cols = [col for col in cols_to_clean if col in out]

    # making sure of the type
    for col in present_cols:
        arr = np.asarray(out[col])
        if arr.dtype == object:
            arr = np.asarray([_flatten_awkward_cell(v) for v in arr], dtype=np.float64)
        else:
            arr = arr.astype(np.float64, copy=False)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        out[col] = np.nan_to_num(arr, nan=MISSING_VALUE, posinf=MISSING_VALUE, neginf=MISSING_VALUE)

    # 
    for feat in training_features:
        # transform log = log + minmax. Result is [0,1].
        feat_arr = np.asarray(out[feat], dtype=np.float64)
        valid = (feat_arr != MISSING_VALUE) & np.isfinite(feat_arr)
        transform = feature_transforms.get(feat, "none")

        if transform == "log":
            positive = valid & (feat_arr > 0)
            feat_arr[valid & ~positive] = MISSING_VALUE
            valid = positive
            feat_arr[valid] = np.log(feat_arr[valid])
        
        out[feat] = _safe_minmax_scale(feat_arr, valid)

    if Constraint not in training_features:
        constraint_arr = np.asarray(out[Constraint], dtype=np.float64)
        valid_constraint = constraint_arr != MISSING_VALUE
        out[Constraint] = _safe_minmax_scale(constraint_arr, valid_constraint)

    return out

def log_transform(quantity):
    # NOTE: Negatives are rounded-m² artifacts, physically ~0. Clipping at 0 instead of -0.999
    # keeps log1p(-0.999)=-6.9 from dominating the min-max range that follows.
    return np.log1p(np.clip(quantity, a_min=0, a_max=None))

# def _safe_minmax_scale(values, valid_mask):
#     scaled = np.zeros_like(values, dtype=np.float64)
#     scaler = MinMaxScaler()
#     scaled[valid_mask] = scaler.fit_transform(values[valid_mask].reshape(-1, 1)).ravel()
#     return scaled

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


# def normalize_class_weights(sig_events, bkg_events):
#     sig = {k: np.asarray(v).copy() for k, v in sig_events.items()}
#     bkg = {k: np.asarray(v).copy() for k, v in bkg_events.items()}
#     sig = sig_events ; bkg = bkg_events

#     sig_weight_sum = np.sum(sig["weight"])
#     bkg_weight_sum = np.sum(bkg["weight"])

#     sig["weight"] = sig["weight"] / sig_weight_sum
#     bkg["weight"] = bkg["weight"] / bkg_weight_sum
#     return sig, bkg

def normalize_weights(events):
    events["weight"] = events.weight / sum(events.weight)
    return events

# def make_dataloaders(data, training_features, Constraint, batch_size):
#     labels_str = np.asarray(data["label"]).astype(np.int32).astype(str)
#     stratify_col = "proc_idx" if "proc_idx" in data else "file_idx"
#     sample_str = np.asarray(data[stratify_col]).astype(np.int64).astype(str)
#     stratify_key = np.char.add(np.char.add(labels_str, "_"), sample_str)
#     logging.info("Using stratification column '%s' for train/val split", stratify_col)

#     unique_keys, unique_counts = np.unique(stratify_key, return_counts=True)
#     rare_keys = unique_keys[unique_counts < 2]
#     if len(rare_keys) > 0:
#         rare_mask = np.isin(stratify_key, rare_keys)
#         stratify_key[rare_mask] = np.char.add(labels_str[rare_mask], "_rare")
#         logging.info(f"Collapsed {len(rare_keys)} rare strata (<2 events) into label-level rare bins for stable splitting")

#     all_indices = np.arange(_data_length(data))

#     train_idx, val_idx = train_test_split(
#         all_indices,
#         test_size=0.2,
#         random_state=42,
#         stratify=stratify_key,
#     )

#     logging.info(f"Feature column order: {dict(enumerate(training_features))}")
#     feature_matrix = np.column_stack([data[f] for f in training_features]).astype(np.float32, copy=False)
#     constraint_values = np.asarray(data[Constraint], dtype=np.float32).reshape(-1, 1)
#     labels = np.asarray(data["label"], dtype=np.float32)
#     weights = np.asarray(data["weight"], dtype=np.float32)

#     train_loader = get_dataloader(
#         dnn_input_data=torch.from_numpy(feature_matrix[train_idx]),
#         constraint_data=torch.from_numpy(constraint_values[train_idx]),
#         labels=torch.from_numpy(labels[train_idx]),
#         weights=torch.from_numpy(weights[train_idx]),
#         batch_size=batch_size,
#         use_sampler=False,
#     )

#     val_loader = get_dataloader(
#         dnn_input_data=torch.from_numpy(feature_matrix[val_idx]),
#         constraint_data=torch.from_numpy(constraint_values[val_idx]),
#         labels=torch.from_numpy(labels[val_idx]),
#         weights=torch.from_numpy(weights[val_idx]),
#         batch_size=batch_size,
#         use_sampler=False,
#         is_validation=True,
#     )

#     return train_loader, val_loader

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
    return train_loader, valid_loader


def run_training(
    model,
    train_loader,
    val_loader,
    output_dir,
    max_epochs,
    flavor,
    devices=[0],
    check_val_every_n_epoch=1,
    early_stopping_patience=20,
    early_stopping_min_delta=1e-4,
):
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(run_dir), name=flavor)

    checkpoint = ModelCheckpoint(
        dirpath=str(Path(logger.log_dir) / "checkpoints"),
        filename=f"{model.flavor}-abcdisco-{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint, lr_monitor]
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices if use_gpu else None,
        logger=logger,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer


def _concat_sig_bkg(sig_events, bkg_events):
    column_order = list(sig_events.keys())
    for key in bkg_events.keys():
        if key not in sig_events:
            column_order.append(key)

    combined = {}
    for key in column_order:
        sig_col = np.asarray(sig_events.get(key, np.array([])))
        bkg_col = np.asarray(bkg_events.get(key, np.array([])))
        combined[key] = np.concatenate([sig_col, bkg_col])

    n_sig = _data_length(sig_events)
    n_bkg = _data_length(bkg_events)
    combined["label"] = np.concatenate(
        [
            np.ones(n_sig, dtype=np.int32),
            np.zeros(n_bkg, dtype=np.int32),
        ]
    )
    return combined


def _write_csv_numpy(data, output_csv):
    columns = list(data.keys())
    arrays = [np.asarray(data[col]) for col in columns]

    if not arrays:
        Path(output_csv).write_text("\n", encoding="utf-8")
        return

    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All output columns must have the same length, got lengths={sorted(lengths)}")

    all_numeric = all(np.issubdtype(arr.dtype, np.number) for arr in arrays)
    if all_numeric:
        matrix = np.column_stack(arrays)
        np.savetxt(
            output_csv,
            matrix,
            delimiter=",",
            header=",".join(columns),
            comments="",
            fmt="%.10g",
        )
        return

    def _format_cell(val):
        if isinstance(val, (np.floating, float)):
            return f"{float(val):.10g}"
        if isinstance(val, (np.integer, int)):
            return str(int(val))
        if isinstance(val, (np.bool_, bool)):
            return str(int(val))
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    output_path = Path(output_csv)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in zip(*arrays):
            writer.writerow([_format_cell(v) for v in row])


def _latest_version_dir(output_dir: Path, flavor) -> Path:
    logs_dir = output_dir / flavor
    version_dirs = [p for p in logs_dir.glob("version_*") if p.is_dir()]
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directories found under {logs_dir}")

    def _version_key(path: Path):
        suffix = path.name.replace("version_", "")
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, path.stat().st_mtime)

    return sorted(version_dirs, key=_version_key)[-1]


def _latest_checkpoint_from_config(cfg, flavor) -> Path:
    output_dir = Path(cfg.get("output", "simple_abcdisco_output"))
    version_dir = _latest_version_dir(output_dir, flavor=flavor)
    ckpt_dir = version_dir / "checkpoints"
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]


def _inference_output_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _build_model_from_config(cfg, flavor, input_size, checkpoint_path, device):
    model = ABCDLightningModule(
        input_size=input_size,
        hidden_layers=cfg.get("architecture", [64, 32, 16]),
        learning_rate=cfg.get("learning_rate", 1e-3),
        bce_weight=cfg.get("bce_weight", 1.0),
        disco_lambda=cfg.get("disco_lambda", 0.0),
        flavor=flavor,
        use_batchnorm=cfg.get("use_batchnorm", True),
        dropout=cfg.get("dropout", 0.0),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()
    return model


def _batched_scores(model, features_tensor, batch_size, flavor, device):
    scores_0 = []
    scores_1 = []

    with torch.no_grad():
        for start in tqdm(range(0, len(features_tensor), batch_size), desc="Inferring events"):
            batch = features_tensor[start:start + batch_size].to(device)
            logits = model(batch)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores_0.append(probs[:, 0])
            if flavor == "double":
                scores_1.append(probs[:, 1])

    out_0 = np.concatenate(scores_0) if scores_0 else np.array([], dtype=np.float32)
    out_1 = np.concatenate(scores_1) if scores_1 else np.array([], dtype=np.float32)
    return out_0, out_1


def _plot_roc_curves(data, flavor, output_path):
    labels = np.asarray(data["label"])
    weights = np.asarray(data["weight"]) if "weight" in data else None

    plt.figure(figsize=(8, 7))

    roc_series = [
        ("DNN", "dnn_score")
    ] if flavor == "single" else [
        ("DNN 0", "dnn_0_score"),
        ("DNN 1", "dnn_1_score"),
    ]

    for label_name, score_col in roc_series:
        fpr, tpr, _ = roc_curve(labels, np.asarray(data[score_col]), sample_weight=weights)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label_name} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({flavor} flavor)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_score_densities(data, flavor, output_path):
    score_cols = ["dnn_score"] if flavor == "single" else ["dnn_0_score", "dnn_1_score"]
    fig, axes = plt.subplots(1, len(score_cols), figsize=(8 * len(score_cols), 6), squeeze=False)

    labels = np.asarray(data["label"])
    sig_mask = labels == 1
    bkg_mask = labels == 0

    if "weight" in data:
        weights = np.asarray(data["weight"])
        sig_w = weights[sig_mask]
        bkg_w = weights[bkg_mask]
    else:
        sig_w = None
        bkg_w = None

    for idx, col in enumerate(score_cols):
        ax = axes[0, idx]
        values = np.asarray(data[col])
        ax.hist(
            values[sig_mask],
            bins=50,
            range=(0, 1),
            weights=sig_w,
            density=True,
            histtype="step",
            linewidth=2,
            label="Signal",
            color="tab:red",
        )
        ax.hist(
            values[bkg_mask],
            bins=50,
            range=(0, 1),
            weights=bkg_w,
            density=True,
            histtype="step",
            linewidth=2,
            label="Background",
            color="tab:blue",
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"Score density: {col}")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_inference(args, cfg, flavor, sig_events, bkg_events, training_features, feature_transforms, Constraint, derived_vars_cfg, is_data=False, inference_data=None):
    logging.info("Starting inference steps...")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint_from_config(cfg, flavor=flavor)
    logging.info("Using checkpoint: %s", checkpoint_path)

    if is_data:
        full_inference_data = inference_data
    else:
        full_inference_data = _concat_sig_bkg(sig_events, bkg_events)
        
    full_inference_data = apply_derived_vars(full_inference_data, derived_vars_cfg)

    logging.info("Preprocessing inference features...")
    processed = preprocess_data(
        full_inference_data,
        training_features=training_features,
        feature_transforms=feature_transforms,
        Constraint=Constraint,
    )

    feature_matrix = np.column_stack([processed[f] for f in training_features]).astype(np.float32, copy=False)
    features_tensor = torch.from_numpy(feature_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model_from_config(
        cfg=cfg,
        flavor=flavor,
        input_size=len(training_features),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    score_0, score_1 = _batched_scores(model, features_tensor, cfg.get("batch_size", 8192), flavor, device)

    if flavor == "single":
        full_inference_data["dnn_score"] = score_0
    else:
        full_inference_data["dnn_0_score"] = score_0
        full_inference_data["dnn_1_score"] = score_1

    if is_data:
        default_out = _inference_output_dir_from_checkpoint(checkpoint_path) / f"predictions_{flavor}_data.csv"
    else:
        default_out = _inference_output_dir_from_checkpoint(checkpoint_path) / f"predictions_{flavor}.csv"

    output_csv = Path(args.output_csv) if args.output_csv else default_out
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing inference CSV to %s", output_csv)
    _write_csv_numpy(full_inference_data, output_csv)
    logging.info("Done. Wrote %d rows.", _data_length(full_inference_data))

    if not is_data:
        roc_path = output_csv.with_name(f"{output_csv.stem}_roc.png")
        density_path = output_csv.with_name(f"{output_csv.stem}_score_density.png")
        _plot_roc_curves(full_inference_data, flavor=flavor, output_path=roc_path)
        _plot_score_densities(full_inference_data, flavor=flavor, output_path=density_path)
        logging.info("Saved ROC plot to %s", roc_path)
        logging.info("Saved score density plot to %s", density_path)


def main():
    parser = ArgumentParser(epilog="Ex) python main.py --config single/run2_2L_1FJ.yaml --flavor single")
    parser.add_argument('-c', "--config"    , required=True      , help="Path to YAML config")
    parser.add_argument('-f', "--flavor"    , default="single"   , choices=["single", "double"], help="Training flavor: single (one output) or double (two outputs). Later prob should put this in config.")
    parser.add_argument('-d', "--data"      , action="store_true", help="Run inference data (without training) using the latest checkpoint from config")
    parser.add_argument('-i', "--infer"     , action="store_true", help="Skip training and run inference only")
    parser.add_argument('-p', "--checkpoint", default=None       , help="Path to model checkpoint (.ckpt) for inference. If omitted, auto-picks newest checkpoint.")
    parser.add_argument('-o', "--output_csv", default=None       , help="Output CSV path")
    args = parser.parse_args()

    # —————————— Load config ————————————————————————————————————————————————————————————
    print(f"\n—————————— Processing {Path(args.config).stem}——————————")
    cfg = load_config(args.config)
    TrainingFeatures, FeatureTransforms, constraint, split_prefixes = parse_training_features(cfg, args.flavor)
    derived_vars_cfg, DerivedVars, input_for_derivedVars, TrainingFeatures, FeatureTransforms = parse_derived_vars(cfg, TrainingFeatures, FeatureTransforms)

    # —————————— Loading samples ———————————————————————————————————————————
    VarsToLoad = TrainingFeatures + [constraint] + input_for_derivedVars + cfg.get("extra_vars", [])
    # branches are stored unsplit; the _1/_2 forms only exist after LoadEvents splits them
    unsplit = lambda f: f[:-2] if f.startswith(split_prefixes) and f.endswith(("_1", "_2")) else f
    derived_bases = {unsplit(d) for d in DerivedVars}
    VarsToLoad = list(dict.fromkeys(unsplit(f) for f in VarsToLoad if unsplit(f) not in derived_bases))
    new_load_start = time.time()
    sig_paths = local_paths_from_json(cfg["sample_json"], cfg["local_base_path"], "sig")
    bkg_paths = local_paths_from_json(cfg["sample_json"], cfg["local_base_path"], "bkg")
    sig_events = LoadEvents(sig_paths, VarsToLoad, split_prefixes, 1)
    bkg_events = LoadEvents(bkg_paths, VarsToLoad, split_prefixes, 0)
    print(f"Loaded sig+bkg in {time.time()-new_load_start:.1f} s")
    

    # —————————— Something about data... ———————————————————————————————————————————
    # if args.data:
    #     io_workers = int(cfg.get("io_workers", min(2, os.cpu_count() or 1)))
    #     if not args.infer:
    #         parser.error("--data can only be used with --infer")
        
    #     data_base = paths_from_json(cfg["sample_json"], cfg["base_path"], "data")
    #     if "data_path" not in cfg:
    #         parser.error("Config must contain 'data_path' when using --data")

    #     data_paths = resolve_paths("*.root", data_base)
    #     logging.info("Loading real data...")
    #     real_data = load_data(data_paths, VarsToLoad, io_workers, split_prefixes)
        
    #     logging.info("Data samples: %d", _data_length(real_data))
    #     if cfg.get("preselection"):
    #         data_mask = _evaluate_preselection_mask(real_data, cfg["preselection"])
    #         real_data = _apply_mask(real_data, data_mask)
    #     logging.info(
    #         "After preselection - Data samples: %d",
    #         _data_length(real_data),
    #     )

    #     logging.info("Skipping training. Running inference on data only...")
    #     run_inference(args, cfg, args.flavor, None, None, TrainingFeatures, FeatureTransforms, constraint, derived_vars_cfg, is_data=True, inference_data=real_data)
    #     return


    # —————————— Preselection cut ———————————————————————————————————————————
    i_sigW = sum(sig_events.weight) ; i_bkgW = sum(bkg_events.weight)
    print(f"\033[1mBefore preselection\033[0m: sig:{len(sig_events.weight)} (weighted: {i_sigW:.3f})  bkg:{len(bkg_events.weight)} (weighted: {i_bkgW:.3f})")
    if cfg.get("preselection"):
        sig_events = evaluate_preselection(sig_events, cfg["preselection"])
        bkg_events = evaluate_preselection(bkg_events, cfg["preselection"])
    sig_percent = round(sum(sig_events.weight)/i_sigW*100, 1) ; bkg_percent = round(sum(bkg_events.weight)/i_bkgW*100, 1)
    print(f"\033[1mAfter preselection\033[0m : sig:{len(sig_events)} (weighted: {sum(sig_events.weight):.3f}, {sig_percent}% survived)  bkg:{len(bkg_events)} (weighted: {sum(bkg_events.weight):.3f}, {bkg_percent}% survived)")
    
    # Keep copies of the original data (with raw weights and features) for inference
    raw_sig_events = ak.copy(sig_events)
    raw_bkg_events = ak.copy(bkg_events)

    sig_events = normalize_weights(sig_events)
    bkg_events = normalize_weights(bkg_events)

    # —————————— Preprocess data ———————————————————————————————————————————
    data = ak.concatenate([sig_events, bkg_events])
    data = apply_derived_vars(data, derived_vars_cfg)
    TrainingFeatures = split_features(TrainingFeatures, split_prefixes)
    data = preprocess(data, TrainingFeatures, FeatureTransforms, constraint)


    # —————————— Preprocess data ———————————————————————————————————————————
    # print("Preprocessing data...")
    # # combined_keys = set(sig_events.keys()) | set(bkg_events.keys())
    # # data = {
    # #     key: np.concatenate([sig_events.get(key, np.array([])), bkg_events.get(key, np.array([]))])
    # #     for key in combined_keys
    # # }
    # data = apply_derived_vars(data, derived_vars_cfg)
    # old_list = TrainingFeatures.copy()
    # for f in old_list:
    #     if f.startswith(split_prefixes):
    #         ix = TrainingFeatures.index(f)
    #         TrainingFeatures[ix : ix + 1] = [f.replace(f, f+"_1"), f.replace(f, f+"_2")]
    # data = preprocess_data(data, TrainingFeatures, FeatureTransforms, constraint)
    # sys.exit()

    if args.infer:
        logging.info("Skipping training. Running inference only...")
        run_inference(args, cfg, args.flavor, raw_sig_events, raw_bkg_events, TrainingFeatures, FeatureTransforms, constraint, derived_vars_cfg)
        return

    train_loader, valid_loader = makeDataLoaders(
        events     = data,
        features   = TrainingFeatures,
        constraint = constraint,
        batch_size = cfg.get("batch_size", 4096),
    )

    # logging.info("Creating data loaders...")
    # train_loader, valid_loader = make_dataloaders(
    #     data=data,
    #     training_features=TrainingFeatures,
    #     Constraint=constraint,
    #     batch_size=cfg.get("batch_size", 4096),
    # )

    os.makedirs("dataset", exist_ok=True)
    torch.save(train_loader.dataset, f"dataset/{Path(args.config).stem}_train.pt")
    torch.save(valid_loader.dataset, f"dataset/{Path(args.config).stem}_valid.pt")
    sys.exit()
    
    # lightning_model = ABCDLightningModule(
    #     input_size=len(TrainingFeatures),
    #     hidden_layers=cfg.get("architecture", [64, 32, 16]),
    #     learning_rate=cfg.get("learning_rate", 1e-3),
    #     bce_weight=cfg.get("bce_weight", 1.0),
    #     disco_lambda=cfg.get("disco_lambda", 0.0),
    #     flavor=args.flavor,
    #     use_batchnorm=cfg.get("use_batchnorm", True),
    #     dropout=cfg.get("dropout", 0.0),
    #     weight_decay=cfg.get("weight_decay", 1e-2),
    #     label_smoothing=cfg.get("label_smoothing", 0.0),
    #     use_lr_scheduler=cfg.get("use_lr_scheduler", True),
    #     lr_scheduler_patience=cfg.get("lr_scheduler_patience", 40),
    #     lr_scheduler_factor=cfg.get("lr_scheduler_factor", 0.5),
    #     lr_scheduler_min_lr=cfg.get("lr_scheduler_min_lr", 1e-6),
    # )

    # output_dir = cfg.get("output", "simple_abcdisco_output")

    # run_training(
    #     model=lightning_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     output_dir=output_dir,
    #     max_epochs=cfg.get("n_epochs", 100),
    #     flavor=args.flavor,
    #     devices=cfg.get("devices", [0]),
    #     check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1),
    #     early_stopping_patience=cfg.get("early_stopping_patience", 40),
    #     early_stopping_min_delta=cfg.get("early_stopping_min_delta", 1e-4),
    # )

    # logging.info("Training finished.")

    # logging.info("Running inference...")
    # run_inference(args, cfg, args.flavor, raw_sig_events, raw_bkg_events, TrainingFeatures, FeatureTransforms, constraint, derived_vars_cfg)
    # logging.info("Inference finished.")

if __name__ == "__main__":
    main()