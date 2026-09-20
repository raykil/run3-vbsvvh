"""
Assigns ABCDNet scores to every MC event and writes them back into the parquet,
so downstream scripts (decorrelation.py, abcd_scan.py) never reload the model.
"""

from TrainingTools import *

@torch.no_grad()
def inferABCDscore(model, loader):
    """ Returns SCORES [0,1] avg 0.315, DISCOS (vbs_score) [0,1] avg 0.237, LABELS, WEIGHTS O[1e-10, 1e-2]. """
    device = next(model.parameters()).device
    SCORES, DISCOS, LABELS, WEIGHTS = [], [], [], []
    for features, disco, labels, weights in loader: # Loops over batches. nIteration = nEvents / batch_size
        SCORES.append(torch.sigmoid(model(features.to(device))).cpu()) # This is where ABCDNet Score is assigned.
        DISCOS.append(disco)
        LABELS.append(labels)
        WEIGHTS.append(weights)
    SCORES  = torch.cat(SCORES).reshape(-1).numpy()
    DISCOS  = torch.cat(DISCOS).reshape(-1).numpy()
    LABELS  = torch.cat(LABELS).reshape(-1).numpy()
    WEIGHTS = torch.cat(WEIGHTS).reshape(-1).numpy()
    return SCORES, DISCOS, LABELS, WEIGHTS

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag"   , default='run2_2L_1FJ'  , choices=["run2_2L_1FJ", "run2_2L_2FJ", "run3_2L_1FJ", "run3_2L_2FJ"])
    parser.add_argument("-s", "--signal", default='c2v1p0_c3_1p0', choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"])
    args = parser.parse_args()

    # —————————— Load Dataset ——————————————————————————————————————————————————
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    basePath = f"{scriptPath}/dataset/{args.signal}/{args.tag}"
    splits = [torch.load(f"{basePath}_{split}.pt", map_location="cpu", weights_only=False) for split in ("train", "valid")]
    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(splits), batch_size=4096, shuffle=False) # [train, valid] order, matching the parquet rows

    # —————————— Load Model ——————————————————————————————————————————————————
    nFeatures = splits[0].data.shape[1]
    model = makeModel(nFeatures)[0]
    modelPath = f"{scriptPath}/models/{args.signal}/{args.tag}_best_model.pt"
    model.load_state_dict(torch.load(modelPath, map_location=next(model.parameters()).device))
    model.eval() # switch from train mode to eval mode. Dropout stops dropping neurons, and batch-norm uses stored running stats.

    # —————————— Evaluate ABCDNet Score ——————————————————————————————————————————————————
    SCORES, DISCOS, LABELS, WEIGHTS = inferABCDscore(model, loader) # DISCOS=norm_VBSscore, LABELS=label, WEIGHTS=norm_weight are only for checking alignment.
    print(f"\n\033[1m—————————— {args.tag} | {args.signal} ——————————\033[0m")
    print(f"nEvents={len(SCORES)} | nFeatures={nFeatures} | ABCDscore avg={SCORES.mean():.3f}")

    # —————————— Write back ——————————————————————————————————————————————————
    parqPath = f"{basePath}.parq"
    events = ak.from_parquet(parqPath)
    aligned = np.array_equal(ak.to_numpy(events.label), LABELS) and np.allclose(events.norm_weight, WEIGHTS) and np.allclose(events.norm_VBSscore, DISCOS)
    assert aligned, f"{parqPath} rows are not aligned with the .pt files"
    events = ak.with_field(events, SCORES, where="ABCDscore")
    ak.to_parquet(events, parqPath)
    print(f"\033[1;32mSaved ABCDscore into abcd/{os.path.relpath(parqPath, scriptPath)}!\033[0m")