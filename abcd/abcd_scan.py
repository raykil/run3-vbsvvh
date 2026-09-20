from TrainingTools import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag"   , default='run2_2L_1FJ'  , choices=["run2_2L_1FJ", "run2_2L_2FJ", "run3_2L_1FJ", "run3_2L_2FJ"])
    parser.add_argument("-s", "--signal", default='c2v1p0_c3_1p0', choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"])
    args = parser.parse_args()

    # —————————— Load all MC ——————————————————————————————————————————————————
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    splits = [torch.load(f"{scriptPath}/dataset/{args.signal}/{args.tag}_{split}.pt", map_location="cpu", weights_only=False) for split in ("train", "valid")]
    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(splits), batch_size=4096, shuffle=False)

    # —————————— Load Model ——————————————————————————————————————————————————
    nFeatures = splits[0].data.shape[1]
    model = makeModel(nFeatures)[0]
    model.load_state_dict(torch.load(f"{scriptPath}/models/{args.signal}/{args.tag}_best_model.pt", map_location=next(model.parameters()).device))
    model.eval() # switch from train mode to eval mode. Dropout stops dropping neurons, and batch-norm uses stored running stats.

    # —————————— Evaluate ABCDNet Score ——————————————————————————————————————————————————
    SCORES, DISCOS, LABELS, WEIGHTS = evalABCDscore(model, loader)
    nBkg, nSig = np.count_nonzero(LABELS==0), np.count_nonzero(LABELS==1)
    print(f"\n\033[1m—————————— {args.tag} | {args.signal} ——————————\033[0m")
    print(f"nEvents={len(LABELS)} | bkg={nBkg}(weighted:{WEIGHTS[LABELS==0].sum():.3f}) | sig={nSig}(weighted:{WEIGHTS[LABELS==1].sum():.3f}) | nFeatures={nFeatures}")
