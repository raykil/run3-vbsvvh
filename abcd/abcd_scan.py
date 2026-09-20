from TrainingTools import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag"   , default='run2_2L_1FJ'  , choices=["run2_2L_1FJ", "run2_2L_2FJ", "run3_2L_1FJ", "run3_2L_2FJ"])
    parser.add_argument("-s", "--signal", default='c2v1p0_c3_1p0', choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"])
    args = parser.parse_args()

    # —————————— Load all MC ——————————————————————————————————————————————————
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    parqPath = f"{scriptPath}/dataset/{args.signal}/{args.tag}.parq"
    events = ak.from_parquet(parqPath) # train+valid, since the scan measures yields rather than validating the model

    SCORES , DISCOS  = ak.to_numpy(events.ABCDscore), ak.to_numpy(events.VBSscore) # raw VBS score keeps cut values interpretable
    LABELS , WEIGHTS = ak.to_numpy(events.label)    , ak.to_numpy(events.weight)   # raw weights, so yields are physical
    nBkg, nSig = np.count_nonzero(LABELS==0), np.count_nonzero(LABELS==1)
    print(f"\n\033[1m—————————— {args.tag} | {args.signal} ——————————\033[0m")
    print(f"nEvents={len(LABELS)} | bkg={nBkg}(weighted:{WEIGHTS[LABELS==0].sum():.3f}) | sig={nSig}(weighted:{WEIGHTS[LABELS==1].sum():.3f})")
