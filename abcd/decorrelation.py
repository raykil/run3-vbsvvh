from TrainingTools import *
import mplhep as hep
from datetime import datetime

def separate(quantity, labels):
    isSig = labels == 1 ; quantity_sig = quantity[isSig]
    isBkg = labels == 0 ; quantity_bkg = quantity[isBkg]
    return quantity_sig, quantity_bkg

def plotScoreDensity(SCORES, LABELS, WEIGHTS):
    """ Goal is to see bkg piled near 0 and sig near 1, which means ABCDNet separates the two """
    SCORES_sig , SCORES_bkg  = separate(SCORES , LABELS)
    WEIGHTS_sig, WEIGHTS_bkg = separate(WEIGHTS, LABELS)
    binEdges = np.arange(0, 1.01, 0.02)
    hist_sig = np.histogram(SCORES_sig, bins=binEdges, weights=WEIGHTS_sig, density=True)[0]
    hist_bkg = np.histogram(SCORES_bkg, bins=binEdges, weights=WEIGHTS_bkg, density=True)[0]

    hep.histplot([hist_sig, hist_bkg], binEdges, color=["tab:red", "tab:blue"], label=["Signal", "Background"])
    hep.cms.label("Preliminary", data=True, com=com, lumi=lumi)
    plt.xlabel("ABCDNet score")
    plt.ylabel("Density")
    plt.xlim(-0.1, 1.1)
    plt.grid()
    plt.legend()

    plotPath = f"{outPath}/score_density.png"
    plt.savefig(plotPath, dpi=300, bbox_inches="tight")
    print(f"\033[1;32mSaved abcd/{plotPath}!\033[0m")

def TProfile(xVar, yVar, weights, binEdges):
    weights = weights.astype(np.float64) # there are very small weights**2 that it treats as 0 at float32.
    binCenters = 0.5 * (binEdges[:-1] + binEdges[1:])
    weightedXSum  = np.histogram(xVar, bins=binEdges, weights=weights)[0] # sum of weights per ABCDNet Score bin
    weightedYSum  = np.histogram(xVar, bins=binEdges, weights=weights*yVar)[0] # sum of weighted vbs bdt score per ABCDNet Score bin
    weightSqSum   = np.histogram(xVar, bins=binEdges, weights=weights**2)[0]
    weightedY2Sum = np.histogram(xVar, bins=binEdges, weights=weights*yVar**2)[0]

    # calculating points
    isFilled = weightedXSum != 0
    Xs = binCenters[isFilled]
    Ys = weightedYSum[isFilled]/weightedXSum[isFilled]

    # Calculating errs
    Xerrs = 0.5 * np.diff(binEdges)[isFilled] # ROOT draws x errors as the half bin width.
    variance   = np.maximum(weightedY2Sum[isFilled]/weightedXSum[isFilled] - Ys**2, 0) # Clamped; float error can dip below 0.
    nEffective = weightedXSum[isFilled]**2 / weightSqSum[isFilled] # ROOT's effective entries, (sum w)^2 / sum w^2.
    Yerrs = np.sqrt(variance/nEffective) # ROOT's default kERRORMEAN: error on the mean, not the spread.
    return Xs, Ys, Xerrs, Yerrs

def plotVBS_vs_ABCD(SCORES, DISCOS, LABELS, WEIGHTS):
    """ Goal is to see flat points, which means ABCDNet Score is decorrelated from vbs_score. """
    SCORES_sig , SCORES_bkg  = separate(SCORES , LABELS)
    DISCOS_sig , DISCOS_bkg  = separate(DISCOS , LABELS)
    WEIGHTS_sig, WEIGHTS_bkg = separate(WEIGHTS, LABELS)
    binEdges = np.arange(0, 1.01, 0.02)
    hist2d_sig = np.histogram2d(SCORES_sig, DISCOS_sig, bins=binEdges, weights=WEIGHTS_sig, density=True)[0]
    hist2d_bkg = np.histogram2d(SCORES_bkg, DISCOS_bkg, bins=binEdges, weights=WEIGHTS_bkg, density=True)[0]
    SIG_DICT = {'kind': 'sig', 'hist': hist2d_sig, 'scores': SCORES_sig, 'discos': DISCOS_sig, 'weights': WEIGHTS_sig, 'cmap': 'Greens'}
    BKG_DICT = {'kind': 'bkg', 'hist': hist2d_bkg, 'scores': SCORES_bkg, 'discos': DISCOS_bkg, 'weights': WEIGHTS_bkg, 'cmap': 'Blues'}

    for DICT in [SIG_DICT, BKG_DICT]:
        fig, ax = plt.subplots()
        plotPath = f"{outPath}/VBS_vs_ABCD_{DICT['kind']}.png"
        hep.hist2dplot(DICT['hist'], binEdges, binEdges, cmap=DICT['cmap'])
        Xs, Ys, Xerrs, Yerrs = TProfile(DICT['scores'], DICT['discos'], DICT['weights'], binEdges)
        plt.errorbar(Xs, Ys, xerr=Xerrs, yerr=Yerrs, fmt='o', ms=3, label=r"$\langle$VBS$\rangle$ per ABCDNet bin $\pm$ std.err.", color='tab:red', capsize=2)

        hep.cms.label(data=False, com=com, lumi=lumi)
        plt.xlabel('ABCDNet Score')
        plt.ylabel('VBS BDT Score')
        plt.legend()
        plt.savefig(plotPath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\033[1;32mSaved abcd/{plotPath}!\033[0m")

def plotDiscoHist(SCORES, DISCOS, LABELS, WEIGHTS):
    """ Goal is to see same shapes over ranges of ABCDNet Score. """
    ABCDNet_binEdges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    SCORES_sig , SCORES_bkg  = separate(SCORES , LABELS)
    DISCOS_sig , DISCOS_bkg  = separate(DISCOS , LABELS)
    WEIGHTS_sig, WEIGHTS_bkg = separate(WEIGHTS, LABELS)
    SIG_DICT = {'kind': 'Sig', 'scores': SCORES_sig, 'discos': DISCOS_sig, 'weights': WEIGHTS_sig, 'cmap': 'Greens'}
    BKG_DICT = {'kind': 'Bkg', 'scores': SCORES_bkg, 'discos': DISCOS_bkg, 'weights': WEIGHTS_bkg, 'cmap': 'Blues' }
    binEdges = np.arange(0, 1.01, 0.02)

    for DICT in [BKG_DICT, SIG_DICT]:
        plt.figure()
        colors = plt.get_cmap(DICT['cmap'])(np.linspace(0.35, 1.0, len(ABCDNet_binEdges)-1))
        for idx, binedge in enumerate(ABCDNet_binEdges[:-1]):
            lo, hi = binedge, ABCDNet_binEdges[idx+1]
            inSlice = (lo <= DICT['scores']) & (DICT['scores'] < hi)
            if not inSlice.any(): continue
            hist = np.histogram(DICT['discos'][inSlice], bins=binEdges, weights=DICT['weights'][inSlice], density=True)[0]
            hep.histplot(hist, binEdges, color=colors[idx], label=f"{DICT['kind']} {lo:.1f}$\\leq$s<{hi:.1f}")

        hep.cms.label(data=False, com=com, lumi=lumi)
        plt.xlabel('VBS BDT Score')
        plt.ylabel('Density')
        plt.grid()
        plt.legend(ncol=1, fontsize=12, borderaxespad=1.5)

        plotPath = f"{outPath}/discoHist_{DICT['kind']}.png"
        plt.savefig(plotPath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\033[1;32mSaved abcd/{plotPath}!\033[0m")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag"   , default='run2_2L_1FJ'  , choices=["run2_2L_1FJ", "run2_2L_2FJ", "run3_2L_1FJ", "run3_2L_2FJ"])
    parser.add_argument("-s", "--signal", default='c2v1p0_c3_1p0', choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"])
    parser.add_argument('-m', '--make'  , default='111', type=str, help='1=draw, 0=not draw, in order of (1. ScoreDensity, 2.VBS_vs_ABCD, 3.DiscoHist).')
    args = parser.parse_args()

    # —————————— Load Dataset ——————————————————————————————————————————————————
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    datasetPath = f"{scriptPath}/dataset/{args.signal}/{args.tag}_valid.pt"
    validation_dataset = torch.load(datasetPath, map_location="cpu", weights_only=False)
    validation_loader  = torch.utils.data.DataLoader(validation_dataset, batch_size=4096, shuffle=False)

    nEvents, nFeatures = validation_dataset.data.shape
    nBkg, nSig = np.count_nonzero(validation_dataset.labels==0), np.count_nonzero(validation_dataset.labels==1)
    print(f"\n\033[1m—————————— {args.tag} | {args.signal} ——————————\033[0m")
    print(f"nEvents={nEvents}(weighted:{sum(validation_dataset.weights):.3f}) | bkg={nBkg}({100*nBkg/nEvents:.3f}%) | sig={nSig}({100*nSig/nEvents:.3f}%) | nFeatures={nFeatures}")

    # —————————— Load Model ——————————————————————————————————————————————————
    model = makeModel(nFeatures)[0]
    modelPath = f"{scriptPath}/models/{args.signal}/{args.tag}_best_model.pt"
    device = "cpu"
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.to(device) # Load model param tensors to device.
    model.eval() # switch from train mode to eval mode. Dropout stops dropping neurons, and batch-norm uses stored running stats.

    # —————————— Evaluate ABCDNet Score ——————————————————————————————————————————————————
    SCORES, LABELS, WEIGHTS, DISCOS = [], [], [], []
    torch.set_grad_enabled(False)
    for features, disco, labels, weights in validation_loader: # Loops over batches. nIteration = nEvents / batch_size
        ABCDNetScore = model(features.to(device)) # This is where ABCDNet Score is assigned.
        ABCDNetScore = torch.sigmoid(ABCDNetScore).cpu()
        SCORES.append(ABCDNetScore)
        DISCOS.append(disco)
        LABELS.append(labels)
        WEIGHTS.append(weights)
    SCORES  = torch.cat(SCORES).reshape(-1).numpy() # ABCDNet Scores. [0, 1] avg: 0.315
    DISCOS  = torch.cat(DISCOS).reshape(-1).numpy() # vbs_score. [0, 1] avg: 0.237
    LABELS  = torch.cat(LABELS).reshape(-1).numpy()
    WEIGHTS = torch.cat(WEIGHTS).reshape(-1).numpy() # O(w) ~ [1e-10, 1e-2]

    # —————————— Plotting ——————————————————————————————————————————————————
    makeScoreDensity, makeVBS_vs_ABCD, makeDiscoHist = (flag == '1' for flag in args.make)
    plt.style.use(hep.style.CMS)
    lumiText = {
        'run2': {'com': 13  , 'lumi': int(round(16.81+19.52+41.48+59.83, 0))},
        'run3': {'com': 13.6, 'lumi': int(round(28.28+34.76+109.95+110.84, 0))}
    }
    com, lumi = lumiText[args.tag[:4]]['com'], lumiText[args.tag[:4]]['lumi']
    outPath = f"{datetime.now():%y%m%d}_plots/{args.signal}/{args.tag}"
    os.makedirs(outPath, exist_ok=True)

    if makeScoreDensity: plotScoreDensity(SCORES, LABELS, WEIGHTS)
    if makeVBS_vs_ABCD : plotVBS_vs_ABCD(SCORES, DISCOS, LABELS, WEIGHTS)
    if makeDiscoHist   : plotDiscoHist(SCORES, DISCOS, LABELS, WEIGHTS)