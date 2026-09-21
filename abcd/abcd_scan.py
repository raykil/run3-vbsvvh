from TrainingTools import *

def prettyPrint(matrix, name='', binEdges=None):
    """ Row/col labels are the cut thresholds, not bin ranges: entry [i,j] is a whole region's yield for the cut pair (binEdges[i], binEdges[j]). """
    matrix = np.asarray(matrix)
    below, above = lambda edge: f"[0.00-{edge:.2f}]", lambda edge: f"[{edge:.2f}-1.00]"
    rowFmt = below if name[-1:] in ("B", "D") else above # B and D keep events below the ABCDNet cut
    colFmt = below if name[-1:] in ("C", "D") else above # C and D keep events below the VBS cut
    rowCuts = [rowFmt(edge) for edge in binEdges[:matrix.shape[0]]]
    colCuts = [colFmt(edge) for edge in binEdges[:matrix.shape[1]]]
    width = max(len(colCuts[0]), max(len(f"{value:.4f}") for value in matrix.flat))
    print(f"---------- {name} ----------")
    print(f"{'VBS':>13} " + " ".join(f"{cut:>{width}}" for cut in colCuts))
    for cut, row in zip(rowCuts, matrix):
        print(f"{cut:>13} " + " ".join(f"{value:{width}.4f}" for value in row))
    print(f"{'ABCDNet':<13}\n")

def regionYields(xVar, yVar, weights, binEdges):
    hist2d = np.histogram2d(xVar, yVar, bins=binEdges, weights=weights)[0] # nEvents per ABCD/VBS 2d bin.
    reversed_hist2d = hist2d[::-1, ::-1] # b/c cumsum works from low idx to high idx
    vertical_sum    = np.cumsum(reversed_hist2d, axis=0) # for each col, nth row contains sum of 0th to nth row in reversed_hist2d.
    horizontal_sum  = np.cumsum(vertical_sum, axis=1) # for each row, last row contains cumsum of the whole row.
    summed_hist2d   = horizontal_sum[::-1, ::-1] # lower index is region A
    A = summed_hist2d # outer axis = ABCDNetscore, inner axis = VBSscore, higher index, tighter cut
    B = A[0:1, :] - A # subtracting 0th row of A by nth row of A
    C = A[:, 0:1] - A # subtracting 0th col of A by nth col of A
    D = A[0, 0] - A[0:1, :] - A[:, 0:1] + A
    return A, B, C, D

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag"   , default='run2_2L_1FJ'  , choices=["run2_2L_1FJ", "run2_2L_2FJ", "run3_2L_1FJ", "run3_2L_2FJ"])
    parser.add_argument("-s", "--signal", default='c2v1p0_c3_1p0', choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"])
    parser.add_argument("-n", "--minCount" , default=10 , type=int  , help="Guardrail: min unweighted MC events required in every region.")
    parser.add_argument("-c", "--maxNonClosure", default=0.2, type=float, help="Guardrail: max |B*C/D / A - 1| tolerated in background MC.")
    parser.add_argument("-x", "--maxSigConta" , default=0.01, type=float, help="Guardrail: max sig/bkg tolerated in any control region, since signal there biases B*C/D.")
    args = parser.parse_args()

    # —————————— Load MC ——————————————————————————————————————————————————
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    parqPath = f"{scriptPath}/dataset/{args.signal}/{args.tag}.parq"
    events  = ak.from_parquet(parqPath)
    SCORES  = ak.to_numpy(events.ABCDscore) # numpy from here on, else awkward intercepts np.histogram2d and every result stays an ak.Array
    LABELS  = ak.to_numpy(events.label)
    DISCOS  = ak.to_numpy(events.VBSscore)
    WEIGHTS = ak.to_numpy(events.weight) # hist2d of sig and bkg sums to WEIGHTS, as it should be.

    nBkg, nSig = ak.count_nonzero(LABELS==0), ak.count_nonzero(LABELS==1)
    print(f"\n\033[1m—————————— {args.tag} | {args.signal} ——————————\033[0m")
    print(f"nEvents={len(LABELS)} | bkg={nBkg}(weighted:{sum(WEIGHTS[LABELS==0]):.3f}) | sig={nSig}(weighted:{sum(WEIGHTS[LABELS==1]):.3f})")

    # —————————— Scan the plane ——————————————————————————————————————————————————
    binEdges = np.arange(0, 1.001, 0.01) # len=101, len(cutPairs)=101*101=10201
    isSig = LABELS == 1
    isBkg = LABELS == 0

    # sigA: for significance
    # sigB/C/D: For contamination
    # bkgA: Answer key to bkgB * bkgC / bkgD = predA
    # raw*: unweighted, for the stats guardrail
    sigA, sigB, sigC, sigD = regionYields(SCORES[isSig] , DISCOS[isSig] , WEIGHTS[isSig] , binEdges)
    bkgA, bkgB, bkgC, bkgD = regionYields(SCORES[~isSig], DISCOS[~isSig], WEIGHTS[~isSig], binEdges)
    rawA, rawB, rawC, rawD = regionYields(SCORES[~isSig], DISCOS[~isSig], None           , binEdges)

    with np.errstate(divide='ignore', invalid='ignore'):
        predA = bkgB * bkgC / bkgD # the ABCD prediction, which MC's own A is the answer key for
        closure = predA/bkgA - 1 # +: overpredict, -: underpredict.
        # signif = sigA/np.sqrt(bkgA)
        signif = np.sqrt(2*((sigA + bkgA)*np.log1p(sigA/bkgA) - sigA)) # Asimov; reduces to s/sqrt(b) for s<<b, but stays valid at low bkgA
        sigConta = np.maximum.reduce([sigB/bkgB, sigC/bkgC, sigD/bkgD]) # worst control region; signal there inflates predA

    verbose = False
    if verbose:
        prettyPrint(sigA, "sigA", binEdges)
        prettyPrint(bkgA, "bkgA", binEdges)
        prettyPrint(bkgB, "bkgB", binEdges)
        prettyPrint(bkgC, "bkgC", binEdges)
        prettyPrint(bkgD, "bkgD", binEdges)

        prettyPrint(rawA, "rawA", binEdges)
        prettyPrint(rawB, "rawB", binEdges)
        prettyPrint(rawC, "rawC", binEdges)
        prettyPrint(rawD, "rawD", binEdges)

        prettyPrint(predA, "predA", binEdges)
        prettyPrint(closure, "closure", binEdges)

    # —————————— Apply guardrails and rank ——————————————————————————————————————————————————
    minCountGuard = np.minimum.reduce([rawA, rawB, rawC, rawD]) >= args.minCount # takes min of A,B,C, and D at all cutPairs.
    nonCloseGuard = np.abs(closure) <= args.maxNonClosure
    finitSigGuard = np.isfinite(signif)
    sigContaGuard = sigConta <= args.maxSigConta # nan (empty control region) fails this, as it should
    passes = minCountGuard & nonCloseGuard & finitSigGuard & sigContaGuard
    if not passes.any(): sys.exit(f"\033[1;31mNo cut pair passes minCount={args.minCount}, maxNonClosure={args.maxNonClosure} and maxSigConta={args.maxSigConta}.\033[0m")

    ranksToDisplay = 3
    ranked = np.argsort(np.where(passes, signif, -np.inf), axis=None)[::-1][:ranksToDisplay]
    print(f"\n\033[1mTop {ranksToDisplay} of {passes.sum()} cut pairs passing minCount={args.minCount}, |nonClosure|<={args.maxNonClosure}, sigConta<={args.maxSigConta}\033[0m")
    print(f"{'ABCD>':>6}{'VBS>':>6}{'signif':>8}{'sigA':>10}{'bkgA':>10}{'predA':>10}{'closure':>9}{'sigConta':>10}{'rawA':>8}")
    for idx in ranked:
        i, j = np.unravel_index(idx, signif.shape)
        print(f"{binEdges[i]:>6.2f}{binEdges[j]:>6.2f}{signif[i,j]:>8.3f}{sigA[i,j]:>10.4f}{bkgA[i,j]:>10.3f}{predA[i,j]:>10.3f}{closure[i,j]:>+9.3f}{sigConta[i,j]:>10.5f}{int(rawA[i,j]):>8d}")