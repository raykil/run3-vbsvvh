from PlotterTools import *
from glob import glob
from datetime import datetime
from subprocess import run
import json

scriptPath = os.path.dirname(os.path.abspath(__file__))
abcdPath = f"{scriptPath}/../abcd"
sys.path.insert(0, abcdPath)
from makeDataset_parq import LooseLeptons, LEPTON_ID_BRANCHES

def LoadEvents(path, include=['*'], exclude=[], columns=None):
    """Removes overlap events on the fly."""
    included = {f for p in include for f in glob(f"{path}/{p}.parq")}
    excluded = {f for p in exclude for f in glob(f"{path}/{p}.parq")}
    names = {os.path.basename(f).removesuffix('.parq'): f for f in sorted(included - excluded)}
    event_dict = {name: ak.from_parquet(f, columns=columns) for name, f in names.items() if SHORT_TO_FULL[name] not in chooseOverlaps()}
    return event_dict

def makeHistogram(events, variable, binEdges, power=1):
    clipped_var = np.clip(events[variable], binEdges[0], binEdges[-1])
    hist = np.histogram(clipped_var, binEdges, weights=events.weight**power)[0]
    return ak.to_numpy(hist) # np.histogram keeps ak inputs as ak, which rejects in-place edits like nBGMC[nBGMC == 0]

if __name__ == "__main__":
    parser = ArgumentParser(description='Kinematics distribution plotter for preselection & training.')
    parser.add_argument('-c', '--channel', type=str, default='run2_1FJ', choices=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'])
    parser.add_argument('-v', '--variable', type=str, default='vbs_score', help='x-axis variable (RDF branch).')
    parser.add_argument('-i', '--index'   , type=int, default=None, help='Object index for jagged branches, e.g. 0 for leading lepton_pt.')
    parser.add_argument('-s', '--selections', type=str, default='presel_nom', choices=['presel_nom', 'training', 'B', 'C', 'D'], help='Cuts on top of the nominal channel cut; regions A-D to be added.')
    parser.add_argument('-d', '--draw'    , action="store_true", help='If True, runs hist_plotter.py to plot the json.')
    args = parser.parse_args()

    # ── Load samples ─────────────────────────────────────────────────────────
    samplePath = f"/groups/cjessop/users/jkil/HVV_2L_RDF_parq/{args.channel}"
    varsToLoad = [args.variable, 'weight', f'passes_2lep_*FJ_nom', 'fatjet_Hbb', 'njet', *LEPTON_ID_BRANCHES]
    DATA = LoadEvents(samplePath, include=['*_Run20*'], columns=varsToLoad)
    BKGS = LoadEvents(samplePath, exclude=['*_Run20*', 'VBS*H_*'], columns=varsToLoad)
    SIGS = LoadEvents(samplePath, include=['VBS*H_*'], columns=varsToLoad)

    # ── Common cuts ──────────────────────────────────────────────────────────
    DoCut = lambda cut: ({name: cut(events) for name, events in sample.items()} for sample in (DATA, BKGS, SIGS))
    passFlag = f'passes_2lep_{args.channel.split('_')[-1]}_nom'
    DATA, BKGS, SIGS = DoCut(lambda events: events[events[passFlag]]) # no op for DATA, BKGS b/c systs was off; ~0.9% reduction for SIGS, since systs was on for sig only.
    DATA, BKGS, SIGS = DoCut(lambda events: events[events.njet >= 2]) # TEMP: skims predate njet >= 2 in passFlag; drops vbs_* = -999

    # ── Additional cuts ──────────────────────────────────────────────────────
    if args.selections=='training': # rest of makeDataset_parq.py's cutflow
        LooseLepCut = lambda events: LooseLeptons(events, events)
        WeightCut   = lambda events: events[events.weight > 0]
        HbbScoreCut = lambda events: events[ak.any(events.fatjet_Hbb >= 0.1, axis=1)] # yaml preselection; any() also covers 2FJ's _1 | _2
        DATA, BKGS, SIGS = DoCut(LooseLepCut)
        DATA, BKGS, SIGS = DoCut(WeightCut)
        DATA, BKGS, SIGS = DoCut(HbbScoreCut)
        # After this block len(BKGS)+len(SIGS) = len(abcd/dataset/c2v*_c3_*/run*_2L_*FJ.parq) ✅

    elif args.selections in 'ABCD':
        abcd_thld = 0.71 ; vbs_thld = 0.83
        LooseLepCut = lambda events: LooseLeptons(events, events)
        WeightCut   = lambda events: events[events.weight > 0]
        HbbScoreCut = lambda events: events[ak.any(events.fatjet_Hbb >= 0.1, axis=1)] # yaml preselection; any() also covers 2FJ's _1 | _2
        DATA, BKGS, SIGS = DoCut(LooseLepCut)
        DATA, BKGS, SIGS = DoCut(WeightCut)
        DATA, BKGS, SIGS = DoCut(HbbScoreCut)

    # ── Make histograms ──────────────────────────────────────────────────────
    BGBG = CategorizeSamples(BKGS)
    SGSG = CategorizeSamples(SIGS)
    DATA = ak.concatenate(DATA.values())
    BKGS = ak.concatenate(BKGS.values())
    SIGS = ak.concatenate(SIGS.values())

    binEdges = getBinEdges(args.variable)
    nDATA = makeHistogram(DATA, args.variable, binEdges)
    nBGMC = makeHistogram(BKGS, args.variable, binEdges)
    mcmcStatUnc = np.sqrt(makeHistogram(BKGS, args.variable, binEdges, 2))
    nBGMC[nBGMC == 0] = np.inf

    # ── Export to JSON ───────────────────────────────────────────────────────
    run_ = args.channel.split('_')[0]
    HIST_DICT = {
        'channel'         : f"{args.channel}, {args.selections}",
        'lumitext'        : f"{LUMI[run_]:.0f} fb$^{{-1}}$ ({'13' if run_ == 'run2' else '13.6'} TeV)",
        'binEdges'        : binEdges.tolist(),
        'nMCMC'           : {cat: {'color': BG['color'], 'hist': makeHistogram(ak.concatenate(BG['events'].values()), args.variable, binEdges).tolist()} for cat, BG in BGBG.items()},
        'nSGMC'           : {sig: {'color': SG['color'], 'hist': makeHistogram(ak.concatenate(SG['events'].values()), args.variable, binEdges).tolist()} for sig, SG in SGSG.items()},
        'nDATA'           : nDATA.tolist(),
        'ratio'           : (nDATA / nBGMC).tolist(),
        'dataStatUnc_ax1' : np.sqrt(nDATA).tolist(),
        'mcmcStatUnc_ax1' : mcmcStatUnc.tolist(),
        'ratioStatUnc_ax2': (np.sqrt(nDATA) / nBGMC).tolist(), # data stat only; MC stat is the band
        'mcmcStatUnc_ax2' : (mcmcStatUnc / nBGMC).tolist(),
        'xlabel'          : args.variable,
    }
    jsonPath = f"{scriptPath}/{datetime.now().strftime('%y%m%d')}_{args.selections}/{args.channel}/jsons/{args.variable}.json"
    os.makedirs(os.path.dirname(jsonPath), exist_ok=True)
    with open(jsonPath, 'w') as j: json.dump(HIST_DICT, j, indent=4)
    print(f"{jsonPath.replace(f'{scriptPath}/', '')} saved!")

    if args.draw:
        run(["python", f"{scriptPath}/hist_plotter.py", "-j", jsonPath])