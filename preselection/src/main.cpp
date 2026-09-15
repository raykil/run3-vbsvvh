#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RLogger.hxx"

#include "weights.h"
#include "corrections.h"
#include "selections.h"
#include "utils.h"
#include "genSelections.h"

#include "argparser.hpp"
#include "cutflow.h"

#include "spanet.h"
#include "spanet_run2.h"
#include "btag_efficiencies.h"
#include "btag_settings.h"

#include <optional>
#include <set>


struct MyArgs : public argparse::Args {
    std::string &spec       = kwarg("i,input", "spec.json path");
    std::string &ana        = kwarg("a,ana", "Which skim selection channel for event selection");
    std::string &name       = kwarg("n,name", "Naming tag for the output").set_default("rdf_output");
    std::string &outdir     = kwarg("o,outdir", "Path to output").set_default(".");
    std::string &run_number = kwarg("r,run_number", "Run number (2 or 3)");
    
    int &batch_size = kwarg("b,batch_size", "batch size for spanet inference").set_default(64);
    int &nthread    = kwarg("j,nthread", "number of threads for ROOT").set_default(0);

    bool &progress = flag("progress", "Show progress bar").set_default(false);
    bool &dumpInput              = flag("dump_input", "Dump all input branches to output ROOT file").set_default(false);
    bool &makeSpanetTrainingdata = flag("spanet_training", "Only make training data for SPANet").set_default(false);
    bool &runSPANetInference     = flag("spanet_infer", "Run SPANet inference").set_default(false);
    bool &storeHLT = flag("store_hlt", "Store HLT trigger branches in output").set_default(false);
    bool &cutflow = flag("cutflow", "Print cutflow").set_default(false);
    bool &makeBTagEfficiencies = flag("btag_eff", "Write selected-AK4 b-tag efficiency histograms (MC only)").set_default(false);
    bool &skipBTagScaleFactors = flag("skip_btag_sf,skip-btag-sf", "Skip b-tag SF application (normally enabled)").set_default(false);
    bool &systs = flag("systs", "Store JES/JER variation branches; nominal-only by default").set_default(false);
    bool &no_jetveto = flag("no_jetveto", "DEBUG ONLY: compute Jet_vetoMap but do not apply it (no Run 3 event veto, no jet masking). NOT for analysis production").set_default(false);
};

RNode runAnalysis(RNode df, std::string ana, std::string run_number, bool isSignal, bool isData, SPANet::SPANetInference *spanet_inference, SPANetRun2::SPANetInference *spanet_inference_run2, bool runSPANetInference = false, bool makeSpanetTrainingdata = false)
{
    std::cout << " -> Run " << ana << "::runAnalysis()" << std::endl;

    df = runPreselection(df, ana, makeSpanetTrainingdata, isData);
    
    if (isSignal) {
        df = GenSelections(df);
    }

    if (!makeSpanetTrainingdata && runSPANetInference) {
        std::cout << "Running spanet" << std::endl;
        if (run_number == "2"){
            df = spanet_inference_run2->RunSPANetInference(df);
            df = spanet_inference_run2->ParseSpanetInference(df);
        }
        else {
            df = spanet_inference->RunSPANetInference(df);
            df = spanet_inference->ParseSpanetInference(df);
        }
    }
    return df;
}

int main(int argc, char** argv) {
    // Read input args
    auto args = argparse::parse<MyArgs>(argc, argv);
    std::string input_spec = args.spec;
    std::string output_file = args.name;

    setApplyJetVetoMaps(!args.no_jetveto);
    if (args.no_jetveto) {
        std::cout << "\n"
                  << "  ############################################################\n"
                  << "  ##  WARNING: --no_jetveto is a DEBUGGING flag             ##\n"
                  << "  ############################################################\n"
                  << "  Jet_vetoMap is computed but NOT applied: the Run 3 event\n"
                  << "  veto is off and no jet is masked by it. This output is NOT\n"
                  << "  veto-map corrected -- it keeps jets in known-bad (eta, phi)\n"
                  << "  regions and, in Run 3, the spurious MET the veto removes.\n"
                  << "  DO NOT use it for an analysis production. Intended only for\n"
                  << "  the veto map validation study, which needs the un-vetoed\n"
                  << "  'before' state (see vbs-coffea/scripts/jetveto_studies).\n"
                  << "  ############################################################\n"
                  << std::endl;
    }

    if (args.nthread > 64) {
        std::cerr << "Error: nthread cannot exceed 64 (requested: " << args.nthread << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::string> channels = {
        "all_events",
        "0lep_0FJ",
        "0lep_1FJ",
        "0lep_1FJ_met",
        "0lep_2FJ",
        "0lep_2FJ_met",
        "0lep_3FJ",
        "1lep_1FJ",
        "1lep_2FJ",
        "2lep_1FJ", // Currently shared between SF and OF
        "2lepSS",
        "2lep_2FJ",
        "3lep",
        "4lep",
    };
    if (std::find(channels.begin(), channels.end(), args.ana) == channels.end()) {
        if (!args.makeSpanetTrainingdata) {
            std::cerr << "Did not recognize analysis tag: " << args.ana << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    if (args.run_number != "2" && args.run_number != "3") {
        throw std::runtime_error("Invalid run_number: must be 2 or 3");
    }

    const bool channelBTagScaleFactors =
        std::find(channels.begin(), channels.end(), args.ana) != channels.end()
        ? bTagScaleFactorsEnabled(args.ana) : false;

    // UParTAK4 has no matching fixed-WP calibration for the NanoAODv12
    // 2022/2023 eras.  Do not derive efficiencies from the unrelated 2024
    // thresholds used by the legacy selection configuration.
    std::optional<BTagEfficiencyMetadata> btag_efficiency_metadata;
    if (args.makeBTagEfficiencies) {
        btag_efficiency_metadata = getSingleSampleBTagEfficiencyMetadata(input_spec);
        const auto &year = btag_efficiency_metadata->year;
        if (year == "2022Re-recoBCD" || year == "2022Re-recoE+PromptFG" ||
            year == "2023PromptC" || year == "2023PromptD") {
            throw std::runtime_error(
                "--btag_eff is not supported for " + year +
                ": UParTAK4 fixed-WP thresholds/SFs are unavailable for NanoAODv12. "
                "Use a supported tagger with a matched implementation, or run 2024/2025 production.");
        }
    }

    // Create output only after validating the requested b-tag workflow.
    std::string output_dir = setOutputDirectory(args.outdir, args.makeSpanetTrainingdata);
    std::cout << " -> Running analysis for Run " << args.run_number << std::endl;
    
    std::unique_ptr<SPANet::SPANetInference> spanet_inference;
    std::unique_ptr<SPANetRun2::SPANetInference> spanet_inference_run2;

    if (args.runSPANetInference) {
        std::string  model_path;

        model_path = "spanet/run2/v31/model.onnx";
        std::cout << " -> Loading Run 2 ONNX model from: " << model_path << std::endl;
        spanet_inference_run2 = std::make_unique<SPANetRun2::SPANetInference>(model_path, args.batch_size);
        std::cout << "    ONNX session loaded successfully." << std::endl;
        
        model_path = "spanet/v2/model.onnx";
        std::cout << " -> Loading Run 3 ONNX model from: " << model_path << std::endl;
        spanet_inference = std::make_unique<SPANet::SPANetInference>(model_path, args.batch_size);
        std::cout << "    ONNX session loaded successfully." << std::endl;
    }

    if (args.runSPANetInference && args.nthread > 1) {
        std::cout << " -> SPANet inference requires single-threaded execution, setting nthread=1" << std::endl;
        args.nthread = 1;
    }

    if (args.nthread > 1) {
        ROOT::EnableImplicitMT(args.nthread);
        ROOT::EnableThreadSafety();
    }

    // Load df
    ROOT::RDataFrame df_ = ROOT::RDF::Experimental::FromSpec(input_spec);
    if (args.progress) { // progress bar isn't needed if using condor so turn off by default
        ROOT::RDF::Experimental::AddProgressBar(df_);
    }

    // Get sample category from config file
    std::string kind = getCategoryFromConfig(input_spec);
    std::cout << " -> Sample kind from config: " << kind << std::endl;

    // Set output file name and input type based on kind
    bool isData = false;
    bool isSignal = false;
    if (kind.find("data") != std::string::npos) {
        isData = true;
        if (output_file.empty()) {
            output_file = "data";
        }
    }
    else if (kind.find("sig") != std::string::npos) {
        // Handle categories like "sig", "sig_c2v_1p5_c3_1p0", etc.
        isSignal = true;
        if (output_file.empty()) {
            output_file = "sig";
        }
    }
    else if (kind.find("bkg") != std::string::npos) {
        // Handle categories like "bkg", "bkg_QCD", "bkg_ttbar", etc.
        if (output_file.empty()) {
            output_file = "bkg";
        }
    }
    else {
        std::cerr << "Unknown kind in config: " << kind << std::endl;
        std::cerr << "Expected: data, sig, or bkg*" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (args.makeBTagEfficiencies && isData) {
        std::cerr << "B-tag efficiencies can only be measured from MC samples" << std::endl;
        return EXIT_FAILURE;
    }
    // Background MC is normally nominal-only: the analysis is data driven, so the bkg
    // variations are never used. Warn but honour the flag -- a deliberate bkg syst pass
    // is a legitimate cross-check, it just should not happen by accident.
    if (args.systs && !isData && !isSignal) {
        std::cout << "\n"
                  << "  ############################################################\n"
                  << "  ##  WARNING: --systs on a BACKGROUND sample               ##\n"
                  << "  ############################################################\n"
                  << "  kind = '" << kind << "'. The JES/JER variation branches will be\n"
                  << "  written for background MC. The statistical analysis is data\n"
                  << "  driven, so these are not used by it -- this costs a large\n"
                  << "  multiple of the disk and CPU of the nominal-only production.\n"
                  << "  If that was not deliberate, drop --systs and submit signal\n"
                  << "  separately (run_rdf.py --kinds sig --systs).\n"
                  << "  ############################################################\n"
                  << std::endl;
    }
    setStoreSysts(args.systs);
    std::cout << " -> JES/JER variation branches: " << (args.systs ? "STORED (--systs)" : "not stored (nominal only)") << std::endl;

    bool makeSpanetTrainingdata = args.makeSpanetTrainingdata;
    if (!isSignal) {
        makeSpanetTrainingdata = false; // do not make training data for non-signal samples
    }

    // Define metadata
    auto df = defineMetadata(df_, isData);

    Cutflow::SetWeightCol(isData ? "1" : "weight");

    if (args.cutflow) Cutflow::Enable();

    // Run analysis
    if (isData) {
        std::cout << " -> Running data analysis" << std::endl;
        df = applyDataCorrections(df);
        df = runAnalysis(df, args.ana, args.run_number, isSignal, isData, spanet_inference.get(), spanet_inference_run2.get(), args.runSPANetInference);
        df = applyDataWeights(df);
        //df = removeDuplicates(df);
    } else {
        std::cout << " -> Running MC analysis" << std::endl;
        df = applyMCCorrections(df);
        df = runAnalysis(df, args.ana, args.run_number, isSignal, isData,
                         spanet_inference.get(), spanet_inference_run2.get(),
                         args.runSPANetInference, makeSpanetTrainingdata);
        if (args.makeBTagEfficiencies) {
            const int nslots = args.nthread > 1 ? args.nthread : 1;
            std::cout << " -> Saving raw b-tag efficiency histograms" << std::endl;
            saveBTagEfficiencyHistograms(df, output_dir, output_file, args.ana,
                                         btag_efficiency_metadata->year, btag_efficiency_metadata->sample, nslots);
            return 0;
        }
        const bool applyBTagScaleFactors = channelBTagScaleFactors && !args.skipBTagScaleFactors;
        const auto btag_working_points = applyBTagScaleFactors
            ? bTagWorkingPointsForChannel(args.ana) : std::vector<std::string>{};
        const auto btag_contexts = applyBTagScaleFactors
            ? prepareBTagEfficiencyContexts(getBTagEfficiencyMetadata(input_spec), args.ana, btag_working_points)
            : BTagEfficiencyContexts{};
        if (args.skipBTagScaleFactors)
            std::cout << " -> B-tag SF application disabled by --skip-btag-sf" << std::endl;
        else if (!channelBTagScaleFactors)
            std::cout << " -> B-tag SF application disabled for " << args.ana
                      << " by applybtag.yaml" << std::endl;
        else
            std::cout << " -> Applying b-tag SFs" << std::endl;
        df = applyMCWeights(df, args.ana, applyBTagScaleFactors, btag_working_points, btag_contexts);
    }

    Cutflow::Add(df, "After SFs and corrections");

    if (isSignal && makeSpanetTrainingdata) {
        std::cout << " -> Saving SPANet training data" << std::endl;
        saveSpanetSnapshot(df, output_dir, output_file);
        Cutflow::Print();
        return 0; // Exit after saving training data
    }

    saveSnapshot(df, output_dir, output_file, isSignal, args.dumpInput, args.storeHLT);
    Cutflow::Print();

    return 0;
}
