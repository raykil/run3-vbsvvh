# Script to build a giant string of logic for RDF to evaluate


DS_DICT_HT = {
    "2016" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "ds_name_1" : [
                "trg_name_1",
            ],
        },
    },
    "2017" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "ds_name_1" : [
                "trg_name_1",
            ],
        },
    },
    "2018" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "ds_name_1" : [
                "trg_name_1",
            ],
        },
    },
}

DS_DICT_0lep0FJ = {
    "2016" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetHT" : [
                "HLT_QuadPFJet_BTagCSV_p016_VBF_Mqq460",
                "HLT_QuadPFJet_BTagCSV_p016_VBF_Mqq500",
                "HLT_QuadPFJet_BTagCSV_p016_p11_VBF_Mqq200",
                "HLT_QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240",

                "HLT_PFHT400_SixJet30_DoubleBTagCSV_p056",
            ],
        },
    },
    "2017" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetHT" : [
                "HLT_PFHT380_SixJet32_DoubleBTagCSV_p075",

                "HLT_PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
                "HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2",
                

                # don't exist in data
                # "HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2",
                # "HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1",
                # "HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2",
                # "HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1",
                # "HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2",
                # "HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1",
                # "HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2",
                # "HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1",
            ],
        },
    },
    "2018" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetHT" : [
                "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2",

                "HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                "HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2"
            ],
        },
    },
    "2022" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
                "HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2",
                "HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2",
                "HLT_QuadPFJet70_50_40_30_PFBTagParticleNet_2BTagSum0p65",
                "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                "HLT_QuadPFJet70_50_45_35_PFBTagParticleNet_2BTagSum0p65",
                "HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet98_83_71_15_PFBTagDeepJet_1p3_VBF2",

                "HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                "HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94",
            ],
        },
    },
    "2023" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
                "HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2",
                "HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2",
                "HLT_QuadPFJet70_50_40_30_PFBTagParticleNet_2BTagSum0p65",
                "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                "HLT_QuadPFJet70_50_45_35_PFBTagParticleNet_2BTagSum0p65",
                "HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                "HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_QuadPFJet98_83_71_15_PFBTagDeepJet_1p3_VBF2",

                "HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                "HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94",

                #single btag
                # "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59",
                # "HLT_PFHT450_SixPFJet36_PFBTagDeepJet_1p59",
            ],
        },
    },

    "2024" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1",
                "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2",
                "HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1",
                "HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2",
                "HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1",
                "HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2",

                "HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50",
                # "HLT_PFHT450_SixPFJet36_PNetBTag0p35",
            ],
        },
    },

    # 2025 reuses the 2024 paths.
    "2025" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1",
                "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2",
                "HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1",
                "HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2",
                "HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1",
                "HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2",

                "HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50",
                # "HLT_PFHT450_SixPFJet36_PNetBTag0p35",
            ],
        },
    },
}

DS_DICT_0lep1FJ = {
    "2016" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetHT" : [
                "HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
                "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50",
                "HLT_AK8PFJet360_TrimMass30",
                "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5",
                "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5",
                "HLT_PFHT800",
                "HLT_PFHT900",
                "HLT_PFJet450",
            ],
        },
    },
    "2017" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetHT" : [
                "HLT_AK8PFHT800_TrimMass50",
                "HLT_AK8PFJet330_BTagCSV_p17",
                "HLT_AK8PFJet400_TrimMass30",
                "HLT_AK8PFJet420_TrimMass30",
                "HLT_AK8PFJet500",
                "HLT_PFHT1050",
                "HLT_PFJet500",

            ],
        },
    },
    "2018" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetHT" : [
                "HLT_AK8PFHT800_TrimMass50",
                "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2",
                "HLT_AK8PFJet400_TrimMass30",
                "HLT_AK8PFJet420_TrimMass30",
                "HLT_AK8PFJet500",
                "HLT_PFHT1050",
                "HLT_PFJet500",
            ],
        },
    },
    "2022" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                "HLT_AK8PFJet425_SoftDropMass40",
            ],
        },
    },
    "2023" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06",
                "HLT_AK8PFJet425_SoftDropMass40",
            ],
        },
    },

    "2024" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06",
                "HLT_AK8PFJet425_SoftDropMass30",
            ],
        },
    },

    # 2025 reuses the 2024 paths.
    "2025" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "JetMET" : [
                "HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06",
                "HLT_AK8PFJet425_SoftDropMass30",
            ],
        },
    },
}


DS_DICT_MET = {
    "2016" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "ds_name_1" : [
                "trg_name_1",
            ],
        },
    },
    "2017" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "ds_name_1" : [
                "trg_name_1",
            ],
        },
    },
    "2018" : {
        "ds_prio_lst" : None,
        "ds_trg_dict" : {
            "ds_name_1" : [
                "trg_name_1",
            ],
        },
    },
}

DS_DICT_SINGLELEP = {
    "2016" : {
        "ds_prio_lst" : ["SingleMuon", "SingleElectron"],
        "ds_trg_dict" : {
            "SingleMuon" : [
                "HLT_IsoMu24",
                "HLT_IsoTkMu24",
                "HLT_IsoMu22",
                "HLT_IsoTkMu22",
                "HLT_IsoMu20",
                "HLT_IsoTkMu20",
            ],
            "SingleElectron" : [
                "HLT_Ele27_eta2p1_WPTight_Gsf",
                "HLT_Ele27_WPTight_Gsf",
                "HLT_Ele25_eta2p1_WPTight_Gsf",
                "HLT_Ele27_eta2p1_WPLoose_Gsf",
            ],
        },
    },
    "2017" : {
        "ds_prio_lst" : ["SingleMuon", "SingleElectron"],
        "ds_trg_dict" : {
            "SingleMuon" : [
                "HLT_IsoMu27",
            ],
            "SingleElectron" : [
                "HLT_Ele32_WPTight_Gsf_L1DoubleEG",
                "HLT_Ele35_WPTight_Gsf",
                "HLT_Ele38_WPTight_Gsf",
                "HLT_Ele40_WPTight_Gsf",
            ],
        },
    },
    "2018" : {
        "ds_prio_lst" : ["SingleMuon", "EGamma"],
        "ds_trg_dict" : {
            "SingleMuon" : [
                "HLT_IsoMu24",
            ],
            "EGamma" : [
                "HLT_Ele32_WPTight_Gsf",
            ],
        },
    },
    "2024" : {
        "ds_prio_lst" : ["Muon", "EGamma"],
        "ds_trg_dict" : {
            "Muon" : [
                "HLT_IsoMu24",
            ],
            "EGamma" : [
                "HLT_Ele30_WPTight_Gsf",
            ],
        },
    },
    "2025" : {
        "ds_prio_lst" : ["Muon", "EGamma"],
        "ds_trg_dict" : {
            "Muon" : [
                "HLT_IsoMu24",
            ],
            "EGamma" : [
                "HLT_Ele30_WPTight_Gsf",
            ],
        },
    },

}
DS_DICT_MULTILEP = {
    "2016" : {
        "ds_prio_lst" : ["DoubleMuon", "MuonEG", "DoubleEG", "SingleMuon", "SingleElectron"],
        "ds_trg_dict" : {
            "DoubleMuon" : [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
                "HLT_TripleMu_12_10_5",
            ],
            "MuonEG" : [
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu17_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_DiMu9_Ele9_CaloIdL_TrackIdL",
                "HLT_Mu8_DiEle12_CaloIdL_TrackIdL",
            ],
            "DoubleEG" : [
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL",
                "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
            ],
            "SingleMuon" : [
                "HLT_IsoMu24",
                "HLT_IsoTkMu24",
                "HLT_IsoMu22_eta2p1",
                "HLT_IsoTkMu22_eta2p1",
                "HLT_IsoMu22",
                "HLT_IsoTkMu22",
                "HLT_IsoMu27",
                "HLT_IsoMu20",
                "HLT_IsoTkMu20",
            ],
            "SingleElectron" : [
                "HLT_Ele27_WPTight_Gsf",
                "HLT_Ele25_eta2p1_WPTight_Gsf",
                "HLT_Ele27_eta2p1_WPLoose_Gsf",
            ],
        },
    },

    "2017" : {
        "ds_prio_lst" : ["DoubleMuon", "MuonEG", "DoubleEG", "SingleMuon", "SingleElectron"],
        "ds_trg_dict" : {
            "DoubleMuon" : [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
                "HLT_TripleMu_12_10_5",
                "HLT_TripleMu_10_5_5_D2",
            ],
            "MuonEG" : [
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
                "HLT_Mu8_DiEle12_CaloIdL_TrackIdL",
                "HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            ],
            "DoubleEG" : [
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL",
                "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
            ],
            "SingleMuon" : [
                "HLT_IsoMu24",
                "HLT_IsoMu27",
            ],
            "SingleElectron" : [
                "HLT_Ele32_WPTight_Gsf",
                "HLT_Ele35_WPTight_Gsf",
                "HLT_Ele38_WPTight_Gsf",
                "HLT_Ele40_WPTight_Gsf",
            ],
        },
    },

    "2018" : {
        "ds_prio_lst" : ["DoubleMuon", "MuonEG", "SingleMuon", "EGamma"],
        "ds_trg_dict" : {
            "DoubleMuon" : [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            ],
            "MuonEG" : [
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            ],
            "SingleMuon" : [
                "HLT_IsoMu24",
                "HLT_IsoMu27",
            ],
            "EGamma" : [
                "HLT_Ele32_WPTight_Gsf",
                "HLT_Ele35_WPTight_Gsf",
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_DoubleEle25_CaloIdL_MW",
            ],
        },
    },
    "2024" : {
        "ds_prio_lst" : ["Muon", "MuonEG", "EGamma"],
        "ds_trg_dict" : {
            "Muon" : [
                "HLT_IsoMu24",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
                "HLT_TripleMu_10_5_5_DZ",
                "HLT_TripleMu_12_10_5",
            ],
            "MuonEG" : [
                # From ZZ Run 3 CMS AN-25-159
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
                "HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ",
            ],
            "EGamma" : [
                "HLT_Ele30_WPTight_Gsf",
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_DoubleEle25_CaloIdL_MW",
            ],
        },
    },
    "2025" : {
        # Just copied 2024 for now
        "ds_prio_lst" : ["Muon", "MuonEG", "EGamma"],
        "ds_trg_dict" : {
            "Muon" : [
                "HLT_IsoMu24",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            ],
            "MuonEG" : [
                # From ZZ Run 3 CMS AN-25-159
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
                "HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ",
            ],
            "EGamma" : [
                "HLT_Ele30_WPTight_Gsf",
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"
            ],
        },
    },

}

# Takes a lits of triggers, makes a string of "(trg1 == true) || (trg2 == true)..." 
def get_or_of_trgs(trg_lst):
    out_str = ""
    for i,trg in enumerate(trg_lst):
        if i < (len(trg_lst) - 1):
            out_str = out_str + f"({trg} == true) || "
        else:
            out_str = out_str + f"({trg} == true)"

    if out_str == "":
        return None
    else:
        return f"({out_str})"


# Given a ds name and a priority ordered list of ds names, return all ds with higher priority
#     - Used for finding the list of triggers to check for overlap removal
#     - We will extract the priority ordered list of ds names from ds_dict given a year
#     - Then e.g., if my_ds is dsC, and ds_prio_lst is [dsA, dsB, dsC, dsD], will return trgs for dsA and dsB
#     - Assumes my_ds only shows up once in the list (should always be true)
def get_higher_priority_ds_trgs(ds_dict, my_ds, year):

    # Get the ds_prio_lst from the ds_dict
    ds_prio_lst = ds_dict[year]["ds_prio_lst"]

    # Get the triggers in ds that are higher priority than my_ds
    out_lst = []
    for ds_name in ds_prio_lst:
        if ds_name == my_ds: break
        else: out_lst = out_lst + ds_dict[year]["ds_trg_dict"][ds_name]

    return out_lst



# Main wrapper function for making the logical OR string for a given dataset dictionary
def dump_logical_or_string(ds_dict,do_overlap_removal):

    # The final output string
    out_str = ""

    # Loop over the years
    for i,year in enumerate(ds_dict):

        # Check for self-consistency in the ds_dict for dataset priority
        if do_overlap_removal:
            ds_names_prio_lst = ds_dict[year]["ds_prio_lst"]
            ds_names_from_keys = ds_dict[year]["ds_trg_dict"].keys()
            if len(ds_names_prio_lst) != len(ds_names_from_keys): raise Exception(f"Mismatch length between ds_prio_lst and ds_trg_dict keys in year: {year}")
            if set(ds_names_prio_lst) != set(ds_names_from_keys): raise Exception(f"Mismatch names between ds_prio_lst and ds_trg_dict keys in year: {year}")

        # The string we will build up for this year
        passes_no_overlap = ""

        # Loop over datasets in this year
        for j, ds_name in enumerate(ds_dict[year]["ds_trg_dict"]):

            # Grab the list of triggers for this ds
            trgs_for_this_ds = ds_dict[year]["ds_trg_dict"][ds_name]

            # Build a string of ORs between all of the triggers that pass
            trg_passes   = get_or_of_trgs(trgs_for_this_ds)

            # Build a string of ORs between all of the triggers that overlap
            if do_overlap_removal:
                trgs_for_higher_priority_ds = get_higher_priority_ds_trgs(ds_dict, ds_name,year)
                trg_overlaps = get_or_of_trgs(trgs_for_higher_priority_ds)
            else:
                trg_overlaps = None


            # Append this to the string for this year (note short dataset name e.g. "MuonEG" is called shortname in the RDF)
            if trg_overlaps is None:
                # No overlap to remove (either this is the highest priority dataset, or we are not doing trigger overlap removal)
                passes_no_overlap = passes_no_overlap + f"( ((shortname==\\\"{ds_name}\\\") || !isData)  && {trg_passes} )"
            else:
                # We are removing overlap, so build a "passes trigger and !overlap" type of string
                passes_no_overlap = passes_no_overlap + f"( (((shortname==\\\"{ds_name}\\\") || !isData) && {trg_passes}) && !({trg_overlaps} && isData) )"


            # Append and OR if this is not the last one
            if j < (len(ds_dict[year]["ds_trg_dict"]) - 1):
                passes_no_overlap = passes_no_overlap + " || "
            # Otherwise if we're done, wrap the whole thing in parentheses
            else:
                passes_no_overlap = f"({passes_no_overlap})"

        # Append to the final out string
        out_str = out_str + f"(is{year} && {passes_no_overlap})"

        # Append the OR if this is not the last one
        if i < len(ds_dict)-1:
            out_str = out_str + " || "


    # Dump the final output string to the screen
    print("")
    print(out_str)



############# Main function #############

def main():

    # dump_logical_or_string(DS_DICT_HT,do_overlap_removal=False)
    # dump_logical_or_string(DS_DICT_MET,do_overlap_removal=False)
    # dump_logical_or_string(DS_DICT_SINGLELEP,do_overlap_removal=True)
    dump_logical_or_string(DS_DICT_MULTILEP,do_overlap_removal=True)
    # dump_logical_or_string(DS_DICT_0lep0FJ, do_overlap_removal=False)
    # dump_logical_or_string(DS_DICT_0lep1FJ, do_overlap_removal=False)

main()





