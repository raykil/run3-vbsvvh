import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def chooseOverlaps():
    OPTIONS = {
        'run2': {
            'DY_M-50': {
                'A': [
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v5_NANOAODSIM",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2_NANOAODSIM",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2_NANOAODSIM"
                ],
                'B': [
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'WToLNu': {
                'A': [
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'WW': {
                'A': [
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'WZ': {
                'A': [
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ZZ': {
                'A': [
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ttW': {
                'A': [
                    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ttZ': {
                'A': [
                    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ZH_HToWW': {
                'A': [
                    "HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            }
        },
        'run3': {
            'DY_M-50': {
                'A': [
                    "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4_NANOAODSIM",
                    "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4_NANOAODSIMSummer24for2025",
                    "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6_NANOAODSIM",
                    "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6_NANOAODSIMSummer24for2025",
                    "DYto2Tau-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v7_NANOAODSIM",
                    "DYto2Tau-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v7_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'QCD': {
                'A': [
                    "QCD_Bin-PT-1000to1500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-1000to1500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-120to170_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-120to170_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-1500to2000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-1500to2000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-170to300_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-170to300_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-2000to2500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-2000to2500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-2500to3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-2500to3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-300to470_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-300to470_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-470to600_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-470to600_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-50to80_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-50to80_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-600to800_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-600to800_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-800to1000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-800to1000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-80to120_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-80to120_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "QCD-4Jets_Bin-HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'WToLNu': {
                'A': [
                    "WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'WW': {
                'A': [
                    "WW_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WW_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'WZ': {
                'A': [
                    "WZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WZtoL3Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZtoL3Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'ZZ': {
                'A': [
                    "ZZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto2Nu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto2Nu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto4Q-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto4Q-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'QCD_ZZTo4L': {
                'A': [
                    "ZZJJto4L-QCD_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZJJto4L-QCD_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'EWK_SSWW': {
                'A': [
                    "VBS-SSWW-LL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "VBS-SSWW-LL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "VBS-SSWW-TL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "VBS-SSWW-TL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WWJJto2L2Nu-SS-noTop-EWK_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWJJto2L2Nu-SS-noTop-EWK_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            }
        }
    }
    CHOICES = {
        'run2': {
            'DY_M-50': 'A',
            'WToLNu': 'A',
            'WW': 'A',
            'WZ': 'A',
            'ZZ': 'A',
            'ttW': 'A',
            'ttZ': 'A',
            'ZH_HToWW': 'A'
        },
        'run3': {
            'DY_M-50': 'A',
            'QCD': 'A',
            'WToLNu': 'A',
            'WW': 'A',
            'WZ': 'A',
            'ZZ': 'A',
            'QCD_ZZTo4L': 'A',
            'EWK_SSWW': 'A'
        }
    }
    return {sample for run, groups in OPTIONS.items() for group, opts in groups.items() for opt, samples in opts.items() if opt != CHOICES[run][group] for sample in samples}

class ABCDDataSet(Dataset):
    def __init__(self, data, constraint_data, labels, weights):
        self.data = data
        self.constraint_data = constraint_data
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.constraint_data[index],
            self.labels[index],
            self.weights[index],
        )

def get_dataloader(dnn_input_data, constraint_data, labels, weights, batch_size, use_sampler=True, is_validation=False):
    dataset = ABCDDataSet(dnn_input_data, constraint_data, labels, weights)

    if use_sampler:
        safe_weights = weights.clone().detach().float()
        safe_weights = torch.clamp(safe_weights, min=0)
        if torch.sum(safe_weights) <= 0:
            safe_weights = torch.ones_like(safe_weights)

        sampler = WeightedRandomSampler(
            weights=safe_weights,
            num_samples=len(safe_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=not is_validation)


def distance_corr(
        var_1:torch.tensor,
        var_2:torch.tensor,
        normedweight:torch.tensor,
        power=1,
        )->torch.tensor:

    # Normalize the weights
    normedweight = normedweight/torch.sum(normedweight)*len(var_1)

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    return dCorr

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, use_batchnorm=True, dropout=0.0):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")

        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

class ABCDModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        learning_rate=1e-3,
        bce_weight=1.0,
        disco_lambda=0.0,
        flavor="single",
        use_batchnorm=True,
        dropout=0.0,
        weight_decay=1e-2,
        label_smoothing=0.0,
        use_lr_scheduler=True,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        lr_scheduler_min_lr=1e-6,
    ):
        super().__init__()

        if flavor not in {"single", "double"}:
            raise ValueError("flavor must be either 'single' or 'double'")

        self.flavor = flavor
        self.bce_weight = bce_weight
        self.disco_lambda = disco_lambda
        self.label_smoothing = label_smoothing

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr

        # ---- model ----
        if flavor == "single":
            self.model = MLP(
                input_size=input_size,
                hidden_layers=hidden_layers,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )
        else:
            self.model = nn.ModuleList([
                MLP(
                    input_size=input_size,
                    hidden_layers=hidden_layers,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                ),
                MLP(
                    input_size=input_size,
                    hidden_layers=hidden_layers,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                ),
            ])

    # ---------------- forward ----------------
    def forward(self, x):
        if self.flavor == "single":
            return self.model(x)

        return torch.stack(
            [self.model[0](x), self.model[1](x)],
            dim=1
        )

    # ---------------- loss ----------------
    def compute_loss(self, batch):
        data, constraint_data, labels, weights = batch

        logits = self(data)
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        # label smoothing
        smoothed_labels = labels.float()
        if self.label_smoothing > 0:
            smoothed_labels = (
                smoothed_labels * (1.0 - self.label_smoothing)
                + 0.5 * self.label_smoothing
            )

        scores = torch.sigmoid(logits)

        safe_weights = torch.clamp(weights, min=0.0)
        safe_weights = safe_weights / (torch.mean(safe_weights) + 1e-12)

        # BCE over heads
        bce_components = []
        for h in range(logits.shape[1]):
            bce_components.append(
                F.binary_cross_entropy_with_logits(
                    logits[:, h],
                    smoothed_labels,
                    weight=safe_weights,
                )
            )
        bce = torch.stack(bce_components).sum()

        # ---------------- disco term ----------------
        bkg_mask = labels < 0.5
        disco_term = torch.zeros((), device=logits.device)

        if bkg_mask.sum() > 0:
            bkg_scores = scores[bkg_mask]
            bkg_constraint = constraint_data[bkg_mask, 0]
            bkg_weights = safe_weights[bkg_mask]

            bkg_score_0 = bkg_scores[:, 0]

            if self.flavor == "single":
                if (
                    bkg_score_0.max() - bkg_score_0.min() > 1e-8
                    and bkg_constraint.max() - bkg_constraint.min() > 1e-8
                ):
                    disco_term = distance_corr(
                        bkg_score_0,
                        bkg_constraint,
                        bkg_weights,
                        power=2,
                    )
            else:
                bkg_score_1 = bkg_scores[:, 1]

                if (
                    bkg_score_0.max() - bkg_score_0.min() > 1e-8
                    and bkg_score_1.max() - bkg_score_1.min() > 1e-8
                ):
                    disco_term = distance_corr(
                        bkg_score_0,
                        bkg_score_1,
                        bkg_weights,
                        power=2,
                    )

        total_loss = self.bce_weight * bce + self.disco_lambda * disco_term

        return total_loss, bce, disco_term, bce_components


def _batched_scores(model, features_tensor, flavor, device):
    scores_0 = []
    scores_1 = []

    with torch.no_grad():
        for batch, x, y, z in features_tensor:
            batch = batch.to(device)
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

def _plot_roc_curves(data, flavor, output_path, tag=""):
    labels = np.asarray(data["label"])
    weights = np.asarray(data["weight"])

    plt.figure(figsize=(8, 7))
    roc_series = [("DNN", "dnn_score")] if flavor == "single" else [("DNN 0", "dnn_0_score"), ("DNN 1", "dnn_1_score")]
    for label_name, score_col in roc_series:
        fpr, tpr, _ = roc_curve(labels, np.asarray(data[score_col]), sample_weight=weights)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label_name} (AUC={auc(fpr, tpr):.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({flavor} flavor, {tag})")
    plt.legend(loc="lower right")
    plt.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.4)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
