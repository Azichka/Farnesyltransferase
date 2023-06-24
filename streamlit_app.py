# Import start
import streamlit as st
from streamlit_ketcher import st_ketcher

import copy

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdDistGeom
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem import AllChem
from rdkit import RDConfig
from rdkit.Chem.PandasTools import ChangeMoleculeRendering
from rdkit.Geometry import Point3D

IPythonConsole.drawOptions.comicMode=True

import mordred
from mordred import Calculator, descriptors
import espsim

from joblib import Parallel, delayed
import pickle
import numpy as np
from scipy.spatial import distance
import sklearn
from sklearn.manifold import TSNE
import os
import pandas as pd
from IPython.display import SVG

from bokeh.plotting import figure,ColumnDataSource
from bokeh.models import HoverTool

param = rdDistGeom.ETKDGv3()
param.pruneRmsThresh = 0
param.boxSizeMult = 2
param.useRandomCoords = True
param.enforceChirality = True
param.useSmallRingTorsions = True
param.useMacrocycleTorsions = True
param.randomSeed = 1
param.numThreads = -1

fdefName = 'BaseFeatures.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
sigFactory = SigFactory(factory,minPointCount=2,maxPointCount=3, trianglePruneBins=False)
sigFactory.SetBins([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9), (9, 100)])
sigFactory.Init()

filerParams = FilterCatalogParams()
filerParams.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
catalog = FilterCatalog(filerParams)
# Import end

num_confs = 3

@st.cache_resource()
def load_models():
    model_0 = pickle.load(open('models_and_scalers/model_0.pkl', 'rb'))
    model_1 = pickle.load(open('models_and_scalers/model_1.pkl', 'rb'))
    model_2 = pickle.load(open('models_and_scalers/model_2.pkl', 'rb'))
    model_3 = pickle.load(open('models_and_scalers/model_3.pkl', 'rb'))
    model_4 = pickle.load(open('models_and_scalers/model_4.pkl', 'rb'))

    minmax_0 = pickle.load(open('models_and_scalers/minmax_0.pkl', 'rb'))
    minmax_1 = pickle.load(open('models_and_scalers/minmax_1.pkl', 'rb'))
    minmax_2 = pickle.load(open('models_and_scalers/minmax_2.pkl', 'rb'))
    minmax_3 = pickle.load(open('models_and_scalers/minmax_3.pkl', 'rb'))
    minmax_4 = pickle.load(open('models_and_scalers/minmax_4.pkl', 'rb'))

    tsne_model = pickle.load(open('models_and_scalers/tsne_res.pkl', 'rb'))

    models = [model_0, model_1, model_2, model_3, model_4]
    scalers = [minmax_0, minmax_1, minmax_2, minmax_3, minmax_4]

    return models, scalers, tsne_model

@st.cache_data()
def load_data():

    AD = pickle.load(open('data/AD.pkl', 'rb'))
    DESCRIPTORS = pickle.load(open('data/DESCRIPTORS.pkl', 'rb'))
    NNN = pickle.load(open('data/NNN.pkl', 'rb'))

    return AD, DESCRIPTORS, NNN

@st.cache_data()
def load_reference_mols():
    sup = Chem.SDMolSupplier('data/reference_mols.sdf')
    mols = [Chem.AddHs(mol, addCoords = True) for mol in sup]
    return mols

@st.cache_data()
def load_modelling_set():
    df = pd.read_csv('data/modelling_set.csv')
    df_0 = df[df['bioclass'] == 0]
    df_1 = df[df['bioclass'] == 1]
    mols_0 = [Chem.MolFromSmiles(x) for x in df_0['Smiles']]
    mols_1 = [Chem.MolFromSmiles(x) for x in df_1['Smiles']]
    fps_active = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    j,2,2048, useFeatures=False, useBondTypes=True) for j in mols_1]
    fps_inactive = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    j,2,2048, useFeatures=False, useBondTypes=True) for j in mols_0]
    r_mols = [Chem.MolFromSmiles(x) for x in df['Smiles']]
    return df, fps_active, fps_inactive, r_mols

@st.cache_data()
def init_calc():

    calc1 = Calculator()

    calc1.register(mordred.MolecularId.MolecularId('hetero', False, 1e-10))
    calc1.register(mordred.TopologicalCharge.TopologicalCharge('mean', 3))
    calc1.register(mordred.BalabanJ.BalabanJ())
    calc1.register(mordred.MoeType.SlogP_VSA(2))
    calc1.register(mordred.Autocorrelation.ATS(3, 'd'))
    calc1.register(mordred.EState.AtomTypeEState('count', 'dsN'))
    calc1.register(mordred.InformationContent.ComplementaryIC(3))
    calc1.register(mordred.MoeType.PEOE_VSA(5))
    calc1.register(mordred.CarbonTypes.CarbonTypes(1, 1))
    calc1.register(mordred.Lipinski.Lipinski())
    calc1.register(mordred.BaryszMatrix.BaryszMatrix('i', 'SM1'))
    calc1.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
    calc1.register(mordred.LogS.LogS())
    calc1.register(mordred.Autocorrelation.ATSC(3, 'p'))
    calc1.register(mordred.Autocorrelation.MATS(3, 'v'))
    calc1.register(mordred.CarbonTypes.CarbonTypes(3, 3))
    calc1.register(mordred.EState.AtomTypeEState('count', 'sNH2'))
    calc1.register(mordred.Autocorrelation.ATSC(5, 'p'))
    calc1.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
    calc1.register(mordred.Autocorrelation.AATS(7, 'i'))
    calc1.register(mordred.Autocorrelation.ATSC(1, 'Z'))
    calc1.register(mordred.Autocorrelation.MATS(6, 'i'))
    calc1.register(mordred.RingCount.RingCount(10, False, True, None, True))
    calc1.register(mordred.RingCount.RingCount(12, True, True, None, None))
    calc1.register(mordred.Autocorrelation.AATS(3, 'v'))
    calc1.register(mordred.Autocorrelation.ATSC(1, 'c'))
    calc1.register(mordred.Autocorrelation.GATS(2, 'm'))
    calc1.register(mordred.Autocorrelation.MATS(2, 'c'))
    calc1.register(mordred.Autocorrelation.GATS(1, 'dv'))
    calc1.register(mordred.AtomCount.AtomCount('N'))
    calc1.register(mordred.Autocorrelation.MATS(5, 'se'))
    calc1.register(mordred.InformationContent.ComplementaryIC(5))
    calc1.register(mordred.Autocorrelation.AATS(4, 'i'))

    calc2 = Calculator()

    calc2.register(mordred.MoeType.PEOE_VSA(12))
    calc2.register(mordred.Autocorrelation.ATSC(2, 'm'))
    calc2.register(mordred.Autocorrelation.AATS(3, 'd'))
    calc2.register(mordred.EState.AtomTypeEState('count', 'sssCH'))
    calc2.register(mordred.InformationContent.ComplementaryIC(3))
    calc2.register(mordred.BaryszMatrix.BaryszMatrix('se', 'SM1'))
    calc2.register(mordred.Autocorrelation.AATS(3, 'v'))
    calc2.register(mordred.Autocorrelation.AATS (7, 'i'))
    calc2.register(mordred.CPSA.RNCG())
    calc2.register(mordred.BCUT.BCUT('i', 0))
    calc2.register(mordred.Autocorrelation.ATSC(1, 'dv'))
    calc2.register(mordred.BCUT.BCUT('c', -1))
    calc2.register(mordred.TopologicalCharge.TopologicalCharge('mean', 3))
    calc2.register(mordred.MoeType.SlogP_VSA(8))
    calc2.register(mordred.Chi.Chi('path', 6, 'dv', True))
    calc2.register(mordred.Autocorrelation.ATS(0, 's'))
    calc2.register(mordred.Autocorrelation.MATS(3, 'd'))
    calc2.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
    calc2.register(mordred.HydrogenBond.HBondAcceptor())
    calc2.register(mordred.RingCount.RingCount(None, False, False, None, True))
    calc2.register(mordred.BCUT.BCUT('s', 0))
    calc2.register(mordred.Autocorrelation.GATS(2, 'v'))
    calc2.register(mordred.MoeType.VSA_EState(1))
    calc2.register(mordred.MolecularId.MolecularId('hetero', False, 1e-10))
    calc2.register(mordred.RingCount.RingCount(7, False, False, None, None))
    calc2.register(mordred.MoeType.SlogP_VSA(6))
    calc2.register(mordred.Autocorrelation.ATSC(0, 'Z'))
    calc2.register(mordred.EState.AtomTypeEState('count', 'dO'))
    calc2.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
    calc2.register(mordred.Autocorrelation.AATS(4, 'i'))
    calc2.register(mordred.Autocorrelation.ATSC(7, 'i'))
    calc2.register(mordred.CarbonTypes.CarbonTypes(2, 2))
    calc2.register(mordred.RingCount.RingCount(6, False, False, True, True))
    calc2.register(mordred.ExtendedTopochemicalAtom.EtaDeltaEpsilon('B'))
    calc2.register(mordred.Autocorrelation.ATSC(8, 'v'))
    calc2.register(mordred.RingCount.RingCount(None, False, True, False, None))
    calc2.register(mordred.Autocorrelation.ATS(3, 'd'))
    calc2.register(mordred.TopologicalCharge.TopologicalCharge('raw', 9))
    calc2.register(mordred.RotatableBond.RotatableBondsRatio())
    calc2.register(mordred.Autocorrelation.MATS(6, 'i'))
    calc2.register(mordred.Autocorrelation.MATS(4, 'Z'))
    calc2.register(mordred.Autocorrelation.ATSC(2, 'p'))
    calc2.register(mordred.Autocorrelation.GATS(1, 'i'))
    calc2.register(mordred.Autocorrelation.AATS(8, 'i'))
    calc2.register(mordred.Autocorrelation.ATS(7, 's'))
    calc2.register(mordred.MoeType.SlogP_VSA(4))

    calc3 = Calculator()

    calc3.register(mordred.BCUT.BCUT('dv', 0))
    calc3.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
    calc3.register(mordred.CarbonTypes.CarbonTypes(2, 3))
    calc3.register(mordred.Autocorrelation.GATS(4, 'se'))
    calc3.register(mordred.Autocorrelation.GATS(3, 'p'))
    calc3.register(mordred.Autocorrelation.GATS(2, 'p'))
    calc3.register(mordred.Autocorrelation.MATS(5, 'se'))
    calc3.register(mordred.Autocorrelation.GATS(7, 'Z'))
    calc3.register(mordred.Autocorrelation.AATS(4, 'dv'))
    calc3.register(mordred.EState.AtomTypeEState('count', 'aaS'))
    calc3.register(mordred.Autocorrelation.GATS(4, 'm'))
    calc3.register(mordred.Chi.Chi('cluster', 4, 'd', False))
    calc3.register(mordred.MoeType.SMR_VSA(6))
    calc3.register(mordred.RingCount.RingCount(11, False, True, None, None))
    calc3.register(mordred.BCUT.BCUT('c', -1))
    calc3.register(mordred.MoeType.SlogP_VSA(3))
    calc3.register(mordred.RingCount.RingCount(None, False, True, False, None))
    calc3.register(mordred.WalkCount.WalkCount(5, False, True))
    calc3.register(mordred.EState.AtomTypeEState('count', 'sNH2'))
    calc3.register(mordred.Autocorrelation.AATS(3, 'Z'))
    calc3.register(mordred.RingCount.RingCount(10, False, True, None, True))
    calc3.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
    calc3.register(mordred.MoeType.PEOE_VSA(8))
    calc3.register(mordred.MoeType.SMR_VSA(3))
    calc3.register(mordred.TopologicalCharge.TopologicalCharge('raw', 10))
    calc3.register(mordred.Autocorrelation.GATS(3, 's'))
    calc3.register(mordred.MoeType.SlogP_VSA(10))
    calc3.register(mordred.Autocorrelation.ATSC(2, 'dv'))
    calc3.register(mordred.MoeType.PEOE_VSA(9))
    calc3.register(mordred.RingCount.RingCount(None, False, False, True, True))
    calc3.register(mordred.TopologicalCharge.TopologicalCharge('raw', 9))
    calc3.register(mordred.Autocorrelation.MATS(6, 'i'))
    calc3.register(mordred.MoeType.PEOE_VSA(4))
    calc3.register(mordred.Autocorrelation.GATS(2, 'c'))
    calc3.register(mordred.MoeType.EState_VSA(3))
    calc3.register(mordred.CarbonTypes.CarbonTypes(1, 3))
    calc3.register(mordred.Autocorrelation.GATS(1, 'd'))
    calc3.register(mordred.MoeType.PEOE_VSA(12))
    calc3.register(mordred.Autocorrelation.ATSC(7, 'd'))
    calc3.register(mordred.RingCount.RingCount(10, False, True, True, True))
    calc3.register(mordred.Autocorrelation.AATS(7, 'i'))
    calc3.register(mordred.Autocorrelation.ATS(7, 's'))
    calc3.register(mordred.Autocorrelation.ATS(5, 'dv'))
    calc3.register(mordred.Autocorrelation.AATS(6, 'dv'))
    calc3.register(mordred.ExtendedTopochemicalAtom.EtaDeltaAlpha('B'))
    calc3.register(mordred.Autocorrelation.MATS(3, 'd'))
    calc3.register(mordred.Autocorrelation.ATSC(7, 'Z'))
    calc3.register(mordred.Autocorrelation.AATS(7, 'v'))
    calc3.register(mordred.HydrogenBond.HBondAcceptor())
    calc3.register(mordred.Autocorrelation.ATSC(2, 'm'))

    calc4 = Calculator()

    calc4.register(mordred.RingCount.RingCount(None, False, False, None, True))
    calc4.register(mordred.BCUT.BCUT('Z', 0))
    calc4.register(mordred.MoeType.PEOE_VSA(9))
    calc4.register(mordred.Autocorrelation.ATS(5, 'dv'))
    calc4.register(mordred.BCUT.BCUT('s', -1))
    calc4.register(mordred.MoeType.SlogP_VSA(10))
    calc4.register(mordred.MoeType.PEOE_VSA(4))
    calc4.register(mordred.Autocorrelation.AATSC(3, 'i'))
    calc4.register(mordred.MoeType.EState_VSA(5))
    calc4.register(mordred.Autocorrelation.ATSC(1, 'c'))
    calc4.register(mordred.EState.AtomTypeEState('count', 'ssCH2'))
    calc4.register(mordred.Chi.Chi('path', 6, 'dv', True))
    calc4.register(mordred.Autocorrelation.ATSC(2, 'dv'))
    calc4.register(mordred.HydrogenBond.HBondAcceptor())
    calc4.register(mordred.MoeType.EState_VSA(4))
    calc4.register(mordred.Autocorrelation.MATS(5, 'se'))
    calc4.register(mordred.Autocorrelation.ATS(7, 's'))
    calc4.register(mordred.BaryszMatrix.BaryszMatrix('se', 'SM1'))
    calc4.register(mordred.RingCount.RingCount(None, False, False, True, True))
    calc4.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
    calc4.register(mordred.Autocorrelation.ATS(8, 'dv'))
    calc4.register(mordred.EState.AtomTypeEState('count', 'dO'))
    calc4.register(mordred.RotatableBond.RotatableBondsRatio())
    calc4.register(mordred.Autocorrelation.ATSC(8, 'v'))
    calc4.register(mordred.EState.AtomTypeEState('count', 'aasN'))
    calc4.register(mordred.BCUT.BCUT('i', 0))
    calc4.register(mordred.AcidBase.AcidicGroupCount())
    calc4.register(mordred.RingCount.RingCount(12, True, True, None, None))
    calc4.register(mordred.Autocorrelation.AATS(7, 'd'))
    calc4.register(mordred.ExtendedTopochemicalAtom.EtaDeltaBeta(False))
    calc4.register(mordred.Autocorrelation.GATS(2, 'v'))
    calc4.register(mordred.Autocorrelation.GATS(1, 'm'))
    calc4.register(mordred.Autocorrelation.GATS(4, 'i'))
    calc4.register(mordred.EState.AtomTypeEState('count', 'sssCH'))
    calc4.register(mordred.EState.AtomTypeEState('count', 'aaO'))
    calc4.register(mordred.Autocorrelation.GATS(1, 'i'))
    calc4.register(mordred.TopologicalCharge.TopologicalCharge('mean', 3))
    calc4.register(mordred.RingCount.RingCount(None, False, False, True, None))
    calc4.register(mordred.PathCount.PathCount(5, True, False, True))

    calc5 = Calculator()

    calc5.register(mordred.MoeType.SlogP_VSA(3))
    calc5.register(mordred.EState.AtomTypeEState('sum', 'sCl'))
    calc5.register(mordred.RingCount.RingCount(6, False, False, False, True))
    calc5.register(mordred.Autocorrelation.ATSC(5, 'i'))
    calc5.register(mordred.Autocorrelation.GATS(4, 'c'))
    calc5.register(mordred.RingCount.RingCount(7, False, False, None, None))
    calc5.register(mordred.MoeType.PEOE_VSA(10))
    calc5.register(mordred.TopologicalCharge.TopologicalCharge('raw', 9))
    calc5.register(mordred.Autocorrelation.MATS(3, 'd'))
    calc5.register(mordred.Autocorrelation.GATS(2, 'p'))
    calc5.register(mordred.CarbonTypes.CarbonTypes(1, 3))
    calc5.register(mordred.InformationContent.ComplementaryIC(3))
    calc5.register(mordred.Autocorrelation.GATS(2, 'are'))
    calc5.register(mordred.EState.AtomTypeEState('count', 'aasN'))
    calc5.register(mordred.BCUT.BCUT('v', 0))
    calc5.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
    calc5.register(mordred.CPSA.RPCG())
    calc5.register(mordred.RingCount.RingCount(10, False, True, None, True))
    calc5.register(mordred.RingCount.RingCount(6, False, False, True, True))
    calc5.register(mordred.MoeType.EState_VSA(5))
    calc5.register(mordred.BalabanJ.BalabanJ())
    calc5.register(mordred.MolecularDistanceEdge.MolecularDistanceEdge(2, 2, 'C'))
    calc5.register(mordred.Autocorrelation.GATS(1, 'p'))
    calc5.register(mordred.RotatableBond.RotatableBondsRatio())
    calc5.register(mordred.Framework.Framework())
    calc5.register(mordred.MoeType.PEOE_VSA(2))
    calc5.register(mordred.MoeType.VSA_EState(3))
    calc5.register(mordred.Autocorrelation.GATS(5, 'se'))
    calc5.register(mordred.Autocorrelation.ATSC(8, 'v'))
    calc5.register(mordred.Autocorrelation.AATS(2, 'v'))
    calc5.register(mordred.EState.AtomTypeEState('count', 'sssCH'))
    calc5.register(mordred.ExtendedTopochemicalAtom.EtaVEMCount('s', True))

    return [calc1, calc2, calc3, calc4, calc5]

def apply_ml(mols, index):
    temp_dict = {}
    prb_desc = [np.array(calcs[index](m)) for m in mols]
    ref_desc = np.array(DESCRIPTORS[index])
    prb_desc = scalers[index].transform(prb_desc)
    for num in range(len(mols)):
        kn = 0
        try:
            dist = [distance.euclidean(prb_desc[num], ref_desc[x, :]) for x in range(ref_desc.shape[0])]
            vals = list(np.argsort(np.argsort(dist)))
            for n in range(NNN[index]):
                if dist[vals.index(n)] <= AD[index] * 2:
                    kn += 1

            if kn == NNN[index]:
                res  = models[index].predict_proba(prb_desc[num, :].reshape(1, -1))
                temp_dict.setdefault(num, res[0][1])
            else:
                temp_dict.setdefault(num, None)

        except:
            temp_dict.setdefault(num, None)
    return temp_dict

def create_confs(clean_mols):
    prbMols = []

    for mta in clean_mols:
        mta = Chem.AddHs(mta, addCoords=True)
        AllChem.EmbedMultipleConfs(mta, num_confs, param)
        if mta.GetNumConformers() == num_confs:
            AllChem.MMFFOptimizeMoleculeConfs(mta, maxIters=100, numThreads = -1)
            Chem.rdmolops.SetAromaticity(mta)
            prbMols.append(mta)
        else:
            prbMols.append(None)

    return prbMols

def align(reference_mols, prbMols, crippen_refs):

    alignedMols = []

    for prbMol in prbMols:

        if prbMol != None:

            alignedPrbMol = Chem.Mol(prbMol)
            alignedPrbMol.RemoveAllConformers()
            crippen_prb = rdMolDescriptors._CalcCrippenContribs(prbMol)

            for ID in range(len(reference_mols)):

                for prb_conf_id in [prb_conf.GetId() for prb_conf in prbMol.GetConformers()]:

                    prb_mol = Chem.Mol(prbMol, confId = prb_conf_id)

                    Chem.rdMolAlign.GetCrippenO3A(prb_mol, reference_mols[ID], crippen_prb,
                                                crippen_refs[ID], prb_conf_id, 0, maxIters=100).Align()

                    alignedPrbMol.AddConformer(prb_mol.GetConformer(), assignId = True)

            alignedMols.append(alignedPrbMol)

        else:

            alignedMols.append(None)

    return alignedMols

@st.cache_data()
def shape_and_electro(reference_mols, alignedMols):

    results_shape = {}
    results_electro = {}
    for x in range(len(alignedMols)):
        results_electro.setdefault(x, [])
        results_shape.setdefault(x, [])

    for alignedMol_id in range(len(alignedMols)):

        if alignedMols[alignedMol_id] != None:

            for alignedConf_id in [alignedConf.GetId() for alignedConf in alignedMols[alignedMol_id].GetConformers()]:

                if alignedConf_id % num_confs == 0:
                    refMol = reference_mols[alignedConf_id // num_confs]

                results_shape[alignedMol_id].append(1 - AllChem.ShapeTanimotoDist(alignedMols[alignedMol_id], refMol,
                                            confId1 = alignedConf_id, confId2 = 0, ignoreHs=True, vdwScale=1.0))

                results_electro[alignedMol_id].append(espsim.GetEspSim(alignedMols[alignedMol_id], refMol,
                                    metric = 'tanimoto', integrate = 'gauss', partialCharges = 'gasteiger',
                                    renormalize = True, randomseed = 1, prbCid = alignedConf_id, refCid = 0))

        else:

            results_shape[alignedMol_id].append(None)
            results_electro[alignedMol_id].append(None)

    mean_shape = [np.median(results_shape[key]) for key in list(results_shape) if results_shape[key] != [None]]

    max_electro = [np.max(results_electro[key]) for key in list(results_electro) if results_shape[key] != [None]]

    return mean_shape, max_electro

@st.cache_data()
def ph4(reference_mols, clean_mols):

    reference_mols_copy = copy.deepcopy(reference_mols)

    ref_fps = []

    for x in reference_mols_copy:
        x.RemoveAllConformers()
        ref_fps.append(Generate.Gen2DFingerprint(x,sigFactory))

    res = []

    for clean_mol in clean_mols:

        prb_fp = Generate.Gen2DFingerprint(clean_mol,sigFactory)

        sims = np.mean([DataStructs.TanimotoSimilarity(prb_fp, y) for y in ref_fps])

        res.append(sims)

    return res

def background_color(Ser):
    colors = []

    if Ser.name == 'ML':
        for x in Ser:
            if x == 'NA' or x == 'None' or x <= 0.3:
                colors.append('background-color: #ffc6c4')
            elif x > 0.7:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'SS':
        for x in Ser:
            if x > 0.4636:
                colors.append('background-color: #4ee44e')
            elif  x < 0.2814 or x == None:
                colors.append('background-color: #ffc6c4')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'PS':
        for x in Ser:
            if x >= 0.05:
                colors.append('background-color: #4ee44e')
            elif  x < 0.01 or x == None:
                colors.append('background-color: #ffc6c4')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'ES':
        for x in Ser:
            if x > 0.3698:
                colors.append('background-color: #4ee44e')
            elif  x < 0.2514 or x == None:
                colors.append('background-color: #ffc6c4')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'Subs':
        for x in Ser:
            colors.append('background-color: #ffffff')

    elif Ser.name == 'AA':
        for x in Ser:
            colors.append('background-color: #ffffff')

    elif Ser.name == 'HA':
        for x in Ser:
            colors.append('background-color: #ffffff')

    elif Ser.name == 'FCsp3':
        for x in Ser:
            colors.append('background-color: #ffffff')

    elif Ser.name == 'TIA':
        for x in Ser:
            if x >= 0.321:
                colors.append('background-color: #4ee44e')
            elif  x < 0.246:
                colors.append('background-color: #ffc6c4')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'TII':
        for x in Ser:
            if x >= 0.321:
                colors.append('background-color: #ffc6c4')
            elif  x < 0.246:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'qed':
        for x in Ser:
            if x <= 0.3:
                colors.append('background-color: #ffc6c4')
            elif x > 0.7:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffff80')

    elif Ser.name == 'TPSA':
        for x in Ser:
            if x <= 140:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'NRB':
        for x in Ser:
            if x <= 10:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'NHD':
        for x in Ser:
            if x <= 5:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'NHA':
        for x in Ser:
            if x <= 10:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'MW':
        for x in Ser:
            if x <= 500:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'LogP':
        for x in Ser:
            if x <= 5:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'Veber':
        for x in Ser:
            if x == 0:
                colors.append('background-color: #4ee44e')
            else:
                colors.append('background-color: #ffc6c4')

    elif Ser.name == 'Lipinski':
        for x in Ser:
            if x == 0:
                colors.append('background-color: #4ee44e')
            elif x == 1:
                colors.append('background-color: #ffff80')
            else:
                colors.append('background-color: #ffc6c4')

    return np.array(colors)

def text_color(Ser):
    colors = ['color: black' for x in range(len(Ser))]

    return np.array(colors)

def clearMols(mol):
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(rdMolStandardize.Uncharger().uncharge(
            rdMolStandardize.FragmentParent(rdMolStandardize.IsotopeParent(
                rdMolStandardize.Cleanup(mol)), skipStandardize=True)))
    return mol

def calc_desc(mol):
    desc = {}
    TPSA = Descriptors.TPSA(mol)
    desc['TPSA'] = TPSA
    NRB = Descriptors.NumRotatableBonds(mol)
    desc['NRB'] = NRB
    NHD = Descriptors.NumHDonors(mol)
    desc['NHD'] = NHD
    NHA = Descriptors.NumHAcceptors(mol)
    desc['NHA'] = NHA
    MW = Descriptors.MolWt(mol)
    desc['MW'] = MW
    LogP = Descriptors.MolLogP(mol)
    desc['LogP'] = LogP
    desc['FCsp3'] = Descriptors.FractionCSP3(mol)
    desc['AA'] = len(mol.GetAromaticAtoms())
    desc['HA'] = Descriptors.HeavyAtomCount(mol)
    desc['qed'] = Descriptors.qed(mol)
    desc['Veber'] = 2 - sum([TPSA <= 140, NRB <= 10])
    desc['Lipinski'] = 4 - sum([NHD <= 5, NHA <= 10, MW <= 500, LogP <= 5])

    return desc

def substructureFilter(mol):
    entries = list(catalog.GetMatches(mol))
    if len(entries) > 0:
        return(set([x.GetDescription() for x in entries]))
    else:
        return {'None'}

@st.cache_data()
def ml_result(mols):
    clean_mols = Parallel(n_jobs = -1, prefer='processes')(delayed(clearMols)(x) for x in mols)
    dicts = Parallel(n_jobs = -1, prefer='processes')(delayed(apply_ml)(clean_mols, index) for index in range(len(calcs)))
    ensemble = {}
    for key in list(dicts[0]):
        ensemble.setdefault(key, [x[key] for x in dicts])

    results = {'ML': []}
    for key in list(ensemble):
        temp_res = ensemble[key]
        if len(temp_res) >= 3 and None not in temp_res:
            results['ML'].append(np.mean(ensemble.pop(key)))
        else:
            results['ML'].append('None')

    desc = pd.DataFrame(results)

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
                j,2,2048, useFeatures=False, useBondTypes=True) for j in clean_mols]

    sim_active = [max(DataStructs.BulkTanimotoSimilarity(fp, fps_active)) for fp in fps]

    sim_inactive = [max(DataStructs.BulkTanimotoSimilarity(fp, fps_inactive)) for fp in fps]

    desc['TIA'] = sim_active
    desc['TII'] = sim_inactive

    return desc, clean_mols

@st.cache_data()
def additional_props(clean_mols, desc):

    add_props = Parallel(n_jobs = -1, prefer='processes')(delayed(calc_desc)(x) for x in clean_mols)
    add_props_df = {}
    for key in list(add_props[0]):
        add_props_df.setdefault(key, [x[key] for x in add_props])

    add_props_df = pd.DataFrame(add_props_df)

    subs = Parallel(n_jobs = -1, prefer='processes')(delayed(
            substructureFilter)(x) for x in clean_mols)

    desc = pd.concat([desc, add_props_df], axis = 1)
    desc['Subs'] = subs

    return desc

def moltosvg(mol,molSize=(300,200)):
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))

@st.cache_resource()
def image(df, desc, _tsne_model):

    mod_res = df[['X', 'Y', 'bioclass']]
    X_drug = mod_res.loc[968, 'X']
    Y_drug = mod_res.loc[968, 'Y']
    mod_res = mod_res.drop(index = 968)
    prb_res = _tsne_model.transform(desc[['TPSA', 'NRB', 'NHD', 'NHA', 'MW', 'LogP']].values)
    tsne_df_prb = pd.DataFrame(prb_res, columns=["X","Y"])
    tsne_df_prb['bioclass'] = 'NA'
    tsne_df = pd.concat([mod_res, tsne_df_prb], ignore_index = True)
    drug_mol= r_mols.pop(968)
    svgs_ref = [moltosvg(m).data for m in r_mols]
    svgs_prb = [moltosvg(m).data for m in mols]
    svgs = svgs_ref + svgs_prb
    svg_drug = moltosvg(drug_mol).data

    colors =  {0: "red", 1: "green", 'NA': "blue"}
    tsne_df['colors'] = tsne_df['bioclass'].map(colors)

    ChangeMoleculeRendering(renderer='PNG')

    source = ColumnDataSource(data=dict(x=tsne_df['X'], y=tsne_df['Y'], svgs=svgs, bio = tsne_df['bioclass'], colors = tsne_df['colors'] ))

    source_drug = ColumnDataSource(data=dict(x=X_drug, y=Y_drug, svgs=svg_drug, bio = 'drug', colors = 'purple' ))

    hover = HoverTool(tooltips="""
        <div>
            <div>@svgs{safe}
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">bioclass:  @bio</span>
                </div>
            </div>
        </div>
        """)
    interactive_map = figure(width=800, height=800, tools=['reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan',hover],
        title="Chemical Space")

    interactive_map.circle('x', 'y', size=8, source=source, color = 'colors',
        fill_alpha=0.2)
    interactive_map.circle('x', 'y', size=8, source=source_drug, color = 'colors',
        fill_alpha=0.2)

    return interactive_map

models, scalers, tsne_model = load_models()
AD,  DESCRIPTORS, NNN = load_data()
reference_mols = load_reference_mols()
crippen_refs = [rdMolDescriptors._CalcCrippenContribs(rm) for rm in reference_mols]
df, fps_active, fps_inactive, r_mols = load_modelling_set()
calcs = init_calc()

st.header('FARNESYLTRANSFERASE Project')
st.subheader('Here you can find farnesyltransferase related project which was created to facilitate \
more effective prioritization of compounds.' )
st.write('You can input your molecules in SMILES format in the field on the left side. Each molecule \
will be preprocessed which includes functional group standartisation, molecular neutralization and \
tautomer canonicalization. After preprocession step descriptive statistics for each molecule will be \
obtained which includes ML based bioactivity class prediction (98.3% accuracy on the validation set), \
scores of 2D pharmacophore (83% accuracy), shape (100% accuracy) and electrostatics (83% accuracy) alignment and some \
basic physicochemical properties including 6 descriptors that are included in Lipinski and Veber rules \
as well as QED, SP3 carbon fraction, number of heavy atoms and number of aromatic atoms. Also list of \
of unwanted substructures is included.')

default = 'C12C=C(Br)C=NC=1C(C1CCN(C(CC3CCN(C(=O)N)CC3)=O)CC1)C1C(Br)=CC(Cl)=CC=1CC2'
molecule = st.text_input("Molecule", default)
smiles = st_ketcher(molecule)

st.markdown(f"Smiles: ``{smiles}``")

with st.sidebar:
    st.header('Here you can input one or more SMILES to obtain prediction for them')
    sm = st.text_area('Input your smiles here. _**Every smiles must be in new row**_ or they will be perceived as wrong.', value = 'c1ccccc1\nC12C=C(Br)C=NC=1C(C1CCN(C(CC3CCN(C(=O)N)CC3)=O)CC1)C1C(Br)=CC(Cl)=CC=1CC2')
    sm = sm.split('\n')
    sm = [x for x in sm if x != '']
    st.write('If you want you can use 3D functionality. It includes estimation of shape, electrostatical potential and pharmacophore overlay. It is not particulary fast (around 1 second for each molecule) but you can give it a try.')
    checker = st.checkbox('Use 3D functionality?')
    st.write('Interactive chemical space visualization can be created. It is based on TSNE and \
    must be recalculated for each new set of molecules. Calculation will take around 10 seconds')
    space = st.checkbox('Create chemical space visualization?')

mols = []
for s in sm:
    try:
        m = Chem.MolFromSmiles(s)
        mols.append(m)
    except:
        continue

mols = [x for x in mols if x != None]

if len(mols) != len(sm):
    st.warning('Some smiles cannot be processed by RDKit. Either some smiles\
     not in new row or they are complitely invalid')

if len(mols) > 0:

    desc, clean_mols = ml_result(mols)

    desc = additional_props(clean_mols, desc)

    if checker:

        prbMols = create_confs(clean_mols)

        alignedMols = align(reference_mols, prbMols, crippen_refs)

        mean_shape, max_electro = shape_and_electro(reference_mols, alignedMols)

        mean_ph4 = ph4(reference_mols, clean_mols)

        desc.insert(3, 'SS', mean_shape)
        desc.insert(3, 'ES', max_electro)
        desc.insert(3, 'PS', mean_ph4)

    st.dataframe(desc.style.apply(background_color).apply(text_color).format(precision = 2))

    st.write('Every cell has color interpritation. Green is desired value, yellow is acceptable, red is undesired. \
    White value of the cell means that it is hard to give unambiguous interpretation.')
    st.write('ML column: results of ensemble of 7 ML models. Somitimes it gives none or NA which means that requered descriptors can not be calculated \
    for particular molecule or molecule out of aplicability domain;')
    st.write('TIA column: tanimoto similarity based on ECFP2 2048 bits. Shows structurual similarity of particular molecule to those of active set;')
    st.write('TII column: tanimoto similarity based on ECFP2 2048 bits. Shows structurual similarity of particular molecule to those of inactive set;')
    st.write('PS column: pharmacophore similarity. It is based on 2D pharmacophore fingerprints. Algorithm very sensitive to noise so obtained values \
    will be small.;')
    st.write('SS column: shape similarity;')
    st.write('ES column: electrostatics similarity;')
    st.write('TPSA column: topological polar surface area of the molecule;')
    st.write('NRB column: number of rotatable bonds;')
    st.write('NHD column: number of hydrogen donors;')
    st.write('NHA column: number of hydrogen acceptors;')
    st.write('MW column: molecular wheight;')
    st.write('LogP column: log10 of octanol-1/water partition;')
    st.write('FCsp3 column: fraction of sp3 carbons;')
    st.write('AA column: aromatic atoms;')
    st.write('HA column: heavy atoms;')
    st.write('qed column: quantative estimation of druglikeness score;')
    st.write('Veber column: number of Veber rule violations;')
    st.write('Lipinski column: number of Lipinski rule violations;')
    st.write('Subs column: unwanted substuctures;')

    if space:

        interactive_map = image(df, desc, tsne_model)

        st.bokeh_chart(interactive_map)

    st.download_button('Press here to download data',
     data = desc.to_csv().encode('utf-8'), file_name='farnesyltransferase.csv', mime='text/csv')

else:
    st.error('Either no smiles were given or all smiles are invalid.')
