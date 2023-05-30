# Import start
import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.MolStandardize import rdMolStandardize
from joblib import Parallel, delayed
from mordred import Calculator, descriptors
import pickle
import numpy as np
from scipy.spatial import distance
import sklearn
import mordred
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdDistGeom
from rdkit.Chem.FeatMaps import FeatMaps
import os
from rdkit import RDConfig

param = rdDistGeom.ETKDGv3()
param.pruneRmsThresh = 0
param.randomSeed = 1
param.boxSizeMult = 2
param.useRandomCoords = True
param.enforceChirality = True
param.useSmallRingTorsions = True
param.useMacrocycleTorsions = True

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
fmParams = {}
for k in factory.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fparams.radius = 2
    fparams.width = 1.0
    fparams.FeatProfile = 0
    fmParams[k] = fparams

filerParams = FilterCatalogParams()
filerParams.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
catalog = FilterCatalog(filerParams)
# Import end

# check shape with mean
min_shape = 0.1461
max_shape = 0.2901
# check electro with max_shape
min_electro = 0.4952
max_electro = 0.6702
# check electro with max_shape
min_ph4 = 0.4952
max_ph4 = 0.6702

@st.cache_resource()
def load_models():
    model_0 = pickle.load(open('model_0.pkl', 'rb'))
    model_1 = pickle.load(open('model_1.pkl', 'rb'))
    model_2 = pickle.load(open('model_2.pkl', 'rb'))
    model_3 = pickle.load(open('model_3.pkl', 'rb'))
    model_4 = pickle.load(open('model_4.pkl', 'rb'))
    model_5 = pickle.load(open('model_5.pkl', 'rb'))
    model_6 = pickle.load(open('model_6.pkl', 'rb'))

    minmax_0 = pickle.load(open('minmax_0.pkl', 'rb'))
    minmax_1 = pickle.load(open('minmax_1.pkl', 'rb'))
    minmax_2 = pickle.load(open('minmax_2.pkl', 'rb'))
    minmax_3 = pickle.load(open('minmax_3.pkl', 'rb'))
    minmax_4 = pickle.load(open('minmax_4.pkl', 'rb'))
    minmax_5 = pickle.load(open('minmax_5.pkl', 'rb'))
    minmax_6 = pickle.load(open('minmax_6.pkl', 'rb'))

    models = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]
    scalers = [minmax_0, minmax_1, minmax_2, minmax_3, minmax_4, minmax_5, minmax_6]

    return models, scalers

@st.cache_data()
def load_data():

    AD = pickle.load(open('AD.pkl', 'rb'))
    FEATURES = pickle.load(open('FEATURES.pkl', 'rb'))
    DESCRIPTORS = pickle.load(open('DESCRIPTORS.pkl', 'rb'))
    NNN = pickle.load(open('NNN.pkl', 'rb'))

    return AD, FEATURES, DESCRIPTORS, NNN

@st.cache_data()
def load_reference_mols():
    sup = Chem.SDMolSupplier('reference_mols.sdf')
    return [mol for mol in sup]

@st.cache_data()
def load_modelling_set():
    df = pd.read_csv('modelling_set.csv')
    df_0 = df[df['bioclass'] == 0]
    df_1 = df[df['bioclass'] == 1]
    mols_0 = [Chem.MolFromSmiles(x) for x in df_0['Smiles']]
    mols_1 = [Chem.MolFromSmiles(x) for x in df_1['Smiles']]
    fps_active = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    j,2,2048, useFeatures=False, useBondTypes=True) for j in mols_1]
    fps_inactive = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    j,2,2048, useFeatures=False, useBondTypes=True) for j in mols_0]
    return df_1, df_0, fps_active, fps_inactive

models, scalers = load_models()
AD, FEATURES, DESCRIPTORS, NNN = load_data()
reference_mols = load_reference_mols()
df_1, df_0, fps_active, fps_inactive = load_modelling_set()

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
            rdMolStandardize.Cleanup(mol)))))
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
        return ['None']

st.header('Here you can find project dedicated to farnesyltransferase.')
st.subheader('You can input one or more smiles of molecules you are interested in. You can use drawer to \
obtain smiles.')
st.write('After receiving smiles, the algorithm will return a Data Frame containing the prediction \
of an ensemble of machine learning models, some physico-chemical descriptors, the fulfilment \
of the Lipinski and Weber rules and the presence of undesirable substructures. Optionally, the algorithm will \
return an estimate of the superposition of the shape, pharmacophore and electrostatic \
potential of the test molecules relative to the reference ones.')

default = 'C=12CCC=3C=C(C=C(C3[C@H](C1N=CC(=C2)Br)C4CCN(CC4)C(=O)CC5CCN(CC5)C(N)=O)Br)Cl'
molecule = st.text_input("Molecule", default)
smiles = st_ketcher(molecule)

st.markdown(f"Smiles: ``{smiles}``")

with st.sidebar:
    st.header('Here you can input one or more smiles to obtain prediction for them')
    sm = st.text_area('Input your smiles here. _**Every smiles must be in new row**_ or they will be perceived as wrong.', value = 'c1ccccc1\nC=12CCC=3C=C(C=C(C3[C@H](C1N=CC(=C2)Br)C4CCN(CC4)C(=O)CC5CCN(CC5)C(N)=O)Br)Cl')
    sm = sm.split('\n')
    sm = [x for x in sm if x != '']
    st.write('If you want you can use 3D functionality. It includes estimation of shape, electrostatical potential and pharmacophore overlay. It is not particulary fast but you can give it a try.')
    props = st.checkbox('Calculate additional properties?')
    checker = st.checkbox('Use 3D functionality?')

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

    clean_mols = Parallel(n_jobs = -1, prefer='processes')(delayed(clearMols)(x) for x in mols)

    ensemble = {}
    for x in range(len(clean_mols)):
        ensemble.setdefault(x, [])

    for j in range(len(DESCRIPTORS)):

        if j == 0:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.MolecularId.MolecularId('hetero', False, 1e-10))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('mean', 3))
            mordred_calc.register(mordred.BalabanJ.BalabanJ())
            mordred_calc.register(mordred.MoeType.SlogP_VSA(2))
            mordred_calc.register(mordred.Autocorrelation.ATS(3, 'd'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'dsN'))
            mordred_calc.register(mordred.InformationContent.ComplementaryIC(3))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(5))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(1, 1))
            mordred_calc.register(mordred.Lipinski.Lipinski())
            mordred_calc.register(mordred.BaryszMatrix.BaryszMatrix('i', 'SM1'))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
            mordred_calc.register(mordred.LogS.LogS())
            mordred_calc.register(mordred.Autocorrelation.ATSC(3, 'p'))
            mordred_calc.register(mordred.Autocorrelation.MATS(3, 'v'))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(3, 3))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'sNH2'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(5, 'p'))
            mordred_calc.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
            mordred_calc.register(mordred.Autocorrelation.AATS(7, 'i'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(1, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.MATS(6, 'i'))
            mordred_calc.register(mordred.RingCount.RingCount(10, False, True, None, True))
            mordred_calc.register(mordred.RingCount.RingCount(12, True, True, None, None))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'v'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(1, 'c'))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'm'))
            mordred_calc.register(mordred.Autocorrelation.MATS(2, 'c'))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'dv'))
            mordred_calc.register(mordred.AtomCount.AtomCount('N'))
            mordred_calc.register(mordred.Autocorrelation.MATS(5, 'se'))
            mordred_calc.register(mordred.InformationContent.ComplementaryIC(5))
            mordred_calc.register(mordred.Autocorrelation.AATS(4, 'i'))

        elif j == 1:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.MoeType.PEOE_VSA(12))
            mordred_calc.register(mordred.Autocorrelation.ATSC(2, 'm'))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'd'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'sssCH'))
            mordred_calc.register(mordred.InformationContent.ComplementaryIC(3))
            mordred_calc.register(mordred.BaryszMatrix.BaryszMatrix('se', 'SM1'))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'v'))
            mordred_calc.register(mordred.Autocorrelation.AATS (7, 'i'))
            mordred_calc.register(mordred.CPSA.RNCG())
            mordred_calc.register(mordred.BCUT.BCUT('i', 0))
            mordred_calc.register(mordred.Autocorrelation.ATSC(1, 'dv'))
            mordred_calc.register(mordred.BCUT.BCUT('c', -1))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('mean', 3))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(8))
            mordred_calc.register(mordred.Chi.Chi('path', 6, 'dv', True))
            mordred_calc.register(mordred.Autocorrelation.ATS(0, 's'))
            mordred_calc.register(mordred.Autocorrelation.MATS(3, 'd'))
            mordred_calc.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
            mordred_calc.register(mordred.HydrogenBond.HBondAcceptor())
            mordred_calc.register(mordred.RingCount.RingCount(None, False, False, None, True))
            mordred_calc.register(mordred.BCUT.BCUT('s', 0))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'v'))
            mordred_calc.register(mordred.MoeType.VSA_EState(1))
            mordred_calc.register(mordred.MolecularId.MolecularId('hetero', False, 1e-10))
            mordred_calc.register(mordred.RingCount.RingCount(7, False, False, None, None))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(6))
            mordred_calc.register(mordred.Autocorrelation.ATSC(0, 'Z'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'dO'))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
            mordred_calc.register(mordred.Autocorrelation.AATS(4, 'i'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'i'))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(2, 2))
            mordred_calc.register(mordred.RingCount.RingCount(6, False, False, True, True))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaDeltaEpsilon('B'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(8, 'v'))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, True, False, None))
            mordred_calc.register(mordred.Autocorrelation.ATS(3, 'd'))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 9))
            mordred_calc.register(mordred.RotatableBond.RotatableBondsRatio())
            mordred_calc.register(mordred.Autocorrelation.MATS(6, 'i'))
            mordred_calc.register(mordred.Autocorrelation.MATS(4, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(2, 'p'))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'i'))
            mordred_calc.register(mordred.Autocorrelation.AATS(8, 'i'))
            mordred_calc.register(mordred.Autocorrelation.ATS(7, 's'))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(4))

        elif j == 2:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.BCUT.BCUT('dv', 0))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(2, 3))
            mordred_calc.register(mordred.Autocorrelation.GATS(4, 'se'))
            mordred_calc.register(mordred.Autocorrelation.GATS(3, 'p'))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'p'))
            mordred_calc.register(mordred.Autocorrelation.MATS(5, 'se'))
            mordred_calc.register(mordred.Autocorrelation.GATS(7, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.AATS(4, 'dv'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aaS'))
            mordred_calc.register(mordred.Autocorrelation.GATS(4, 'm'))
            mordred_calc.register(mordred.Chi.Chi('cluster', 4, 'd', False))
            mordred_calc.register(mordred.MoeType.SMR_VSA(6))
            mordred_calc.register(mordred.RingCount.RingCount(11, False, True, None, None))
            mordred_calc.register(mordred.BCUT.BCUT('c', -1))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(3))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, True, False, None))
            mordred_calc.register(mordred.WalkCount.WalkCount(5, False, True))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'sNH2'))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'Z'))
            mordred_calc.register(mordred.RingCount.RingCount(10, False, True, None, True))
            mordred_calc.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(8))
            mordred_calc.register(mordred.MoeType.SMR_VSA(3))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 10))
            mordred_calc.register(mordred.Autocorrelation.GATS(3, 's'))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(10))
            mordred_calc.register(mordred.Autocorrelation.ATSC(2, 'dv'))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(9))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, False, True, True))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 9))
            mordred_calc.register(mordred.Autocorrelation.MATS(6, 'i'))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(4))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'c'))
            mordred_calc.register(mordred.MoeType.EState_VSA(3))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(1, 3))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'd'))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(12))
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'd'))
            mordred_calc.register(mordred.RingCount.RingCount(10, False, True, True, True))
            mordred_calc.register(mordred.Autocorrelation.AATS(7, 'i'))
            mordred_calc.register(mordred.Autocorrelation.ATS(7, 's'))
            mordred_calc.register(mordred.Autocorrelation.ATS(5, 'dv'))
            mordred_calc.register(mordred.Autocorrelation.AATS(6, 'dv'))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaDeltaAlpha('B'))
            mordred_calc.register(mordred.Autocorrelation.MATS(3, 'd'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.AATS(7, 'v'))
            mordred_calc.register(mordred.HydrogenBond.HBondAcceptor())
            mordred_calc.register(mordred.Autocorrelation.ATSC(2, 'm'))

        elif j == 3:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.RingCount.RingCount(None, False, False, None, True))
            mordred_calc.register(mordred.BCUT.BCUT('Z', 0))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(9))
            mordred_calc.register(mordred.Autocorrelation.ATS(5, 'dv'))
            mordred_calc.register(mordred.BCUT.BCUT('s', -1))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(10))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(4))
            mordred_calc.register(mordred.Autocorrelation.AATSC(3, 'i'))
            mordred_calc.register(mordred.MoeType.EState_VSA(5))
            mordred_calc.register(mordred.Autocorrelation.ATSC(1, 'c'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'ssCH2'))
            mordred_calc.register(mordred.Chi.Chi('path', 6, 'dv', True))
            mordred_calc.register(mordred.Autocorrelation.ATSC(2, 'dv'))
            mordred_calc.register(mordred.HydrogenBond.HBondAcceptor())
            mordred_calc.register(mordred.MoeType.EState_VSA(4))
            mordred_calc.register(mordred.Autocorrelation.MATS(5, 'se'))
            mordred_calc.register(mordred.Autocorrelation.ATS(7, 's'))
            mordred_calc.register(mordred.BaryszMatrix.BaryszMatrix('se', 'SM1'))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, False, True, True))
            mordred_calc.register(mordred.EState.AtomTypeEState('sum', 'ssNH'))
            mordred_calc.register(mordred.Autocorrelation.ATS(8, 'dv'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'dO'))
            mordred_calc.register(mordred.RotatableBond.RotatableBondsRatio())
            mordred_calc.register(mordred.Autocorrelation.ATSC(8, 'v'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aasN'))
            mordred_calc.register(mordred.BCUT.BCUT('i', 0))
            mordred_calc.register(mordred.AcidBase.AcidicGroupCount())
            mordred_calc.register(mordred.RingCount.RingCount(12, True, True, None, None))
            mordred_calc.register(mordred.Autocorrelation.AATS(7, 'd'))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaDeltaBeta(False))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'v'))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'm'))
            mordred_calc.register(mordred.Autocorrelation.GATS(4, 'i'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'sssCH'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aaO'))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'i'))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('mean', 3))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, False, True, None))
            mordred_calc.register(mordred.PathCount.PathCount(5, True, False, True))

        elif j == 4:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.MoeType.SlogP_VSA(3))
            mordred_calc.register(mordred.EState.AtomTypeEState('sum', 'sCl'))
            mordred_calc.register(mordred.RingCount.RingCount(6, False, False, False, True))
            mordred_calc.register(mordred.Autocorrelation.ATSC(5, 'i'))
            mordred_calc.register(mordred.Autocorrelation.GATS(4, 'c'))
            mordred_calc.register(mordred.RingCount.RingCount(7, False, False, None, None))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(10))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 9))
            mordred_calc.register(mordred.Autocorrelation.MATS(3, 'd'))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'p'))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(1, 3))
            mordred_calc.register(mordred.InformationContent.ComplementaryIC(3))
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'are'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aasN'))
            mordred_calc.register(mordred.BCUT.BCUT('v', 0))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 5))
            mordred_calc.register(mordred.CPSA.RPCG())
            mordred_calc.register(mordred.RingCount.RingCount(10, False, True, None, True))
            mordred_calc.register(mordred.RingCount.RingCount(6, False, False, True, True))
            mordred_calc.register(mordred.MoeType.EState_VSA(5))
            mordred_calc.register(mordred.BalabanJ.BalabanJ())
            mordred_calc.register(mordred.MolecularDistanceEdge.MolecularDistanceEdge(2, 2, 'C'))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'p'))
            mordred_calc.register(mordred.RotatableBond.RotatableBondsRatio())
            mordred_calc.register(mordred.Framework.Framework())
            mordred_calc.register(mordred.MoeType.PEOE_VSA(2))
            mordred_calc.register(mordred.MoeType.VSA_EState(3))
            mordred_calc.register(mordred.Autocorrelation.GATS(5, 'se'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(8, 'v'))
            mordred_calc.register(mordred.Autocorrelation.AATS(2, 'v'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'sssCH'))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaVEMCount('s', True))

        elif j == 5:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'i'))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaVEMCount('ns_d', False))
            mordred_calc.register(mordred.MoeType.VSA_EState(2))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 8))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aaS'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'd'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(8, 'i'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(2, 'p'))
            mordred_calc.register(mordred.RingCount.RingCount(10, False, True, None, True))
            mordred_calc.register(mordred.Autocorrelation.ATSC(1, 'c'))
            mordred_calc.register(mordred.Framework.Framework())
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aasN'))
            mordred_calc.register(mordred.BCUT.BCUT('i', -1))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(11))
            mordred_calc.register(mordred.Autocorrelation.MATS(6, 'i'))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(3, 3))
            mordred_calc.register(mordred.MoeType.EState_VSA(7))
            mordred_calc.register(mordred.Autocorrelation.ATS(3, 'd'))
            mordred_calc.register(mordred.InformationContent.ComplementaryIC(3))
            mordred_calc.register(mordred.Autocorrelation.GATS(1, 'm'))
            mordred_calc.register(mordred.Autocorrelation.GATS(7, 'd'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aaNH'))
            mordred_calc.register(mordred.RingCount.RingCount(7, False, False, None, None))
            mordred_calc.register(mordred.Autocorrelation.MATS(2, 'c'))
            mordred_calc.register(mordred.Autocorrelation.GATS(4, 'i'))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'i'))
            mordred_calc.register(mordred.CarbonTypes.CarbonTypes(1, 3))
            mordred_calc.register(mordred.Autocorrelation.ATSC(5, 'i'))
            mordred_calc.register(mordred.Autocorrelation.AATS(8, 'i'))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(10))
            mordred_calc.register(mordred.Autocorrelation.GATS(4, 'c'))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaDeltaBeta(False))
            mordred_calc.register(mordred.MoeType.EState_VSA (10))

        elif j == 6:

            mordred_calc = Calculator()

            mordred_calc.register(mordred.Autocorrelation.GATS(7, 'd'))
            mordred_calc.register(mordred.RingCount.RingCount(6, False, False, False, True))
            mordred_calc.register(mordred.BCUT.BCUT('s', 0))
            mordred_calc.register(mordred.RingCount.RingCount(9, False, True, False, None))
            mordred_calc.register(mordred.RingCount.RingCount(10, False, True, None, True))
            mordred_calc.register(mordred.MoeType.PEOE_VSA(9))
            mordred_calc.register(mordred.MoeType.VSA_EState(1))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'sNH2'))
            mordred_calc.register(mordred.BCUT.BCUT('are', 0))
            mordred_calc.register(mordred.PathCount.PathCount(5, True, False, True))
            mordred_calc.register(mordred.MoeType.EState_VSA(3))
            mordred_calc.register(mordred.Constitutional.ConstitutionalMean('i'))
            mordred_calc.register(mordred.Chi.Chi('path', 7, 'd', False))
            mordred_calc.register(mordred.Chi.Chi('path', 1, 'dv', True))
            mordred_calc.register(mordred.RotatableBond.RotatableBondsRatio())
            mordred_calc.register(mordred.BaryszMatrix.BaryszMatrix('p', 'SpDiam'))
            mordred_calc.register(mordred.Autocorrelation.MATS(4, 'Z'))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 8))
            mordred_calc.register(mordred.Autocorrelation.ATSC(5, 'i'))
            mordred_calc.register(mordred.TopologicalCharge.TopologicalCharge('raw', 10))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, False, True, None))
            mordred_calc.register(mordred.MoeType.SMR_VSA(3))
            mordred_calc.register(mordred.Autocorrelation.ATS(8, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.AATS(4, 'd'))
            mordred_calc.register(mordred.Autocorrelation.ATS(7, 's'))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(10))
            mordred_calc.register(mordred.AcidBase.BasicGroupCount())
            mordred_calc.register(mordred.AtomCount.AtomCount('X'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(8, 'i'))
            mordred_calc.register(mordred.Autocorrelation.AATS(3, 'v'))
            mordred_calc.register(mordred.HydrogenBond.HBondDonor())
            mordred_calc.register(mordred.Autocorrelation.GATS(2, 'c'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(0, 'Z'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'i'))
            mordred_calc.register(mordred.CPSA.RPCG())
            mordred_calc.register(mordred.Framework.Framework())
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'd'))
            mordred_calc.register(mordred.Autocorrelation.ATS(7, 'i'))
            mordred_calc.register(mordred.Autocorrelation.AATSC(3, 'i'))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aasN'))
            mordred_calc.register(mordred.ExtendedTopochemicalAtom.EtaCompositeIndex(False, True, False))
            mordred_calc.register(mordred.MolecularId.MolecularId('hetero', False, 1e-10))
            mordred_calc.register(mordred.Autocorrelation.GATS(5, 'v'))
            mordred_calc.register(mordred.RingCount.RingCount(6, False, False, True, True))
            mordred_calc.register(mordred.BaryszMatrix.BaryszMatrix('se', 'SM1'))
            mordred_calc.register(mordred.MoeType.SlogP_VSA(2))
            mordred_calc.register(mordred.RingCount.RingCount(None, False, True, False, None))
            mordred_calc.register(mordred.EState.AtomTypeEState('count', 'aaNH'))
            mordred_calc.register(mordred.Autocorrelation.ATSC(7, 'v'))

        prb_desc = [np.array(mordred_calc(m)) for m in clean_mols]

        ref_desc = np.array(DESCRIPTORS[j])

        prb_desc = scalers[j].transform(prb_desc)

        for num in range(len(mols)):

            kn = 0

            try:

                dist = [distance.euclidean(prb_desc[num], ref_desc[x, :]) for x in range(ref_desc.shape[0])]
                vals = list(np.argsort(np.argsort(dist)))

                for n in range(NNN[j]):
                    if dist[vals.index(n)] <= AD[j]:
                        kn += 1

                if kn == NNN[j]:

                    res  = models[j].predict_proba(prb_desc[num, :].reshape(1, -1))

                    if num in list(ensemble):
                        ensemble[num].append(res[0][1])
                    else:
                        ensemble.setdefault(num, [res[0][1]])

            except:

                if num in list(ensemble):
                    ensemble[num].append(None)
                else:
                    ensemble.setdefault(num, [None])

    results = {'ML': []}
    for key in list(ensemble):
        temp_res = ensemble[key]
        if len(temp_res) > 3 and None not in temp_res:
            results['ML'].append(np.mean(ensemble.pop(key)))
        elif None in temp_res:
            results['ML'].append('None')
        else:
            results['ML'].append('NA')

    desc = pd.DataFrame(results)

    modelling_mols_ative = [Chem.MolFromSmiles(x) for x in df_1['Smiles']]
    modelling_mols_inative = [Chem.MolFromSmiles(x) for x in df_0['Smiles']]

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
    j,2,2048, useFeatures=False, useBondTypes=True) for j in mols]

    sim_active = [max(DataStructs.BulkTanimotoSimilarity(fp, fps_active)) for fp in fps]

    sim_inactive = [max(DataStructs.BulkTanimotoSimilarity(fp, fps_inactive)) for fp in fps]

    desc['TIA'] = sim_active
    desc['TII'] = sim_inactive

    if props:

        add_props = Parallel(n_jobs = -1, prefer='processes')(delayed(calc_desc)(x) for x in clean_mols)
        add_props_df = {}
        for key in list(add_props[0]):
            add_props_df.setdefault(key, [x[key] for x in add_props])

        add_props_df = pd.DataFrame(add_props_df)

        subs = Parallel(n_jobs = -1, prefer='processes')(delayed(
                    substructureFilter)(x) for x in clean_mols)

        desc = pd.concat([desc, add_props_df], axis = 1)
        desc['Subs'] = subs

    st.dataframe(desc.style.apply(background_color).apply(text_color).format(precision = 2))

else:
    st.error('Either no smiles were given or all smiles are invalid.')









#model = joblib.load('xgbpipe.joblib')

#def predict():
#    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked])
#    X = pd.DataFrame([row], columns = columns)
#    prediction = model.predict(X)
#    if prediction[0] == 1:
#        st.success('Passenger Survived :thumbsup:')
#    else:
#        st.error('Passenger did not Survive :thumbsdown:')
