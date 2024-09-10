import itertools
import random
from typing import List
import pandas as pd
import numpy as np
import os



import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, rdMolAlign
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
from crem.crem import grow_mol, mutate_mol
import umap





def create_list_of_molecules(sdf_file: str) -> list:
    """
    Create a list of molecules from an SDF file
    :param sdf_file:
    :return:
    """
    list_of_molecules = []
    with Chem.SDMolSupplier(sdf_file) as suppl:
        for mol in suppl:
            if mol is None: continue
            list_of_molecules.append(mol)
    return list_of_molecules


def align_molecules(ligands_sdf_file: str, output_file_path: str) -> None:
    """
    Align a list of molecules to the first molecule in the list
    :param ligands_sdf_file:
    :param output_file_path:
    :return:
    """
    list_of_ligands = Chem.SDMolSupplier(ligands_sdf_file)
    molecules = [molecule for molecule in list_of_ligands if molecule is not None]

    molecules = [Chem.AddHs(molecule) for molecule in molecules]
    [AllChem.EmbedMolecule(molecule, AllChem.ETKDG()) for molecule in molecules]
    reference_mol = molecules[0]

    # Align all molecules to the reference molecule
    for molecule in molecules[1:]:
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)

        # GetO3A: Generate O3A object
        o3a = rdMolAlign.GetO3A(molecule, reference_mol)
        o3a.Align()

    # Save the aligned molecules to a new SDF file
    writer = Chem.SDWriter(output_file_path)
    for mol in molecules:
        writer.write(mol)
    writer.close()


def design_new_molecules(list_of_molecules: str, number_of_molecules: int) -> list:
    """
    Design new molecules from a list of molecules using Brics Build
    :param list_of_molecules:
    :param number_of_molecules:
    :return: List of molecules
    """
    fragment_list = []
    random.seed(42)

    for mol in list_of_molecules:
        fragment_list.append(list(Chem.BRICS.BRICSDecompose(mol)))
    # fragment list is a list of lists, so we need to flatten it
    merged_molecules_list = list(itertools.chain(*fragment_list))
    fragments = [Chem.MolFromSmiles(x) for x in merged_molecules_list if x is not None]
    build = BRICS.BRICSBuild(fragments)

    products = [next(build) for _ in range(number_of_molecules)]
    return products


def create_sdf_file_from_molecules(list_of_molecules: list, file_name: str) -> None:
    """
    Create an SDF file from a list of molecules
    :param list_of_molecules:
    :param file_name:
    :return:
    """

    writer = Chem.SDWriter(file_name)
    for molecule in list_of_molecules:
        try:
            molecule.UpdatePropertyCache()
            molecule = Chem.AddHs(molecule)

            AllChem.EmbedMolecule(molecule)
            AllChem.MMFFOptimizeMolecule(molecule)
            writer.write(molecule)

        except Exception as e:
            print(e)
            print(f"Error in writing molecule {Chem.MolToSmiles(molecule)}")
    writer.close()


def get_scaffold(rdkit_mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(rdkit_mol)
    return Chem.MolToSmiles(scaffold)


class CreateProlifFingerPrint:
    """
    Create a fingerprint of a molecule using Prolif
    """

    def __init__(self, receptor_file: str):

        self.receptor_file = receptor_file

    def create_fingerprints_of_docking_poses(self, sdf_file: str) -> plf.Fingerprint:
        """
        Create a protein ligand fingerprint using receptor file and all poses after a docking run
        :param sdf_file:
        :return:
        """
        mda_protein_mol = mda.Universe(self.receptor_file)
        protein_mol = plf.Molecule.from_mda(mda_protein_mol)
        docking_pose_mol = plf.sdf_supplier(sdf_file)
        fp = plf.Fingerprint()

        fp.run_from_iterable(docking_pose_mol, protein_mol)
        return fp

    def create_fingerprint_of_reference_ligand(self, sdf_file: str, index: int) -> plf.Fingerprint:
        """
        Create a protein ligand fingerprint of a reference ligand and receptor
        :param sdf_file:
        :param index:
        :return:
        """
        mda_protein_mol = mda.Universe(self.receptor_file)
        protein_mol = plf.Molecule.from_mda(mda_protein_mol)
        ref_ligand_mol = plf.sdf_supplier(sdf_file)[index]
        fp = plf.Fingerprint()
        fp.run_from_iterable([ref_ligand_mol], protein_mol)
        return fp
    

def smiles_to_fp(smiles_series: pd.Series) -> List[AllChem.GetMorganFingerprintAsBitVect]:
    fingerprints = []
    for smile in smiles_series:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
    return fingerprints


def calculate_tanimoto_similarities(reference_smiles: str, original_smiles_list: List[str]) -> List[float]:
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reference_smiles), radius=2, nBits=1024)
    fps_list = smiles_to_fp(pd.Series(original_smiles_list))
    return BulkTanimotoSimilarity(ref_fp, fps_list)


def generate_molecules(smi: str) -> List[str]:
        m = Chem.MolFromSmiles(smi)
        random_number = random.random()
        db_name = "replacements02_sa2.db"
        data_dir = ".\data\crem_db"
        db_path = os.path.join(data_dir, db_name)


        if random_number < 0.33333:
            mols = list(mutate_mol(m, db_name=db_path))
        elif random_number < 0.66666:
            mols = list(grow_mol(m, db_name=db_path))
        else:
            mutated = random.choice(list(mutate_mol(m, db_name=db_path)))
            m = Chem.MolFromSmiles(mutated)
            mols = list(grow_mol(m, db_name=db_path))

        return random.choices(mols, k=15)


def _smiles_to_fp(smiles_series: pd.Series) -> List[AllChem.GetMorganFingerprintAsBitVect]:
    fingerprints = []
    for smile in smiles_series:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
    return fingerprints

def create_umap_df(smiles_list, generation_list):
    fps = _smiles_to_fp(smiles_list)
            
    fps_array = np.array(fps)
    umap_reducer = umap.UMAP(n_neighbors=50, random_state=20)
    umap_results = umap_reducer.fit_transform(fps_array)
    umap_df = pd.DataFrame(umap_results, columns=["UMAP1", "UMAP2"])
    umap_df["Source"] = [label for label, fp in zip(generation_list, fps) if fp is not None]

    return umap_df
