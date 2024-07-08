import itertools
import random

import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, rdMolAlign


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
    build = BRICS.BRICSBuild(fragments, onlyCompleteMols=True)

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









