import os
from protkit.structure import Protein
from protkit.file_io import PDBIO, ProtIO
from protkit.properties import DihedralAngles
from protkit.download import Download
from joblib import Parallel, delayed
from tqdm import tqdm
import math

from src.paths import *
from src.utils import Utils

class CalculateAngles:
    BACKBONE_ATOMS = {"N", "CA", "C"}
    RESIDUE_LIST = ["ALA", "CYS", "ASP", "GLU", "PHE",
                "GLY", "HIS", "ILE", "LYS", "LEU",
                "MET", "ASN", "PRO", "GLN", "ARG",
                "SER", "THR", "VAL", "TRP", "TYR",
                ]


    @staticmethod
    def set_has_missing_heavy_atoms(protein: Protein):
        for residue in protein.residues:
            atom_types = set()
            for atom in residue.atoms:
                if atom.element in ["C", "CA", "N"]:
                    atom_types.add(atom.atom_type)
            missing_backbone_atoms = CalculateAngles.BACKBONE_ATOMS - atom_types
            if len(missing_backbone_atoms) > 0:
                residue.set_attribute("has_missing_backbone_atoms", True)
            else:
                residue.set_attribute("has_missing_backbone_atoms", False)

        return protein

    @staticmethod
    def remove_residues_with_missing_backbone_atoms(protein: Protein):
        for chain in protein.chains:
            chain._residues = [residue for residue in chain._residues if
                               not residue.get_attribute("has_missing_backbone_atoms")]

        return protein

    @staticmethod
    def remove_unknown_residues(protein: Protein):
        for chain in protein.chains:
            chain._residues = [residue for residue in chain._residues if
                               residue.residue_type in CalculateAngles.RESIDUE_LIST]

        return protein

    @staticmethod
    def remove_ambiguous_residues(protein: Protein):
        for chain in protein.chains:
            chain._residues = [residue for residue in chain._residues if
                               residue.residue_type not in ["GLX", "ASX", "UNK"]]

        return protein

    @staticmethod
    def extract_angles(protein: Protein):
        angle_dict = {}
        for residue in protein.residues:
            angles = residue.get_attribute("dihedral_angles")

            for key, value in angles.items():
                if value is not None:
                    value = math.radians(value)
                    if key in angle_dict:
                        angle_dict[key].append(value)
                    else:
                        angle_dict[key] = [value]
        protein.set_attribute("angles", angle_dict)

    @staticmethod
    def process_entry(entry):
        pdb_id = entry["pdb_id"]
        chain = entry["chain"]
        pdb_file_path = entry["pdb_file_path"]
        protein_file_path = entry["protein_file_path"]

        if not os.path.isfile(pdb_file_path):
            Download.download_pdb_file_from_rcsb(pdb_id, pdb_file_path)
        if not os.path.isfile(protein_file_path):
            protein = PDBIO.load(file_path=pdb_file_path, pdb_id=pdb_id)[0]
            protein.remove_water_residues()
            protein.remove_hetero_residues()
            protein = CalculateAngles.remove_unknown_residues(protein)
            protein.fix_disordered_atoms()
            protein = CalculateAngles.set_has_missing_heavy_atoms(protein)
            protein = CalculateAngles.remove_residues_with_missing_backbone_atoms(protein)
            protein = CalculateAngles.remove_ambiguous_residues(protein)
            DihedralAngles.dihedral_angles_of_protein(protein, assign_attribute=True)
            CalculateAngles.extract_angles(protein)
            ProtIO.save(protein, protein_file_path, compress=False)
            angles = protein.get_attribute("angles")


    @staticmethod
    def batch_process_entries(experiment_directory, entries_file_path):
        pdb_directory = os.path.join(experiment_directory, "pdb")
        protein_directory = os.path.join(experiment_directory, "protein")
        # Create directories if they do not exist
        if not os.path.exists(protein_directory):
            os.mkdir(protein_directory)
        if not os.path.exists(pdb_directory):
            os.mkdir(pdb_directory)
        entries = Utils.load_entries(entries_file_path)
        Parallel(n_jobs=1)(delayed(CalculateAngles.process_entry)(pdb) for pdb in tqdm(entries))
