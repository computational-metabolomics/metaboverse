import sys
import os
from rdkit import Chem
import sqlite3
import pickle

metaboverse_path = os.path.join("..", "..", "..", "metaboverse")

sys.path.append(os.path.join(metaboverse_path, "metaboverse"))
from databases import SubstructureDb, get_substructure_bond_idx, get_substructure, calculate_exact_mass, get_elements


def initialise_neutralisation_reactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
        )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def neutralise_radicals(mol):
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() == 1 and a.GetFormalCharge() == 1:
            a.SetNumRadicalElectrons(0)
            a.SetFormalCharge(0)

    return mol


def neutralise_charges(smiles):
    mol = Chem.MolFromSmiles(smiles)

    for reactant, product in initialise_neutralisation_reactions():
        while mol.HasSubstructMatch(reactant):
            rms = Chem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    return Chem.MolToSmiles(neutralise_radicals(mol))


def remove_charges(mol):
    for a in mol.GetAtoms():
        a.SetNumRadicalElectrons(0)
        a.SetFormalCharge(0)

    return mol


def update_mfrontier_database(mol, fn_db, path_sdfs):
    conn = sqlite3.connect(fn_db)
    cursor = conn.cursor()

    suppl = Chem.SDMolSupplier(path_sdfs)
    for substructure in suppl:
        if substructure is None:
            continue

        Chem.SanitizeMol(substructure)
        substructure = remove_charges(substructure)

        try:
            Chem.SanitizeMol(substructure)
        except ValueError:
            continue

        edge_idxs = get_substructure_bond_idx(substructure, mol)

        if edge_idxs is None:
            continue

        lib = get_substructure(mol, edge_idxs)
        if lib is None:
            continue
        smiles_rdkit = Chem.MolToSmiles(lib["mol"])

        exact_mass = calculate_exact_mass(lib["mol"])
        els = get_elements(lib["mol"])

        pkl_lib = pickle.dumps(lib)
        sub_smi_dict = {'smiles': smiles_rdkit,
                        'exact_mass': exact_mass,
                        'count': 0,
                        'length': sum([els[atom] for atom in els if atom != "*"]),
                        "valence": lib["valence"],
                        "valence_atoms": str(lib["degree_atoms"]),
                        "atoms_available": lib["atoms_available"],
                        "lib": pkl_lib}

        sub_smi_dict["exact_mass__1"] = round(sub_smi_dict["exact_mass"], 0)
        sub_smi_dict["exact_mass__0_1"] = round(sub_smi_dict["exact_mass"], 1)
        sub_smi_dict["exact_mass__0_01"] = round(sub_smi_dict["exact_mass"], 2)
        sub_smi_dict["exact_mass__0_001"] = round(sub_smi_dict["exact_mass"], 3)
        sub_smi_dict["exact_mass__0_0001"] = round(sub_smi_dict["exact_mass"], 4)

        sub_smi_dict.update(els)
        sub_smi_dict["heavy_atoms"] = sum([els[atom] for atom in els if atom != "H" and atom != "*"])

        cursor.execute("""INSERT OR IGNORE INTO substructures (
                                              smiles, 
                                              heavy_atoms, 
                                              length, 
                                              exact_mass__1, 
                                              exact_mass__0_1, 
                                              exact_mass__0_01, 
                                              exact_mass__0_001, 
                                              exact_mass__0_0001, 
                                              exact_mass, count, 
                                              C, 
                                              H, 
                                              N, 
                                              O, 
                                              P, 
                                              S, 
                                              valence, 
                                              valence_atoms, 
                                              atoms_available, 
                                              lib)
                                          values (
                                              :smiles,
                                              :heavy_atoms,
                                              :length,
                                              :exact_mass__1,
                                              :exact_mass__0_1,
                                              :exact_mass__0_01,
                                              :exact_mass__0_001,
                                              :exact_mass__0_0001,
                                              :exact_mass,
                                              :count,
                                              :C,
                                              :H,
                                              :N,
                                              :O,
                                              :P,
                                              :S,
                                              :valence,
                                              :valence_atoms,
                                              :atoms_available,:lib)
                                       """, sub_smi_dict)

        cursor.execute("""INSERT OR IGNORE INTO hmdbid_substructures (
                                              hmdbid, 
                                              smiles_rdkit) 
                                          VALUES ("%s", "%s")
                                       """ % (mol.GetProp("HMDB_ID"), smiles_rdkit))

    conn.commit()
    conn.close()


def build_mfrontier_database(name, path_db, path_sdf):
    db = SubstructureDb(path_db, "", "")
    db.create_compound_database()
    db.close()

    smi_to_id = {}
    suppl = Chem.SDMolSupplier(os.path.join(path_sdf, name + ".sdf"))
    for i, mol in enumerate(suppl):
        Chem.SanitizeMol(mol)
        smi_to_id[Chem.MolToSmiles(mol, False)] = mol.GetProp("HMDB_ID")

    suppl = Chem.SDMolSupplier(os.path.join(path_sdf, name + "_filtered_by_MassFrontier.sdf"))
    for i, mol in enumerate(suppl):
        Chem.SanitizeMol(mol)
        try:
            mol.SetProp("HMDB_ID", smi_to_id[Chem.MolToSmiles(mol, False)])
        except KeyError:
            mol.SetProp("HMDB_ID", "NA" + str(i))

        update_mfrontier_database(mol, path_db, os.path.join(path_sdf, name + "_output_" + str(i + 1) + ".sdf"))

    db = SubstructureDb(path_db, "", "")
    db.create_indexes()
    db.close()


def incorporate_mfrontier_substructures(db_path, mfrontier_db_path):
    db = SubstructureDb(db_path, "")
    db.cursor.execute("attach database '%s' as mfrontier" % mfrontier_db_path)

    db.cursor.execute("drop table if exists mfrontier_substructures")
    db.cursor.execute("""create table mfrontier_substructures as 
                            select * from mfrontier.substructures""")
    db.close()
