import os
import sys
import urllib.request
import csv
import tempfile
import networkx as nx
from statistics import median
from rdkit import Chem
from shutil import rmtree
import pickle
from msp2db.db import create_db
from msp2db.parse import LibraryData
import sqlite3

sys.path.append(os.path.join("..", "..", "..", "metaboblend", "metaboblend"))
from databases import update_substructure_database, filter_records, parse_xml, SubstructureDb, get_elements, \
    calculate_exact_mass
from build_structures import build

sys.path.append(os.path.join("..", "functions"))
from gen_aux import get_from_hmdb


def parse_testing_data(csv_path, hmdb_path):
    """
    Method for converting a CSV file, containing information on MS2 peaks, into a dictionary to be used for running
    the tool on.

    :param csv_path: The path of the CSV file to be parsed.

    :param hmdb_path: The directory containing HMDB xmls to acquire further data from.

    :return: A dictionary containing dictionaries for each "category" of data; each of these contains a dictionary of
        an HMDB record.
    """

    with open(csv_path) as csv_file:
        csv_data = csv.reader(csv_file)

        data_categories = {}
        for line in csv_data:
            if line[0] + ".xml" not in os.listdir(hmdb_path):
                get_from_hmdb(line[0], line[0], hmdb_path)

            # setup ms data category
            if line[4] not in data_categories:
                data_categories[line[4]] = {}

            # setup compound dict
            if line[0] not in data_categories[line[4]]:
                data_categories[line[4]][line[0]] = {"name": line[1], "precursor_ion_mass": float(line[3]), "peaks": []}

                for record_dict in filter_records(parse_xml(os.path.join(hmdb_path, line[0] + ".xml"))):
                    assert record_dict["HMDB_ID"] == line[0]

                    data_categories[line[4]][line[0]]["mc"] = [record_dict["C"], record_dict["H"], record_dict["N"],
                                                               record_dict["O"], record_dict["P"], record_dict["S"]]
                    data_categories[line[4]][line[0]]["exact_mass"] = record_dict["exact_mass"]
                    mol = Chem.MolFromSmiles(Chem.MolToSmiles(record_dict["mol"], isomericSmiles=False))
                    Chem.SanitizeMol(mol)
                    data_categories[line[4]][line[0]]["mol"] = mol
                    data_categories[line[4]][line[0]]["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False)

            assert data_categories[line[4]][line[0]]["exact_mass"] is not None
            data_categories[line[4]][line[0]]["peaks"].append(float(line[2]))

    return data_categories


class MspDatabase:
    """
    Class for parsing msp files for MS/MS data.

    :param path_db: The path of the SQLite substructure database to be generated.

    :param path_msp: The path of the msp file to be parsed. If given, a database is automatically generated upon
        initialisation of the class.

    :param schema: The schema of the msp file. "mona" or "massbank".
    """

    def __init__(self, path_db, path_msp=None, schema="mona"):
        self.path_db = path_db
        self.schema = schema

        if path_msp is not None:
            print("Generating DB")
            self.lib_data = self.generate_db(path_msp)
        else:
            assert os.path.exists(path_db)
            self.lib_data = None
            print("Using existing DB")

        self.conn = sqlite3.connect(self.path_db)
        self.cursor = self.conn.cursor()

    def generate_db(self, path_msp):
        """
        Generate the SQLite database using msp2db.

        :param path_msp: The path of the msp database to be parsed.

        :return: LibraryData class.
        """

        create_db(file_pth=self.path_db)

        return LibraryData(msp_pth=path_msp, db_pth=self.path_db, db_type='sqlite', schema=self.schema)

    def get_fragments(self, precursor_type="[M+H]+", ms_level=2, max_mass=300, min_mass=100, snr=2.0):
        """
        Get MS data from the SQLite database.

        :param precursor_type: The precursor type to filter by.

        :param ms_level: The MS level of datasets to be obtained.

        :param max_mass: The maximum mass of compounds to be considered.

        :param min_mass: The minimum mass of compounds to be considered.

        :param snr: Signal to noise ratio to filter MS/MS datasets by.

        :return: Yields a list including MS/MS datasets from the database.
        """

        self.cursor.execute("""select distinct inchikey_id, name, exact_mass, smiles
                               from metab_compound where exact_mass > {0} and exact_mass < {1}
                            """.format(str(min_mass), str(max_mass)))

        compounds = self.cursor.fetchall()

        for compound in compounds:
            self.cursor.execute("""select id, accession, precursor_mz, polarity, precursor_type 
                                   from library_spectra_meta 
                                   where inchikey_id = '{0}' and ms_level = {1} and precursor_type = '{2}'
                                   """.format(compound[0], str(ms_level), precursor_type))

            spectra_meta = self.cursor.fetchone()
            if spectra_meta is None:
                continue
            elif len(spectra_meta) == 0:
                continue

            mz = []
            intensity = []

            self.cursor.execute("""select mz, i from library_spectra 
                                   where library_spectra_meta_id = %s""" % spectra_meta[0])

            for spectra in self.cursor.fetchall():
                mz.append(spectra[0])
                intensity.append(spectra[1])

            # inchikey_id, name, exact_mass, smiles, accession, precursor_mz, mzs, itensities
            # med_snr = median(intensity) * snr
            # indices = [i for i in range(len(mz)) if intensity[i] > med_snr]
            indices = sorted(range(len(intensity)), key=lambda i: intensity[i])[-15:]  # top 15 results
            mz, intensity = [m for i, m in enumerate(mz) if i in indices], \
                            [inten for i, inten in enumerate(intensity) if i in indices]

            if len(intensity) == 0:
                continue

            yield list(compound) + list(spectra_meta[1:3]) + [mz] + [intensity]

    def close(self):
        """
        Close connection to the SQLite database.
        """

        self.conn.close()


def parse_msp_testing_data(paths_msp_db, names_msp, path_hmdb_ids, hmdb_path, path_full_hmdb, no_hmdb=False):
    """
    See parse_testing_data. Generatesa dictionary for running the MS2 build method on.

    :param paths_msp_db: Path to MSP files to be parses.

    :param names_msp: The name of the categories/datasets.

    :param path_hmdb_ids: Path to a CSV file that relates each compound to a HMDB ID.

    :param hmdb_path: Path to HMDB XML files.

    :return: A dictionary containing a dictionary for each dataset passed to the function. These dictionaries contain
        dictionaries, each of which contain relevant meta and MS2 data.
    """

    with open(path_hmdb_ids, "r") as hmdb_ids:
        hmdb_ids_csv = csv.reader(hmdb_ids)

        hmdb_dict = {}
        for row in hmdb_ids_csv:
            hmdb_dict[row[1]] = row[4]

    with open(path_full_hmdb, "r", encoding='utf-8') as full_hmdb:
        full_hmdb_csv = csv.reader(full_hmdb)

        full_hmdb_dict = {}
        for row in full_hmdb_csv:
            full_hmdb_dict[row[9]] = row[0]

    seen_hmdbs = set()
    data_categories = {}
    for path_msp_db, name_msp in zip(paths_msp_db, names_msp):
        data_categories[name_msp] = {}

        msp_db = MspDatabase(path_msp_db)

        # 0            1     2           3       4          5             6    7
        # inchikey_id, name, exact_mass, smiles, accession, precursor_mz, mzs, itensities
        msp_data = msp_db.get_fragments(precursor_type="[M+H]+")

        for spectra in msp_data:
            if spectra[0] in data_categories[name_msp]:
                continue

            mol = Chem.MolFromSmiles(spectra[3])
            try:
                Chem.SanitizeMol(mol)
            except:
                continue

            if mol is None:
                continue

            if mol.GetNumHeavyAtoms() < 4:
                continue

            atom_check = [True for atom in mol.GetAtoms() if atom.GetSymbol() not in ["C", "H", "N", "O", "P", "S"]]
            if len(atom_check) > 0:
                continue

            if "+" in Chem.MolToSmiles(mol, isomericSmiles=False) or "-" in Chem.MolToSmiles(mol, isomericSmiles=False):
                continue

            try:
                hmdb_id = hmdb_dict[spectra[0]]
            except KeyError:
                try:
                    hmdb_id = full_hmdb_dict[spectra[0]]
                except KeyError:
                    if no_hmdb:
                        hmdb_id = spectra[0]
                    else:
                        continue

            if hmdb_id in seen_hmdbs:
                continue
            elif hmdb_id is None:
                if no_hmdb:
                    hmdb_id = spectra[0]
                else:
                    continue
            elif hmdb_id == "":
                if not no_hmdb:
                    continue

            else:
                if hmdb_id != "":
                    seen_hmdbs.add(hmdb_id)

            if hmdb_id + ".xml" not in os.listdir(hmdb_path):
                if not no_hmdb:
                    get_from_hmdb(hmdb_id, hmdb_id, hmdb_path)

            if no_hmdb:
                data_categories[name_msp][hmdb_id]["exact_mass"] = float(spectra[5]) - 1.007276
            else:
                for record_dict in filter_records(parse_xml(os.path.join(hmdb_path, hmdb_id + ".xml"))):
                    data_categories[name_msp][hmdb_id] = {"exact_mass": record_dict["exact_mass"]}

                try:
                    data_categories[name_msp][hmdb_id]["exact_mass"]
                except KeyError:
                    print("Could not retrieve exact mass from HMDB record: " + hmdb_id)
                    continue

            data_categories[name_msp][hmdb_id]["name"] = spectra[1]
            data_categories[name_msp][hmdb_id]["inchikey_id"] = spectra[0]
            data_categories[name_msp][hmdb_id]["precursor_ion_mass"] = float(spectra[5])
            data_categories[name_msp][hmdb_id]["peaks"] = []
            data_categories[name_msp][hmdb_id]["accession"] = hmdb_id
            data_categories[name_msp][hmdb_id]["actual_accession"] = spectra[4]

            data_categories[name_msp][hmdb_id]["mol"] = mol
            data_categories[name_msp][hmdb_id]["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False)

            data_categories[name_msp][hmdb_id]["chemical_formula"] = get_elements(mol)

            chemical_formula = []
            for element in ["C", "H", "N", "O", "P", "S"]:
                data_categories[name_msp][hmdb_id][element] = data_categories[name_msp][hmdb_id]["chemical_formula"][
                    element]
                chemical_formula.append(data_categories[name_msp][hmdb_id]["chemical_formula"][element])

            data_categories[name_msp][hmdb_id]["mc"] = chemical_formula
            data_categories[name_msp][hmdb_id]["chemical_formula"] = ""

            data_categories[name_msp][hmdb_id]["peaks"] = spectra[6]
            data_categories[name_msp][hmdb_id]["intensities"] = spectra[7]

        msp_db.close()

    return data_categories
