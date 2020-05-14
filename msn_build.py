import os
import sys
import urllib.request
import csv
import tempfile
import networkx as nx
from rdkit import Chem
from shutil import rmtree
import pickle

metaboverse_path = os.path.join("..", "..", "..", "metaboverse")

sys.path.append(os.path.join(metaboverse_path, "metaboverse"))
from databases import reformat_xml, update_substructure_database, filter_records, parse_xml, SubstructureDb, get_elements, calculate_exact_mass
from build_structures import build, gen_subs_table

sys.path.append(os.path.join("..", "functions"))
from gen_aux import get_uniq_subs, subset_substructures
from msn_aux import add_small_substructures


def test_build(out_dir, mc, exact_mass, mol, hmdb_id, path_subs, path_k_graphs, path_pkls, heavy_atoms, max_valence,
               accuracy, fragment_masses, ppm, hydrogenation_allowance=2, max_atoms_available=2, max_n_substructures=3):
    mol_smi = Chem.MolToSmiles(mol)
    pre_reccurence = {}

    for fragment_mass in fragment_masses:
        smi_out = os.path.join(out_dir, "{}_".format(hmdb_id) + str(round(fragment_mass, 4)) + ".smi")
        open(smi_out, "w").close()

        for j in range(0 - hydrogenation_allowance, hydrogenation_allowance + 1):
            hydrogenated_fragment_mass = fragment_mass + (j * 1.007825)
            build(mc, exact_mass, smi_out, heavy_atoms, max_valence, accuracy, max_atoms_available, max_n_substructures,
                  path_k_graphs, path_pkls, path_subs, hydrogenated_fragment_mass, ppm, out_mode="a",
                  table_name="msn_subset")

        get_uniq_subs(smi_out, ignore_substructures=True)
        with open(smi_out, mode="r") as smis:
            for line in smis:
                if len(line) > 0:
                    try:
                        pre_reccurence[line.strip()] += 1
                    except KeyError:
                        pre_reccurence[line.strip()] = 1

    with open(os.path.join(out_dir, "structure_ranks.csv"), "w", newline="") as ranks_out:
        ranks_csv = csv.writer(ranks_out)
        ranks_csv.writerow(["kekule_smiles", "occurence"])

        num_recurrence = {}
        num_struct = 0
        for smi in pre_reccurence.keys():
            ranks_csv.writerow([smi, pre_reccurence[smi]])
            num_struct += 1
            try:
                num_recurrence[pre_reccurence[smi]] += 1

            except KeyError:
                num_recurrence[pre_reccurence[smi]] = 1
    try:
        recurrence = str(pre_reccurence[mol_smi])

    except KeyError:
        recurrence = "0"
        better_candidates = 0
        for num in num_recurrence.keys():
            better_candidates += num_recurrence[num]

    else:
        better_candidates = 0
        for num in num_recurrence.keys():
            if num >= pre_reccurence[mol_smi]:
                better_candidates += num_recurrence[num]

    if len(pre_reccurence.values()) > 0:
        max_recurrence = str(max(pre_reccurence.values()))
    else:
        max_recurrence = str(0)

    return [
        str(hmdb_id),
        str(mol_smi),
        str(exact_mass),
        str(len(fragment_masses)),  # number of fragments
        recurrence,  # true structure recurrence
        str(better_candidates),  # number of structures above/equal to candidate
        max_recurrence,  # maximal structure recurrence,
        str(num_struct), # tot structures
        str(fragment_masses),  # fragment neutral masses
        str(heavy_atoms),
        str(max_valence),
        str(accuracy),
        str(ppm)
    ]


def run_test(out_dir, ms_data, test_name, heavy_atoms, max_valence, accuracy, ppm, db_path, test_type="ind_exp", subset=True, max_atoms_available=2):
    with open(os.path.join(out_dir, "results.csv"), newline="", mode="w") as overall_results:
        results_csv = csv.writer(overall_results)
        if test_type == "ind_exh_exp":
            results_csv.writerow([
                "category",
                "HMDB_ID",
                "SMILES",
                "Precursor_Mass",
                "Reconstructed",
                "Total_Structures",
                "Unique_Structures",
                "Heavy_Atoms",
                "Max_Valence",
                "Accuracy"
            ])
        else:
            results_csv.writerow([
                "category",
                "HMDB_ID",
                "SMILES",
                "Precursor_Mass",
                "Num_Peaks",
                "True_Recurrence",
                "Structure_Ranking",
                "Maximal_Recurrence",
                "Total_Structures",
                "Fragment_Masses",
                "Heavy_Atoms",
                "Max_Valence",
                "Accuracy",
                "ppm"
            ])

        if test_type == "ind_exp":
            run_ind_exp(out_dir, ms_data, test_name, heavy_atoms, max_valence, accuracy, ppm, results_csv, db_path, subset=subset, max_atoms_available=max_atoms_available)
        elif test_type == "ind_exh_exp":
            run_ind_exh_exp(out_dir, ms_data, test_name, heavy_atoms, max_valence, accuracy, results_csv, db_path, subset=subset, max_atoms_available=max_atoms_available)


def run_ind_exp(out_dir, ms_data, test_name, heavy_atoms, max_valence, accuracy, ppm, csv, db_path, subset=True, max_atoms_available=2):
    results = []
    for category in ms_data.keys():
        os.mkdir(os.path.join(out_dir, category))
        for hmdb in ms_data[category].keys():
            os.mkdir(os.path.join(out_dir, category, hmdb))

            ms_data[category][hmdb]["neutral_precursor_ion_mass"] = ms_data[category][hmdb][
                                                                        "precursor_ion_mass"] - 1.007276
            ms_data[category][hmdb]["neutral_peaks"] = [peak - 1.007276 for peak in ms_data[category][hmdb]["peaks"]]

            subset_substructures([hmdb], db_path, "subset.sqlite", subset=subset)
            add_small_substructures("subset.sqlite")

            db = SubstructureDb("subset.sqlite", "")
            gen_subs_table(db, heavy_atoms, max_valence, max_atoms_available, table_name="msn_subset")
            db.close()

            results.append([category] + test_build(
                out_dir=os.path.join(out_dir, category, hmdb),
                mc=ms_data[category][hmdb]["mc"],
                exact_mass=ms_data[category][hmdb]["neutral_precursor_ion_mass"],
                mol=ms_data[category][hmdb]["mol"],
                hmdb_id=hmdb,
                path_subs="subset.sqlite",
                path_k_graphs="../../Data/databases/k_graphs.sqlite",
                path_pkls="../../Data/databases/pkls",
                heavy_atoms=heavy_atoms, max_valence=max_valence, accuracy=accuracy,
                fragment_masses=ms_data[category][hmdb]["neutral_peaks"],
                ppm=ppm,
                max_atoms_available=max_atoms_available
            ))

    csv.writerows(results)


def run_ind_exh_exp(out_dir, ms_data, test_name, heavy_atoms, max_valence, accuracy, csv, db_path, subset=True, max_atoms_available=2):
    results = []
    for category in ms_data.keys():
        for hmdb in ms_data[category].keys():

            ms_data[category][hmdb]["neutral_precursor_ion_mass"] = ms_data[category][hmdb]["precursor_ion_mass"] - 1.007276

            subset_substructures([hmdb], db_path, "subset.sqlite", subset=subset)

            db = SubstructureDb("subset.sqlite", "")
            gen_subs_table(db, heavy_atoms, max_valence, max_atoms_available, table_name="msn_subset")
            db.close()

            build(ms_data[category][hmdb]["mc"], ms_data[category][hmdb]["neutral_precursor_ion_mass"],
                  "temp_structures.smi", heavy_atoms, max_valence, accuracy, max_atoms_available, 3,
                  path_db="subset.sqlite", path_db_k_graphs="../../Data/databases/k_graphs.sqlite",
                  path_pkls="../../Data/databases/pkls",  out_mode="w", table_name="msn_subset")

            i = 0
            with open("temp_structures.smi") as smi:
                for i, l in enumerate(smi):
                    pass

            total_structures = i + 1

            reconstructed = False
            mol_smi = Chem.MolToSmiles(ms_data[category][hmdb]["mol"])
            get_uniq_subs("temp_structures.smi", ignore_substructures=True)
            with open("temp_structures.smi") as smi:
                for i, line in enumerate(smi):
                    if line.strip() == mol_smi:
                        reconstructed = True

            uniq_structures = i + 1

            results.append([
                str(category),
                str(hmdb),
                str(mol_smi),
                str(str(ms_data[category][hmdb]["neutral_precursor_ion_mass"])),
                str(reconstructed),
                str(total_structures),
                str(uniq_structures),
                str(heavy_atoms),
                str(max_valence),
                str(accuracy)
            ])

    csv.writerows(results)

def run_group_exp(out_dir, ms_data, test_name, heavy_atoms, max_valence, accuracy, ppm, csv):
    results = []
    for category in ms_data.keys():
        if category != "Catecholamines" and category != "Hexose and Pentose saccharides":
            continue

        group_ids = []
        for hmdb in ms_data[category].keys():
            group_ids.append(hmdb)

        os.mkdir(os.path.join(out_dir, category))
        for hmdb in ms_data[category].keys():
            os.mkdir(os.path.join(out_dir, category, hmdb))

            ms_data[category][hmdb]["neutral_precursor_ion_mass"] = ms_data[category][hmdb]["precursor_ion_mass"] - 1.007276
            ms_data[category][hmdb]["neutral_peaks"] = [peak - 1.007276 for peak in ms_data[category][hmdb]["peaks"]]

            removed_group_ids = [id for id in group_ids if id != hmdb]
            assert len(removed_group_ids) < len(group_ids)

            subset_substructures(removed_group_ids, "../databases/" + test_name + ".sqlite", "../databases/subset.sqlite")
            add_small_substructures("../databases/subset.sqlite")

            results.append([category] + test_build(
                out_dir=os.path.join(out_dir, category, hmdb),
                mc=ms_data[category][hmdb]["mc"],
                exact_mass=ms_data[category][hmdb]["neutral_precursor_ion_mass"],
                mol=ms_data[category][hmdb]["mol"],
                hmdb_id=hmdb,
                path_subs="../databases/" + test_name + ".sqlite",
                path_k_graphs="../databases/k_graphs.sqlite",
                path_pkls="../databases/pkls",
                heavy_atoms=heavy_atoms, max_valence=max_valence, accuracy=accuracy,
                fragment_masses=ms_data[category][hmdb]["neutral_peaks"],
                ppm=ppm
            ))

    csv.writerows(results)
