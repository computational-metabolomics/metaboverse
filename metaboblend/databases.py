#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019-2020 Ralf Weber
#
# This file is part of MetaboBlend.
#
# MetaboBlend is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MetaboBlend is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MetaboBlend.  If not, see <https://www.gnu.org/licenses/>.
#

import io
import os
import pickle
import sqlite3
import tempfile
import itertools
import subprocess
import networkx as nx
from typing import Sequence, Dict, Union

from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import BRICS

from parse import parse_xml
from auxiliary import calculate_complete_multipartite_graphs, graph_to_ri, graph_info, sort_subgraphs


class SubstructureDb:
    """
    Methods for interacting with the SQLITE3 substructure and connectivity databases. Provides a connection to the
    substructure database and, if provided, the connectivity database.

    :param path_substructure_db: Path to the substructure database.

    :param path_connectivity_db: Path to the connectivity database.

    :param conn: A :py:meth:`sqlite3.connection` to the substructure database; the connectivity database will be attached
        as 'graphs'.

    :param cursor: A :py:meth:`sqlite3.connection.cursor` for the substructure database; the connectivity database will
        be attached as "graphs".
    """

    def __init__(self, path_substructure_db, path_connectivity_db=None):
        """Constructor method"""

        self.path_substructure_db = path_substructure_db
        self.path_connectivity_db = path_connectivity_db

        self.conn = sqlite3.connect(self.path_substructure_db)
        self.cursor = self.conn.cursor()

        if self.path_connectivity_db is not None:
            self.cursor.execute("ATTACH DATABASE '%s' as 'graphs';" % self.path_connectivity_db)

        self.temporary_table_names = []

    def select_compounds(self, cpds=[]):
        """
        Select all keys from the compounds table of the substructure database, filtered by HMDB IDs if provided.

        :param cpds: A list of HMDB IDs by which to filter the compounds; applies no filter if not provided.

        :return: Returns the result of the compounds table query via :py:meth:`sqlite3.connection.cursor.fetchall`;
            essentially provides a list containing a list for the values of each row.
        """

        if len(cpds) > 0:
            sql = " WHERE hmdbid in ('%s')" % ("', '".join(map(str, cpds)))
        else:
            sql = ""

        self.cursor.execute("""SELECT DISTINCT hmdbid, exact_mass, formula, C, H, N, O, P, S, smiles 
                               FROM compounds%s""" % sql)

        return self.cursor.fetchall()

    def filter_hmdbid_substructures(self, min_node_weight=2):
        """
        Filters `hmdbid_substructures` table in the substructure database by the frequency of the substructure; allows
        for the removal of non-unique (min_node_weight = 2) or infrequent (min_node_weight > 2) substructures. Generates
        a new table, *unique_hmdbid*, which contains distinct hmdbids and *filtered_hmdbid_substructures* which contains
        the filtered substructures with their frequency.

        :param min_node_weight: Minimal count of the substructure within 'hmdbid_substructures'.
        """

        self.cursor.execute("DROP TABLE IF EXISTS filtered_hmdbid_substructures")

        self.cursor.execute("""CREATE TABLE filtered_hmdbid_substructures AS
                                    SELECT * FROM hmdbid_substructures 
                                    WHERE substructure_id IN (
                                        SELECT substructure_id FROM hmdbid_substructures
                                        GROUP BY substructure_id HAVING COUNT(*) >=%s
                                    )
                            """ % min_node_weight)

    def generate_substructure_network(self, min_node_weight=2, return_networkx=False):
        """
        Generate networks to explore the co-occurence of substructures in the substructure database.

        :param min_node_weight: Minimum frequency of substructures to be included in the network; a minimum node weight
            of 2 means that all non-unique substructures will be present.

        :param return_networkx: If True, returns a :py:meth:`networkx.Graph` containing the generated network.

        :return: If return_networkx, A :py:meth:`networkx.Graph` object containing the generated network.
        """

        self.cursor.execute("DROP TABLE IF EXISTS substructure_graph")

        self.cursor.execute("""CREATE TABLE substructure_graph (
                                   substructure_id_1 INTEGER,
                                   substructure_id_2 INTEGER,
                                   weight INTEGER NOT NULL DEFAULT 0,
                                   PRIMARY KEY (substructure_id_1, substructure_id_2)
                               )
                            """)

        self.filter_hmdbid_substructures(min_node_weight)

        # add edge for each linked parent structure and substructure
        self.cursor.execute("SELECT DISTINCT hmdbid FROM filtered_hmdbid_substructures")

        for hmdb_id in self.cursor.fetchall():
            self.cursor.execute("SELECT substructure_id FROM filtered_hmdbid_substructures WHERE hmdbid = '%s'" % hmdb_id)
            substructures = self.cursor.fetchall()

            seen_substructures = []
            for adj1 in substructures:
                for adj2 in seen_substructures:
                    if adj1 == adj2:
                        continue

                    self.cursor.execute("""INSERT OR IGNORE INTO substructure_graph (
                                               substructure_id_1,
                                               substructure_id_2
                                           ) VALUES ({}, {})""".format(
                                               min(adj1[0], adj2[0]),
                                               max(adj1[0], adj2[0])
                                           ))

                    self.cursor.execute("""UPDATE substructure_graph
                                               SET weight = weight + 1
                                               WHERE substructure_id_1 = '{}'
                                               AND substructure_id_2 = '{}'
                                        """.format(
                                               min(adj1[0], adj2[0]),
                                               max(adj1[0], adj2[0])
                                        ))

                seen_substructures.append(adj1)

        self.conn.commit()

        if return_networkx:
            return self.get_substructure_network()

    def get_substructure_network(self):
        """
        Converts the SQLite 3 network representation to a :py:meth:`networkx.Graph` object.

        :return: A :py:meth:`networkx.Graph` object containing the generated network.
        """

        substructure_graph = nx.Graph()

        self.cursor.execute("""SELECT substructure_id, COUNT(*) 
                                   FROM filtered_hmdbid_substructures
                                   GROUP BY substructure_id
                            """)

        for substructure in self.cursor.fetchall():
            substructure_graph.add_node(substructure[0], weight=substructure[1])

        self.cursor.execute("SELECT * FROM substructure_graph")

        for edge in self.cursor.fetchall():
            substructure_graph.add_edge(edge[0], edge[1], weight=edge[2])

        substructure_graph.remove_nodes_from(list(nx.isolates(substructure_graph)))

        return substructure_graph

    def get_single_edge(self, substructure_ids):
        """
        Get the edge weight corresponding to a subset of substructures

        :param substructure_ids: A list of substructure IDs; weights will be obtained for the combinations.
        """

        substructure_weights = {}

        for substructure_id_1 in substructure_ids:
            for substructure_id_2 in substructure_ids:

                small_id = min(substructure_id_1, substructure_id_2)
                large_id = max(substructure_id_1, substructure_id_2)

                if substructure_id_1 == substructure_id_2:
                    edge_weight = None

                else:
                    self.cursor.execute("""SELECT COUNT(*) 
                                               FROM hmdbid_substructures 
                                               WHERE substructure_id = {}
                                               AND hmdbid IN (
                                                   SELECT hmdbid
                                                       FROM hmdbid_substructures
                                                       WHERE substructure_id = {}
                                               )
                                        """.format(substructure_id_1, substructure_id_2))

                    edge_weight = self.cursor.fetchall()[0][0]

                try:
                    substructure_weights[small_id][large_id] = edge_weight
                except KeyError:
                    substructure_weights[small_id] = {large_id: edge_weight}

        return substructure_weights

    def select_mass_values(self, accuracy, masses, table_name):
        """
        Gets mass values from a table of substructures. Can limit results based on mass at integer level. Used by
        subset_sum to filter substructures prior to building.

        :param accuracy: To which decimal places of accuracy results are to be limited to.

            * **1** Integer level
            * **0_0001** Four decimal places

        :param masses: A list of integers to limit the query by. If a non-empty list of masses is given, the query
            is returned as a list of lists for each mass query, as opposed to a single list of masses.

        :param table_name: Name of the substructure table to be queried.

        :return: Sorted list of mass values from the substructure database, filtered by the supplied parameters.
        """

        if table_name is None:
            table_name = "substructures"
        else:
            table_name += "_substructures"

        if type(masses) == list and len(masses) > 0:

            mass_values = []

            for m in masses:
                self.cursor.execute("""SELECT DISTINCT exact_mass__{} 
                                           FROM {}
                                           WHERE exact_mass__1 = {}
                                    """.format(accuracy, table_name, m))

                m_values = [record[0] for record in self.cursor.fetchall()]
                m_values.sort()

                mass_values.append(m_values)

            return mass_values

        else:

            self.cursor.execute("SELECT DISTINCT exact_mass__{} FROM {}".format(accuracy, table_name))
            mass_values = [record[0] for record in self.cursor.fetchall()]
            mass_values.sort()

            return mass_values

    def select_mfs(self, exact_mass, table_name, accuracy):
        """
        Select elemental compositions based on an exact mass; allows for inexact mass searches based on error (ppm).

        :param exact_mass: The exact mass used to query the substructure database.

        :param table_name: Name of the substructure table to be queried.

        :param accuracy: To which decimal places of accuracy results are to be limited to.

            * **1** Integer level
            * **0_0001** Four decimal places

        :return: Returns the elemental compositions associated with the `exact_mass` via
            :py:meth:`sqlite3.connection.cursor.fetchall`; essentially provides a list containing a list for the values
            of each row (C, H, N, O, P, S).
        """

        if table_name is None:
            table_name = "substructures"
        else:
            table_name += "_substructures"

        self.cursor.execute("""SELECT DISTINCT C, H, N, O, P, S
                                   FROM {}
                                   WHERE exact_mass__{} = {}
                            """.format(
                                   table_name,
                                   accuracy,
                                   str(exact_mass)
                               ))

        return self.cursor.fetchall()

    def k_configs(self):
        """
        Obtains strings detailing the valences for each substructure in a connectivity graph and the ID of the related
        PKL file. Used to match a set of substructures to the correct set of non-isomorphic graphs in the connectivity
        database.

        :return: Dictionary containing the valences as keys and PKL IDs as values.
        """

        self.cursor.execute("""SELECT root, nodes_valences
                                   FROM subgraphs""")

        records = self.cursor.fetchall()
        configs = {}

        for record in records:
            configs[str(record[1])] = []

            for path in self.paths(pickle.loads(record[0])):
                configs[str(record[1])].append(path)

        return configs

    def paths(self, tree, cur=()):
        """
        Parses a tree structure within a dictionary, representing a set of non-isomorphic graphs
        to be used to connect substructures together to generate molecules.

        :param tree: A dictionary containing a set of non-isomorphic graphs for a particular connectivity configuration.

        :param cur: Tuple for results to be appended to.

        :return: For each graph contained within *tree*, generates a tuple of bonds to be formed between substructures.
        """

        if tree == {}:
            yield cur

        else:
            for n, s in tree.items():
                for path in self.paths(s, cur + (n,)):
                    yield path

    def select_substructures(self, l_atoms, table_name):
        """
        Selects specific substructures from a substructure table based on elemental composition. Used to obtain
        sets substructures to be connected.

        :param l_atoms: A list of lists containing the elemental compositions of the queries, in the format
            [C, H, N, O, P, S].

        :param table_name: Name of the substructure table to be queried.

        :return: A list of lists containing the libs of the substructures obtained by the query; the lib is a
            dictionary containing details about the substructure, as generated by
            :py:meth:`metaboblend.databases.get_substructure`.
        """

        if table_name is None:
            table_name = "substructures"
        else:
            table_name += "_substructures"

        subsets = []
        for i in range(len(l_atoms)):

            self.cursor.execute("""SELECT DISTINCT 
                                   smiles,
                                   mol, 
                                   bond_types, 
                                   valence_atoms,  
                                   valence, 
                                   atoms_available,
                                   dummies 
                                       FROM {}
                                       WHERE C = {} 
                                       AND H = {} 
                                       AND N = {} 
                                       AND O = {}
                                       AND P = {}
                                       AND S = {}
                                """.format(table_name,
                                           l_atoms[i][0], l_atoms[i][1], l_atoms[i][2],
                                           l_atoms[i][3], l_atoms[i][4], l_atoms[i][5]))
            records = self.cursor.fetchall()
            if len(records) == 0:
                return []

            ss = []
            for record in records:
                ss.append({
                    "smiles": record[0],
                    "mol": Chem.Mol(record[1]),
                    "bond_types": eval(record[2]),
                    "degree_atoms": eval(record[3]),
                    "valence": record[4],
                    "atoms_available": record[5],
                    "dummies": eval(record[6])
                })

            subsets.append(ss)

        return subsets

    def calculate_possible_hydrogenations(self):
        """
        Calculate likely hydrogen re-arrangements, as per
        :py:meth:`metaboblend.databases.calculate_hydrogen_rearrangements`. Inserts these into the substructure ions
        table.
        """

        self.cursor.execute("SELECT ROWID, smiles, mol, bond_types, exact_mass FROM substructures")

        for substructure in self.cursor.fetchall():

            mol = Chem.Mol(substructure[2])
            bond_types = eval(substructure[3])

            fragment_ions = []
            for i, fragment_ion in enumerate(bond_types.keys()):

                atom_symbol = mol.GetAtomWithIdx(fragment_ion).GetSymbol()

                for bond_type in bond_types[fragment_ion]:
                    fragment_ions.append((atom_symbol, bond_type == 2))

            positive_hydrogenations = set()
            negative_hydrogenations = set()

            for fragment_ion_permutation in itertools.permutations(fragment_ions):

                positive_hydrogenations.update(calculate_hydrogen_rearrangements(fragment_ion_permutation, "+"))
                negative_hydrogenations.update(calculate_hydrogen_rearrangements(fragment_ion_permutation, "-"))

            self.insert_substructure_ion(substructure, positive_hydrogenations, 1)
            self.insert_substructure_ion(substructure, negative_hydrogenations, 0)

        self.conn.commit()

    def insert_substructure_ion(self, substructure, possible_hydrogenations, ion_mode):
        """
        Insert substructure ions into the substructure_ions table.

        :param substructure: List, [ROWID (int), smiles (str), mol (RDKit Mol), bond_types (Dict), exact_mass (float]

        :param possible_hydrogenations: Number of hydrogens to add compared to the mass of the neutral substructure.

        :param ion_mode: If True, assumes positive ion mode, else is configured for negative ion mode.
        """

        for possible_hydrogenation in possible_hydrogenations:

            self.cursor.execute("""INSERT INTO substructure_ions (
                                       substructure_id, 
                                       hydrogen_modification, 
                                       ion_mode_positive,
                                       modified_exact_mass__1, 
                                       modified_exact_mass__0_0001)
                                   values ({}, {}, {}, {}, {})""".format(
                                       substructure[0],
                                       possible_hydrogenation,
                                       ion_mode,
                                       round(substructure[4] + (possible_hydrogenation * 1.007825), 0),
                                       round(substructure[4] + (possible_hydrogenation * 1.007825), 4)
                                   ))

    def create_compound_database(self):
        """Generates a substructure database, removing previously existing tables if they are present."""

        self.cursor.execute("DROP TABLE IF EXISTS compounds")
        self.cursor.execute("DROP TABLE IF EXISTS substructures")
        self.cursor.execute("DROP TABLE IF EXISTS substructure_ions")
        self.cursor.execute("DROP TABLE IF EXISTS hmdbid_substructures")

        self.cursor.execute("""CREATE TABLE compounds (
                                   hmdbid TEXT PRIMARY KEY,
                                   exact_mass INTEGER,
                                   formula TEXT,
                                   C INTEGER,
                                   H INTEGER,
                                   N INTEGER,
                                   O INTEGER,
                                   P INTEGER,
                                   S INTEGER,
                                   smiles TEXT)""")

        self.cursor.execute("""CREATE TABLE substructures (
                                   substructure_id INTEGER PRIMARY KEY,
                                   smiles TEXT NOT NULL UNIQUE, 
                                   heavy_atoms INTEGER,
                                   length INTEGER,
                                   exact_mass__1 INTEGER,
                                   exact_mass__0_0001 REAL,
                                   exact_mass REAL,
                                   C INTEGER,
                                   H INTEGER,
                                   N INTEGER,
                                   O INTEGER,
                                   P INTEGER,
                                   S INTEGER,
                                   valence INTEGER,
                                   valence_atoms TEXT,
                                   atoms_available INTEGER,
                                   bond_types TEXT,
                                   dummies TEXT,
                                   mol BLOB)""")

        self.cursor.execute("""CREATE TABLE substructure_ions (
                                   substructure_id INTEGER, 
                                   hydrogen_modification INTEGER, 
                                   ion_mode_positive BOOLEAN,
                                   modified_exact_mass__1 INTEGER, 
                                   modified_exact_mass__0_0001 REAL,
                                   PRIMARY KEY (substructure_id, hydrogen_modification, ion_mode_positive),
                                   FOREIGN KEY (substructure_id) REFERENCES substructures(substructure_id))""")

        self.cursor.execute("""CREATE TABLE hmdbid_substructures (
                                   hmdbid TEXT,
                                   substructure_id INTEGER,
                                   PRIMARY KEY (hmdbid, substructure_id),
                                   FOREIGN KEY (substructure_id) REFERENCES substructures(substructure_id),
                                   FOREIGN KEY (hmdbid) REFERENCES compounds(hmdbid))""")

    def create_indexes(self):
        """Creates indexes for the `substructures` table for use by the build method."""

        self.cursor.execute("DROP INDEX IF EXISTS modified_exact_mass__1")
        self.cursor.execute("DROP INDEX IF EXISTS modified_exact_mass__0_0001")
        self.cursor.execute("DROP INDEX IF EXISTS exact_mass__1")
        self.cursor.execute("DROP INDEX IF EXISTS exact_mass__0_0001")
        self.cursor.execute("DROP INDEX IF EXISTS atoms")

        self.cursor.execute("""CREATE INDEX modified_exact_mass__1
                               ON %s (modified_exact_mass__1)""" % "substructure_ions")

        self.cursor.execute("""CREATE INDEX modified_exact_mass__0_0001
                               ON %s (modified_exact_mass__0_0001)""" % "substructure_ions")

        self.cursor.execute("""CREATE INDEX exact_mass__1
                               ON %s (exact_mass__1)""" % "substructures")

        self.cursor.execute("""CREATE INDEX exact_mass__0_0001
                               ON %s (exact_mass__0_0001)""" % "substructures")

        self.cursor.execute("""CREATE INDEX atoms ON %s (C, H, N, O, P, S);""" % "substructures")

        self.cursor.execute("DROP INDEX IF EXISTS heavy_atoms__valence__atoms_available__exact_mass__1")
        self.cursor.execute("DROP INDEX IF EXISTS smiles__heavy_atoms__valence__atoms_available__exact_mass__1")

        self.cursor.execute("""CREATE INDEX heavy_atoms__valence__atoms_available__exact_mass__1
                               ON %s (heavy_atoms, atoms_available, valence, exact_mass__1)""" % "substructures")

        self.cursor.execute("""CREATE INDEX smiles__heavy_atoms__valence__atoms_available__exact_mass__1
                                   ON %s (smiles, heavy_atoms, atoms_available, valence, exact_mass__1)""" % "substructures")

    def create_temp_indexes(self, table_name):
        """ Creates indexes for a temporary substructure tables. """

        self.cursor.execute("DROP INDEX IF EXISTS %s_modified_exact_mass__1" % table_name)
        self.cursor.execute("DROP INDEX IF EXISTS %s_modified_exact_mass__0_0001" % table_name)
        self.cursor.execute("DROP INDEX IF EXISTS %s_atoms" % table_name)
        self.cursor.execute("DROP INDEX IF EXISTS %s_exact_mass__1" % table_name)
        self.cursor.execute("DROP INDEX IF EXISTS %s_exact_mass__0_0001" % table_name)

        self.cursor.execute("""CREATE INDEX {}_modified_exact_mass__1
                               ON {} (modified_exact_mass__1)
                            """.format(table_name, table_name + "_substructure_ions"))

        self.cursor.execute("""CREATE INDEX {}_modified_exact_mass__0_0001
                               ON {} (modified_exact_mass__0_0001)
                            """.format(table_name, table_name + "_substructure_ions"))

        self.cursor.execute("""CREATE INDEX {}_exact_mass__1
                               ON {} (exact_mass__1)
                            """.format(table_name, table_name + "_substructures"))

        self.cursor.execute("""CREATE INDEX {}_exact_mass__0_0001
                               ON {} (exact_mass__0_0001)
                            """.format(table_name, table_name + "_substructures"))

        self.cursor.execute("""CREATE INDEX {}_atoms ON {} (C, H, N, O, P, S)
                            """.format(table_name, table_name + "_substructures"))

    def close(self):
        """ Remove temporary tables from the database and close the connection. """

        for temporary_table_name in self.temporary_table_names:
            self.cursor.execute("DROP TABLE IF EXISTS %s" % temporary_table_name)

        self.temporary_table_names = []
        self.conn.close()


def get_substructure(mol, idxs_edges_subgraph, isomeric_smiles=False):
    """
    Generates information for the substructure database from a reference molecule and the bond IDs of a substructure.

    :param mol: An :py:meth:`rdkit.Chem.Mol` object containing a reference molecule that has been fragmented.

    :param idxs_edges_subgraph: Either a list of atom indices within the reference molecule that make up the
        substructure (as returned by :py:meth:`metaboblend.databases.get_sgs`) or an integer representing the index
        of a single atom.

    :param isomeric_smiles: If True, returns smiles with non-structural isomeric information.

    :return: A list of lists containing the libs of the substructures obtained by the query; the lib is a
        dictionary containing details about the substructure, in the format:

        * "**smiles**": Substructure smiles string

        * "**mol**": Substructure :py:meth:`rdkit.Chem.Mol`

        * "**bond_types**": The type of bonds to be formed by dummy atoms - see
            :py:meth:`metaboblend.build_structures.add_bonds` and :py:meth:`Chem.rdchem.BondType`. Is a dictionary
            whose keys are atom indices and values are bond types, as follows:

            * **1.0** Single
            * **1.5** Aromatic
            * **2.0** Double

        * "**degree_atoms**": A dictionary containing indices of the atoms connected to dummy atoms that can form bonds
            during structure generation as keys, and the number of bonds they can form as values.

        * "**valence**": The total number of bonds that can be formed by the substructure
            (the product of `degree_atoms` and `atoms_available`).

        * "**atoms_available**": The total number of degree atoms.

        * "**dummies**": List of the indices of atoms that may be removed to form bonds during structure generation,
            represented by `*`.
    """

    # convert list of bond indices to list of atom indices
    if isinstance(idxs_edges_subgraph, int):  # small substructure addition
        atom_idxs_subgraph = [idxs_edges_subgraph]
    else:
        atom_idxs_subgraph = []
        for bIdx in idxs_edges_subgraph:
            b = mol.GetBondWithIdx(bIdx)
            a1 = b.GetBeginAtomIdx()
            a2 = b.GetEndAtomIdx()

            if a1 not in atom_idxs_subgraph:
                atom_idxs_subgraph.append(a1)
            if a2 not in atom_idxs_subgraph:
                atom_idxs_subgraph.append(a2)

    # identify atoms which will become dummy elements in the final substructure
    atoms_to_dummy = []
    for idx in atom_idxs_subgraph:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            if atom.GetIdx() not in atom_idxs_subgraph:
                atoms_to_dummy.append(atom.GetIdx())

    mol_edit = Chem.EditableMol(mol)
    degree_atoms = {}

    for atom in reversed(mol.GetAtoms()):

        if atom.GetIdx() in atoms_to_dummy:
            mol_edit.ReplaceAtom(atom.GetIdx(), Chem.Atom("*"))

    mol = mol_edit.GetMol()
    mol_edit = Chem.EditableMol(mol)

    for atom in reversed(mol.GetAtoms()):
        if atom.GetIdx() not in atom_idxs_subgraph and atom.GetSymbol() != "*":
            mol_edit.RemoveAtom(atom.GetIdx())

    mol_out = mol_edit.GetMol()

    dummies = [atom.GetIdx() for atom in mol_out.GetAtoms() if atom.GetSymbol() == "*"]

    # get bond degrees
    for atom in mol_out.GetAtoms():

        if atom.GetIdx() in dummies:

            for atom_n in atom.GetNeighbors():

                if atom_n.GetSymbol() == "*":
                    continue  # do not count dummies for valence calculations
                elif atom_n.GetIdx() not in degree_atoms:
                    degree_atoms[atom_n.GetIdx()] = 1
                else:
                    degree_atoms[atom_n.GetIdx()] += 1

    # returns the type of the bond as a double (i.e. 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE)
    bond_types = {}

    for b in mol_out.GetBonds():

        begin_atom = mol_out.GetAtomWithIdx(b.GetBeginAtomIdx())
        end_atom = mol_out.GetAtomWithIdx(b.GetEndAtomIdx())

        # use bond types to dummy atoms to inform future structure building from compatible substructures
        if begin_atom.GetSymbol() == "*" and end_atom.GetSymbol() != "*":  # do not count dummy-dummy bonds for valence calculations
            if b.GetEndAtomIdx() not in bond_types:
                bond_types[b.GetEndAtomIdx()] = [b.GetBondTypeAsDouble()]
            else:
                bond_types[b.GetEndAtomIdx()].append(b.GetBondTypeAsDouble())

        elif end_atom.GetSymbol() == "*" and begin_atom.GetSymbol() != "*":
            if b.GetBeginAtomIdx() not in bond_types:
                bond_types[b.GetBeginAtomIdx()] = [b.GetBondTypeAsDouble()]
            else:
                bond_types[b.GetBeginAtomIdx()].append(b.GetBondTypeAsDouble())

    try:
        mol_out.UpdatePropertyCache()  # alternative to Chem.SanitizeMol that updates valence information
    except:
        return

    valence = 0
    for atom_available in bond_types.values():
        for bond_type in atom_available:
            if bond_type == 2:
                valence += 2
            else:
                valence += 1

    return {"smiles": Chem.MolToSmiles(mol_out, isomericSmiles=isomeric_smiles),  # REORDERED ATOM INDEXES
            "mol": mol_out,
            "bond_types": bond_types,
            "degree_atoms": degree_atoms,
            "valence": valence,
            "atoms_available": len(degree_atoms.keys()),
            "dummies": dummies}


def get_elements(mol, elements=None):
    """
    Gets the elemental composition of a molecule.

    :param mol: An :py:meth:`rdkit.Chem.Mol` object containing the molecule of interest.

    :param elements: A dictionary whose keys are strings representing an element and values are 0. Unspecified,
        defaults to `{"C": 0, "H": 0, "N": 0, "O": 0, "P": 0, "S": 0, "*": 0}`.

    :return: The dictionary specified by **elements**, with the number of atoms corresponding to each element as keys.
    """

    if not elements:
        elements = {"C": 0, "H": 0, "N": 0, "O": 0, "P": 0, "S": 0, "*": 0}

    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():

        elements[atom.GetSymbol()] += 1

    return elements


def calculate_exact_mass(mol, exact_mass_elements=None):
    """
    Gets the exact mass of a molecule.

    :param mol: An :py:meth:`rdkit.Chem.Mol` object containing the molecule of interest.

    :param exact_mass_elements: A dictionary whose keys are strings representing an element and values are the exact masses of
        each element. Unspecified, defaults to :
        `{"C": 12.0, "H": 1.007825, "N": 14.003074, "O": 15.994915, "P": 30.973763, "S": 31.972072, "*": -1.007825}`

    :return: The exact mass of the molecule.
    """

    if not exact_mass_elements:
        exact_mass_elements = {"C": 12.0, "H": 1.007825, "N": 14.003074, "O": 15.994915, "P": 30.973763, "S": 31.972072}

    exact_mass = 0.0
    mol = Chem.AddHs(mol)

    for atom in mol.GetAtoms():

        atom_symbol = atom.GetSymbol()

        if atom_symbol != "*":
            exact_mass += exact_mass_elements[atom_symbol]

    return exact_mass


def filter_records(records, isomeric_smiles=False):
    """
    Filters records generated by :py:meth:`parse_xml` to ensure they are compatible with the MetaboBlend workflow.

    :param records: A dictionary containing information about the molecule, as generated by :py:meth:`parse_xml`.

    :param isomeric_smiles: If True, returns smiles with non-structural isomeric information.

    :return: Generates a dictionary containing key information extracted from the record.

        * "**HMDB_ID**": The HMDB ID of the molecule, in the format "HMDBXXXXXXX".

        * "**formula**": The elemental composition of the molecule, in the format "CXHXNXOXPXSX".

        * "**exact_mass**": The exact mass of the molecule, rounded to 6d.p.

        * "**smiles**": A string containing the smiles representation of the molecule.

        * "**smiles_rdkit**": A string containing the smiles representation of the molecule.

        * "**smiles_rdkit_kek**": A string containing the kekule smiles representation of the molecule.

        * "C", "H", "N", "O", "P", "S": Integers referring to elemental composition.

        * "**mol**": An :py:meth:`rdkit.Chem.Mol` object containing the molecule.
    """

    for record in records:

        if "smiles" in record:
            mol = Chem.MolFromSmiles(record["smiles"])

            try:
                Chem.SanitizeMol(mol)  # confirm mols are legitimate
            except:
                continue

            if mol is None:
                continue

            if mol.GetNumHeavyAtoms() < 4:
                continue  # structures with <3 heavy atoms will not produce useful substructures

            atom_check = [True for atom in mol.GetAtoms() if atom.GetSymbol() not in ["C", "H", "N", "O", "P", "S"]]
            if len(atom_check) > 0:
                continue

            smiles_rdkit = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)

            if "+" in smiles_rdkit or "-" in smiles_rdkit:
                continue  # only neutral molecules are compatible

            els = get_elements(mol)
            exact_mass = calculate_exact_mass(mol)

            record_dict = {'HMDB_ID': record['accession'],
                           'formula': record["chemical_formula"],
                           'exact_mass': round(exact_mass, 6),
                           'smiles': record['smiles'],
                           'smiles_rdkit': smiles_rdkit,
                           'smiles_rdkit_kek': Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles, kekuleSmiles=True),
                           'C': els['C'],
                           'H': els['H'],
                           'N': els['N'],
                           'O': els['O'],
                           'P': els['P'],
                           'S': els['S'],
                           'mol': mol}

            yield record_dict


def get_substructure_bond_idx(prb_mol, ref_mol):
    """
    Takes a substructure and the original molecule from which it was generated and matches the substructure to its
    original position in the reference molecule. Will only find a single solution, whilst multiple are possible.

    :param prb_mol: Substructure (without dummy atoms) as an :py:meth:`rdkit.Chem.Mol` object.

    :param ref_mol: Original molecule for the substructure to be matched in as an :py:meth:`rdkit.Chem.Mol` object.

    :returns: None if there is no match for the **prb_mol** in the **ref_mol**, else returns a tuple of the bond indices
        in the **ref_mol* matched by the **prb_mol**.
    """

    if ref_mol.HasSubstructMatch(prb_mol):
        atom_idx = ref_mol.GetSubstructMatch(prb_mol)
    else:
        return None

    bond_idx = ()
    for atom in ref_mol.GetAtoms():
        if atom.GetIdx() in atom_idx:
            for bond in atom.GetBonds():
                if bond.GetBeginAtomIdx() in atom_idx and bond.GetEndAtomIdx() in atom_idx:
                    if bond.GetIdx() not in bond_idx:
                        bond_idx = (*bond_idx, bond.GetIdx())

    return bond_idx


def subset_sgs_sizes(sgs, n_min, n_max):
    """
    Some substructure generation methods require that their results be filtered to ensure that they are within the
    databvase substructure size requirements.

    :param sgs: A list of lists containing the indices of edges in the original molecule that represent its
        substructures.

    :param n_min: The minimum number of bonds (edges) for a valid substructure.

    :param n_max: The maximum number of bonds (edges) for a valid substructure.

    :return: The original sgs list, with those substructures that are not within n_min and n_max (inclusive) removed.
    """

    sgs_new = []

    for i, edge_idxs in enumerate(sgs):
        edge_idxs_new = []

        for j, bonds in enumerate(edge_idxs):

            if n_max is None:
                if n_min <= len(bonds):
                    edge_idxs_new.append(bonds)

            else:
                if n_min <= len(bonds) <= n_max:
                    edge_idxs_new.append(bonds)

        if len(edge_idxs_new) > 0:
            sgs_new.append(edge_idxs_new)

    return sgs_new


def get_sgs(record_dict, n_min, n_max, method="exhaustive"):
    """
    Generates substructures based on an original molecule, which are described by sets of its bond indices.

    :param record_dict: A dictionary of key information about the original molecule, as generated by
        :py:meth:`metaboblend.databases.filter_records`. Includes HMDBID, smiles and the related
        :py:meth:`rdkit.Chem.Mol` representation.

    :param n_min: The minimum number of bonds (edges) for a valid substructure.

    :param n_max: The maximum number of bonds (edges) for a valid substructure.

    :param method: The method by which to fragment molecules. Substructures must have an exact substructure match in
        the original molecule in order to be considered valid.

        * **exhaustive** The default method for substructure generation. Generates all substructures for a molecule
            within the size range. See :py:meth:`rdkit.Chem.FindAllSubgraphsOfLengthMToN`.

        * **RECAP**  Generates substructures using the retrosynthetic combinatorial analysis procedure; fragments are
            identified that are likely to be useful for drug synthesis. See :py:meth:`rdkit.Chem.RECAP`.

        * **BRICS** Generates substructures by breaking retrosynthetically interesting chemical substructures; fragments
            are identified that are likely to be useful for drug synthesis.. See :py:meth:`rdkit.Chem.BRICS`.

    :return: A list of lists of bond indices referring to substructures of the original molecule.
    """

    if method == "exhaustive":

        if n_max is None:
            n_max = 1000

        return Chem.FindAllSubgraphsOfLengthMToN(record_dict["mol"], n_min, n_max)

    elif method == "RECAP":

        hierarchy = Recap.RecapDecompose(record_dict["mol"])
        sgs = []
        for substructure in hierarchy.GetAllChildren().values():

            substructure = Chem.DeleteSubstructs(substructure.mol, Chem.MolFromSmarts('[#0]'))
            edge_idxs = get_substructure_bond_idx(substructure, record_dict["mol"])
            if edge_idxs is not None:
                sgs.append(edge_idxs)

        return subset_sgs_sizes(sgs=[sgs], n_min=n_min, n_max=n_max)

    elif method == "BRICS":

        substructures = BRICS.BRICSDecompose(record_dict["mol"])
        sgs = []
        for substructure in substructures:
            substructure = Chem.DeleteSubstructs(Chem.MolFromSmiles(substructure), Chem.MolFromSmarts('[#0]'))
            edge_idxs = get_substructure_bond_idx(substructure, record_dict["mol"])

            if edge_idxs is not None:
                sgs.append(edge_idxs)

        return subset_sgs_sizes(sgs=[sgs], n_min=n_min, n_max=n_max)


def create_substructure_database(hmdb_paths: Union[str, bytes, os.PathLike],
                                 path_substructure_db: Union[str, bytes, os.PathLike],
                                 ha_min: Union[int, None] = None,
                                 ha_max: Union[int, None] = None,
                                 max_degree: Union[int, None] = 6,
                                 max_atoms_available: Union[int, None] = 2,
                                 method: str = "exhaustive",
                                 substructures_only: bool = False,
                                 isomeric_smiles: bool = False) -> None:
    """
    Creates a substructure database by fragmenting one or more input molecules. Combinations of
    substructures in this database are used to build new molecules. Fragmentation is carried out by selecting
    connected sets bonds in the supplied compound(s). Creates the database before calling
    'metaboverse.databases.update_substructure_database' to add substructures for each input molecule. Generates
    indexes on the substructure table.

    :param hmdb_paths: The paths of the HMDB XML records detailing molecules to be fragmented.

    :param path_substructure_db: The path of the SQLite 3 substructure database to be created.

    :param ha_min: The minimum size (number of heavy atoms) of substructures to be added to the substructure
        database. If None, no limit is applied.

    :param ha_max: The maximum size (number of heavy atoms) of substructures to be added to the substructure
        database. None, no limit is applied.

    :param max_atoms_available: The maximum number of  atoms available of each substructure to be considered for
        building molecules. `atoms_available` refers to the number of atoms on a substructure involved in forming
        chemical bonds (e.g. single or double bonds). Atoms available are also limited by the extensivity of the
        supplied connectivity database.

    :param max_degree: The maximum allowable degree of substructures to be considered for building structures. We
        define degree as the product of `atoms_available` and the degree of their bonds (bond types, where 1 = single,
        2 = double, etc.). Maximum degree is also limited by the extensivity of the supplied connectivity database. For
        instance, a substructure that has 3 `atoms_available`, each of their bond types being single bonds, would have
        a total degree of 3.

    :param method: The method by which to fragment molecules. Substructures must have an exact substructure match in
        the original molecule in order to be considered valid.

        * **exhaustive** The default method for substructure generation. Generates all substructures for a molecule
            within the size range. See :py:meth:`rdkit.Chem.FindAllSubgraphsOfLengthMToN`.

        * **RECAP**  Generates substructures using the retrosynthetic combinatorial analysis procedure; fragments are
            identified that are likely to be useful for drug synthesis. See :py:meth:`rdkit.Chem.RECAP`.

        * **BRICS** Generates substructures by breaking retrosynthetically interesting chemical substructures; fragments
            are identified that are likely to be useful for drug synthesis.. See :py:meth:`rdkit.Chem.BRICS`.

    :param substructures_only: Whether to generate all tables or only the substructures table. Retains necessary
        information for building and reduces database size.

    :param isomeric_smiles: If True, generates a database using smiles with non-structural isomeric information.
    """

    db = SubstructureDb(path_substructure_db)
    db.create_compound_database()
    db.close()

    for hmdb_path in hmdb_paths:
        update_substructure_database(hmdb_path=hmdb_path, path_substructure_db=path_substructure_db, ha_min=ha_min,
                                     ha_max=ha_max, method=method, max_atoms_available=max_atoms_available,
                                     max_degree=max_degree, substructures_only=substructures_only,
                                     isomeric_smiles=isomeric_smiles)

    db = SubstructureDb(path_substructure_db)
    db.calculate_possible_hydrogenations()
    db.create_indexes()
    db.close()


def update_substructure_database(hmdb_path: Union[str, bytes, os.PathLike, None],
                                 path_substructure_db: Union[str, bytes, os.PathLike],
                                 ha_min: Union[int, None] = None,
                                 ha_max: Union[int, None] = None,
                                 max_atoms_available: Union[int, None] = None,
                                 max_degree: Union[int, None] = None,
                                 method: str = "exhaustive",
                                 substructures_only: bool = False,
                                 records: Union[Sequence[Dict], None] = None,
                                 isomeric_smiles: bool = False) -> None:
    """
    Add entries to the substructure database by fragmenting a molecule or set of molecules. Combinations of
    substructures in this database are used to build new molecules. Fragmentation is carried out by selecting
    connected sets bonds in the supplied compound(s).

    :param hmdb_path: The path of the HMDB XML record(s) detailing molecules to be fragmented. Can take HMDB records for
        individual metabolites or the entirety of HMDB. Will be overriden by
        `records` parameter, if provided.

    :param path_substructure_db: The path of the existing SQLite 3 substructure database to be updated.

    :param ha_min: The minimum size (number of heavy atoms) of substructures to be added to the substructure
        database. If None, no limit is applied.

    :param ha_max: The maximum size (number of heavy atoms) of substructures to be added to the substructure
        database. None, no limit is applied.

    :param max_atoms_available: The maximum number of  atoms available of each substructure to be considered for
        building molecules. `atoms_available` refers to the number of atoms on a substructure involved in forming
        chemical bonds (e.g. single or double bonds). Atoms available are also limited by the extensivity of the
        supplied connectivity database.

    :param max_degree: The maximum allowable degree of substructures to be considered for building structures. We
        define degree as the product of `atoms_available` and the degree of their bonds (bond types, where 1 = single,
        2 = double, etc.). Maximum degree is also limited by the extensivity of the supplied connectivity database. For
        instance, a substructure that has 3 `atoms_available`, each of their bond types being single bonds, would have
        a total degree of 3.

    :param method: The method by which to fragment molecules. Substructures must have an exact substructure match in
        the original molecule in order to be considered valid.

        * **exhaustive** The default method for substructure generation. Generates all substructures for a molecule
            within the size range. See :py:meth:`rdkit.Chem.FindAllSubgraphsOfLengthMToN`.

        * **RECAP**  Generates substructures using the retrosynthetic combinatorial analysis procedure; fragments are
            identified that are likely to be useful for drug synthesis. See :py:meth:`rdkit.Chem.RECAP`.

        * **BRICS** Generates substructures by breaking retrosynthetically interesting chemical substructures; fragments
            are identified that are likely to be useful for drug synthesis.. See :py:meth:`rdkit.Chem.BRICS`.

    :param substructures_only: Whether to generate all tables or only the substructures table. Retains necessary
        information for building and reduces database size.

    :param records: Records of molecules to be fragmented. Must be a list containing dictionaries containing key
        information about the molecules, as generated by :py:meth:`metaboblend.databases.parse_xml`; if records
        is not supplied, the records will be obtained from the XML at `hmdb_path`.

    :param isomeric_smiles: If True, generates a database using smiles with non-structural isomeric information.
    """

    conn = sqlite3.connect(path_substructure_db)
    cursor = conn.cursor()

    if records is None:
        records = parse_xml(hmdb_path, reformat=False)

    if ha_min is None:
        ha_min = 1

    if ha_max is None:
        ha_max = 9999

    for record_dict in filter_records(records, isomeric_smiles=isomeric_smiles):
        if not substructures_only:
            cursor.execute("""INSERT OR IGNORE INTO compounds (
                                   hmdbid, 
                                   exact_mass, 
                                   formula, 
                                   C, H, N, O, P, S, 
                                   smiles)
                               VALUES (
                                   :HMDB_ID, 
                                   :exact_mass,
                                   :formula, 
                                   :C, :H, :N, :O, :P, :S, 
                                   :smiles)""", record_dict)

        # Returns a tuple of 2-tuples with bond IDs
        for sgs in get_sgs(record_dict=record_dict, n_min=ha_min-1, n_max=ha_max-1, method=method):
            for edge_idxs in sgs:
                lib = get_substructure(record_dict["mol"], edge_idxs, isomeric_smiles=isomeric_smiles)  # convert bond IDs to substructure mol

                # insert substructure obtained from get_sgs
                insert_substructure(lib, cursor, record_dict, substructures_only, max_atoms_available, max_degree,
                                    isomeric_smiles)

        if ha_min <= 1:
            for atom in record_dict["mol"].GetAtoms():
                lib = get_substructure(record_dict["mol"], atom.GetIdx(), isomeric_smiles=isomeric_smiles)

                # insert single atom substructures
                insert_substructure(lib, cursor, record_dict, substructures_only, max_atoms_available, max_degree,
                                    isomeric_smiles)

    conn.commit()
    conn.close()


def insert_substructure(lib, cursor, record_dict, substructures_only, max_atoms_available, max_degree, isomeric_smiles):
    """
    Converts the details of a single substructure into an entry in a substructure database. See
    :py:meth:`update_substructure_database`.

    :param lib: A dictionary containing details about the substructure, as returned by
        :py:meth:`metaboblend.databases.get_substructure`, in the format:

        * "**smiles**": Substructure smiles string

        * "**mol**": Substructure :py:meth:`rdkit.Chem.Mol`

        * "**bond_types**": The type of bonds to be formed by dummy atoms - see
            :py:meth:`metaboblend.build_structures.add_bonds` and :py:meth:`Chem.rdchem.BondType`. Is a dictionary
            whose keys are atom indices and values are bond types, as follows:

            * **1.0** Single
            * **1.5** Aromatic
            * **2.0** Double

        * "**degree_atoms**": A dictionary containing indices of the atoms connected to dummy atoms that can form bonds
            during structure generation as keys, and the number of bonds they can form as values.

        * "**valence**": The total number of bonds that can be formed by the substructure
            (the product of `degree_atoms` and `atoms_available`).

        * "**atoms_available**": The total number of degree atoms.

        * "**dummies**": List of the indices of atoms that may be removed to form bonds during structure generation,
            represented by `*`.

    :param cursor: SQLite3 cursor connected to the substructure database. Used to insert substructures.

    :param record_dict: Record of molecule to be fragmented. Must be a dictionary containing key
        information about the molecule, as generated by :py:meth:`metaboblend.databases.parse_xml`.

    :param substructures_only: Whether to generate all tables or only the substructures table. Retains necessary
        information for building and reduces database size.

    :param max_atoms_available: The maximum number of  atoms available of each substructure to be considered for
        building molecules. `atoms_available` refers to the number of atoms on a substructure involved in forming
        chemical bonds (e.g. single or double bonds). Atoms available are also limited by the extensivity of the
        supplied connectivity database.

    :param max_degree: The maximum allowable degree of substructures to be considered for building structures. We
        define degree as the product of `atoms_available` and the degree of their bonds (bond types, where 1 = single,
        2 = double, etc.). Maximum degree is also limited by the extensivity of the supplied connectivity database. For
        instance, a substructure that has 3 `atoms_available`, each of their bond types being single bonds, would have
        a total degree of 3.

    :param isomeric_smiles: If True, generates database entries using smiles with non-structural isomeric information.
    """

    if lib is None:
        return

    if lib["valence"] == 0:
        return

    if max_atoms_available is not None:
        if lib["atoms_available"] > max_atoms_available:
            return

    if max_degree is not None:
        if lib["valence"] > max_degree:
            return

    smiles_rdkit = Chem.MolToSmiles(lib["mol"], isomericSmiles=isomeric_smiles)  # canonical rdkit smiles

    exact_mass = calculate_exact_mass(lib["mol"])
    els = get_elements(lib["mol"])

    sub_smi_dict = {'smiles': smiles_rdkit,
                    'exact_mass': exact_mass,
                    'length': sum([els[atom] for atom in els if atom != "*"]),
                    "valence": lib["valence"],
                    "valence_atoms": str(lib["degree_atoms"]),
                    "atoms_available": lib["atoms_available"],
                    "mol": lib["mol"].ToBinary(),
                    "bond_types": str(lib["bond_types"]),
                    "dummies": str(lib["dummies"])}

    sub_smi_dict["exact_mass__1"] = round(sub_smi_dict["exact_mass"], 0)
    sub_smi_dict["exact_mass__0_0001"] = round(sub_smi_dict["exact_mass"], 4)

    sub_smi_dict.update(els)
    sub_smi_dict["heavy_atoms"] = sum([els[atom] for atom in els if atom != "H" and atom != "*"])

    cursor.execute("""INSERT OR IGNORE INTO substructures (
                          smiles, 
                          heavy_atoms, 
                          length, 
                          exact_mass__1, 
                          exact_mass__0_0001, 
                          exact_mass, 
                          C, 
                          H, 
                          N, 
                          O, 
                          P, 
                          S, 
                          valence, 
                          valence_atoms, 
                          atoms_available, 
                          bond_types,
                          dummies,
                          mol)
                      values (
                          :smiles,
                          :heavy_atoms,
                          :length,
                          :exact_mass__1,
                          :exact_mass__0_0001,
                          :exact_mass,
                          :C,
                          :H,
                          :N,
                          :O,
                          :P,
                          :S,
                          :valence,
                          :valence_atoms,
                          :atoms_available,
                          :bond_types,
                          :dummies,
                          :mol)""", sub_smi_dict)

    if not substructures_only:
        cursor.execute("SELECT substructure_id FROM substructures WHERE smiles = '%s'" % sub_smi_dict["smiles"])

        cursor.execute("""INSERT OR IGNORE INTO hmdbid_substructures (
                              hmdbid, 
                              substructure_id) 
                          VALUES ('{}', {})""".format(record_dict['HMDB_ID'], cursor.fetchall()[0][0]))


def calculate_hydrogen_rearrangements(fragment_ions, ion_mode):
    """
    Calculate MS-FINDER re-arrangement possibilities.

    :param fragment_ions: The element symbol of the assumed ionised atom.

    :param ion_mode: If True, assumes positive ion mode, else is configured for negative ion mode.

    :return: A set of integers referring to likely hydrogenation modifiers; i.e. if the set contains 1, this means that
        the substructure could have one less hydrogen than expected.
    """

    which_rule = {"+": {True:  {"C": ["P1"], "N": ["P2"], "O": ["P2"], "P": ["P1", "P2"], "S": ["P1", "P2"]},
                        False: {"C": ["P3", "P4"], "N": ["P3", "P4"], "O": ["P3", "P4"], "P": ["P3", "P4"], "S": ["P3", "P4"]}},
                  "-": {True:  {"C": ["N1", "N2"], "N": ["N1"], "O": ["N1"], "P": ["N2"], "S": ["N1", "N3"]},
                        False: {"C": ["N4", "N5"], "N": ["N4", "N5"], "O": ["N4", "N5"], "P": ["N4", "N5"], "S": ["N4", "N5"]}}
                  }

    fragment_ion_rules = []
    for i, fragment_ion in enumerate(fragment_ions):

        fragment_ion_rules.append(which_rule[ion_mode][i == 0][fragment_ion[0]])

        if fragment_ion[1]:  # double bond (i.e. two hydrogens displaced)
            fragment_ion_rules.append(which_rule[ion_mode][False][fragment_ion[0]])

    return get_hydrogenation_modifiers(fragment_ion_rules)


def get_hydrogenation_modifiers(rules_list):
    """
    Convert rule names to hydrogen modifiers.

    :param: A list of list of rules. E.g. `[["P1"], ["P3", P4"]]`.

    :return: A set of integers referring to likely hydrogenation modifiers; i.e. if the set contains 1, this means that
        the substructure could have one less hydrogen than expected. If the example above (`[["P1"], ["P3", P4"]]`)
        was given as input, the output would be a set containing
        `sum([rule_hydrogenations["P1"], rule_hydrogenations["P3"]])` and
        `sum([rule_hydrogenations["P1"], rule_hydrogenations["P4"]])`.
    """

    # the rules modified to account for the fact we are using the neutralised peak mass
    rule_hydrogenations = {"P1": -1, "P2": +1, "P3": +1, "P4": -1,
                           "N1": +1, "N2": -1, "N3": +0, "N4": +1, "N5": -1}

    possible_hydrogenations = set()
    for rule_set in itertools.product(*rules_list):

        # note that we don't consider dummy atoms ("*") to have mass, so we do not have to account for these
        # in the hydrogenation modifier (as MS-FINDER do)
        possible_hydrogenations.add(sum([rule_hydrogenations[rule] for rule in rule_set]))

    return possible_hydrogenations


def create_connectivity_database(path_connectivity_db: Union[str, bytes, os.PathLike], max_n_substructures: int = 3,
                                 max_atoms_available: int = 2, path_ri: Union[str, bytes, os.PathLike, None] = None
                                 ) -> None:
    """
    Generates a connectivity database containing sets of possible combinations of substructures; these combinations are
    represented by graphs whose vertices correspond to substructures and edges to bonds. We use geng, part of the nauty
    package, along with RI3.6 to ensure that the generated graphs are non-isomorphic - i.e. we only generate each
    combination of substructures once. These graphs are pickled in order to be stored in the final column of the
    SQLite 3 connectivity database.

    :param path_connectivity_db: The path at which to generate the SQLite 3 database.

    :param max_n_substructures: The maximal number of substructures (vertices). At least two substructures must be
        available for bonding for a graph to be created.

    :param max_atoms_available: The maximum number of  atoms available of each substructure to be considered for
        building molecules. `atoms_available` refers to the number of atoms on a substructure involved in forming
        chemical bonds (e.g. single or double bonds).

    :param path_ri: The path of RI, a required tool for verifying subgraph isomorphism.
    """
    
    conn = sqlite3.connect(path_connectivity_db)
    cursor = conn.cursor()

    cursor.execute("""DROP TABLE IF EXISTS subgraphs""")
    cursor.execute("""CREATE TABLE subgraphs (
                          id_pkl INTEGER,
                          n_graphs INTEGER,
                          graph6 TEXT,
                          k INTEGER,
                          k_partite TEXT,
                          k_valences TEXT,
                          nodes_valences TEXT,
                          n_nodes INTEGER,
                          n_edges INTEGER,
                          root BLOB,
                          PRIMARY KEY (graph6, k_partite, nodes_valences)
                   );""")
    conn.commit()

    id_pkl = 0

    for g, p in calculate_complete_multipartite_graphs(max_atoms_available, max_n_substructures):

        # get complete set of non-isomorphic graphs, using geng, from a distinct multipartite graph as input
        proc = subprocess.Popen(["geng", str(len(g.nodes)), "-d1", "-D2", "-q"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)  # max valence for single atom of 2
        geng_out, err = proc.communicate()

        proc.stdout.close()
        proc.stderr.close()

        # pipe geng output to RI to generate mappings (complete set of non-isomorphic configurations)
        for i, line_geng in enumerate(geng_out.split()):

            s_g = nx.read_graph6(io.BytesIO(line_geng))

            k_gfu = tempfile.NamedTemporaryFile(mode="w", delete=False)
            k_gfu.write(graph_to_ri(g, "k_graph"))
            k_gfu.seek(0)

            s_gfu = tempfile.NamedTemporaryFile(mode="w", delete=False)
            s_gfu.write(graph_to_ri(s_g, "subgraph"))
            s_gfu.seek(0)

            proc = subprocess.Popen([path_ri, "mono", "geu", k_gfu.name, s_gfu.name], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)  # TODO: add ri as dependency
            ri_out, err = proc.communicate()

            k_gfu.close()
            s_gfu.close()

            mappings = []
            subgraphs = {}

            for line in ri_out.decode("utf-8").splitlines():
                if line[0] == "{":
                    mappings.append(eval(line))

            if len(mappings) > 0:
                gi = graph_info(p, s_g, mappings, )  # convert mappings to valence/connectivity specifications

                for vn in gi:
                    if vn not in subgraphs:
                        subgraphs[vn] = gi[vn]

                    else:
                        for es in gi[vn]:
                            if es not in subgraphs[vn]:
                                subgraphs[vn].append(es)

            if len(subgraphs) > 0:
                for vn in subgraphs:  # for each valence configuration
                    subgraphs[vn] = sort_subgraphs(subgraphs[vn])  # sort to remove duplicate configurations
                    root = {}  # graph to be pickled

                    for fr in subgraphs[vn]:
                        parent = root
                        for e in fr:
                            parent = parent.setdefault(e, {})

                    vt = tuple([sum(v) for v in eval(vn)])

                    id_pkl += 1
                    cursor.execute("""INSERT INTO subgraphs (
                                          id_pkl, 
                                          n_graphs, 
                                          graph6,
                                          k,
                                          k_partite,
                                          k_valences,
                                          nodes_valences,
                                          n_nodes, n_edges,
                                          root)
                                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                                          id_pkl,
                                          len(subgraphs[vn]),
                                          line_geng,
                                          len(p),
                                          str(p),
                                          str(vt),
                                          str(vn),
                                          s_g.number_of_nodes(),
                                          s_g.number_of_edges(),
                                          pickle.dumps(root)))

            conn.commit()
    conn.close()
