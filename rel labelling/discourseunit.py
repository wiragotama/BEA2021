"""
by  Jan Wira Gotama Putra

This is the data structure to represent annotated essays 
"""
import os
import numpy as np
import csv
import re
import sys
import codecs
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from copy import deepcopy
from nltk.tokenize import word_tokenize

"""
Global constants
"""
NO_TARGET_CONSTANT = -1 #-1 means the discourse unit in question has no target (no outgoing link)
DELIMITER = "\t"
NO_REL_SYMBOL = '' # empty string 
RESTATEMENT_SYMBOL = '='
str_to_bool = lambda s: s.lower() in ["true", "t", "yes", "1"] # convert boolean string to boolean


class DiscourseUnit:
    """
    DiscourseUnit data structure
    """

    def __init__(self):
        global DELIMITER
        self.ID = "" # unit ID (original order)
        self.text = ""
        self.targetID = ""
        self.rel_name = ""
        self.dropping = ""

    def __display_targetID(self):
        return str(self.targetID) if self.targetID!=-1 else ""

    def __str__(self):
        return str(self.ID) + DELIMITER + self.text + DELIMITER + self.__display_targetID() + DELIMITER + str(self.rel_name) + DELIMITER + str(self.dropping)



class Essay:
    """
    Essay data structure
    
    Args:
        filepath (str): file that contains the essay annotation

    Attributes:
        essay_code (str): the essay code of the current object
        units (:obj:`list` of :obj:`DiscourseUnit`): list of discourse untits 
        scores (:obj:`list` of :obj:`int`): essay score(s)
        score_types (:obj:`list` of :obj:`str`): singleton variable of score types

    """

    score_types = ["Content (/12)", "Organization (/12)", "Vocabulary (/12)", "Language Use (/12)", "Mechanics (/12)", "Total 1 (%)", "Total 2 (Weighted %)"]

    def __init__(self, filepath):
        """
        :param string filepath: file that contains the essay annotation
        """
        global NO_TARGET_CONSTANT
        self.essay_code, self.units = self.__process_file(filepath) #  automatically detects html or tsv and then process them
        self.scores =  [] # essay scores, there are many as defined by score_types (singleton variable)


    def n_ACs(self):
        """
        Returns: 
            the number of argumentative components (non-dropped) in the essay
        """
        total = 0
        for unit in self.units:
            if unit.dropping == False:
                total += 1
        return total

    def n_non_ACs(self):
        """
        Returns:
            the number of non-argumentative components (non-dropped) in the essay
        """
        total = 0
        for unit in self.units:
            if unit.dropping == True:
                total += 1
        return total


    def n_rel(self, label):
        """
        Args:
            label (str): relation label

        Returns:
            the number of relations with corresponding label existing in the essay
        """
        total = 0
        for unit in self.units:
            if unit.rel_name == label:
                total += 1
        return total


    def n_tokens(self):
        """
        Returns:
            the number of total tokens in the essay
        """
        total = 0
        for unit in self.units:
            total += len(word_tokenize(unit.text))
        return total


    def n_tokens_per_sentence(self):
        """
        Returns:
            the number of total tokens in the essay per sentence
        """
        output = []
        for unit in self.units:
            output.append(len(word_tokenize(unit.text)))
        return output


    def get_pairwise_link_labelling_data(self, encode_sentences=False, encoder=None, normalised=True):
        """
        Get the list of relations for pairwise link labelling task
        Given a pair of (source, target), determine the relation label that links source -> target
        The assumption is that links between source and target has been determined before
        
        Args:
            encode_sentences (bool): whether to encode sentences in the essay 
            encoder (obj): any kind of encoder object, it needs to have "text_to_vec(str)" function
            normalised (bool): True if we want to use the original text without repair, False if use repaired text

        Returns:
            {
                list of list of {str or float--if embedding}, containing source and target sentences (texts)
                list of str, containing the relation labels that connect them
            }
        """
        source_target_sentences = []
        source_target_rels = []
        texts = self.get_texts(order="original", normalised=normalised) # must be in original order to be aligned with the adj_matrix

        if encode_sentences:
            sent_embs = encoder.text_to_vec(texts)

        adj_matrix = self.adj_matrix() # adj matrix uses the unit ID (so it corresponds to original order) for ease of debug
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] != NO_REL_SYMBOL:
                    if encode_sentences:
                        source_target_sentences.append([sent_embs[i].tolist(), sent_embs[j].tolist()])
                    else: # this is verbose, since we already get normalised texts in get_texts(), consider fixing this part
                        if normalised:
                            source_target_sentences.append([ Essay.discard_text_repair(texts[i]), 
                                                            Essay.discard_text_repair(texts[j]) ])
                        else:
                            source_target_sentences.append([ Essay.use_text_repair(texts[i]), 
                                                            Essay.use_text_repair(texts[j]) ])
                    source_target_rels.append(adj_matrix[i, j])

        return source_target_sentences, source_target_rels


    def get_rel_distances(self, mode="reordering", include_non_arg_units=False):
        """
        Get the list of relation distance (+ relation labels, this is useful for stats) for all sentences, according to mode
            if mode=="reordering", use the ordering as annotated, while sorted if "original"
            minus sign means pointing backward (to something appeared before), while plus means forward

        Args:
            mode (str): {reordering, original}
            include_non_arg_units (bool): True of False

        Returns:
            {
                distances from each sentence to its target,
                labels from each sentence to its target
            }
        """
        units_copy = deepcopy(self.units)
        assert mode in {"reordering", "original"}
        if mode == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        order = []
        for unit in units_copy:
            order.append(int(unit.ID))

        distances = []
        rels = []
        curr_pos = 0
        for unit in units_copy:
            if unit.dropping == False:
                if  unit.targetID != NO_TARGET_CONSTANT: # connected
                    target_pos = order.index(int(unit.targetID))
                    dist = target_pos - curr_pos # minus means pointing backward, plus means forward
                    rels.append(unit.rel_name)
                else: # not connected
                    dist = 0 # major claim points to itself
                    rels.append("major claim")
                distances.append(dist)
            else: # dropped = non-AC 
                if include_non_arg_units:
                    dist = 0
                    rels.append("non-AC")
                    distances.append(dist)
            curr_pos += 1

        return distances, rels


    def get_texts(self, order="reordering", normalised=False):
        """
        get the list of sentences in the essay

        Args:
            order (str, optional, defaults to 'reordering'): {"reordering", "original"}
            normalised(bool, optional, defaults to False): True if we use the original version of the text without repair, False if we use the text repair
        
        Returns:
            list of sentences in the essay
        """
        units_copy = deepcopy(self.units)
        if order == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        output = []
        for x in units_copy:
            if normalised:
                output.append(Essay.discard_text_repair(x.text))
            else:
                output.append(Essay.use_text_repair(x.text))
        return output


    def min_edit_distance(self):
        """
        Calculate the min_edit_distance between original ordering and annotated ordering of sentences

        Returns:
            {
                minimum edit distance score,
                sentence original order (sentence IDs)
                sentence reordered (sentence IDs)
            }
        """
        units_copy = deepcopy(self.units)
        units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        order_annotated = []
        for unit in self.units:
            order_annotated.append(int(unit.ID))

        order_ori = []
        for unit in units_copy:
            order_ori.append(int(unit.ID))

        # Create a table to store results of subproblems
        n = len(order_annotated)
        dp = [[0 for x in range(n+1)] for x in range(n+1)] 
      
        # Fill d[][] in bottom up manner 
        for i in range(n+1): 
            for j in range(n+1): 
                if i == 0: 
                    dp[i][j] = j    # Min. operations = j 
                elif j == 0: 
                    dp[i][j] = i    # Min. operations = i 
                elif order_ori[i-1] == order_annotated[j-1]: 
                    dp[i][j] = dp[i-1][j-1] 
                else: # if different
                    dp[i][j] = 1 + min(dp[i][j-1],  # Insert 
                                       dp[i-1][j],  # Remove 
                                       dp[i-1][j-1]) # Replace 

        return dp[n][n], order_ori, order_annotated


    def get_score(self, score_name):
        """
        Args:
            score_name (str): 

        Returns:
            the corresponding score for score_name
        """
        try:
            return self.scores[self.score_types.index(score_name)]
        except:
            return np.nan


    @staticmethod
    def open_html(path):
        """
        Read tsv from external file

        Args:
            path (str): path to file

        Returns:
            BeautifulSoup
        """
        f = codecs.open(path, 'r', 'utf-8')
        soup= BeautifulSoup(f.read(), 'html.parser')
        return soup


    def __process_html(self, soup):
        """
        Process html file (annotated with TIARA annotation tool) to internal data structure

        Args:
            soup (BeautifulSoup): html document object
        """
        try:
            essay_code = soup.find("h4", id="essay_code").get_text().strip() # current format
        except:
            essay_code = soup.find("h4", id="essay_code_ICNALE").get_text().strip() # old format
        units = []

        unit_annotation = soup.find_all("div", class_="flex-item") # sentences, clause, or clause-like segments (discourse unit)
        for x in unit_annotation:
            unit = DiscourseUnit() # initialization            
            unit.ID = int(x.find('span', class_="sentence-id-number").get_text().strip())
            unit.text = x.find("textarea").get_text().strip()

            target = x.find("span", id="target"+str(unit.ID)).get_text().strip()
            if target != "":
                unit.targetID = int(re.sub("[^0-9]", "", target))
            else:
                unit.targetID = NO_TARGET_CONSTANT 

            unit.rel_name = x.find("span", id="relation"+str(unit.ID)).get_text().strip()

            dropping_flag = x.find("input", id="dropping"+str(unit.ID))["value"]
            if dropping_flag == "non-drop":
                unit.dropping = False
            else:
                unit.dropping = True

            units.append(unit)

        return essay_code, units


    def __process_tsv(self, filepath):
        """
        Process tsv file into internal data structure

        Args:
            filepath (str): tsv filepath
        """
        # use filename as essay_code
        filename, file_extension = os.path.splitext(filepath)
        essay_code = filename.split("/")[-1]

        # open file
        with open(filepath, 'r') as f:
            data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t')]
        del data[0] # delete header
        
        # process
        n_sentences = len(data)
        units = [] # sentences
        for i in range(n_sentences):
            row = data[i]
            unit = DiscourseUnit()
            unit.ID = int(row[1])
            unit.text = row[2]
            unit.targetID = int(row[3]) if row[3]!='' else NO_TARGET_CONSTANT 
            unit.rel_name = row[4]
            unit.dropping = str_to_bool(row[5])
        
            units.append(unit) 
        return essay_code, units


    def __process_file(self, filepath):
        """
        Load a file into internal data structure

        Args:
            filepath (str): 
        """
        filename, file_extension = os.path.splitext(filepath)
        if file_extension == ".html":
            return self.__process_html(Essay.open_html(filepath))
        elif file_extension == ".tsv":
            return self.__process_tsv(filepath)
        else:
            raise Exception('unsupported file', filepath)


    def to_tsv(self):
        """
        Convert the essay to tsv

        Returns:
            str
        """
        header = "essay code" + DELIMITER + "unit id" + DELIMITER + "text" + DELIMITER + "target" + DELIMITER + "relation" + DELIMITER + "drop_flag" + "\n"
        tsv = header
        for i in range(len(self.units)):
            tsv = tsv + self.essay_code + DELIMITER + str(self.units[i]) + "\n"
        return tsv


    def to_tsv_sorted(self):
        """
        Convert the essay to tsv, sorted according to original unit ID

        Returns:
            str
        """
        header = "essay code" + DELIMITER + "unit id" + DELIMITER + "text" + DELIMITER + "target" + DELIMITER + "relation" + DELIMITER + "drop_flag" + "\n"
        tsv = header
        units_copy = deepcopy(self.units)
        units_copy.sort(key=lambda x: x.ID)
        for i in range(len(units_copy)):
            tsv = tsv + self.essay_code + DELIMITER + str(units_copy[i]) + "\n"
        return tsv


    def get_dropping_sorted(self):
        """
        Get dropping info sorted according to unitID

        Returns:
            list of bool
        """
        drop_flags = []        
        units_copy = deepcopy(self.units)
        units_copy.sort(key=lambda x: x.ID)
        for i in range(len(units_copy)):
            drop_flags.append(units_copy[i].dropping)
        return drop_flags


    def adj_matrix(self):
        """
        Convert the relations in essay into adj matrix

        Returns:
            numpy.ndarray
        """
        n = len(self.units)
        adj_matrix = np.zeros((n, n), dtype="<U5")
        for i in range(n):
            source = self.units[i].ID - 1
            target = self.units[i].targetID - 1
            if target != NO_TARGET_CONSTANT - 1:
                adj_matrix[source][target] = self.units[i].rel_name

        return adj_matrix


    @staticmethod
    def discard_text_repair(sentence):
        """
        Use the original version of the repaired part

        Args:
            sentence (str)

        Return:
            sentence normalised without text repair
        """
        while sentence.find("[") != -1:
            left = sentence.find("[")
            mid = sentence.find("|")
            right = sentence.find("]")
            old = sentence[left+1:mid].strip()
            want_to_replace = sentence[left:right+1]
            sentence = sentence.replace(want_to_replace, " "+old+" ") # give space before and after to avoid tokenization problem
        # remove multiple spaces
        sentence = re.sub(' +', ' ', sentence)
        return sentence.strip()


    @staticmethod
    def use_text_repair(sentence):
        """
        Use the repaired version of the repaired part

        Args:
            sentence (str)

        Return:
            sentence normalised without text repair
        """
        while sentence.find("[") != -1:
            left = sentence.find("[")
            mid = sentence.find("|")
            right = sentence.find("]")
            use = sentence[mid+1:right].strip()
            want_to_replace = sentence[left:right+1]
            sentence = sentence.replace(want_to_replace, " "+use+" ") # give space before and after to avoid tokenization problem
        # remove multiple spaces
        sentence = re.sub(' +', ' ', sentence)
        return sentence.strip()


    @staticmethod
    def rel_combination(adj_matrix):
        """
        Convert adj_matrix or adj_matrix_with inference to cohen's kappa format

        Args:
            adj_matrix (numpy.ndarray)
        
        Returns:
            list of string
        """
        retval = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if i != j: # a sentence cannot point to itself
                    retval.append(adj_matrix[i][j])

        return retval


    @staticmethod
    def list_paths(adj_matrix):
        """
        List all paths in the graph, from leaf to root node

        Args:
            adj_matrix (numpy.ndarray)
        
        Returns:
            list[list(int, str)], node and its relation to its parent
        """
        n_nodes = len(adj_matrix)
        pointed = [False] * n_nodes

        # pointed flag
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i][j] != NO_REL_SYMBOL:
                    pointed[j] = True

        # list paths from leaf to root
        all_paths = []
        for i in range(n_nodes):
            if not pointed[i]:
                visited = [False] * n_nodes
                Essay.parent_traversal(i, visited, n_nodes, adj_matrix, [], all_paths)

        return all_paths


    @staticmethod
    def depth(adj_matrix):
        """
        Return the depth of the structure (adj_matrix)
        
        Args:
            adj_matrix (numpy.ndarray)
        """
        paths = Essay.list_paths(adj_matrix)
        max_depth = -1
        for path in paths:
            if len(path) > max_depth:
                max_depth = len(path)
        return max_depth-1 # because root is 0


    @staticmethod
    def list_subpaths(adj_matrix):
        """
        List all paths in the graph (including subpath), from leaf to root node
        Discard the subpaths that contain only one element

        Args:
            adj_matrix (numpy.ndarray)

        Returns:
            list[list[int]], nodes involved in subpaths
        """
        paths = Essay.list_paths(adj_matrix)
        retval = []
        for i in range(len(paths)):
            retval.append(paths[i])
            n = len(paths[i])
            for j in range(1, n):
                start = 0
                end = start + j
                while end <= n:
                    retval.append(paths[i][start:end])
                    start += 1
                    end = start + j

        # only take the node ID
        retval_clean = []
        for path in retval:
            path_clean = []
            for n, rel_name in path:
                path_clean.append(n)
            path_clean = tuple(path_clean)
            retval_clean.append(path_clean)

        # take only unique paths
        retval_clean = set(retval_clean)

        # delete subpaths that consist of only one element
        retval_filter = set()
        for e in retval_clean:
            if len(e) > 1:
                retval_filter.add(e)

        return retval_filter


    @staticmethod
    def parent_traversal(current_node, visited, n_nodes, adj_matrix, curr_path, all_paths):
        """
        DFS traversal from a node until reaching root, we discard dropped nodes (no incoming and outgoing connection)

        Args:
            current_node (int)
            visited (:obj:`list` of :obj:`bool`)
            n_nodes (int)
            adj_matrix (numpy.ndarray)
            curr_path (:obj:`list` of :obj:`(int, str)`)
            all_paths (:obj:`list` of :obj:`list` of :obj:`(int, str)`): variable to save all found paths, this is MUTABLE
        """
        vis_copy = deepcopy(visited)
        if not vis_copy[current_node]:
            vis_copy[current_node] = True
            outgoing = adj_matrix[current_node]
            flag = False
            for i in range(n_nodes):
                if outgoing[i] != NO_REL_SYMBOL:
                    flag = True
                    path = deepcopy(curr_path)
                    path.append((current_node, outgoing[i]))
                    Essay.parent_traversal(i, vis_copy, n_nodes, adj_matrix, path, all_paths)
            if not flag:
                curr_path.append((current_node, ""))
                if len(curr_path) > 1: # not a dropped node
                    all_paths.append(curr_path)
        else: # cycle detected, this is useful for graph after restatement inference
            if len(curr_path) > 1: # not a dropped node
                all_paths.append(curr_path)


    @staticmethod
    def list_substructures(adj_matrix):
        """
        Generate substructure information (set of nodes: own + descendant) for each node in the matrix

        Args:
            adj_matrix (numpy.ndarray)

        Returns:
            list[list[int]]
        """
        n_nodes = len(adj_matrix)
        pointed = [False] * n_nodes

        # pointed flag
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i][j] != NO_REL_SYMBOL:
                    pointed[j] = True

        # list paths from leaf to root
        subtrees = [set()] * n_nodes
        for i in range(n_nodes):
            if not pointed[i]:
                visited = [False] * n_nodes
                Essay.substructure_parsing(i, visited, n_nodes, adj_matrix, set(), subtrees)
        return subtrees


    @staticmethod
    def substructure_parsing(current_node, visited, n_nodes, adj_matrix, curr_subtree, subtrees):
        """
        DFS traversal from a node until reaching root, we USE dropped nodes (no incoming and outgoing connection)
        While traversing, pass the information of own descendants to parent

        Args:
            current_node (int)
            visited (:obj:`list` of :obj:`bool`)
            n_nodes (int)
            adj_matrix (numpy.ndarray)
            curr_subtree (set)
            all_paths (:obj:`list` of :obj:`list` of :obj:`int`): variable to save all found paths, this is MUTABLE
        """
        vis_copy = deepcopy(visited)
        if not vis_copy[current_node]:
            vis_copy[current_node] = True
            outgoing = adj_matrix[current_node]
            flag = False
            for i in range(n_nodes):
                if outgoing[i] != NO_REL_SYMBOL:
                    flag = True
                    local_subtree = deepcopy(curr_subtree)
                    local_subtree.add(current_node)
                    subtrees[current_node] = subtrees[current_node].union(local_subtree)
                    Essay.substructure_parsing(i, vis_copy, n_nodes, adj_matrix, subtrees[current_node], subtrees)
            if not flag:
                local_subtree = deepcopy(curr_subtree)
                local_subtree.add(current_node)
                if len(local_subtree) > 1: # not a dropped node
                    subtrees[current_node] = subtrees[current_node].union(local_subtree)
        else: # cycle detected, this is useful for graph after restatement inference
            if len(curr_subtree) > 1: # not a dropped node
                subtrees[current_node] = subtrees[current_node].union(curr_subtree)


    @staticmethod
    def list_siblinghood(adj_matrix):
        """
        List nodes appearing per each level
        Note that restatement is considered the same level as its target

        Args:
            adj_matrix (numpy.ndarray)
        
        Returns:
            list[set[int]]
        """
        paths = Essay.list_paths(adj_matrix)
        siblinghood = []
        node_at_level = []

        # initialisation
        for i in range(len(adj_matrix)):
            siblinghood.append(set())
            node_at_level.append(set())
        
        # get siblinghood info
        for path in paths:
            curr_level = 0
            for j in reversed(range(len(path))):
                if j == len(path)-1:
                    node_at_level[curr_level].add(path[j][0])
                else:
                    if path[j][1] != RESTATEMENT_SYMBOL:
                        curr_level += 1
                    node_at_level[curr_level].add(path[j][0])

        # list of siblings for each node
        for nodes in node_at_level:
            for n in nodes:
                siblinghood[n] = nodes - {n} # other members of the set

        return siblinghood


    @staticmethod
    def all_pairs_shortest_path(adj_matrix):
        """
        Calculate all pairs shortest path of the input matrix
        
        Args:
            adj_matrix (numpy.ndarray)
        
        Returns:
            numpy.ndarray
        """
        dist_matrix = deepcopy(adj_matrix)
        n_nodes = len(dist_matrix)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dist_matrix[i][j] == NO_REL_SYMBOL:
                    dist_matrix[i][j] = float('inf')
                else:
                    dist_matrix[i][j] = 1
        dist_matrix = dist_matrix.astype(float)

        # FloydWarshall
        for k in range(n_nodes): 
            # pick all vertices as source one by one 
            for i in range(n_nodes): 
                # pick all vertices as destination for the 
                # above picked source 
                for j in range(n_nodes): 
                    # if vertex k is on the shortest path from  
                    # i to j, then update the value of dist[i][j] 
                    dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k]+dist_matrix[k][j])
        return dist_matrix


    @staticmethod
    def restatement_inference(adj_matrix):
        """
        Add additional relations to the adj_matrix as the result of restatement inference

        Args:
            adj_matrix (numpy.ndarray)

        Returns:
            numpy.ndarray
        """
        return Restatement.inference(adj_matrix)


    @staticmethod
    def double_att_inference(adj_matrix):
        """
        Add additional relations to the adj_matrix as the result of double_att inference
        
        Args:
            adj_matrix (numpy.ndarray)

        Returns:
            numpy.ndarray
        """
        return DoubleAtt.inference(adj_matrix)


class Restatement: 
    """
    For handling restatement inference, since it is complicated
    A piece of wisdom: it is better to to touch this class AT ALL.
    """

    @staticmethod
    def inference(adj_matrix):
        """
        Add additional relations to the adj_matrix as the result of restatement inference

        Args:
            adj_matrix (numpy.ndarray)

        Returns:
            numpy.ndarray
        """
        inp_matrix = deepcopy(adj_matrix)
        n_nodes = len(inp_matrix)
        pointed = [False] * n_nodes

        # pointed flag
        for i in range(n_nodes):
            for j in range(n_nodes):
                if inp_matrix[i][j] != NO_REL_SYMBOL:
                    pointed[j] = True

        # restatement resolution, creating restatement chains
        restatement_chains = []
        for i in range(n_nodes):
            if not pointed[i]: # leaf
                visited = [False] * n_nodes
                Restatement.traverse(i, visited, inp_matrix, n_nodes, restatement_chains, True, False)

        # normalisation
        restatement_chains = Restatement.normalise_restatement_chains(restatement_chains)

        # copy incoming and outgoing relations in restatement chains, until convergence
        first_pass = True
        condition = True
        while first_pass or condition:
            first_pass = False

            copy = deepcopy(inp_matrix)
            for i in range(len(restatement_chains)):
                outgoings = Restatement.merge_outgoing(restatement_chains[i], n_nodes, inp_matrix)
                incomings = Restatement.merge_incoming(restatement_chains[i], n_nodes, inp_matrix)
                Restatement.override_relation(restatement_chains[i], n_nodes, outgoings, incomings, inp_matrix)

            condition = not (copy==inp_matrix).all()

        return inp_matrix


    @staticmethod
    def traverse(start, visited, adj_matrix, n_nodes, restatement_chains, chain_begin, prev_chain):
        """
        Finding chain of restatements in graph

        Args:
            start (int): current node
            visited (:obj:`list` of :obj:`bool`)
            adj_matrix (numpy.ndarray)
            n_nodes (int)
            restatement_chains (:obj:`list` of :obj:`list` of :obj:`int`): list of chain of restatement (mutable)
            chain_begin (bool): whether the current node is potentially the beginning node of a chain
            prev_chain (bool): whether the relation of previously visited node was a restatement
        """
        if not visited[start]:
            for j in range(n_nodes):
                if adj_matrix[start][j] != NO_REL_SYMBOL:
                    visited[start] = True # set flag

                    if adj_matrix[start][j] == "=": # restatement detected
                        if chain_begin: 
                            restatement_chains.append([start])
                        else:
                            restatement_chains[len(restatement_chains)-1].append(start)
                        chain_begin = False
                        prev_chain = True
                        Restatement.traverse(j, visited, adj_matrix, n_nodes, restatement_chains, chain_begin, prev_chain)
                        prev_chain = False
                    else:
                        if prev_chain: # not a root, not a restatement, but still part of previous chain
                            restatement_chains[len(restatement_chains)-1].append(start)
                            prev_chain = False
                        chain_begin = True
                        Restatement.traverse(j, visited, adj_matrix, n_nodes, restatement_chains, chain_begin, prev_chain)

            if prev_chain: # root case
                restatement_chains[len(restatement_chains)-1].append(start)
                prev_chain = False


    @staticmethod
    def normalise_restatement_chains(restatement_chains):
        """
        Normalising restatement chains. The chain is computed from the leaf to the root (directed). However, the resulting chains might not be correct
        e.g., 1 -> 4 <- 7 becomes [1,4] and [7,4]. We need to normalise then using transitive closure of restatements. In this case, they become a single chain [1,4,7]
        
        Args:
            restatement_chains (:obj:`list` of :obj:`list` of :obj:`int`): list of chain of restatement

        Returns:
            list[list[int]]
        """
        first_pass = True
        condition = True
        while first_pass or condition:
            first_pass = False

            copy = deepcopy(restatement_chains)
            flag = True
            for i in range(len(restatement_chains)):
                for j in range(len(restatement_chains[i])):
                    for k in range(len(copy)):
                        if i!=k:
                            if restatement_chains[i][j] in copy[k]:
                                restatement_chains[i].extend(copy[k])
                                restatement_chains[i] = list(set(restatement_chains[i]))
                                restatement_chains.pop(k)
                                flag = False
                                break
                        if not flag:
                            break
                if not flag:
                    break

            condition = not (copy==restatement_chains)

        return restatement_chains


    @staticmethod
    def merge_outgoing(chain, n_nodes, adj_matrix):
        """
        Merge outgoing relations between chain of restatements

        Args:
            chain (:obj:`list` of :obj:`int`): a chain of restatement nodes
            n_nodes (int)
            adj_matrix (numpy.ndarray)
        
        Returns:
            list[str], denoting the merge of outgoing relations among nodes in the chain 
        """
        merger = [''] * n_nodes
        for i in range(1, len(chain)):
            for j in range(n_nodes):
                if i==1:
                    merger[j] = Restatement.merge_rel(adj_matrix[chain[i]][j], adj_matrix[chain[i-1]][j])
                else:
                    merger[j] = Restatement.merge_rel(adj_matrix[chain[i]][j], merger[j])
        return merger


    @staticmethod
    def merge_incoming(chain, n_nodes, adj_matrix):
        """
        Merge incoming relations between chain of restatements
        
        Args:
            chain (:obj:`list` of :obj:`int`): a chain of restatement nodes
            n_nodes (int)
            adj_matrix (numpy.ndarray)

        Returns:
            list[str], denoting the merge of outgoing relations among nodes in the chain 
        """
        merger = [''] * n_nodes
        for i in range(n_nodes):
            for j in range(1, len(chain)):
                if j==1:
                    merger[i] = Restatement.merge_rel(adj_matrix[i][chain[j]], adj_matrix[i][chain[j-1]])
                else:
                    merger[i] = Restatement.merge_rel(adj_matrix[i][chain[j]], merger[i])
        return merger


    @staticmethod
    def merge_rel(rel1, rel2):
        """
        Combine two relations

        Args:
            rel1 (str)
            rel2 (str)
        
        Returns:
            str
        """
        # if rel1 != NO_REL_SYMBOL and rel2!= NO_REL_SYMBOL:
        #     assert(rel1 == rel2)
        if rel1 != NO_REL_SYMBOL:
            return rel1
        elif rel2 != NO_REL_SYMBOL:
            return rel2
        else:
            return rel1


    @staticmethod
    def override_relation(chain, n_nodes, outgoing_rels, incoming_rels, adj_matrix):
        """
        Override the relation, now we consider restatement nodes as equal in terms of incoming and outgoing connections

        Args:
            chain (:obj:`list` of :obj:`int`): a chain of restatement nodes
            n_nodes (int)
            outgoing_rels (:obj:`list` of :obj:`int`): merger result of outgoing relations among all participating nodes
            incoming_rels (:obj:`list` of :obj:`int`): merger result of incoming relations among all participating nodes
            adj_matrix (numpy.ndarray): MUTABLE
        """
        # outgoing
        for i in range(len(chain)):
            for j in range(n_nodes):
                adj_matrix[chain[i]][j] = Restatement.merge_rel(outgoing_rels[j], adj_matrix[chain[i]][j]);

        # incoming
        for i in range(n_nodes):
            for j in range(len(chain)):
                adj_matrix[i][chain[j]] = Restatement.merge_rel(incoming_rels[i], adj_matrix[i][chain[j]]); 



class DoubleAtt:
    """
    Responsible for double_att resolution
    """

    @staticmethod
    def inference(adj_matrix):
        """
        Double attack inference
        First - list all paths from leaf to root (we discard dropped sentences)
        Second - Perform double attack inference by windowing

        Args:
            adj_matrix (numpy.ndarray)

        Returns:
            list[list[(int, str)]], from leaf to node (node, edge label)
        """ 
        all_paths = Essay.list_paths(adj_matrix)
        inp_matrix = deepcopy(adj_matrix)
        
        # add additional relations for double_att inference, i.e., (X - att -> Y - att -> Z) means (X - sup -> Z) implicitly
        # we make the implicit become explicit
        for i in range(len(all_paths)):
            if len(all_paths[i]) >= 3:
                # windowing
                for j in range(2, len(all_paths[i])):
                    if all_paths[i][j-1][1] == 'att' and all_paths[i][j-2][1] == 'att':
                        source = all_paths[i][j-2][0]
                        target = all_paths[i][j][0]
                        inp_matrix[source][target] = 'sup'

        return inp_matrix

