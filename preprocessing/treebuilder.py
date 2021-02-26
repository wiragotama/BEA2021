"""
This is the script converts the output of sequence tagging experimental setting as adjacency matrix, and then perform operations on it
output = d_1, ..., d_N  that denotes where each sentence should be connected to, e.g., [-2, -1, 0, ...]

This is useful for output quality estimation

"""
import numpy as np
from collections import deque
import sys
from copy import deepcopy


class TreeBuilder:
	def __init__(self, dist, mode='strict'):
		"""
		Converts dist as adjacency matrix
		:param numpy.ndarray dist: denoting the distance from a sentence to its target sentence
		:param str mode: 'strict' or 'loose'; strict if we do not allow out_of_bound distances, 'loose' if we ignore them
		"""
		self.adj_matrix = np.zeros(shape=(len(dist), len(dist)), dtype=int)
		for i in range(len(dist)):
			if 0 <= i + dist[i] <= len(dist)-1: # check out of bound distance(s)
				self.adj_matrix[i][i+dist[i]] = 1
			else:
				if mode == 'strict':
					print("[ERROR]", dist)
					raise Exception("Cannot build an adjacency matrix representation: there is/are out of bound distance(s)")
				else: # loose
					pass

	def adj_matrix(self):
		"""
		:return: numpy.ndarray: adjacency matrix representation of the structure
		"""
		return self.adj_matrix


	def n_nodes(self):
		"""
		:return: int (the number of nodes in the graph)
		"""
		return len(self.adj_matrix)


	def flatten(self):
		"""
		Converts internal adj_matrix representation as a list 
		:return: numpy.ndarray
		"""
		return self.adj_matrix.flatten('C')


	@staticmethod
	def cycle_exist(adj_matrix, start):
		"""
		Check if cycle exist in adj_matrix using BFS
		:param numpy.ndarray adj_matrix
		:param int start: source node
		:return: boolean
		"""
		visited = [False] * len(adj_matrix)
		queue = deque([start])
		cycle = False

		while queue and not cycle:
			curr_node = queue.popleft()
			if visited[curr_node] == False:
				visited[curr_node] = True
				in_links = adj_matrix[:, curr_node]
				for j in range(len(in_links)):
					if j!=curr_node and in_links[j] == 1: # relation exist from j to curr_node
						queue.append(j)
			else:
				cycle = True
				break

		return cycle, visited


	def auto_component_labels(self, AC_breakdown=False, mask_MC=False):
		"""
		Guess the argumentative-component labels from the structure
		- non-arg components: no outgoing and incoming links
		- arg-components : have outgoing links, but it is okay if they do not have incoming links (can be further classified as major claim, claim and premises)
			- major claims: no outgoing links but have incoming links
	
		:param bool AC_breakdown: breakdown AC into AC (leaf) and AC (non-leaf)
		:param bool mask_MC: mask "major claim" as "arg component"
		:return: list[str]
		"""
		n = len(self.adj_matrix)
		labels = []
		for i in range(n):
			in_links = self.adj_matrix[:, i] # slice column
			n_incoming = np.count_nonzero(np.delete(in_links, i)) # exclude self-loop
			out_links = self.adj_matrix[i] # slice row
			n_outgoing = np.count_nonzero(np.delete(out_links, i)) # exclude self-loop

			if n_incoming > 0 and n_outgoing == 0:
				if mask_MC:
					labels.append("arg comp.")
				else:
					labels.append("major claim")
			elif n_incoming == 0 and n_outgoing == 0:
				labels.append("non-arg comp.")
			elif n_incoming == 0 and n_outgoing > 0:
				if AC_breakdown:
					labels.append("arg comp. (leaf)")
				else:
					labels.append("arg comp.")
			else:
				if AC_breakdown:
					labels.append("arg comp. (non-leaf)")
				else:
					labels.append("arg comp.")
		return labels


	def is_tree(self):
		"""
		Check if current adj_matrix is a tree

			1. Check potential roots (only incoming, no outgoing to other nodes)
			2. Check if only one of the potential root is pointed (and others are not referenced)
				- If yes, it is the root
				- If no, then the structure forms a forest
			3. Check if cycle exist (traverse from the root)

		:return: boolean
		"""
		n = len(self.adj_matrix)
		cycle = False
		component_labels = self.auto_component_labels() # automatic component labelling based on the structure

		n_cand = np.count_nonzero(np.asarray(component_labels) == "major claim")
		# print(n_cand)
		if n_cand == 1: # there is only one root candidate
			root = component_labels.index("major claim")
			verdict, visited = TreeBuilder.cycle_exist(self.adj_matrix, root)

			if not verdict: # no cycle
				# if non visited nodes are all non-arg component, then the structure forms a tree. Otherwise, it is a forest
				flag = True
				for i in range(n):
					if not visited[i] and component_labels[i] != "non-arg comp.":
						flag = False
						break
				return flag
			else:
				return False
		else:
			return False


	def tree_depth_and_leaf_proportion(self):
		"""
		Check the depth of tree, as well as the proportion of leaf nodes using BFS
		:return: int (max depth), float (proportion of leaf nodes)
		"""
		if self.is_tree():
			# output
			max_depth = -1
			n_leaves = 0

			# initialization
			visited = [False] * self.n_nodes()
			root = self.auto_component_labels().index("major claim")
			depth = 0
			queue = deque([ (root, depth) ])

			# BFS
			while queue:
				curr_node, curr_depth = queue.popleft()
				# depth
				if curr_depth > max_depth:
					max_depth = curr_depth
				# expand
				if visited[curr_node] == False:
					visited[curr_node] = True
					in_links = self.adj_matrix[:, curr_node]
					# leaf or not leaf
					if np.count_nonzero(in_links) == 0:
						n_leaves += 1
					# explore children
					for j in range(len(in_links)):
						if j!=curr_node and in_links[j] == 1:
							queue.append( (j, curr_depth+1) )

			return max_depth, float(n_leaves) / (self.n_nodes())
		else:
			return None, None


	def BFS_assign_depth(self, visited, depth, target, queue):
		"""
		Perform BFS while assigning the depth of the node (minimum depth, should cycle happens)
		:param list visited: flag, whether node has been visited or not
		:param list depth: node depth information
		:param list target: distance between nodes to their targets
		:param deque queue: BFS queue
		"""
		while queue:
			curr_node, target_dist, curr_depth = queue.popleft()
			if visited[curr_node] == False and depth[curr_node] > curr_depth:
				visited[curr_node] = True
				depth[curr_node] = curr_depth
				target[curr_node] = target_dist

				in_links = self.adj_matrix[:, curr_node]
				for j in range(len(in_links)):
					if j!=curr_node and in_links[j] == 1: # relation exist from j to curr_node
						queue.append( (j, curr_node-j, curr_depth+1) )


	def node_depths(self):
		"""
		Get the depth of each node in the structure, cycle is possible here (we assign the minimum possible depth)
		:return: list[int]
		"""
		components = self.auto_component_labels()
		visited = [False] * len(components)
		depth = [sys.maxsize] * len(components)
		target = [sys.maxsize] * len(components)

		for i in range(len(components)): # starts from major claim
			if components[i] == 'major claim':
				queue = deque([ (i, 0, 0) ])
				self.BFS_assign_depth(visited, depth, target, queue)

		for i in range(len(components)): # non major claim root, cycle possibly happens here
			if not visited[i]:
				queue = deque([ (i, 0, 0) ])
				self.BFS_assign_depth(deepcopy(visited), depth, target, queue)

		return depth



if __name__ == "__main__":
	# dist = [0, 3, 2, 1, 8, 1, -2, -3, -4, 3, -1, -1, -12, -13, 0]
	# dist = [0, -1, -1, -3, 0]
	dist = [1, 1, 1, 1, 4, 1, -2, -1, 0, -5]

	rep = TreeBuilder(dist)
	print("auto component labels", rep.auto_component_labels(AC_breakdown=True))
	# print("tree:", rep.is_tree())
	# print(rep.tree_depth_and_leaf_proportion())
	# print(rep.node_depths())



