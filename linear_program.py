# Load the required modules
import warnings
import itertools 
import numpy as np
import picos as pic
import networkx as nx
import matplotlib.pyplot as plt

'''
	This function declares a set-encoded description of the FR product for a system of 
	2 parties, with 2 measurements per party, and 2 outcomes for each measurement.
'''
def set_fr_product_B222():
	return [
		[  1,  2,  3,  4],
		[  5,  6,  7,  8],
		[  1,  2,  7,  8],
		[  3,  4,  5,  6],
		[  2,  4,  9, 11],
		[  1,  3, 10, 12],
		[  5,  7, 14, 16],
		[  6,  8, 13, 15],
		[  9, 10, 11, 12],
		[  9, 10, 15, 16],
		[ 11, 12, 13, 14],
		[ 13, 14, 15, 16]]

'''
	This function generates the Non-Orthogonality graph of the FR product for a system of 
	two parties, with 2 measurements per party, and 2 outcomes for each measurement.
'''
def nonorthogonality_graph_B222(fr_product):
	# Generate the adjacency graph of the FR product
	adjacency_graph = [list(set(list(range(1,17))).difference(set([item for sublist in list(
		filter(None, [x if (i+1 in x) else None for x in fr_product])) for item in sublist]))) for i in range(16)]
	# Generate the graph
	G = nx.Graph()
	[G.add_node(i+1) for i in range(16)]
	# Generate the networkX form of the NO graph
	[[G.add_edge(i+1, adjacency_graph[i][j]) 
	  for j in range(len(adjacency_graph[i]))] 
		 for i in range(16)]
	v = 0.25 # vertex offset
	o = 0.5  # formal offset
	# Determine the positions of the graph
	pos = { 1: [   0, v+o], 
			2: [   v, v+o],
			3: [   0, 0+o],
			4: [   v, 0+o],
			5: [ 0+o, v+o],
			6: [ v+o, v+o],
			7: [ 0+o, 0+o],
			8: [ v+o, 0+o],
			9: [   0,   v], 
		   10: [   v,   v],
		   11: [   0,   0],
		   12: [   v,   0],
		   13: [   0+o, v],
		   14: [   v+o, v],
		   15: [   0+o, 0],
		   16: [   v+o, 0]}
	# Draw the orthogonality graph
	fig = plt.figure(1,figsize=(8,8))
	nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='white')
	nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=0.5)
	nx.draw_networkx_labels(G, pos, font_size=18, font_family='sans-serif')
	plt.axis('off')
	plt.show()
	return G

'''
	This function produces a linear program that corresponds to a probabilistic model formed
	by clique weightings of the weighted fractional packing number.
'''
def linear_program(k_v,corr_A1B1_v, cliques):
	# Ignoring warnings
	warnings.filterwarnings("ignore")
	# Produce a mapping between the intersections of the probabilities and the cliques
	p_c = {i+1:[] for i in range(16)}
	for i in range(16):
		[p_c[i+1].append(c) for c in range(len(cliques)) if ((i+1) in cliques[c])]
	# Declare the linear program
	QLP  = pic.Problem()
	# Declare the probabilities
	p = [ QLP.add_variable('p[%s]' % i,1) for i in range(16) ]
	# Declare the clique weightings
	c = [ QLP.add_variable('c[%s]' % i,1) for i in range(len(cliques)) ]
	# Constrain all probabilities to non-negative values
	[QLP.add_constraint( 1| p[i] >= 0) for i in range(16)]
	# Constrain all probabilities to the FR product
	[QLP.add_constraint( 1| p[s[0]-1]+p[s[1]-1]+p[s[2]-1]+p[s[3]-1] == 1.00) for s in set_fr_product_B222()]
	c_total = 0
	for i in range(len(cliques)):
		# Constrain the cliques to non-negative values
		QLP.add_constraint( 1|c[i] >= 0)
		c_total += c[i]
	# The weighted fractional packing number requires that the summation of all the clique weightings equals 1
	QLP.add_constraint( 1| c_total <= k_v)
	# Constrain the value of each probability to equal the summation of the cliques that overlap it
	for k, v in p_c.items():
		w = 0
		for l in v:
			w += c[l]
		QLP.add_constraint( 1|p[k-1] == w)
	# Produce a maximally contextual system
	corr_A1B1 = (( p[0] +  p[3]) - ( p[1] +  p[2]))
	corr_A2B1 = (( p[4] +  p[7]) - ( p[5] +  p[6]))
	corr_A1B2 = (( p[8] + p[11]) - ( p[9] + p[10]))
	corr_A2B2 = ((p[12] + p[15]) - (p[13] + p[14]))
	QLP.add_constraint( 1| 0 - corr_A1B1 + corr_A2B1 + corr_A1B2 + corr_A2B2 == corr_A1B1_v)
	QLP.add_constraint( 1| 0 + corr_A1B1 - corr_A2B1 + corr_A1B2 + corr_A2B2 == 0)
	QLP.add_constraint( 1| 0 + corr_A1B1 + corr_A2B1 - corr_A1B2 + corr_A2B2 == 0)
	QLP.add_constraint( 1| 0 + corr_A1B1 + corr_A2B1 + corr_A1B2 - corr_A2B2 == 0)
	QLP.solve(verbose=0)
	# Instantiating values
	return np.array([round(x*1000)/1000 for x in c])

'''
	This function plots the weightings of the cliques for a probabilistic model in question
'''
def plot_weightings_of_p_model(weightings, title):
	fig, ax = plt.subplots(1,1)
	plt.rcParams["font.family"] = 'Times New Roman'
	plt.rcParams["font.weight"] = 'bold'
	plt.rcParams["font.size"] = 24
	plt.rcParams["figure.figsize"] = [20,8]
	red_deft = '#DD4980'
	ax.margins(tight=True)
	ax.grid(which='minor', color='#EEE', linestyle='-', linewidth=0.5)
	ax.grid(which='major', color='#DDD', linestyle='-', linewidth=1)
	ax.plot(weightings, 
			 color=red_deft, marker='D', markersize=7, lw=2)
	ax.set_xticks(range(0,161,10))
	ax.set_xticklabels(range(0,161,10))
	ax.set_ylabel("Weighting")
	ax.set_xlabel("Clique")
	ax.set_title(title, y=1.08)
	fig.subplots_adjust(bottom=0.5)
	plt.draw()
	plt.plot()

'''
	This function evaluates the graph properties that are conjectured to define the cliques that characterise
	all extremal probabilistic models (i.e., maximally contextual probabilistic models) of a system of 
	contextuality scenarios.
'''
def evaluate_graph_properties(cliques_of_length_4):
	# Generate all combinations of cliques of length 4
	combinations = list(itertools.product([0,1], repeat=len(cliques_of_length_4)))
	print("Tested a total of %s combinations:\n" % (len(combinations)))
	for c in range(0, len(combinations)):
		# Define the combination of cliques
		comb_cliques = [cliques_of_length_4[i] for i in range(len(combinations[c])) 
														if (combinations[c][i] > 0)]
		# Determine if this combination of cliques is connected
		cliques_are_connected = False
		if (len(comb_cliques) > 0):
			# The combination of cliques is determined to be connected by attempting to form a
			# single connected component
			comb_cliques_copy = [set(x) for x in comb_cliques]
			connected_cliques = comb_cliques_copy[0]
			comb_cliques_copy.remove(connected_cliques)
			found_intersection = True
			# Continuously loop through the combination of cliques, adding intersecting cliques
			# until no more intersections are present.
			while (found_intersection):
				found_intersection = False
				for a_clique in comb_cliques_copy:
					if (set(a_clique).intersection(set(connected_cliques))):
						comb_cliques_copy.remove(a_clique)
						connected_cliques = connected_cliques.union(set(a_clique))
						found_intersection = True
			# The combination of cliques are connected only if all elements could be intersected
			cliques_are_connected = (not (len(comb_cliques_copy) > 0))

		# Determine if this combination of cliques intersects all vertices in the contextuality scenario
		intersects_all_vertices = (
			list(set([item for sublist in comb_cliques for item in sublist])) 
										== [x+1 for x in range(16)])

		# Determine whether there is a single non-adjacent vertex in each clique
		# by determining the unique vertices in each combination
		comb_cliques_unique = []
		for elem in comb_cliques:
			for vertex in elem:
				duplic = False
				for elem_b in comb_cliques:
					if (vertex in elem_b and (elem_b is not elem)):
						duplic = True
				if not duplic:
					comb_cliques_unique.append(vertex)
		# Determine the number of unique vertices within each combination
		comb_cliques_counts = []
		for i in comb_cliques:
			total = 0
			for num in comb_cliques_unique:
				if (num in i):
					total += 1
			comb_cliques_counts.append(total)
		non_adjacency_in_each_clique = ((len(set(comb_cliques_counts)) == 1) 
											and (sum(comb_cliques_counts) == len(comb_cliques)))

		# Test all three propositions simultaneously
		if (intersects_all_vertices and non_adjacency_in_each_clique and cliques_are_connected):
			print("\tFound a combination of cliques (set form):")
			print("\t%s" % comb_cliques)
			print()
