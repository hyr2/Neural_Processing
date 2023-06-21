from typing import Union
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as isomorphism

def create_graph(n_units: int, edge_table: np.ndarray):
    '''
    create a graph with `n_units` nodes and edges defined in `edge_table`.
    `edge_table` has shape (n_edges, 3) where each row is
    (source id, sink id, connection_type)
    '''
    g = nx.DiGraph()
    g.add_nodes_from(range(n_units))
    edge_table_n = [(u, v, {"type": x}) for (u,v,x) in edge_table]
    g.add_edges_from(edge_table_n)
    return g

def create_graph_no_orphans(edge_table: np.ndarray):
    '''
    create a graph with and edges defined in `edge_table` and involved nodes.
    `edge_table` has shape (n_edges, 3) where each row is
    (source id, sink id, connection_type)
    '''
    g = nx.DiGraph()
    edge_table_n = [(u, v, {"type": x}) for (u,v,x) in edge_table]
    g.add_edges_from(edge_table_n)
    return g

def get_subgraph_keep_nodes(g, edge_type):
    """
    get subgraph from DiGraph `g` where all nodes are retained and edges are
    selected based on `edge_type`
    Returns a new copy
    """
    h = g.copy()
    edges_to_remove = [(u, v) for u,v,d in h.edges(data="type") if d!=edge_type]
    h.remove_edges_from(edges_to_remove)
    return h

def plot_graph(g, node_shanks, node_depths, ax):
    '''
    plot the graph with the nodes placed in a circle and each shank taking up
        90 degrees.
    g: the graph with `n` nodes labeled from 0..n-1
    node_shanks: shanks of nodes; take values in 0..3; a list of size n
    node_depths_normalized: depths of nodes normalized at [0,1]; list of size n
    ax: the axis on which to plot
    '''
    n_nodes = len(g.nodes)
    n_edges = len(g.edges)
    assert len(node_depths)==n_nodes and len(node_shanks)==n_nodes
    # calculate node positions
    node_pos_arc = (
        np.asarray(node_shanks)+np.asarray(node_depths)
    )*np.pi/2
    xs = np.cos(node_pos_arc)
    ys = np.sin(node_pos_arc)
    node_pos_dict = dict(enumerate(zip(xs, ys)))
    # color the nodes by shank
    shank2color = [(1,1,0), (1,0,1), (0,1,1), (0,1,0)]
    node_colors = np.zeros((n_nodes, 4))
    node_colors[:, 3] = 0.3
    for i_ in range(4):
        node_colors[node_shanks==i_, :3] = shank2color[i_]
    # color the edges by type (exc/inh)
    edge_type_arr = np.array([x[2] for x in g.edges.data("type")])
    edge_colors = np.zeros((n_edges, 3))
    edge_colors[edge_type_arr==1, :]  = [1,0,0] # render excitatory edges in red
    edge_colors[edge_type_arr==-1, :] = [0,0,1] # render inhibitory edges in blue
    nx.draw_networkx(
        g, pos=node_pos_dict, edge_color=edge_colors, node_color=node_colors,
        node_size=8,
        arrows=True, with_labels=False, ax=ax
    )
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])

def match_microcircuits(g, gsubs):
    """
    g: connectivity graph
    gsubs: list or dictionary of "microcircuits" (a small networkx graph)
    For each pattern `gsub` in `gsubs`, returns a list of subgraphs of `g` 
    that matches `gsub` in both topology and edge type.
    Return type is list or dictionary depending on `gsubs` type.
    """
    edge_match_func = lambda e1, e2: e1["type"]==e2["type"]
    if isinstance(gsubs, dict):
        ret_dict = {}
        for name, gsub in gsubs.items():
            gm = isomorphism.DiGraphMatcher(g, gsub, edge_match=edge_match_func)
            ret_dict[name] = list(gm.subgraph_isomorphisms_iter())
        return ret_dict
    elif isinstance(gsubs, list):
        ret_list = []
        for gsub in gsubs:
            gm = isomorphism.DiGraphMatcher(g, gsub, edge_match=edge_match_func)
            ret_list.append(list(gm.subgraph_isomorphisms_iter()))
        return ret_list

# microcircuit templates, only involving 2 or 3 edges
MICROCIRCUIT_EDGETABLES = {
    "ff_exc_2": [[0,1,1],[1,2,1]],             # 0 --< 1 --< 2
    "ff_inh_2": [[0,1,1],[1,2,-1]],            # 0 --< 1 --| 2
    "fb_inh_3": [[0,1,1],[1,2,1],[2,0,-1]],    # 0 --< 1 --< 2 --| 0
    "fb_inh_2": [[0,1,1],[1,0,-1]],            # 0 --< 1 --| 0
    "fb_exc_3": [[0,1,1],[1,2,1],[2,0,1]],     # 0 --< 1 --< 2 --< 0
    "fb_exc_2": [[0,1,1],[1,0,1]],             # 0 --< 1 --< 0
}
MICROCIRCUIT_NXGRAPHS = dict()
for k, e in MICROCIRCUIT_EDGETABLES.items():
    g = nx.DiGraph()
    eb = [(u, v, {"type": x}) for (u,v,x) in e]
    g.add_edges_from(eb)
    MICROCIRCUIT_NXGRAPHS[k] = g
    
# def calc_hits(g, normalize=None, return_node_keys=False):
#     assert normalize in ["sum", "ssq", None]
#     # n_nodes = len(g.nodes)
#     v0 = dict((node, 1.0) for node in g.nodes)
#     if normalize=="sum":
#         dict_hu, dict_au = nx.hits(g, normalized=True, nstart=v0)
#     else:
#         dict_hu, dict_au = nx.hits(g, normalized=False, nstart=v0)
#     hu = np.array(list(dict_hu.values()))
#     au = np.array(list(dict_au.values()))
#     # correctly assume the hub and authority scores are presented in the same
#     # node order. See https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/hits_alg.html#hits
    
#     if normalize=="ssq":
#         hu /= np.sqrt(np.sum(hu**2))
#         au /= np.sqrt(np.sum(au**2))
    
#     if return_node_keys:
#         return list(dict_hu.keys()), hu, au
#     return hu, au

def calc_hits(g):
    a_ = nx.adjacency_matrix(g).todense()
    w_h, v_h = np.linalg.eigh(np.dot(a_, a_.T))
    h = v_h[:, np.argmax(w_h)] # (n_nodes, )
    h[np.isclose(h, 0)] = 0 # numerical correction
    if np.any(h<0):
        h *= -1
    if np.any(h<0):
        raise AssertionError("Hub score should be all positive\n", h)

    w_a, v_a = np.linalg.eigh(np.dot(a_.T, a_))
    a = v_a[:, np.argmax(w_a)]
    a[np.isclose(a, 0)] = 0 # numerical correction
    if np.any(a<0):
        a *= -1
    if np.any(a<0):
        raise AssertionError("Authority score should be all positive\n", a)
    return h, a
