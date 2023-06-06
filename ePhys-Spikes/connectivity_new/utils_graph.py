from typing import Union
import numpy as np
import networkx as nx

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
    i_units = set(edge_table[:,0]) | set(edge_table[:,1])
    g.add_nodes_from(i_units)
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
