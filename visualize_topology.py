import matplotlib.pyplot as plt
import networkx as nx

from eon_env.v1.topology import NetworkTopology


def visualize_nsfnet_topology():
    """
    Creates and visualizes the NSFNET topology used in the EON environment.
    The visualization includes nodes, links, and link lengths, and is saved
    to a file.
    """
    print("--- Generating NSFNET Topology Visualization ---")

    # Create the topology instance and get the graph object: --->
    topology = NetworkTopology()
    G = topology.graph

    # Set up the plot style and figure: --->
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    # Determine node positions using a spring layout for better spacing: --->
    pos = nx.spring_layout(G, seed=42, k=0.9)

    # Draw the network components: --->
    # 1. Nodes: --->
    nx.draw_networkx_nodes(
        G, pos, ax = ax,
        node_color = 'skyblue',
        node_size = 800,
        edgecolors = 'black',
        linewidths = 1.0
    )
    # 2. Edges: --->
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=1.5,
        alpha=0.7,
        edge_color='gray'
    )
    # 3. Node labels: --->
    nx.draw_networkx_labels(
        G, pos, ax = ax,
        font_size = 10,
        font_weight = 'bold'
    )
    # 4. Edge labels (displaying link length): --->
    edge_labels = nx.get_edge_attributes(G, 'length_km')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels = edge_labels,
        ax = ax,
        font_color = 'darkred',
        font_size = 8
    )

    # 5. Finalize and save the plot: --->
    ax.set_title("NSFNET Network Topology", fontsize = 20, fontweight = 'bold')
    fig.tight_layout()
    plt.axis('off')

    plt.savefig("nsfnet_topology.png", dpi=300, bbox_inches='tight')
    print("Network topology visualization saved to 'nsfnet_topology.png'")


if __name__ == '__main__':
    visualize_nsfnet_topology()
