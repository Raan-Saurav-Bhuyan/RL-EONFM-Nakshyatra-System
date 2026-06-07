import matplotlib.pyplot as plt
import networkx as nx
import os

from eon_env.v2.environment import EONEnvV2

def visualize_v2_topology(json_path = "nsfnet.json"):
    """
    Visualizes an external JSON topology initialized through the V2 Gym Environment.
    Maps ROADM nodes and multi-layer structural links.
    """
    print(f"--- Generating V2 Topology Visualization from {json_path} ---")

    if not os.path.exists(json_path):
        print(f"Error: Could not find '{json_path}'. Please ensure the GNPy-compatible topology file exists.")
        return

    # Create the V2 Gym environment and extract the loaded graph: --->
    env = EONEnvV2(network_json_path = json_path)
    env.reset()
    G = env.simulator.topology.graph

    # Set up the plot style and figure: --->
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    # Determine node positions: --->
    pos = nx.spring_layout(G, seed=42, k=0.9)

    # Draw nodes (Representing ROADM equipment): --->
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgreen',
                           node_size=800, edgecolors='black', linewidths=1.0)

    # Draw edges (Representing LinkLayers with Fibers, EDFAs, FIFOs): --->
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    # Draw edge labels showing the physical distance configured: --->
    edge_labels = { (u, v): f"{d['weight']} km" for u, v, d in G.edges(data=True) }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_color='darkred', font_size=8)

    ax.set_title("SDM & Flex-Grid EON NSFNET Topology", fontsize = 20, fontweight = 'bold')
    fig.tight_layout()
    plt.axis('off')

    out_file = "visualizations/nsfnet_topology_V2.png"
    plt.savefig(out_file, dpi = 300, bbox_inches = 'tight')
    print(f"Network topology visualization successfully saved to '{out_file}'")

if __name__ == '__main__':
    visualize_v2_topology()
