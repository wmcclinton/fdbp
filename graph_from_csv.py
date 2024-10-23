import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from pyvis.network import Network
import torch
import torch.nn as nn

MAX_DIST = 1
def similarity_loss(state1, state2):
    return torch.max(1 - (nn.MSELoss()(state1, state2) / MAX_DIST), torch.tensor(0.0))

def dissimilarity_loss_dbscan(state1, state2):
    return (nn.MSELoss()(torch.tensor(state1, dtype=torch.float32).unsqueeze(0), torch.tensor(state2, dtype=torch.float32).unsqueeze(0)) / MAX_DIST).item()

start_node = None
final_node = None
def create_edge_list_from_csv(file_path='done.csv'):
    global start_node
    global final_node
    # Load the CSV into a dataframe
    df = pd.read_csv(file_path)

    # Extract the 64-dim numpy arrays from the dataframe
    # Assuming the first column is the label (e.g., "num_action") and the rest are the 64-dimensional vectors
    labels = df.iloc[:, 0]  # first column contains row labels like "num_action"
    vectors = df.iloc[:, 1:].values  # the rest are the 64-dimensional vectors

    # Function to cluster based on L2 norm
    def cluster_by_l2_norm(vectors, threshold):
        # Perform DBSCAN clustering based on the L2 norm (Euclidean distance)
        clustering = DBSCAN(eps=threshold, min_samples=1, metric=dissimilarity_loss_dbscan).fit(vectors)
        
        # Return the clustering labels
        return clustering.labels_

    # Define the L2 norm threshold
    threshold = 5e-4  # Adjust this based on the acceptable L2 distance for clustering

    # Cluster the rows
    cluster_labels = cluster_by_l2_norm(vectors, threshold)

    # Add the cluster labels to the dataframe
    df['cluster'] = cluster_labels

    # Output the dataframe with clusters
    #print(df[['Unnamed: 0', 'cluster']])

    def create_cluster_dict(df):
        """
        This function takes a DataFrame, filters rows where the 'Unnamed: 0' column does not contain underscores,
        and returns a dictionary where the keys are 'Unnamed: 0' values and the values are the corresponding 'cluster' values.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame with 'Unnamed: 0' and 'cluster' columns.

        Returns:
        dict: A dictionary mapping 'Unnamed: 0' values to 'cluster' values.
        """
        # Filter rows where 'Unnamed: 0' doesn't contain underscores
        filtered_df = df[~df['Unnamed: 0'].str.contains('_')]
        
        # Create and return the dictionary
        result_dict = dict(zip(filtered_df['Unnamed: 0'], filtered_df['cluster']))
        
        return result_dict

    cluster_dict = create_cluster_dict(df)
    def inv_cluster_dict(node):
        for k, v in  dict(zip(df['Unnamed: 0'], df['cluster'])).items():
            if v == node:
                return k
        return None
    #print(cluster_dict)

    filtered_df = df[df['Unnamed: 0'].str.contains('_')]
    result_dict = dict(zip(filtered_df['Unnamed: 0'], filtered_df['cluster']))

    # Initialize an empty list to store the edges
    edges = []
    def get_node_vector(node):
        return df.loc[df['Unnamed: 0'] == node, [str(i) for i in range(64)]].to_numpy()[0]
    start_node = cluster_dict['start']
    final_node = cluster_dict['goal']
    for node_action, node2 in result_dict.items():
        node1 = cluster_dict[node_action.split('_')[0]]
        action = node_action.split('_')[1]
        # print(f'Node 1: {node1}, Action: {action}, Node 2: {node2}')
        assert final_node is not None
        similarity1 = None
        similarity2 = None
        similarity3 = None
        val = torch.tensor(get_node_vector(node_action.split('_')[0]), dtype=torch.float32).unsqueeze(0)
        valg = torch.tensor(get_node_vector(inv_cluster_dict(final_node)), dtype=torch.float32).unsqueeze(0)
        similarity1 = similarity_loss(val, valg)
        val2  = torch.tensor(get_node_vector(node_action), dtype=torch.float32).unsqueeze(0)
        similarity2 = similarity_loss(val2, valg)
        try:
            val3 = torch.tensor(get_node_vector(inv_cluster_dict(node2)), dtype=torch.float32).unsqueeze(0)
            similarity3 = similarity_loss(val, val3)
        except:
            pass
        edges.append((str(node1) + '_' + str(similarity1).replace('tensor(','').replace(')',''), str(node2) + '_' + str(similarity2).replace('tensor(','').replace(')',''), str(action) + '_' + str(similarity3).replace('tensor(','').replace(')','')))
            

    # Convert the edge list into a DataFrame
    edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])

    # Save the edge list DataFrame to a CSV file
    output_csv_file = "edges.csv"
    edges_df.to_csv(output_csv_file, index=False)
    print(f"Edge list successfully saved to {output_csv_file}")

def edge_list_to_network(csv_file_path, output_html_file):
    global start_node
    global final_node
    # Load the edge list CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Initialize a PyVis Network object
    net = Network(directed=True, notebook=True, bgcolor="#222222", font_color="white")
    # Customize physics settings (optional)
    # Enable and configure physics options
    net.toggle_physics(True)
    net.set_options("""
    var options = {
    "physics": {
        "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.3,
        "springLength": 200,
        "springConstant": 0.04,
        "damping": 0.09,
        "avoidOverlap": 1
        },
        "minVelocity": 0.75
    }
    }
    """)
    nodes = set()

    # Iterate through the DataFrame and add nodes and edges to the graph
    for _, row in df.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row['Weight']
        
        # Add nodes and edges
        if str(source).split("_")[0] == str(start_node):
            net.add_node(str(source), title="Start node", label="Start")
            nodes.add(str(source))
        else:
            net.add_node(str(source))
            nodes.add(str(source))

        if str(target).split("_")[0] == str(final_node):
            net.add_node(str(target), title="Final node", label="Final")
            nodes.add(str(target))
        else:
            net.add_node(str(target))
            nodes.add(str(target))
        if 'None' in str(weight) or 'tensor(0.)' in str(weight):
            continue
        net.add_edge(str(source), str(target), title="action_" + str(weight)) # 'value' sets edge thickness based on weight
    if str(final_node) not in [node.split("_")[0] for node in nodes]:
        net.add_node(str(final_node), title="Final node", label="Final")
        nodes.add(str(final_node))
    # Generate and save the network graph to an HTML file
    net.show(output_html_file)
    print(f"Graph successfully saved to {output_html_file}")
    print("Nodes:", len(nodes))
    print("Edges:", len(net.edges))
    print("Start node:", start_node)
    print("Final node:", final_node)
    #print(net.get_adj_list())

# Example usage:
create_edge_list_from_csv('graph.csv')
edge_list_to_network('edges.csv', 'first_network.html')

create_edge_list_from_csv('graph_7000.csv')
edge_list_to_network('edges.csv', 'intermediate_network.html')

create_edge_list_from_csv('done.csv')
edge_list_to_network('edges.csv', 'final_network.html')