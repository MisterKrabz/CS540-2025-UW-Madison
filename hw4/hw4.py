# Import necessary libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    """
    Takes in a string with a path to a CSV file and returns the data points as a
    list of dictionaries, preserving the original headers.
    
    Args:
        filepath (str): The path to the CSV file to be read.

    Returns:
        list: A list where each element is a dictionary representing one row of the file.
    """
    data_list = []
    # Use 'utf-8-sig' to handle potential Byte Order Mark (BOM) in the CSV file
    with open(filepath, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        # Directly create a list of dictionaries from the reader, keeping original keys.
        for row in reader:
            data_list.append(row)
    return data_list

def calc_features(row):
    """
    Takes in one row dictionary, calculates the corresponding feature vector, 
    and returns it as a NumPy array of shape (9,).

    Args:
        row (dict): A dictionary representing one country.

    Returns:
        np.ndarray: A NumPy array of shape (9,) and dtype float64.
    """
    # These are the 9 features specified in the project rubric, using the exact
    # headers from the CSV file.
    feature_keys = [
        'child_mort', 'exports', 'health', 'imports', 'income',
        'inflation', 'life_expec', 'total_fer', 'gdpp'
    ]
    
    # Extract the values for the specified keys and convert them to float.
    feature_vector = [float(row[key]) for key in feature_keys]
    
    # Return as a NumPy array with the specified dtype.
    return np.array(feature_vector, dtype=np.float64)

def normalize_features(features):
    """
    Takes a list of feature vectors and computes the normalized values.
    The output is a list of normalized feature vectors.

    Args:
        features (list): A list of NumPy arrays (feature vectors).

    Returns:
        list: A list of normalized NumPy arrays.
    """
    # Convert the list of arrays into a 2D NumPy array for efficient calculation.
    features_np = np.array(features)
    
    # Calculate the mean and standard deviation for each feature column (axis=0).
    means = np.mean(features_np, axis=0)
    stds = np.std(features_np, axis=0)
    
    # Prevent division by zero if a feature has zero standard deviation.
    stds[stds == 0] = 1.0
    
    # Apply z-score normalization: (x - mu) / sigma.
    normalized_features_np = (features_np - means) / stds
    
    # Return the result as a list of 1D NumPy arrays, matching the required output format.
    return [row for row in normalized_features_np]

def hac(features, linkage_type):
    """
    Performs hierarchical agglomerative clustering.

    Args:
        features (list): A list of NumPy arrays of shape (9,).
        linkage_type (str): Linkage type, can be "single" or "complete".

    Returns:
        np.ndarray: A NumPy array of shape (n-1, 4) representing the clustering.
    """
    n = len(features)
    # Convert list of features to a 2D numpy array for distance calculation
    features_np = np.array(features)
    
    # Pre-calculate the initial pairwise Euclidean distance matrix
    # Using a dictionary for distances is easier for dynamic updates
    dist_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(features_np[i] - features_np[j])
            dist_matrix[(i, j)] = dist

    # `clusters` will map a cluster ID to the list of original data points it contains
    clusters = {i: [i] for i in range(n)}
    
    # Z will store the (n-1) merge steps
    Z = np.zeros((n - 1, 4))
    
    # Perform n-1 merges
    for k in range(n - 1):
        # Find the pair of clusters with the minimum distance, adhering to the tie-breaking rule
        min_dist = np.inf
        best_pair = (-1, -1)
        
        # Get current active cluster IDs and sort them to ensure consistent iteration order
        active_cluster_ids = sorted(clusters.keys())
        
        # Iterate through all pairs of active clusters to find the closest pair
        for i in range(len(active_cluster_ids)):
            for j in range(i + 1, len(active_cluster_ids)):
                c1_id, c2_id = active_cluster_ids[i], active_cluster_ids[j]
                
                # Ensure the key is always (smaller_id, larger_id)
                pair_key = (min(c1_id, c2_id), max(c1_id, c2_id))
                
                if dist_matrix[pair_key] < min_dist:
                    min_dist = dist_matrix[pair_key]
                    best_pair = pair_key

        # Unpack the best pair
        c1_id, c2_id = best_pair
        
        # Create the new cluster ID
        new_cluster_id = n + k
        
        # Calculate the size of the new merged cluster
        new_size = len(clusters[c1_id]) + len(clusters[c2_id])
        
        # Store the merge result in the Z matrix
        Z[k, :] = [c1_id, c2_id, min_dist, new_size]

        # Update the distance matrix for the new cluster
        # Get the list of remaining cluster IDs after the merge
        remaining_ids = [cid for cid in active_cluster_ids if cid not in [c1_id, c2_id]]

        for rem_id in remaining_ids:
            # Get distances from the old clusters to the remaining cluster
            dist1 = dist_matrix[tuple(sorted((c1_id, rem_id)))]
            dist2 = dist_matrix[tuple(sorted((c2_id, rem_id)))]
            
            # Calculate new distance based on linkage type
            if linkage_type == 'single':
                new_dist = min(dist1, dist2)
            elif linkage_type == 'complete':
                new_dist = max(dist1, dist2)
            else:
                raise ValueError("Invalid linkage_type. Must be 'single' or 'complete'.")
            
            # Add the new distance to the matrix
            dist_matrix[tuple(sorted((new_cluster_id, rem_id)))] = new_dist

        # Merge the clusters in the `clusters` dictionary
        clusters[new_cluster_id] = clusters[c1_id] + clusters[c2_id]
        
        # Delete the old clusters
        del clusters[c1_id]
        del clusters[c2_id]

        # Clean up the old distances from the distance matrix
        keys_to_remove = [key for key in dist_matrix if c1_id in key or c2_id in key]
        for key in keys_to_remove:
            del dist_matrix[key]
            
    return Z

def fig_hac(Z, names):
    """
    Visualizes the hierarchical agglomerative clustering.

    Args:
        Z (np.ndarray): The linkage matrix from hac. 
        names (list): A list of country names. 
    """
    # Initialize a figure for plotting
    plt.figure(figsize=(10, 20)) # Adjusted size for better label visibility
    
    # Use scipy's dendrogram function to plot
    dendrogram(
        Z,
        labels=names,
        orientation='top', # Match the orientation in the rubric figures
        leaf_rotation=90,  # Rotate labels for readability
        leaf_font_size=8,
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel("Distance")
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

# Main execution block
if __name__ == "__main__": 
    # Path to the data file
    DATA_FILEPATH = 'Country-data.csv'
    
    # 1. Load the data as a list of dictionaries
    data = load_data(DATA_FILEPATH)
    
    # 2. Extract country names and calculate feature vectors for each country
    # This follows the structure suggested in the "Testing" section
    # CORRECTED: Use the exact, case-sensitive key 'Country' (Capital 'C') to get the names.
    country_names = [row['Country'] for row in data]
    features = [calc_features(row) for row in data]
    
    # 3. Normalize the features
    features_normalized = normalize_features(features)
    
    # --- Example for N=20 countries with 'complete' linkage (matches Fig 3) ---
    print("Performing HAC with 'complete' linkage for the first 20 countries...")
    n_test = 20
    Z_complete_20 = hac(features_normalized[:n_test], linkage_type="complete")
    fig_hac(Z_complete_20, country_names[:n_test])
    plt.suptitle("First 20 Countries - Complete Linkage (Normalized)")
    plt.show()

    # --- Example for all countries with 'single' linkage ---
    print("\nPerforming HAC with 'single' linkage for all countries...")
    Z_single_all = hac(features_normalized, linkage_type='single')
    fig_hac(Z_single_all, country_names)
    plt.suptitle("All Countries - Single Linkage (Normalized)")
    plt.show()