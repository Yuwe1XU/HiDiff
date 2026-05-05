import numpy as np
import math
from multiprocessing import Pool
import warnings
from utils.cython_merge.cython_merge import merge_cython

def cython_merge(points, adj_mat):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
    real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations

import numpy as np
import math
from multiprocessing import Pool

class MCTSNode:
    def __init__(self, node_id, parent=None):
        self.node_id = node_id
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = []
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.total_reward / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def expand(self, action):
        child = MCTSNode(action, parent=self)
        self.children[action] = child
        self.untried_actions.remove(action)
        return child
    
    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

def mcts_tour_construction(adj_mat, max_iterations=100, c_param=1.4):
    """
    Construct TSP tour using MCTS instead of greedy selection.
    
    Args:
        adj_mat: Adjacency matrix with probabilities
        max_iterations: Number of MCTS iterations per step
        c_param: UCB exploration parameter
    
    Returns:
        Complete tour as list of node indices
    """
    n_nodes = adj_mat.shape[0]
    tour = [0]  # Start from node 0
    
    while len(tour) < n_nodes + 1:
        current_node = tour[-1]
        
        # Get available next nodes
        available_nodes = np.nonzero(adj_mat[current_node])[0]
        if len(tour) > 1:
            available_nodes = available_nodes[available_nodes != tour[-2]]
        
        if len(available_nodes) == 0:
            break
            
        if len(available_nodes) == 1:
            # Only one choice, take it
            tour.append(available_nodes[0])
            continue
        
        # Use MCTS to select next node
        next_node = mcts_select_node(adj_mat, tour, available_nodes, max_iterations, c_param)
        tour.append(next_node)
    
    return tour

def mcts_select_node(adj_mat, current_tour, available_nodes, max_iterations, c_param):
    """
    Use MCTS to select the best next node for the tour.
    """
    # Initialize root node
    root = MCTSNode(-1)  # Dummy root
    root.untried_actions = list(available_nodes)
    
    for _ in range(max_iterations):
        # Selection & Expansion
        node = root
        temp_tour = current_tour.copy()
        
        # Selection phase - traverse down the tree
        while not node.is_fully_expanded() and len(node.children) > 0:
            if len(node.untried_actions) > 0:
                break
            node = node.best_child(c_param)
            if node.node_id != -1:  # Skip dummy root
                temp_tour.append(node.node_id)
        
        # Expansion phase
        if len(node.untried_actions) > 0:
            action = np.random.choice(node.untried_actions)
            node = node.expand(action)
            temp_tour.append(action)
        
        # Simulation phase - complete the tour randomly and evaluate
        reward = simulate_completion(adj_mat, temp_tour)
        
        # Backpropagation phase
        while node is not None:
            node.update(reward)
            node = node.parent
    
    # Select the most visited child (most robust choice)
    if len(root.children) == 0:
        return np.random.choice(available_nodes)
    
    best_child = max(root.children.values(), key=lambda x: x.visits)
    return best_child.node_id

def mcts_tour_construction_2x(combined_adj_mat, adj_mat_1, adj_mat_2, max_iterations=100, c_param=1.4):
    """
    Construct TSP tour using MCTS with 2x diffusion samples.
    
    Args:
        combined_adj_mat: Combined adjacency matrix from both samples
        adj_mat_1: First sample adjacency matrix
        adj_mat_2: Second sample adjacency matrix
        max_iterations: Number of MCTS iterations per step
        c_param: UCB exploration parameter
    
    Returns:
        Complete tour as list of node indices
    """
    n_nodes = combined_adj_mat.shape[0]
    tour = [0]  # Start from node 0
    
    while len(tour) < n_nodes + 1:
        current_node = tour[-1]
        
        # Get available next nodes from combined matrix
        available_nodes = np.nonzero(combined_adj_mat[current_node])[0]
        if len(tour) > 1:
            available_nodes = available_nodes[available_nodes != tour[-2]]
        
        if len(available_nodes) == 0:
            break
            
        if len(available_nodes) == 1:
            # Only one choice, take it
            tour.append(available_nodes[0])
            continue
        
        # Use MCTS to select next node with sample diversity consideration
        next_node = mcts_select_node_2x(combined_adj_mat, adj_mat_1, adj_mat_2, 
                                       tour, available_nodes, max_iterations, c_param)
        tour.append(next_node)
    
    return tour

def mcts_select_node_2x(combined_adj_mat, adj_mat_1, adj_mat_2, current_tour, 
                       available_nodes, max_iterations, c_param):
    """
    Use MCTS to select the best next node considering both diffusion samples.
    """
    # Initialize root node
    root = MCTSNode(-1)  # Dummy root
    root.untried_actions = list(available_nodes)
    
    for _ in range(max_iterations):
        # Selection & Expansion
        node = root
        temp_tour = current_tour.copy()
        
        # Selection phase - traverse down the tree
        while not node.is_fully_expanded() and len(node.children) > 0:
            if len(node.untried_actions) > 0:
                break
            node = node.best_child(c_param)
            if node.node_id != -1:  # Skip dummy root
                temp_tour.append(node.node_id)
        
        # Expansion phase
        if len(node.untried_actions) > 0:
            action = np.random.choice(node.untried_actions)
            node = node.expand(action)
            temp_tour.append(action)
        
        # Simulation phase - complete the tour and evaluate with both samples
        reward = simulate_completion_2x(combined_adj_mat, adj_mat_1, adj_mat_2, temp_tour)
        
        # Backpropagation phase
        while node is not None:
            node.update(reward)
            node = node.parent
    
    # Select the most visited child (most robust choice)
    if len(root.children) == 0:
        return np.random.choice(available_nodes)
    
    best_child = max(root.children.values(), key=lambda x: x.visits)
    return best_child.node_id

def simulate_completion_2x(combined_adj_mat, adj_mat_1, adj_mat_2, partial_tour):
    """
    Complete the tour randomly and return a reward considering both samples.
    """
    tour = partial_tour.copy()
    n_nodes = combined_adj_mat.shape[0]
    
    # Complete tour with remaining nodes
    visited = set(tour)
    remaining_nodes = [i for i in range(n_nodes) if i not in visited]
    
    while len(remaining_nodes) > 0:
        current_node = tour[-1]
        
        # Get available next nodes from combined matrix
        available = []
        for node in remaining_nodes:
            if combined_adj_mat[current_node, node] > 0:
                available.append(node)
        
        if not available:
            # If no valid connections, pick randomly
            available = remaining_nodes
        
        # Select next node with probability proportional to combined edge weights
        if len(available) == 1:
            next_node = available[0]
        else:
            probs = [combined_adj_mat[current_node, node] for node in available]
            probs = np.array(probs)
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
                next_node = np.random.choice(available, p=probs)
            else:
                next_node = np.random.choice(available)
        
        tour.append(next_node)
        remaining_nodes.remove(next_node)
    
    # Calculate reward considering both samples
    total_reward = 0.0
    diversity_bonus = 0.0
    
    for i in range(len(tour) - 1):
        # Main reward from combined probabilities
        prob_combined = combined_adj_mat[tour[i], tour[i+1]]
        if prob_combined > 0:
            total_reward += np.log(prob_combined + 1e-8)
        
        # Diversity bonus: reward when both samples agree
        prob_1 = adj_mat_1[tour[i], tour[i+1]]
        prob_2 = adj_mat_2[tour[i], tour[i+1]]
        
        if prob_1 > 0 and prob_2 > 0:
            # Both samples support this edge - consistency bonus
            consistency = min(prob_1, prob_2) / max(prob_1, prob_2)
            diversity_bonus += 0.1 * consistency
        elif prob_1 > 0 or prob_2 > 0:
            # Only one sample supports - small penalty for uncertainty
            diversity_bonus -= 0.05
    
    return total_reward + diversity_bonus

def merge_tours_MCTS(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1, 
                    mcts_iterations=100, c_param=1.4):
    """
    Modified version of your original function using MCTS for tour construction.
    
    Args:
        adj_mat: Adjacency matrix with edge probabilities
        np_points: Point coordinates  
        edge_index_np: Edge indices
        sparse_graph: Whether graph is sparse
        parallel_sampling: Number of parallel samples
        mcts_iterations: MCTS iterations per node selection
        c_param: UCB exploration parameter
    """
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    splitted_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat]  # For dense graph
    
    splitted_points = [np_points for _ in range(parallel_sampling)]

    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(cython_merge, zip(splitted_points, splitted_adj_mat))
    else:
        results = [cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)]

    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    tours = []
    for i in range(parallel_sampling):
        # Use MCTS instead of greedy decoding
        tour = mcts_tour_construction(splitted_real_adj_mat[i], mcts_iterations, c_param)
        tours.append(tour)

    merge_iterations = np.mean(splitted_merge_iterations)
    return tours, merge_iterations

def merge_tours_MCTS_2x(adj_mat_samples, np_points, edge_index_np, sparse_graph=False, 
                       parallel_sampling=1, mcts_iterations=100, c_param=1.4, 
                       ensemble_method='average'):
    """
    MCTS with 2x diffusion samples - uses multiple heatmaps for better tour construction.
    
    Args:
        adj_mat_samples: List of 2 adjacency matrices from different diffusion samples
        np_points: Point coordinates  
        edge_index_np: Edge indices
        sparse_graph: Whether graph is sparse
        parallel_sampling: Number of parallel samples
        mcts_iterations: MCTS iterations per node selection
        c_param: UCB exploration parameter
        ensemble_method: How to combine multiple samples ('average', 'max', 'weighted_vote')
    """
    assert len(adj_mat_samples) == 2, "This method expects exactly 2 diffusion samples"
    
    # Process each sample
    processed_samples = []
    all_merge_iterations = []
    
    for sample_idx, adj_mat in enumerate(adj_mat_samples):
        splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
        splitted_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat]
        
        splitted_points = [np_points for _ in range(parallel_sampling)]

        if np_points.shape[0] > 1000 and parallel_sampling > 1:
            with Pool(parallel_sampling) as p:
                results = p.starmap(cython_merge, zip(splitted_points, splitted_adj_mat))
        else:
            results = [cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)]

        splitted_real_adj_mat, splitted_merge_iterations = zip(*results)
        processed_samples.append(splitted_real_adj_mat)
        all_merge_iterations.extend(splitted_merge_iterations)
    
    # Generate tours using ensemble of samples
    tours = []
    for i in range(parallel_sampling):
        # Combine the two adjacency matrices
        adj_mat_1 = processed_samples[0][i]
        adj_mat_2 = processed_samples[1][i]
        
        if ensemble_method == 'average':
            combined_adj_mat = (adj_mat_1 + adj_mat_2) / 2.0
        elif ensemble_method == 'max':
            combined_adj_mat = np.maximum(adj_mat_1, adj_mat_2)
        elif ensemble_method == 'weighted_vote':
            # Weight by confidence (sum of probabilities)
            weight_1 = np.sum(adj_mat_1) / (np.sum(adj_mat_1) + np.sum(adj_mat_2))
            weight_2 = 1.0 - weight_1
            combined_adj_mat = weight_1 * adj_mat_1 + weight_2 * adj_mat_2
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        # Use MCTS with combined probabilities
        tour = mcts_tour_construction_2x(combined_adj_mat, adj_mat_1, adj_mat_2, 
                                        mcts_iterations, c_param)
        tours.append(tour)

    merge_iterations = np.mean(all_merge_iterations)
    return tours, merge_iterations

# Alternative simplified MCTS approach for faster execution
def merge_tours_MCTS_simple(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
    """
    Simplified MCTS approach that uses UCB for node selection without full tree search.
    """
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    splitted_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat]
    
    splitted_points = [np_points for _ in range(parallel_sampling)]

    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(cython_merge, zip(splitted_points, splitted_adj_mat))
    else:
        results = [cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)]

    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    tours = []
    for i in range(parallel_sampling):
        tour = [0]
        node_visits = {}  # Track visits for UCB calculation
        
        while len(tour) < splitted_adj_mat[i].shape[0] + 1:
            current_node = tour[-1]
            n = np.nonzero(splitted_real_adj_mat[i][current_node])[0]
            if len(tour) > 1:
                n = n[n != tour[-2]]
            
            if len(n) == 1:
                tour.append(n[0])
            else:
                # Use simple UCB-like selection
                total_visits = sum(node_visits.get(node, 0) for node in n)
                if total_visits == 0:
                    # First visit - select based on probability
                    probs = splitted_real_adj_mat[i][current_node][n]
                    next_node = n[np.argmax(probs)]
                else:
                    # UCB selection
                    ucb_scores = []
                    for node in n:
                        prob = splitted_real_adj_mat[i][current_node][node]
                        visits = node_visits.get(node, 0)
                        if visits == 0:
                            ucb_score = float('inf')
                        else:
                            exploitation = prob
                            exploration = math.sqrt(2 * math.log(total_visits) / visits)
                            ucb_score = exploitation + 0.1 * exploration
                        ucb_scores.append(ucb_score)
                    
                    next_node = n[np.argmax(ucb_scores)]
                
                node_visits[next_node] = node_visits.get(next_node, 0) + 1
                tour.append(next_node)
        
        tours.append(tour)

    merge_iterations = np.mean(splitted_merge_iterations)
    return tours, merge_iterations