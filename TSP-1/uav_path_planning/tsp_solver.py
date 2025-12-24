import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
import heapq
import time

print("Loading tsp_solver module...")

def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_lp_lower_bound(nodes_indices, dist_matrix, start_node, end_node):
    """
    Calculates the lower bound using Assignment Problem (AP).
    This is the standard 'LP' relaxation for TSP.
    We find a minimum cost path from start_node to end_node visiting all nodes_indices
    by solving an AP on the subgraph.
    Sources: {start_node} U nodes_indices
    Sinks:   nodes_indices U {end_node}
    """
    if not nodes_indices:
        return 0.0
    if len(nodes_indices) == 1:
        return dist_matrix[start_node][nodes_indices[0]] + dist_matrix[nodes_indices[0]][end_node]

    n_u = len(nodes_indices)
    n_dim = n_u + 1
    
    # Cost Matrix for AP
    # Rows: 0 (start_node), 1..n_u (nodes_indices)
    # Cols: 0..n_u-1 (nodes_indices), n_u (end_node)
    
    cost_matrix = np.zeros((n_dim, n_dim))
    LARGE_VAL = 1e9
    
    # 1. Row 0 (start_node) -> Cols (U + E)
    for j in range(n_u):
        u = nodes_indices[j]
        cost_matrix[0][j] = dist_matrix[start_node][u]
    # S -> E is technically possible in AP (skipping U), but valid as relaxation.
    # However, if we must visit U, S->E implies skipping U. 
    # But since every column (U) must be covered, if S->E is picked, 
    # the nodes in U must be covered by other nodes in U.
    # This forms S->E and cycles in U. This is a valid relaxation.
    cost_matrix[0][n_u] = dist_matrix[start_node][end_node]
    
    # 2. Rows 1..n_u (U) -> Cols (U + E)
    for i in range(n_u):
        u = nodes_indices[i]
        for j in range(n_u):
            v = nodes_indices[j]
            if i == j:
                cost_matrix[i+1][j] = LARGE_VAL
            else:
                cost_matrix[i+1][j] = dist_matrix[u][v]
        cost_matrix[i+1][n_u] = dist_matrix[u][end_node]
        
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    lb = cost_matrix[row_ind, col_ind].sum()
    
    return lb

def nearest_neighbor_path(current_path, unvisited, dist_matrix):
    """
    Greedy completion of the path for upper bound calculation.
    """
    path = list(current_path)
    remaining = set(unvisited)
    
    curr = path[-1]
    while remaining:
        nxt = min(remaining, key=lambda x: dist_matrix[curr][x])
        path.append(nxt)
        remaining.remove(nxt)
        curr = nxt
    return path

def calculate_path_length(path, dist_matrix):
    length = 0.0
    for i in range(len(path) - 1):
        length += dist_matrix[path[i]][path[i+1]]
    return length

def branch_and_bound_order(start_pos, centroids, patches_data, dist_fn, start_cost, return_cost, timeout=None):
    """
    Improved B&B TSP Solver using LP-based Lower Bound (Iterative approach).
    :param timeout: Max time in seconds. If None, runs until completion (Exact).
    """
    n = len(centroids)
    
    # Precompute distance matrix including start_pos (index -1 effectively)
    # We will treat start_pos as node n, and we need to return to start_pos
    # effectively making it a TSP tour on indices 0..n-1 plus start/end.
    # However, the input format asks for an order of 0..n-1.
    # We can model this as finding a path from start_pos to start_pos passing through all 0..n-1.
    
    # Construct full distance matrix for indices 0..n-1
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = dist_fn(i, j) if i != j else 0
            
    # Initial Solution (Upper Bound) using Nearest Neighbor + 2-opt
    # We need a tour: start -> p[0] -> ... -> p[n-1] -> start
    # Let's use the provided heuristic to get a good initial T*
    initial_order = nearest_neighbor_order(start_pos, centroids)
    
    # Apply 2-Opt to improve the initial heuristic
    # Node weights (patch lengths) are constant, so we only need to optimize travel distances.
    def dist_fn_2opt(i, j):
        return dist_matrix[i][j]
    
    print("Running 2-Opt optimization on initial heuristic...")
    initial_order = two_opt(initial_order, dist_fn_2opt, start_cost, return_cost)
    
    # Calculate initial upper bound cost
    def get_tour_cost(order):
        cost = start_cost(order[0]) + patches_data[order[0]]['len']
        for k in range(len(order)-1):
            cost += dist_matrix[order[k]][order[k+1]] + patches_data[order[k+1]]['len']
        cost += return_cost(order[-1])
        return cost

    best_order = list(initial_order)
    best_cost = get_tour_cost(best_order)
    
    print(f"Initial Heuristic Cost: {best_cost:.2f}")

    # Priority Queue for B&B: (lower_bound, current_path_tuple)
    # We store path as indices 0..n-1.
    # The cost function includes 'len' of patches.
    # Let's separate the static node costs (patch lengths) from edge costs for LP.
    # Total Cost = Edge Costs + Node Costs (sum of all patch lengths)
    # LP optimizes Edge Costs. Node Costs are constant for a full tour.
    total_patch_len = sum(p['len'] for p in patches_data)
    
    # Initial state
    # We start at 'virtual start node'. But the function signature expects us to return order of centroids.
    # The first node in the path must be one of the centroids.
    # To fit the document's 'path from a to b' model:
    # Here we are finding a tour. We can fix the first node to be the one closest to start_pos to reduce symmetry?
    # Or strictly follow the document: Tree of paths.
    
    # Let's adapt:
    # Root branches: Start -> i (for all i)
    # But start is fixed. So we just need to pick the first centroid.
    
    # To use the LP bound effectively for "Start -> ... -> Return", we can augment the graph:
    # Node n is 'Start'.
    # We need a path from n to n visiting 0..n-1.
    
    # Augmented Distance Matrix (Size n+1 x n+1)
    # Indices 0..n-1 are centroids. Index n is Start.
    aug_n = n + 1
    aug_dist = np.zeros((aug_n, aug_n))
    aug_dist[:n, :n] = dist_matrix
    for i in range(n):
        aug_dist[n][i] = aug_dist[i][n] = start_cost(i) # Assuming start_cost approx equals return_cost or symmetric
        
    # Queue stores: (lower_bound, current_path)
    # current_path is a list of indices in 0..n-1.
    # The path implies: Start -> path[0] -> ... -> path[-1]
    
    pq = []
    
    # Initialize with 1-length paths (Start -> i)
    for i in range(n):
        # Path: [i] (Implicitly Start -> i)
        # Cost so far: start_cost(i) + patch_len[i]
        # Remaining: visit other n-1 nodes and return to Start
        
        # Lower Bound = (Cost so far) + LP_Lower_Bound(on remaining nodes + i + Start)
        # Note: We need a path through remaining nodes, connecting i and Start.
        
        # Unvisited nodes + i + Start
        nodes_subset = [i, n] + [x for x in range(n) if x != i]
        # We need a path from i to n (Start) visiting all others.
        
        # Optimization: Global constant term (sum of all patch lengths) can be added at the end
        # But we need it for comparison with best_cost.
        
        # Current fixed cost (edges): start_cost(i)
        current_edge_cost = start_cost(i)
        
        # LP Bound for remaining edges:
        # Path from i to n (Start) covering unvisited
        rem_indices = [x for x in range(n) if x != i]
        if not rem_indices:
             lb_edges = 0
        else:
             # Use LP Lower Bound
             # Path from i to n (Start) covering rem_indices
             lb_edges = calculate_lp_lower_bound(rem_indices, aug_dist, i, n)
             
        total_lb = current_edge_cost + lb_edges + total_patch_len
        
        if total_lb < best_cost:
            heapq.heappush(pq, (total_lb, [i]))

    import time
    start_time = time.time()
    
    while pq:
        # Check timeout
        if timeout is not None and (time.time() - start_time > timeout):
            print(f"B&B Time Limit Exceeded ({timeout}s). Returning best solution found.")
            break
            
        lb, path = heapq.heappop(pq)
        
        if lb >= best_cost:
            continue
            
        last_node = path[-1]
        unvisited = [x for x in range(n) if x not in path]
        
        # If full tour found
        if not unvisited:
            # Complete tour: ... -> last_node -> Start
            final_cost = calculate_path_length(path, dist_matrix) + start_cost(path[0]) + return_cost(path[-1]) + total_patch_len
            # Recalculate precisely to match get_tour_cost logic (dist_matrix might be slightly diff from start/return functions if not symmetric)
            # Actually get_tour_cost uses patches_data['len'] which is total_patch_len.
            # Just call get_tour_cost
            true_cost = get_tour_cost(path)
            
            if true_cost < best_cost:
                best_cost = true_cost
                best_order = path
                print(f"New Best Cost: {best_cost:.2f}")
            continue
            
        # Branching: Extend path to nearest unvisited neighbors
        # To reduce branching factor, we can just expand ALL unvisited, 
        # but sort them by distance to prioritize good paths (Best-First)
        
        # For the document's method: "Create a new path p(k) by adding to path p_natural it closest node"
        # The document implies a specific iterative expansion. 
        # But standard B&B expands all children. 
        # "Iterative" in the doc might mean maintaining just one tree and expanding the best node.
        
        # Let's expand all children but prune heavily.
        
        # Heuristic: Sort children by distance from last_node
        unvisited.sort(key=lambda x: dist_matrix[last_node][x])
        
        for next_node in unvisited:
            new_path = path + [next_node]
            
            # Calculate Upper Bound for this new branch (using Nearest Neighbor completion)
            # to see if we can update best_cost
            rem_for_ub = [x for x in unvisited if x != next_node]
            if not rem_for_ub:
                ub_path = new_path
            else:
                ub_path = nearest_neighbor_path(new_path, rem_for_ub, dist_matrix)
            
            ub_cost = get_tour_cost(ub_path)
            if ub_cost < best_cost:
                best_cost = ub_cost
                best_order = ub_path
                print(f"Heuristic Update Best Cost: {best_cost:.2f}")
                
            # Calculate Lower Bound
            # Path: new_path (Start -> ... -> last_node -> next_node)
            # Remaining: unvisited (excluding next_node)
            
            rem_unvisited = [x for x in unvisited if x != next_node]
            current_edge_cost = calculate_path_length(new_path, dist_matrix) + start_cost(new_path[0])
            
            if not rem_unvisited:
                lb_edges = 0
            else:
                lb_edges = calculate_lp_lower_bound(rem_unvisited, aug_dist, next_node, n)
            
            total_lb = current_edge_cost + lb_edges + total_patch_len
            
            if total_lb < best_cost:
                heapq.heappush(pq, (total_lb, new_path))
                
    return best_order

def nearest_neighbor_order(start_pos, centroids):
    n = len(centroids)
    unvisited = set(range(n))
    order = []
    cur = min(unvisited, key=lambda i: _dist(start_pos, centroids[i]))
    order.append(cur)
    unvisited.remove(cur)
    while unvisited:
        nxt = min(unvisited, key=lambda i: _dist(centroids[cur], centroids[i]))
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order

def two_opt(order, dist_fn, start_cost, end_cost):
    n = len(order)
    def route_length(o):
        if not o:
            return 0.0
        s = start_cost(o[0])
        s += sum(dist_fn(o[k], o[k+1]) for k in range(n-1))
        s += end_cost(o[-1])
        return s
    improved = True
    while improved:
        improved = False
        for i in range(0, n-2):
            for j in range(i+2, n):
                new_order = order[:i+1] + order[i+1:j+1][::-1] + order[j+1:]
                if route_length(new_order) + 1e-6 < route_length(order):
                    order = new_order
                    improved = True
    return order

def orientation_dp(order, patches_data, dist_matrix, start_dist, return_dist):
    n = len(order)
    dp = np.full((n, 2), float('inf'))
    prev = [[-1, -1] for _ in range(n)]
    first = order[0]
    dp[0][0] = patches_data[first]['len'] + start_dist[first][0]
    dp[0][1] = patches_data[first]['len'] + start_dist[first][1]
    for i in range(1, n):
        a = order[i-1]
        b = order[i]
        for dprev in (0,1):
            for dcur in (0,1):
                c = dp[i-1][dprev] + patches_data[b]['len'] + dist_matrix[a][dprev][b][0 if dcur==0 else 1]
                if c < dp[i][dcur]:
                    dp[i][dcur] = c
                    prev[i][dcur] = dprev
    last_idx = order[-1]
    end_dir = 0 if dp[n-1][0] + return_dist[last_idx][0] <= dp[n-1][1] + return_dist[last_idx][1] else 1
    dirs = [0]*n
    dirs[n-1] = end_dir
    for i in range(n-1,0,-1):
        dirs[i-1] = prev[i][dirs[i]]
    return [(order[i], dirs[i]) for i in range(n)]


