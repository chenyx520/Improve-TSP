
import time
import numpy as np
from uav_path_planning.tsp_solver import branch_and_bound_order, nearest_neighbor_order

def test_bnb_performance(n=16):
    print(f"Testing B&B with N={n}")
    
    # Mock data
    np.random.seed(42)
    centroids = [tuple(np.random.rand(2) * 100) for _ in range(n)]
    start_pos = (0, 0)
    
    # Mock patches_data (just need 'len')
    patches_data = [{'len': np.random.rand() * 10} for _ in range(n)]
    
    def dist_fn(i, j):
        return np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
    
    start_cost = lambda i: np.linalg.norm(np.array(start_pos) - np.array(centroids[i]))
    return_cost = lambda i: np.linalg.norm(np.array(centroids[i]) - np.array(start_pos))
    
    # Run B&B
    start_time = time.time()
    order = branch_and_bound_order(start_pos, centroids, patches_data, dist_fn, start_cost, return_cost)
    end_time = time.time()
    
    print(f"B&B took {end_time - start_time:.4f} seconds")
    print(f"Order: {order}")
    
    # Calculate cost
    cost = start_cost(order[0]) + patches_data[order[0]]['len']
    for k in range(len(order)-1):
        cost += dist_fn(order[k], order[k+1]) + patches_data[order[k+1]]['len']
    cost += return_cost(order[-1])
    print(f"Total Cost: {cost}")

if __name__ == "__main__":
    # Test small scale
    test_bnb_performance(10)
    # Test boundary scale
    test_bnb_performance(15)
    # Test larger scale (LP should help here)
    test_bnb_performance(18)
