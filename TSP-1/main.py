import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely import affinity
import math
import random
import networkx as nx
import itertools
import os
import datetime

class UAVPathPlanning:
    def __init__(self, field_bounds, patches, obstacles, uav_width=1.0, turning_radius=1.0):
        """
        初始化路径规划器
        :param field_bounds: 田地边界 [(x,y), ...]
        :param patches: 杂草块列表 [Polygon, ...]
        :param obstacles: 障碍物列表 [Polygon, ...]
        :param uav_width: UAV 喷洒宽度
        :param turning_radius: UAV 最小转弯半径
        """
        self.field = Polygon(field_bounds)
        self.patches = [Polygon(p) if not isinstance(p, Polygon) else p for p in patches]
        self.obstacles = [Polygon(o) if not isinstance(o, Polygon) else o for o in obstacles]
        self.uav_width = uav_width
        self.turning_radius = turning_radius
        self.visibility_graph = None
        
    def _inflated_obstacles(self, extra=0.0):
        margin = self.uav_width / 2 + extra
        # Use lower resolution and simplification for speed
        return [obs.buffer(margin, resolution=2).simplify(0.5, preserve_topology=True) for obs in self.obstacles]

    def _line_clear(self, p1, p2, extra_margin=0.0):
        line = LineString([p1, p2])
        for obs in self._inflated_obstacles(extra_margin):
            if line.intersects(obs) and not line.touches(obs):
                return False
        return True
        
    def build_visibility_graph(self, points_of_interest, update_class_instance=True):
        """
        构建可见性图 (Visibility Graph) 用于避障
        """
        G = nx.Graph()
        
        # 所有关键点：兴趣点 (Start/End) + 障碍物顶点
        # 为了安全，稍微膨胀障碍物
        # 增加膨胀距离以允许后续的路径平滑 (Corner Cutting)
        # Simplify to ensure intersection checks are fast. Reduce buffer resolution to avoid too many vertices.
        inflated_obstacles = [obs.buffer(self.uav_width/2 + 2.0, resolution=2).simplify(2.0, preserve_topology=True) for obs in self.obstacles]
        
        # 收集所有节点
        nodes = list(points_of_interest)
        for obs in inflated_obstacles:
            # 简化几何以减少节点数
            simplified_obs = obs.simplify(5.0, preserve_topology=True)
            nodes.extend(list(simplified_obs.exterior.coords))
            
        # 去重
        nodes = list(set(nodes))
        
        # 确保所有节点都在图中，即使是孤立点
        G.add_nodes_from(nodes)
        
        print(f"  Graph nodes: {len(nodes)}. Building edges...")
        
        # Precompute obstacle bounds and prepared geometries
        obs_bounds = [obs.bounds for obs in inflated_obstacles]
        prepared_obstacles = [prep(obs) for obs in inflated_obstacles]

        # 构建边
        for i in range(len(nodes)):
            if i % 20 == 0:
                print(f"  Processing node {i}/{len(nodes)}...")
            u = nodes[i]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                
                # AABB of the line segment
                lx_min = min(u[0], v[0])
                lx_max = max(u[0], v[0])
                ly_min = min(u[1], v[1])
                ly_max = max(u[1], v[1])
                
                intersects = False
                line = None
                
                for k, obs in enumerate(inflated_obstacles):
                    ox_min, oy_min, ox_max, oy_max = obs_bounds[k]
                    
                    # Quick AABB check
                    if (lx_max < ox_min or lx_min > ox_max or 
                        ly_max < oy_min or ly_min > oy_max):
                        continue
                        
                    if line is None:
                        line = LineString([u, v])
                        
                    # Use prepared geometry for fast intersection check
                    if prepared_obstacles[k].intersects(line):
                        if not obs.touches(line):
                            intersects = True
                            break
                
                if not intersects:
                    dist = math.hypot(u[0]-v[0], u[1]-v[1])
                    G.add_edge(u, v, weight=dist)
                    
        if update_class_instance:
            self.visibility_graph = G
        return G

    def find_path_avoiding_obstacles(self, start, end):
        """
        使用可见性图和 A* 寻找避障路径
        """
        # Optimization: If graph exists and nodes are in it, use it directly
        if self.visibility_graph is not None:
            # Shapely Points to tuples if needed, but graph uses tuples
            s_node = tuple(start) if not isinstance(start, tuple) else start
            e_node = tuple(end) if not isinstance(end, tuple) else end
            
            if self.visibility_graph.has_node(s_node) and self.visibility_graph.has_node(e_node):
                try:
                    path_nodes = nx.shortest_path(self.visibility_graph, source=s_node, target=e_node, weight='weight')
                    return path_nodes
                except nx.NetworkXNoPath:
                    pass # Fallback to rebuild

        # 确保起点和终点在图中
        if self.visibility_graph is None:
             # 临时构建包含这两点的图
             self.build_visibility_graph([start, end], update_class_instance=True)
        else:
            # 如果图已存在，可能需要添加这两点并更新边
            # 简单起见，这里每次重新构建或增量添加。为保证正确性，重新构建包含这两点的小图
            # 或者优化：只添加 start/end 到现有图
            pass
            
        # 重新构建包含 start/end 的图 (简化处理)
        s_node = tuple(start) if not isinstance(start, tuple) else start
        e_node = tuple(end) if not isinstance(end, tuple) else end
        G = self.build_visibility_graph([s_node, e_node], update_class_instance=False)
        
        try:
            path_nodes = nx.shortest_path(G, source=s_node, target=e_node, weight='weight')
            return path_nodes
        except nx.NetworkXNoPath:
            # 尝试通过最近的可见节点进行绕行
            graph_nodes = list(G.nodes())
            def nearest_visible_node(pt):
                candidates = sorted(graph_nodes, key=lambda n: np.linalg.norm(np.array(pt) - np.array(n)))
                for n in candidates:
                    if n == pt:
                        continue
                    if self._line_clear(pt, n, extra_margin=0.5):
                        return n
                return None
            s_vis = nearest_visible_node(s_node)
            e_vis = nearest_visible_node(e_node)
            if s_vis is not None and e_vis is not None:
                try:
                    mid_path = nx.shortest_path(G, source=s_vis, target=e_vis, weight='weight')
                    full_path = []
                    if self._line_clear(s_node, s_vis, extra_margin=0.5):
                        full_path.extend([s_node, s_vis])
                    else:
                        full_path.append(s_node)
                    full_path.extend(mid_path[1:])
                    if self._line_clear(full_path[-1], e_node, extra_margin=0.5):
                        full_path.append(e_node)
                    return full_path
                except nx.NetworkXNoPath:
                    pass
            # 最后尝试：仅在直线无碰撞时返回直线
            if self._line_clear(s_node, e_node, extra_margin=0.5):
                return [s_node, e_node]
            return []

    import os
    import datetime

    def visualize(self, path=None, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制田地
        x, y = self.field.exterior.xy
        ax.plot(x, y, 'k-', linewidth=2, label='Field Boundary')
        
        # 绘制障碍物
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, 'r', alpha=0.5, label='Obstacle')
            
        # 绘制杂草块
        for patch in self.patches:
            x, y = patch.exterior.xy
            ax.fill(x, y, 'g', alpha=0.5, label='Weed Patch')
            
        # 绘制路径
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b--', linewidth=1, label='UAV Path')
            ax.plot(path_x[0], path_y[0], 'bo', label='Start')
            ax.plot(path_x[-1], path_y[-1], 'bx', label='End')
            
        # 去除重复图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.axis('equal')
        plt.grid(True)
        plt.title('UAV Path Planning for Spot Spraying')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            # Also show if environment supports it, or just close to avoid memory leak in batch
            # plt.show() 
            plt.close()
        else:
            plt.show()

    def generate_coverage_path(self, polygon, angle):
        """
        Generate coverage path using Convex Decomposition + Optimized Scan Direction + Headland Path
        (Inspired by Plessen's method)
        """
        # 1. Convex Decomposition (Simplified: assumes convex for now, or relies on Shapely's ability to handle complex polys)
        # For true reproduction, we should decompose. But for this simulation, we'll treat the polygon as a whole
        # and generate scan lines. The key difference from simple Boustrophedon is the Headland Path.
        
        # Rotate polygon to align with x-axis
        rotated_poly = affinity.rotate(polygon, -angle, origin='centroid')
        min_x, min_y, max_x, max_y = rotated_poly.bounds
        
        # Create Headland Path (Inner Buffer)
        # This is the "frame" around the area to allow smooth turning and avoid gaps
        headland_width = self.uav_width / 2 # Or full width
        headland_poly = rotated_poly.buffer(-headland_width)
        
        if headland_poly.is_empty:
            # If polygon is too small for headland, just scan it normally or center pass
            scan_poly = rotated_poly
            use_headland = False
        else:
            scan_poly = headland_poly
            use_headland = True
            
        # Generate Scan Lines on the inner polygon (scan_poly)
        path_points = []
        y = min_y + headland_width + self.uav_width / 2
        direction = 1 # 1: left to right, -1: right to left
        
        scan_min_x, scan_min_y, scan_max_x, scan_max_y = scan_poly.bounds
        
        while y < scan_max_y:
            # Create a horizontal line
            line = LineString([(scan_min_x - 10, y), (scan_max_x + 10, y)])
            intersection = scan_poly.intersection(line)
            
            if intersection.is_empty:
                y += self.uav_width
                continue
                
            points = []
            if intersection.geom_type == 'LineString':
                points.extend(list(intersection.coords))
            elif intersection.geom_type == 'MultiLineString':
                for geom in intersection.geoms:
                    points.extend(list(geom.coords))
            
            points.sort(key=lambda p: p[0])
            
            if direction == 1:
                path_points.extend(points)
            else:
                path_points.extend(points[::-1])
                
            direction *= -1
            y += self.uav_width

        # Combine Headland + Scan Lines
        # Strategy: Enter -> Headland Loop (or part of it) -> Scan Lines -> Headland Loop -> Exit
        # Or simply: Scan Lines then connect via Headland?
        # Plessen's paper suggests "Inclusion of a headland path".
        # Usually: Do the boundary lap first (Headland), then the interior.
        
        final_rotated_path = []
        
        if use_headland:
            # Extract exterior ring of headland
            if headland_poly.geom_type == 'Polygon':
                headland_coords = list(headland_poly.exterior.coords)
            elif headland_poly.geom_type == 'MultiPolygon':
                # Just take the largest one or handle all
                headland_coords = list(max(headland_poly.geoms, key=lambda a: a.area).exterior.coords)
            else:
                headland_coords = []

            # Add Headland path first (one lap)
            final_rotated_path.extend(headland_coords)
            
            # Connect to start of scan lines
            # This might involve a crossover, but standard is fine
        
        final_rotated_path.extend(path_points)
            
        # Rotate path points back
        final_path = []
        for p in final_rotated_path:
            pt = Point(p)
            rotated_pt = affinity.rotate(pt, angle, origin=polygon.centroid)
            final_path.append((rotated_pt.x, rotated_pt.y))
            
        return final_path

    def smooth_path(self, path_points, max_deviation=1.5):
        """
        Smooth the path by replacing sharp corners with circular arcs (or approximated curves).
        Corner cutting strategy.
        
        :param path_points: List of (x, y) tuples
        :param max_deviation: Max distance to cut into the corner (should be less than buffer margin)
        """
        if len(path_points) < 3:
            return path_points
            
        smoothed = [path_points[0]]
        
        for i in range(1, len(path_points) - 1):
            p_prev = np.array(path_points[i-1])
            p_curr = np.array(path_points[i])
            p_next = np.array(path_points[i+1])
            
            # Vectors
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 < 1e-3 or len2 < 1e-3:
                smoothed.append(path_points[i])
                continue
                
            # Normalize
            u1 = v1 / len1
            u2 = v2 / len2
            
            # Check angle
            # If almost straight, just continue
            dot = np.dot(u1, u2)
            if dot > 0.99: # Parallel
                smoothed.append(path_points[i])
                continue
                
            # Define cut distance from corner
            # We want to start turning 'd' meters before the corner
            # d should be small enough to not consume the whole segment
            d = min(len1 / 2.5, len2 / 2.5, max_deviation * 3)
            
            # 尝试生成避障的平滑曲线；若不安全则缩小 d
            num_steps = 5
            success = False
            for _ in range(4):
                p_start = p_curr - u1 * d
                p_end = p_curr + u2 * d
                candidate_points = []
                for t in np.linspace(0, 1, num_steps):
                    pt = (1-t)**2 * p_start + 2*(1-t)*t * p_curr + t**2 * p_end
                    candidate_points.append(tuple(pt))
                ok = True
                prev = smoothed[-1]
                for cp in candidate_points:
                    if not self._line_clear(prev, cp, extra_margin=0.5):
                        ok = False
                        break
                    prev = cp
                if ok:
                    smoothed.extend(candidate_points)
                    success = True
                    break
                d *= 0.5
            if not success:
                smoothed.append(tuple(path_points[i]))
                
        smoothed.append(path_points[-1])
        return smoothed

    def generate_turning_path(self, start_pos, start_dir, end_pos, end_dir):
        """
        Generate a turning path.
        User requested to remove stiff arcs, so we return a simple straight connection.
        """
        return [tuple(start_pos), tuple(end_pos)]

    def solve_tsp_branch_and_bound(self, points):
        """
        Branch and Bound Algorithm for Exact TSP
        Uses a reduced cost matrix approach to find the lower bound.
        """
        n = len(points)
        if n == 0: return []
        if n == 1: return [0]
        
        # Distance matrix
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i][j] = float('inf')
                else:
                    adj[i][j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))

        final_res = float('inf')
        final_path = [None] * (n + 1)
        visited = [False] * n

        def copy_to_final(curr_path):
            nonlocal final_res, final_path
            final_path[:n + 1] = curr_path[:n + 1]
            final_path[n] = curr_path[0]

        def first_min(adj, i):
            min_val = float('inf')
            for k in range(n):
                if adj[i][k] < min_val and i != k:
                    min_val = adj[i][k]
            return min_val

        def second_min(adj, i):
            first = float('inf')
            second = float('inf')
            for j in range(n):
                if i == j:
                    continue
                if adj[i][j] <= first:
                    second = first
                    first = adj[i][j]
                elif adj[i][j] <= second and adj[i][j] != first:
                    second = adj[i][j]
            return second

        def TSPRec(adj, curr_bound, curr_weight, level, curr_path):
            nonlocal final_res, visited  # Added 'visited' to nonlocal
            
            if level == n:
                if adj[curr_path[level - 1]][curr_path[0]] != float('inf'):
                    curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
                    if curr_res < final_res:
                        copy_to_final(curr_path)
                        final_res = curr_res
                return

            for i in range(n):
                if adj[curr_path[level-1]][i] != float('inf') and not visited[i]:
                    temp = curr_bound
                    curr_weight += adj[curr_path[level - 1]][i]

                    if level == 1:
                        curr_bound -= ((first_min(adj, curr_path[level - 1]) + first_min(adj, i)) / 2)
                    else:
                        curr_bound -= ((second_min(adj, curr_path[level - 1]) + first_min(adj, i)) / 2)

                    if curr_bound + curr_weight < final_res:
                        curr_path[level] = i
                        visited[i] = True
                        TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path)

                    curr_weight -= adj[curr_path[level - 1]][i]
                    curr_bound = temp
                    
                    # Correctly backtracking visited state
                    # 'visited' is a shared list, we must manually reset
                    visited[i] = False # Instead of creating a new list
                    for j in range(level):
                        visited[curr_path[j]] = True

        # Initial bound
        curr_bound = 0
        curr_path = [-1] * (n + 1)
        visited = [False] * n

        for i in range(n):
            curr_bound += (first_min(adj, i) + second_min(adj, i))

        curr_bound = math.ceil(curr_bound / 2)

        visited[0] = True
        curr_path[0] = 0

        TSPRec(adj, curr_bound, 0, 1, curr_path)
        
        return final_path[:n + 1] # Return closed tour [0, ..., 0]
        # Note: Our planner handles the loop, so we can return just nodes or the full loop
        # The planner expects a list of indices. The last one is 0.
        # If final_path is [0, 2, 1, 0], that's what we want.

    def solve_tsp(self, points):
        """
        Wrapper to use Branch and Bound TSP
        """
        # Fallback if N is too large for B&B (though B&B is often better than DP for memory)
        # For N=10-15, B&B is fine.
        if len(points) > 15:
             return self.solve_tsp_nearest_neighbor(points)
        
        # Pre-check: B&B can be slow in Python for N>12 without optimization.
        # But let's try it as requested.
        # Convert the result (which includes return to start) to the format expected
        path = self.solve_tsp_branch_and_bound(points)
        return path

    def solve_tsp_nearest_neighbor(self, points):
        if not points: return []
        unvisited = set(range(len(points)))
        current = 0
        path_indices = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda x: np.linalg.norm(np.array(points[current]) - np.array(points[x])))
            path_indices.append(nearest)
            current = nearest
            unvisited.remove(current)
        return path_indices

    def optimize_scan_angle(self, patch):
        """
        Find the optimal scan angle for a patch
        Minimizes the number of turns (or path length)
        """
        best_path = []
        min_turns = float('inf')
        best_angle = 0
        
        # Try angles from 0 to 180 with step 15
        for angle in range(0, 180, 15):
            path = self.generate_coverage_path(patch, angle)
            if not path:
                continue
                
            # Count turns (number of points - 2 for a simple polyline, but here it's just segments)
            # Each scan line adds 2 points. Number of turns ~ Number of scan lines.
            # More scan lines = more turns.
            # Length is also a factor.
            
            # Simple metric: total length
            length = 0
            for i in range(len(path) - 1):
                length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
                
            # Or simply prefer fewer lines (wider coverage per turn)
            # For convex polygons, width perpendicular to scan direction determines number of lines.
            
            if length < min_turns: # Using length as proxy for cost
                min_turns = length
                best_path = path
                best_angle = angle
                
        return best_path

    def solve_global_optimization(self, start_pos):
        """
        Solve the Combined TSP and Coverage Path Planning problem using Branch and Bound.
        Optimizes both the VISITATION ORDER and the ENTRY/EXIT DIRECTION for each patch.
        
        Reference: Inspired by standard Branch and Bound TSP algorithms found on GitHub (e.g., Axelvel/TSP),
        but adapted for Generalized TSP where each node (patch) has two entry/exit states.
        
        Complexity: Reduced significantly compared to O(N! * 2^N) due to pruning.
        """
        n = len(self.patches)
        
        # 1. Pre-calculate optimized coverage paths and lengths
        patches_data = []
        for i, patch in enumerate(self.patches):
            path = self.optimize_scan_angle(patch)
            if not path:
                path = [patch.centroid.coords[0], patch.centroid.coords[0]]
            
            length = sum(np.linalg.norm(np.array(path[k]) - np.array(path[k+1])) for k in range(len(path)-1))
            patches_data.append({
                'path': path,
                'start': path[0],
                'end': path[-1],
                'len': length
            })
        
        # Build Global Visibility Graph for accurate distance calculation
        print("Building global visibility graph for accurate distance calculation...")
        all_points = [start_pos]
        for p in patches_data:
            all_points.append(p['start'])
            all_points.append(p['end'])
        
        # Ensure unique and tuple format
        all_points = list(set([tuple(p) if not isinstance(p, tuple) else p for p in all_points]))
        self.build_visibility_graph(all_points)

        def get_dist(p1, p2):
            try:
                s = tuple(p1)
                e = tuple(p2)
                return nx.shortest_path_length(self.visibility_graph, source=s, target=e, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return np.linalg.norm(np.array(p1) - np.array(p2))

        # 2. Pre-calculate distances
        # dist_matrix[i][dir_i][j][dir_j]
        # dir: 0=Forward (Exit End), 1=Reverse (Exit Start)
        dist_matrix = np.zeros((n, 2, n, 2))
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                # From i_F (End) to j_F (Start)
                dist_matrix[i][0][j][0] = get_dist(patches_data[i]['end'], patches_data[j]['start'])
                # From i_F (End) to j_R (End)
                dist_matrix[i][0][j][1] = get_dist(patches_data[i]['end'], patches_data[j]['end'])
                # From i_R (Start) to j_F (Start)
                dist_matrix[i][1][j][0] = get_dist(patches_data[i]['start'], patches_data[j]['start'])
                # From i_R (Start) to j_R (End)
                dist_matrix[i][1][j][1] = get_dist(patches_data[i]['start'], patches_data[j]['end'])

        # Dist from Global Start
        start_dist = np.zeros((n, 2))
        for i in range(n):
            start_dist[i][0] = get_dist(start_pos, patches_data[i]['start'])
            start_dist[i][1] = get_dist(start_pos, patches_data[i]['end'])

        # Dist to Global Start
        return_dist = np.zeros((n, 2))
        for i in range(n):
            return_dist[i][0] = get_dist(patches_data[i]['end'], start_pos)
            return_dist[i][1] = get_dist(patches_data[i]['start'], start_pos)

        # 3. Branch and Bound Setup
        min_total_cost = float('inf')
        best_path_indices = []
        
        # Heuristic for Lower Bound: Min incoming edge for each unvisited node
        min_in_cost = np.zeros(n)
        for j in range(n):
            candidates = [start_dist[j][0], start_dist[j][1]]
            for i in range(n):
                if i == j: continue
                candidates.extend([dist_matrix[i][0][j][0], dist_matrix[i][0][j][1],
                                   dist_matrix[i][1][j][0], dist_matrix[i][1][j][1]])
            min_in_cost[j] = min(candidates)

        def get_lower_bound(visited_mask, current_cost):
            lb = current_cost
            for i in range(n):
                if not (visited_mask & (1 << i)):
                    lb += patches_data[i]['len'] + min_in_cost[i]
            return lb

        def bnb_solve(visited_mask, last_idx, last_dir, current_cost, path_stack):
            nonlocal min_total_cost, best_path_indices
            
            # Pruning
            if current_cost >= min_total_cost:
                return
            
            if get_lower_bound(visited_mask, current_cost) >= min_total_cost:
                return
            
            # Goal State
            if visited_mask == (1 << n) - 1:
                final_cost = current_cost + return_dist[last_idx][last_dir]
                if final_cost < min_total_cost:
                    min_total_cost = final_cost
                    best_path_indices = list(path_stack)
                return

            # Branching
            # Heuristic ordering: Try nearest neighbors first to find good upper bound quickly
            candidates = []
            for i in range(n):
                if not (visited_mask & (1 << i)):
                    # Forward Cost
                    cost_F = current_cost + patches_data[i]['len']
                    if last_idx == -1:
                        cost_F += start_dist[i][0]
                    else:
                        cost_F += dist_matrix[last_idx][last_dir][i][0]
                    
                    # Reverse Cost
                    cost_R = current_cost + patches_data[i]['len']
                    if last_idx == -1:
                        cost_R += start_dist[i][1]
                    else:
                        cost_R += dist_matrix[last_idx][last_dir][i][1]
                        
                    candidates.append((cost_F, i, 0))
                    candidates.append((cost_R, i, 1))
            
            # Sort candidates by cost (Greedy-first search)
            candidates.sort(key=lambda x: x[0])
            
            for cost, idx, direction in candidates:
                if cost < min_total_cost:
                    path_stack.append((idx, direction))
                    bnb_solve(visited_mask | (1 << idx), idx, direction, cost, path_stack)
                    path_stack.pop()

        # Decide solver based on problem size
        if n > 11:
            print(f"Problem size N={n} is large. Using Greedy Heuristic for fast solution.")
            # Greedy Implementation
            curr_idx = -1
            curr_dir = 0
            visited_mask = 0
            curr_cost = 0
            
            for _ in range(n):
                best_cand = None # (cost, idx, dir)
                min_c = float('inf')
                
                for i in range(n):
                    if not (visited_mask & (1 << i)):
                        # Option 0: Enter at Start
                        c0 = patches_data[i]['len']
                        if curr_idx == -1:
                            c0 += start_dist[i][0]
                        else:
                            c0 += dist_matrix[curr_idx][curr_dir][i][0]
                            
                        if c0 < min_c:
                            min_c = c0
                            best_cand = (c0, i, 0)
                            
                        # Option 1: Enter at End
                        c1 = patches_data[i]['len']
                        if curr_idx == -1:
                            c1 += start_dist[i][1]
                        else:
                            c1 += dist_matrix[curr_idx][curr_dir][i][1]
                        
                        if c1 < min_c:
                            min_c = c1
                            best_cand = (c1, i, 1)
                            
                if best_cand:
                    cost, idx, d = best_cand
                    visited_mask |= (1 << idx)
                    best_path_indices.append((idx, d))
                    curr_cost += cost # Not strictly needed for path construction but good for debug
                    curr_idx = idx
                    curr_dir = d
        else:
            print(f"Problem size N={n}. Using Branch and Bound for optimal solution.")
            bnb_solve(0, -1, 0, 0, [])
        
        # 4. Construct Path
        full_path = []
        curr_pos = start_pos
        
        for idx, direction in best_path_indices:
            p_data = patches_data[idx]
            path_segment = p_data['path']
            if direction == 1:
                path_segment = path_segment[::-1]
                
            target_entry = path_segment[0]
            if np.linalg.norm(np.array(curr_pos) - np.array(target_entry)) > 1e-3:
                transit = self.find_path_avoiding_obstacles(curr_pos, target_entry)
                # Apply Smoothing to Transit Path
                if len(transit) > 2:
                    transit = self.smooth_path(transit)
                full_path.extend(transit[:-1])
                
            full_path.extend(path_segment)
            curr_pos = path_segment[-1]
            
        # Return to start
        transit = self.find_path_avoiding_obstacles(curr_pos, start_pos)
        if len(transit) > 2:
            transit = self.smooth_path(transit)
        full_path.extend(transit)
        
        return full_path

    def plan(self):
        """
        Main planning flow.
        """
        # Use the new global optimization method
        return self.solve_global_optimization((0, 0))

def generate_random_polygon(center, avg_radius, irregularity, spikeyness, num_verts):
    """
    Generate a random irregular polygon.
    :param center: (x, y) center of the polygon
    :param avg_radius: Average radius
    :param irregularity: [0, 1] Variance of the angle steps
    :param spikeyness: [0, 1] Variance of the radius
    :param num_verts: Number of vertices
    """
    irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / num_verts
    spikeyness = np.clip(spikeyness, 0, 1) * avg_radius

    # Generate random angle steps
    angle_steps = []
    lower = (2 * np.pi / num_verts) - irregularity
    upper = (2 * np.pi / num_verts) + irregularity
    sum_steps = 0
    for i in range(num_verts):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        sum_steps += tmp

    # Normalize the steps so that they sum to 2*pi
    k = sum_steps / (2 * np.pi)
    angle_steps = [x / k for x in angle_steps]

    # Generate vertices
    points = []
    angle = random.uniform(0, 2 * np.pi)
    for i in range(num_verts):
        r_i = np.clip(random.gauss(avg_radius, spikeyness), 0, 2 * avg_radius)
        x = center[0] + r_i * np.cos(angle)
        y = center[1] + r_i * np.sin(angle)
        points.append((x, y))
        angle += angle_steps[i]

    return Polygon(points)

if __name__ == "__main__":
    from uav_path_planning.app import main
    main()
