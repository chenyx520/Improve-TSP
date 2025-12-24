import math
import numpy as np
import networkx as nx
import datetime
from shapely.geometry import Point

from .visibility import build_visibility_graph
from .coverage import generate_coverage_path, optimize_scan_angle
from .tsp_solver import nearest_neighbor_order, two_opt, orientation_dp, branch_and_bound_order

class UAVPathPlanning:
    def __init__(self, field_bounds, patches, obstacles, uav_width=1.0, turning_radius=1.0):
        self.field = Point(0,0).buffer(1.0)
        self.field = self.field.envelope
        self.field = self.field.from_bounds(field_bounds[0][0], field_bounds[0][1], field_bounds[2][0], field_bounds[2][1])
        self.patches = patches
        self.obstacles = obstacles
        self.uav_width = uav_width
        self.turning_radius = turning_radius
        self.visibility_graph = None

    def _inflated_obstacles(self, extra=0.0):
        # Increased base margin for safety in visibility graph
        margin = self.uav_width / 2 + 1.5 + extra
        # Use lower resolution and simplification for speed
        return [obs.buffer(margin, resolution=2).simplify(0.5, preserve_topology=True) for obs in self.obstacles]

    def _line_clear(self, p1, p2, extra_margin=0.0):
        from shapely.geometry import LineString
        line = LineString([p1, p2])
        # Check against strictly inflated obstacles
        for obs in self._inflated_obstacles(extra_margin):
            if line.intersects(obs) and not line.touches(obs):
                return False
        return True

    def build_visibility_graph(self, points_of_interest):
        G = build_visibility_graph(self.obstacles, self.uav_width, points_of_interest)
        self.visibility_graph = G
        return G

    def find_path_avoiding_obstacles(self, start, end):
        if self.visibility_graph is not None:
            s_node = tuple(start) if not isinstance(start, tuple) else start
            e_node = tuple(end) if not isinstance(end, tuple) else end
            if self.visibility_graph.has_node(s_node) and self.visibility_graph.has_node(e_node):
                try:
                    return nx.shortest_path(self.visibility_graph, source=s_node, target=e_node, weight='weight')
                except nx.NetworkXNoPath:
                    pass
        s_node = tuple(start) if not isinstance(start, tuple) else start
        e_node = tuple(end) if not isinstance(end, tuple) else end
        try:
            G = self.build_visibility_graph([s_node, e_node])
            return nx.shortest_path(G, source=s_node, target=e_node, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [start, end]

    def visualize(self, path=None, save_path=None):
        import matplotlib.pyplot as plt
        x = [0, self.field.bounds[2], self.field.bounds[2], 0, 0]
        y = [0, 0, self.field.bounds[3], self.field.bounds[3], 0]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, y, 'k-', linewidth=2, label='Field Boundary')
        for obs in self.obstacles:
            xx, yy = obs.exterior.xy
            ax.fill(xx, yy, 'r', alpha=0.5, label='Obstacle')
        for patch in self.patches:
            xx, yy = patch.exterior.xy
            ax.fill(xx, yy, 'g', alpha=0.5, label='Weed Patch')
            
        # Plot decomposition lines if available.
        # We need to re-run coverage generation or store it.
        # Ideally, we should visualize based on the 'patches_data' stored in the planner after planning.
        # But 'plan' method doesn't store patches_data in self.
        # However, for the sake of this visualization call which might be separate, 
        # we can just re-generate them or accept them as argument?
        # The standard visualize call only takes path.
        # Let's modify plan to store patches_data or pass it to visualize if we want to show lines.
        # Or, we can just quickly re-generate them here for visualization purposes.
        # Re-generating is safer to avoid state management issues.
        
        # But wait, optimize_scan_angle might be slow.
        # Let's check if we can access the latest run data.
        # For now, let's just re-generate for the patches to show the lines.
        # It shouldn't take too long for visualization.
        
        # Actually, let's check if we can modify the class to store the last plan data.
        # if hasattr(self, 'last_patches_data'):
        #     for p_data in self.last_patches_data:
        #         if 'decomposition_lines' in p_data:
        #             for line in p_data['decomposition_lines']:
        #                 lx, ly = line.xy
        #                 ax.plot(lx, ly, 'm-', linewidth=2, label='Decomposition Line')
        
        if path:
            # Determine if we have one path or multiple paths
            paths = []
            if len(path) > 0:
                # Check if the first element is a point (tuple/list of numbers) or a path (list of points)
                first_elem = path[0]
                if isinstance(first_elem, list) and len(first_elem) > 0 and isinstance(first_elem[0], (tuple, list)):
                    # List of lists of points -> Multiple paths
                    paths = path
                else:
                    # List of points -> Single path
                    paths = [path]
            
            colors = ['b', 'c', 'm', 'y', 'k', 'orange', 'purple']
            
            for i, p in enumerate(paths):
                if not p: continue
                path_x, path_y = zip(*p)
                color = colors[i % len(colors)]
                label = f'UAV {i+1} Path' if len(paths) > 1 else 'UAV Path'
                ax.plot(path_x, path_y, color=color, linestyle='--', linewidth=1, label=label)
                # Only plot start/end markers once or for each? For each is better for multi-UAV.
                ax.plot(path_x[0], path_y[0], marker='o', color=color, label='Start' if i==0 else "")
                ax.plot(path_x[-1], path_y[-1], marker='x', color=color, label='End' if i==0 else "")

        
        # Deduplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Move legend outside to the right
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.axis('equal')
        plt.grid(True)
        plt.title('UAV Path Planning for Spot Spraying')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def smooth_path(self, path_points, turning_radius=None):
        """
        Smooth the path by replacing sharp corners with circular arcs.
        Uses provided turning_radius or self.turning_radius.
        """
        if len(path_points) < 3:
            return path_points
            
        from shapely.prepared import prep
        from shapely.geometry import LineString
            
        smoothed = [path_points[0]]
        
        # Prepare obstacles for collision check
        # Use slightly larger buffer than geometric requirement to account for discretization
        safe_obstacles = [obs.buffer(self.uav_width/2.0 + 0.5) for obs in self.obstacles]
        safe_obstacles_prep = [prep(obs) for obs in safe_obstacles]
        
        radius = turning_radius if turning_radius is not None else self.turning_radius
        
        for i in range(1, len(path_points) - 1):
            p_prev = np.array(path_points[i-1])
            p_curr = np.array(path_points[i])
            p_next = np.array(path_points[i+1])
            
            # Vectors pointing OUT from corner
            v_in = p_prev - p_curr # Direction to prev
            v_out = p_next - p_curr # Direction to next
            
            len_in = np.linalg.norm(v_in)
            len_out = np.linalg.norm(v_out)
            
            if len_in < 1e-3 or len_out < 1e-3:
                smoothed.append(path_points[i])
                continue
                
            u_in = v_in / len_in
            u_out = v_out / len_out
            
            # Angle between legs
            dot = np.clip(np.dot(u_in, u_out), -1.0, 1.0)
            theta = np.arccos(dot)
            
            # If almost straight (theta ~ 180 deg) or sharp U-turn (theta ~ 0)
            # theta is angle between vectors. 
            # If straight: u_in = -u_out. dot = -1. theta = pi.
            # If U-turn (retracing): u_in = u_out. dot = 1. theta = 0.
            
            # We care about the INTERIOR angle alpha = theta.
            # No, standard definition:
            # Turn angle is pi - theta?
            # Let's use geometry.
            # Tangent distance L = R / tan(theta/2).
            # If theta = pi (straight), tan(pi/2) -> inf. L -> 0. Correct.
            # If theta = pi/2 (90 deg turn), tan(pi/4) = 1. L = R. Correct.
            
            if abs(theta - np.pi) < 1e-2: # Straight
                smoothed.append(path_points[i])
                continue
                
            if theta < 1e-2: # Sharp U-turn (0 degrees)
                # Cannot put a circle inside 0 degree corner (infinite distance)
                smoothed.append(path_points[i])
                continue
                
            # Calculate required tangent distance
            tan_half = np.tan(theta / 2.0)
            if abs(tan_half) < 1e-6:
                L_req = 0 # Should not happen given check above
            else:
                L_req = radius / tan_half
            
            # Check availability
            # We can use at most half of the leg length
            L_avail = min(len_in, len_out) / 2.0 * 0.99 # 0.99 for safety
            
            # Adaptive Radius Logic for Obstacle Avoidance
            # We assume the path point (corner) is at least 'margin' distance away from the actual obstacle
            # because the Visibility Graph uses a larger inflation (uav_width/2 + 1.0)
            # than the collision check buffer (uav_width/2 + 0.1).
            # Margin = 1.0 - 0.1 = 0.9 meters.
            margin = 0.9
            
            # Calculate maximum cut-in distance allowed: d_cut = R * (csc(theta/2) - 1)
            # We need d_cut < margin => R < margin / (csc(theta/2) - 1)
            sin_half = np.sin(theta / 2.0)
            if sin_half < 1e-3:
                # theta ~ 0 (U-turn), max_R -> 0
                max_safe_R = 0
            elif abs(sin_half - 1.0) < 1e-3:
                # theta ~ pi (Straight), max_R -> inf
                max_safe_R = float('inf')
            else:
                csc_half = 1.0 / sin_half
                if csc_half - 1.0 < 1e-6:
                     max_safe_R = float('inf')
                else:
                    max_safe_R = margin / (csc_half - 1.0)
            
            # Determine actual radius
            # Must satisfy:
            # 1. R <= target_radius
            # 2. Tangent length L <= L_avail
            # 3. Cut-in depth <= margin (max_safe_R)
            
            # Constraint from L_avail: L = R / tan(theta/2) => R_max_L = L_avail * tan(theta/2)
            tan_half = np.tan(theta / 2.0)
            R_max_L = L_avail * tan_half if tan_half > 1e-6 else 0
            
            actual_radius = min(radius, max_safe_R, R_max_L)
            
            # If radius is too small, just keep the corner
            if actual_radius < 0.1:
                smoothed.append(path_points[i])
                continue
                
            # Re-calculate tangent length with constrained radius
            L = actual_radius / tan_half if tan_half > 1e-6 else 0
                
            # Calculate tangent points
            p_start = p_curr + L * u_in
            p_end = p_curr + L * u_out
            
            # Calculate Center
            # Bisector direction
            bisector = u_in + u_out
            bisector_len = np.linalg.norm(bisector)
            if bisector_len < 1e-6:
                 smoothed.append(path_points[i])
                 continue
                 
            u_bisector = bisector / bisector_len
            dist_center = actual_radius / sin_half
            
            center = p_curr + dist_center * u_bisector
            
            # Generate Arc Points
            v_start = p_start - center
            v_end = p_end - center
            
            ang_start = np.arctan2(v_start[1], v_start[0])
            ang_end = np.arctan2(v_end[1], v_end[0])
            
            # Shortest arc logic
            diff = ang_end - ang_start
            while diff <= -np.pi: diff += 2*np.pi
            while diff > np.pi: diff -= 2*np.pi
            
            # Increased resolution for smoothness
            num_steps = max(5, int(abs(diff) * actual_radius * 20)) # 20 points per meter-radian
            
            arc_points = []
            for k in range(num_steps + 1):
                ang = ang_start + diff * (k / num_steps)
                pt = center + actual_radius * np.array([np.cos(ang), np.sin(ang)])
                arc_points.append(tuple(pt))
                
            # Collision Check (Final Safeguard)
            collision = False
            for pt in arc_points:
                point_geom = Point(pt)
                for obs_prep in safe_obstacles_prep:
                    if obs_prep.contains(point_geom):
                        collision = True
                        break
                if collision:
                    break
            
            if not collision:
                smoothed.extend(arc_points)
            else:
                # If colliding, try a smaller radius (0.5x) recursively or just fallback to line
                # Here we fallback to line, but ensure line itself is safe (it should be from previous steps)
                # But strictly speaking, the corner point p_curr might be close to obstacle.
                smoothed.append(path_points[i])

                
        smoothed.append(path_points[-1])
        
        return smoothed

    def get_boundary_samples(self, polygon, num_samples_per_edge=3):
        """
        Generate candidate entry/exit points along the polygon boundary.
        Returns a list of tuples: (point_coords, edge_vector, edge_length)
        """
        candidates = []
        if polygon is None or polygon.is_empty:
            return candidates
            
        if polygon.geom_type == 'MultiPolygon':
            polys = list(polygon.geoms)
        else:
            polys = [polygon]
            
        for poly in polys:
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i+1])
                edge_vec = p2 - p1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-3:
                    continue
                    
                # Add vertices
                candidates.append((coords[i], edge_vec / edge_len, edge_len))
                
                # Add intermediate samples
                for t in np.linspace(0.1, 0.9, num_samples_per_edge):
                    pt = p1 + t * edge_vec
                    candidates.append((tuple(pt), edge_vec / edge_len, edge_len))
                
        return candidates

    def get_full_headland_loop(self, headland_poly, start_pt):
        """
        Return the full headland ring path starting and ending at start_pt.
        Handles MultiPolygon by finding the component containing start_pt.
        """
        if headland_poly is None or headland_poly.is_empty:
            return [start_pt]
            
        target_poly = None
        if headland_poly.geom_type == 'MultiPolygon':
            # Find which polygon start_pt belongs to (on boundary)
            pt = Point(start_pt)
            best_dist = float('inf')
            for poly in headland_poly.geoms:
                dist = poly.distance(pt)
                if dist < best_dist:
                    best_dist = dist
                    target_poly = poly
            
            # If start_pt is not close to any (shouldn't happen if selected from samples), pick first
            if target_poly is None:
                target_poly = headland_poly.geoms[0]
        else:
            target_poly = headland_poly
            
        ring = target_poly.exterior
        total_len = ring.length
        if total_len < 1e-3:
            return [start_pt]
            
        start_d = ring.project(Point(start_pt))
        
        # We want to go from start_d to start_d + total_len
        # Re-use get_arc logic or simplified version
        coords = []
        
        # 1. Add start point
        coords.append(ring.interpolate(start_d).coords[0])
        
        # 2. Add all vertices in order, wrapping around
        # Find index of vertex after start_d
        
        # Simplify: Just get all coords, rotate list to start after start_d, append start_d
        ring_coords = list(ring.coords)[:-1] # Remove duplicate end
        if not ring_coords:
            return [start_pt]
            
        # Find segment containing start_d
        idx = -1
        curr_d = 0
        found_idx = -1
        
        for i in range(len(ring_coords)):
            p1 = ring_coords[i]
            p2 = ring_coords[(i+1) % len(ring_coords)]
            seg_len = np.linalg.norm(np.array(p2) - np.array(p1))
            
            if curr_d <= start_d and (curr_d + seg_len) > start_d:
                found_idx = i
                break
            curr_d += seg_len
            
        if found_idx == -1:
            found_idx = 0 # Should not happen
            
        # Reconstruct sequence:
        # start_pt -> vertex[found_idx+1] -> ... -> vertex[found_idx] -> start_pt
        
        # Vertices starting from found_idx + 1
        ordered_verts = []
        for k in range(1, len(ring_coords) + 1):
            idx = (found_idx + k) % len(ring_coords)
            ordered_verts.append(ring_coords[idx])
            
        # Filter: If start_pt is exactly on a vertex, we might duplicate.
        # Check distance
        if np.linalg.norm(np.array(ordered_verts[-1]) - np.array(start_pt)) < 1e-3:
            ordered_verts.pop() # Remove last if same as start
            
        coords.extend(ordered_verts)
        coords.append(ring.interpolate(start_d).coords[0]) # Close the loop
        
        return coords

    def get_headland_path(self, headland_poly, p1, p2):
        """
        Find shortest path along the headland ring between p1 and p2.
        headland_poly is the Polygon defining the headland boundary.
        """
        if headland_poly is None or headland_poly.is_empty:
            return [p1, p2]
            
        target_poly = None
        if headland_poly.geom_type == 'MultiPolygon':
            # Check if p1 and p2 are on the same polygon
            pt1 = Point(p1)
            pt2 = Point(p2)
            
            # Find poly for p1
            poly1 = None
            min_d1 = float('inf')
            for p in headland_poly.geoms:
                d = p.distance(pt1)
                if d < min_d1:
                    min_d1 = d
                    poly1 = p
            
            # Find poly for p2
            poly2 = None
            min_d2 = float('inf')
            for p in headland_poly.geoms:
                d = p.distance(pt2)
                if d < min_d2:
                    min_d2 = d
                    poly2 = p
            
            if poly1 != poly2:
                # Different islands, just return direct line (fly over)
                return [p1, p2]
            
            target_poly = poly1
        else:
            target_poly = headland_poly
            
        ring = target_poly.exterior
        total_len = ring.length
        if total_len < 1e-3:
            return [p1, p2]

        d1 = ring.project(Point(p1))
        d2 = ring.project(Point(p2))
        
        if abs(d1 - d2) < 1e-3:
            return [p1]

        # Function to get points between two distances
        def get_arc(start_d, end_d):
            coords = []
            # Add start point
            coords.append(ring.interpolate(start_d).coords[0])
            
            # Add intermediate vertices
            # We need to find vertices that fall between start_d and end_d
            # Vertex distances:
            # Note: ring.coords includes the duplicate start/end point.
            # We iterate through segments.
            
            # Efficient way:
            if start_d < end_d:
                # Simple range
                # Find vertices with distance > start_d and < end_d
                # We can pre-calculate vertex distances or do it on the fly
                curr_d = 0
                for i in range(len(ring.coords) - 1):
                    pt = ring.coords[i]
                    next_pt = ring.coords[i+1]
                    seg_len = np.linalg.norm(np.array(next_pt) - np.array(pt))
                    
                    if curr_d > start_d and curr_d < end_d:
                        coords.append(pt)
                    elif curr_d <= start_d and (curr_d + seg_len) > start_d:
                        # This segment contains start_d, but start_d is added via interpolate
                        # If start_d is exactly a vertex, interpolate handles it.
                        # If start_d is between vertices, we don't add the previous vertex.
                        # We only add vertices strictly inside the range.
                        pass
                        
                    curr_d += seg_len
                    
                # Add end point
                coords.append(ring.interpolate(end_d).coords[0])
                return coords, end_d - start_d
            else:
                # Wrap around is handled by caller splitting into two calls or logic
                return [], float('inf')

        # Path 1: Forward (d1 -> d2)
        # If d1 < d2: just d1->d2
        # If d1 > d2: impossible directly without wrapping? 
        # Wait, ring is a loop. "Forward" means increasing distance.
        # But we can go d1 -> End -> 0 -> d2
        
        # Let's consider the two arcs on the circle.
        # Arc 1: From d1 to d2 (CW or CCW depending on definition)
        # Arc 2: From d1 to d2 (the other way)
        
        # Calculate lengths
        len_forward = (d2 - d1) % total_len
        len_backward = (d1 - d2) % total_len
        
        path_points = []
        
        if len_forward <= len_backward:
            # Go forward from d1 to d2
            if d1 <= d2:
                # Simple: d1 to d2
                pts, _ = get_arc(d1, d2)
                path_points = pts
            else:
                # Wrap: d1 to Total, then 0 to d2
                pts1, _ = get_arc(d1, total_len)
                pts2, _ = get_arc(0, d2)
                # pts1 ends at total_len (which is same as 0). pts2 starts at 0.
                # Avoid duplicate
                path_points = pts1 + pts2[1:]
        else:
            # Go backward (which is forward from d2 to d1, then reversed)
            if d2 <= d1:
                # Simple: d2 to d1
                pts, _ = get_arc(d2, d1)
                path_points = pts[::-1]
            else:
                # Wrap: d2 to Total, then 0 to d1
                pts1, _ = get_arc(d2, total_len)
                pts2, _ = get_arc(0, d1)
                full_pts = pts1 + pts2[1:]
                path_points = full_pts[::-1]
                
        return path_points

    def calculate_adaptability_cost(self, exit_pt, entry_pt, exit_edge_len, exit_edge_vec, scan_angle_deg):
        """
        Calculate cost based on document formula 4.4
        """
        # Distance term
        dist = np.linalg.norm(np.array(exit_pt) - np.array(entry_pt))
        
        # Adaptability term
        # Angle between flight vector and coverage path main direction
        flight_vec = np.array(entry_pt) - np.array(exit_pt)
        flight_len = np.linalg.norm(flight_vec)
        
        if flight_len < 1e-3:
            theta = 0
        else:
            flight_dir = flight_vec / flight_len
            # Main direction is determined by scan angle
            scan_rad = np.radians(scan_angle_deg)
            scan_dir = np.array([np.cos(scan_rad), np.sin(scan_rad)])
            
            # Angle between flight_dir and scan_dir
            # We want alignment, so dot product close to 1 or -1?
            # Document says "angle between ... and full-coverage path's main direction".
            # Usually we want to enter parallel to scan lines?
            # Or perpendicular?
            # If we enter "aligned", we might need to turn 90 deg to start scanning?
            # Let's assume theta is the acute angle.
            dot = np.abs(np.dot(flight_dir, scan_dir))
            theta = np.degrees(np.arccos(min(dot, 1.0)))
            
        # Formula 4.4 logic
        # L_exit > 2W ?
        # Assuming W = uav_width
        is_long_enough = exit_edge_len > 2 * self.uav_width
        
        c_adapt = 0.9
        if is_long_enough:
            if theta <= 30:
                c_adapt = 0.1
            elif theta <= 60:
                c_adapt = 0.5
                
        # Total cost (Bi-objective)
        # alpha = 0.6, beta = 0.4 (from document)
        # We need to normalize distance. Let's just use raw weighted sum for relative comparison.
        # Or normalize by some large factor e.g. 100m
        cost = 0.6 * dist + 0.4 * c_adapt * 100 # Scaling c_adapt to be comparable to meters
        return cost

    def solve_global_optimization(self, start_pos, patches=None):
        target_patches = patches if patches is not None else self.patches
        n = len(target_patches)
        if n == 0:
            return []
            
        patches_data = []
        for patch in target_patches:
            # Pass obstacles to enable convex decomposition if needed
            result = optimize_scan_angle(patch, self.uav_width, self.obstacles)
            
            # Unpack result
            path = result['path'] # Combined path (Headland + Scan)
            scan_path = result['scan_path']
            headland_poly = result['headland_poly']
            decomposition_lines = result.get('decomposition_lines', [])
            
            if not path:
                # Fallback
                path = [patch.centroid.coords[0], patch.centroid.coords[0]]
                scan_path = path
                
            length = sum(np.linalg.norm(np.array(path[k]) - np.array(path[k+1])) for k in range(len(path)-1))
            
            # Identify Scan Start/End (for orientation DP)
            # If scan_path is empty, use centroid
            if scan_path:
                s_start = scan_path[0]
                s_end = scan_path[-1]
            else:
                s_start = path[0]
                s_end = path[-1]
                
            patches_data.append({
                'path': path, # Original full path (fallback)
                'scan_path': scan_path,
                'headland_poly': headland_poly,
                'decomposition_lines': decomposition_lines,
                'start': s_start, # Used for TSP heuristics
                'end': s_end,     # Used for TSP heuristics
                'len': length,
                'centroid': patch.centroid
            })
            
        self.last_patches_data = patches_data # Store for visualization
        
        all_points = [start_pos]
        for p in patches_data:
            all_points.append(p['start'])
            all_points.append(p['end'])
        all_points = list(set([tuple(p) if not isinstance(p, tuple) else p for p in all_points]))
        
        use_euclidean_tsp = n > 20
        # Always build visibility graph for final path generation to avoid rebuilding it for every segment
        self.build_visibility_graph(all_points)
        print("Visibility graph built. Calculating distance matrix...")
        
        def get_dist(p1, p2):
            if use_euclidean_tsp:
                return np.linalg.norm(np.array(p1) - np.array(p2))
            try:
                s = tuple(p1)
                e = tuple(p2)
                return nx.shortest_path_length(self.visibility_graph, source=s, target=e, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return np.linalg.norm(np.array(p1) - np.array(p2))
                
        dist_matrix = np.zeros((n, 2, n, 2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist_matrix[i][0][j][0] = get_dist(patches_data[i]['end'], patches_data[j]['start'])
                dist_matrix[i][0][j][1] = get_dist(patches_data[i]['end'], patches_data[j]['end'])
                dist_matrix[i][1][j][0] = get_dist(patches_data[i]['start'], patches_data[j]['start'])
                dist_matrix[i][1][j][1] = get_dist(patches_data[i]['start'], patches_data[j]['end'])
        start_dist = np.zeros((n, 2))
        for i in range(n):
            start_dist[i][0] = get_dist(start_pos, patches_data[i]['start'])
            start_dist[i][1] = get_dist(start_pos, patches_data[i]['end'])
        return_dist = np.zeros((n, 2))
        for i in range(n):
            return_dist[i][0] = get_dist(patches_data[i]['end'], start_pos)
            return_dist[i][1] = get_dist(patches_data[i]['start'], start_pos)
            
        best_path_indices = []
        
        # Decide TSP method based on N
        if n <= 12:
            print(f"Using Exact B&B (LP) for TSP (N={n})...")
            timeout = None
            method_name = "Exact B&B (LP)"
        else:
            print(f"Using Time-Limited B&B (LP) for TSP (N={n}, Timeout=60s)...")
            timeout = 60.0
            method_name = "Time-Limited B&B (LP)"
            
        centroids = [tuple((p['centroid'].x, p['centroid'].y)) for p in patches_data]
        def dist_fn_idx(i, j):
            return get_dist(centroids[i], centroids[j])
        start_cost_fn = lambda i: get_dist(start_pos, centroids[i])
        return_cost_fn = lambda i: get_dist(centroids[i], start_pos)
        
        start_time_tsp = datetime.datetime.now()
        best_order = branch_and_bound_order(start_pos, centroids, patches_data, dist_fn_idx, start_cost_fn, return_cost_fn, timeout=timeout)
        tsp_duration = (datetime.datetime.now() - start_time_tsp).total_seconds()
        
        best_path_indices = orientation_dp(best_order, patches_data, dist_matrix, start_dist, return_dist)
        print(f"TSP solved ({method_name}) in {tsp_duration:.2f}s. Constructing final path...")
        
        # --- Entry/Exit Point Optimization (Document Method) ---
        print("Optimizing Entry/Exit points...")
        full_path = []
        curr_pos = start_pos
        
        for k in range(len(best_path_indices)):
            idx, direction = best_path_indices[k]
            p_data = patches_data[idx]
            
            # 1. Determine Scan Path Orientation
            scan_path = p_data['scan_path']
            if direction == 1:
                scan_path = scan_path[::-1]
            
            # Handle empty scan path
            if not scan_path:
                 full_path.extend(self.find_path_avoiding_obstacles(curr_pos, p_data['start']))
                 curr_pos = p_data['start']
                 continue
                 
            s_start = scan_path[0]
            s_end = scan_path[-1]
            
            # 2. Determine Entry Point
            # If we have a headland polygon, we can choose the best entry point
            # Candidates on headland of THIS patch
            entry_pt = s_start # Default
            
            if p_data['headland_poly']:
                # Generate candidates
                candidates = self.get_boundary_samples(p_data['headland_poly'])
                
                # Find best entry
                # Minimize: Dist(curr_pos, cand) + Dist(cand, s_start)
                # Note: This is simplified. Document says project edges of PREV patch to THIS patch.
                # But here curr_pos is fixed (from previous step).
                # So we just find best entry on this patch closest to curr_pos.
                best_entry = s_start
                min_entry_cost = float('inf')
                
                for cand_pt, _, _ in candidates:
                    # Cost = Flight Dist + Headland Travel Dist
                    # Using Euclidean for speed here
                    flight_dist = np.linalg.norm(np.array(curr_pos) - np.array(cand_pt))
                    headland_dist = np.linalg.norm(np.array(cand_pt) - np.array(s_start)) # Approx
                    cost = flight_dist + headland_dist
                    if cost < min_entry_cost:
                        min_entry_cost = cost
                        best_entry = cand_pt
                entry_pt = best_entry
            
            # 3. Move to Entry
            if np.linalg.norm(np.array(curr_pos) - np.array(entry_pt)) > 1e-3:
                transit = self.find_path_avoiding_obstacles(curr_pos, entry_pt)
                # Quadcopter: Use small smoothing (fillet) to avoid sharp visual turns
                if len(transit) > 2:
                    transit = self.smooth_path(transit, turning_radius=0.5)
                full_path.extend(transit[:-1])
                
            # 4. Move from Entry to Scan Start (Headland Travel)
            # Use the newly implemented get_headland_path to follow the ring
            if p_data['headland_poly']:
                # First, fly the full headland loop for coverage (starting from entry)
                headland_loop = self.get_full_headland_loop(p_data['headland_poly'], entry_pt)
                full_path.extend(headland_loop)
                
                # Then transit to scan start
                headland_segment = self.get_headland_path(p_data['headland_poly'], entry_pt, s_start)
                # headland_segment starts with entry_pt. 
                # headland_loop ends with entry_pt.
                # So we can extend headland_segment[1:]
                if len(headland_segment) > 1:
                    full_path.extend(headland_segment[1:])
                elif len(headland_segment) == 1:
                    # Just in case
                    pass
            else:
                full_path.append(entry_pt)
                if np.linalg.norm(np.array(entry_pt) - np.array(s_start)) > 1e-3:
                    full_path.append(s_start) 
                
            # 5. Execute Scan
            # Apply kinematic smoothing to scan path turns (small radius for quadcopter)
            if len(scan_path) > 2:
                # Use smaller turning radius for scan turns to avoid wide loops but remove sharp corners
                scan_path_smoothed = self.smooth_path(scan_path, turning_radius=0.5)
                full_path.extend(scan_path_smoothed)
            else:
                full_path.extend(scan_path)
            
            # 6. Determine Exit Point (for NEXT transition)
            # We look ahead to the next patch to pick the best exit
            next_target_pos = start_pos # Default if last patch
            if k < len(best_path_indices) - 1:
                next_idx, next_dir = best_path_indices[k+1]
                # We don't know the exact entry of next patch yet, but we know its general location (centroid or start)
                # Use Next Scan Start as target proxy
                next_data = patches_data[next_idx]
                next_scan = next_data['scan_path']
                if next_dir == 1:
                     next_target_pos = next_scan[-1] if next_scan else next_data['start']
                else:
                     next_target_pos = next_scan[0] if next_scan else next_data['start']
            
            exit_pt = s_end # Default
            
            if p_data['headland_poly']:
                candidates = self.get_boundary_samples(p_data['headland_poly'])
                best_exit = s_end
                min_exit_cost = float('inf')
                
                for cand_pt, edge_vec, edge_len in candidates:
                    # Cost = Headland Travel Dist + Flight Dist (weighted with adaptability)
                    headland_dist = np.linalg.norm(np.array(s_end) - np.array(cand_pt))
                    
                    # Adaptability Cost to Next Target
                    # We need "scan_angle" of the CURRENT patch for adaptability?
                    # No, Document says "angle between flight vector and full-coverage path's main direction".
                    # Main direction of THIS patch? Or NEXT?
                    # "Angle between exit_i->entry_j and the full-coverage path's main direction (aligned with patch's longest edge)."
                    # It likely refers to the direction we are LEAVING.
                    
                    # Let's use a dummy angle 0 for now or calculate it.
                    # We can pass scan angle if we stored it.
                    # For now, just use 0.
                    adapt_cost = self.calculate_adaptability_cost(cand_pt, next_target_pos, edge_len, edge_vec, 0)
                    
                    cost = headland_dist + adapt_cost
                    if cost < min_exit_cost:
                        min_exit_cost = cost
                        best_exit = cand_pt
                exit_pt = best_exit
            
            # 7. Move from Scan End to Exit
            if p_data['headland_poly']:
                 headland_segment_exit = self.get_headland_path(p_data['headland_poly'], s_end, exit_pt)
                 full_path.extend(headland_segment_exit)
            else:
                 if np.linalg.norm(np.array(s_end) - np.array(exit_pt)) > 1e-3:
                    full_path.append(exit_pt)
                
            curr_pos = exit_pt

        # Final return to start
        transit = self.find_path_avoiding_obstacles(curr_pos, start_pos)
        if len(transit) > 2:
            transit = self.smooth_path(transit, turning_radius=0.5)
        full_path.extend(transit)
        
        return full_path

    def plan_multi_uav(self, num_uavs=3, start_pos=(0, 0)):
        if not self.patches:
            return []
            
        print(f"Planning for {num_uavs} UAVs...")
        
        # 1. K-Means Clustering with Load Balancing (Area-based)
        centroids = np.array([(p.centroid.x, p.centroid.y) for p in self.patches])
        areas = np.array([p.area for p in self.patches])
        n_patches = len(centroids)
        
        if n_patches < num_uavs:
            # More UAVs than patches
            clusters = [[i] for i in range(n_patches)] + [[] for _ in range(num_uavs - n_patches)]
        else:
            # Step A: Initial Standard K-Means
            indices = np.random.choice(n_patches, num_uavs, replace=False)
            means = centroids[indices]
            clusters = [[] for _ in range(num_uavs)]
            
            for _ in range(20): 
                clusters = [[] for _ in range(num_uavs)]
                for i, p in enumerate(centroids):
                    dists = np.linalg.norm(means - p, axis=1)
                    cluster_idx = np.argmin(dists)
                    clusters[cluster_idx].append(i)
                
                new_means = []
                for i in range(num_uavs):
                    if clusters[i]:
                        new_m = np.mean(centroids[clusters[i]], axis=0)
                        new_means.append(new_m)
                    else:
                        new_means.append(means[i])
                
                new_means = np.array(new_means)
                if np.linalg.norm(new_means - means) < 1e-3:
                    means = new_means
                    break
                means = new_means

            # Step B: Load Balancing (Minimize Area Variance)
            # We want each UAV to cover roughly equal area
            print("Optimizing task allocation (Load Balancing)...")
            total_area = sum(areas)
            target_area = total_area / num_uavs
            
            for _ in range(100): # Max iterations for balancing
                cluster_areas = [sum(areas[idx] for idx in c) for c in clusters]
                min_area = min(cluster_areas)
                max_area = max(cluster_areas)
                
                # If balanced enough (diff < 10% of target or < avg single patch area)
                avg_patch_area = total_area / n_patches
                if (max_area - min_area) < max(target_area * 0.1, avg_patch_area):
                    break
                    
                max_cluster_idx = np.argmax(cluster_areas)
                min_cluster_idx = np.argmin(cluster_areas)
                
                # Try to move a patch from max_cluster to min_cluster
                # Heuristic: Pick patch in max_cluster closest to min_cluster centroid
                max_cluster_indices = clusters[max_cluster_idx]
                if not max_cluster_indices: break # Should not happen
                
                min_centroid = means[min_cluster_idx]
                
                best_p_idx = -1
                min_dist_to_target = float('inf')
                
                # We also want to ensure we don't break spatial coherence too much
                # So we pick the one that is CLOSER to min_cluster than its current center?
                # No, just closest to min_cluster is good enough for boundary patches.
                
                for p_idx in max_cluster_indices:
                    p_pos = centroids[p_idx]
                    dist = np.linalg.norm(p_pos - min_centroid)
                    if dist < min_dist_to_target:
                        min_dist_to_target = dist
                        best_p_idx = p_idx
                
                # Move it
                if best_p_idx != -1:
                    clusters[max_cluster_idx].remove(best_p_idx)
                    clusters[min_cluster_idx].append(best_p_idx)
                    
                    # Update means roughly (optional, but helps keep track)
                    if clusters[max_cluster_idx]:
                        means[max_cluster_idx] = np.mean(centroids[clusters[max_cluster_idx]], axis=0)
                    if clusters[min_cluster_idx]:
                        means[min_cluster_idx] = np.mean(centroids[clusters[min_cluster_idx]], axis=0)

                
        # 2. Plan for each cluster
        all_paths = []
        for i in range(num_uavs):
            patch_indices = clusters[i]
            if not patch_indices:
                print(f"UAV {i+1}: No patches assigned.")
                all_paths.append([])
                continue
                
            cluster_patches = [self.patches[idx] for idx in patch_indices]
            print(f"UAV {i+1}: Planning for {len(cluster_patches)} patches...")
            path = self.solve_global_optimization(start_pos, patches=cluster_patches)
            all_paths.append(path)
            
        # 3. CBS Collision Avoidance
        print("Running Conflict-Based Search for Collision Avoidance...")
        final_paths = self.solve_cbs(all_paths, speed=5.0, safety_dist=3.0)
        
        return final_paths

    def solve_cbs(self, initial_paths, speed=5.0, safety_dist=3.0):
        """
        Simplified CBS for continuous paths.
        Resolves conflicts by inserting wait times at the start or segment boundaries.
        Optimized with spatial pre-computation and dynamic delay calculation.
        """
        import heapq
        from shapely.geometry import LineString

        # Helper to convert path to trajectory segments (without time shift)
        # Returns: list of segments, each is dict(start, end, duration, dist, path_idx_from)
        def get_static_segments(path):
            segs = []
            if not path: return segs
            for i in range(len(path) - 1):
                p1 = np.array(path[i])
                p2 = np.array(path[i+1])
                dist = np.linalg.norm(p2 - p1)
                duration = dist / speed
                segs.append({
                    'start': p1,
                    'end': p2,
                    'duration': duration,
                    'dist': dist,
                    'path_idx': i
                })
            return segs

        # Precompute static segments for all UAVs
        static_segments = [get_static_segments(p) for p in initial_paths]

        # Precompute spatial conflicts
        # potential_conflicts[(i, j)] = list of (seg_idx_i, seg_idx_j)
        potential_conflicts = {}
        for i in range(len(initial_paths)):
            for j in range(i + 1, len(initial_paths)):
                potential_conflicts[(i, j)] = []
                segs_i = static_segments[i]
                segs_j = static_segments[j]
                
                # Check bounding box first if many segments
                # For small number of segments (<1000), nested loop is fine
                for idx_i, s_i in enumerate(segs_i):
                    for idx_j, s_j in enumerate(segs_j):
                        # Distance check between two line segments
                        # Simple check: min distance between endpoints
                        # If endpoints are far, segments might still cross
                        # Use shapely for robust check
                        l1 = LineString([s_i['start'], s_i['end']])
                        l2 = LineString([s_j['start'], s_j['end']])
                        if l1.distance(l2) < safety_dist:
                            potential_conflicts[(i, j)].append((idx_i, idx_j))
                            
        # Function to get time intervals of segments given a start delay
        # Returns: list of (t_start, t_end) for each segment
        def get_time_intervals(uav_idx, start_delay):
            intervals = []
            t = start_delay
            for seg in static_segments[uav_idx]:
                intervals.append((t, t + seg['duration']))
                t += seg['duration']
            return intervals

        def detect_conflict(delays):
            # delays is dict {uav_id: start_delay}
            # Only checking start delays for now
            
            # Cache time intervals for current delays
            uav_intervals = {}
            for i in range(len(initial_paths)):
                d = delays.get(i, {}).get(0, 0.0)
                uav_intervals[i] = get_time_intervals(i, d)

            for i in range(len(initial_paths)):
                for j in range(i + 1, len(initial_paths)):
                    # Check precomputed potential spatial conflicts
                    pairs = potential_conflicts.get((i, j), [])
                    
                    for (idx_i, idx_j) in pairs:
                        # Get time intervals
                        t_start_i, t_end_i = uav_intervals[i][idx_i]
                        t_start_j, t_end_j = uav_intervals[j][idx_j]
                        
                        # Check overlap
                        overlap_start = max(t_start_i, t_start_j)
                        overlap_end = min(t_end_i, t_end_j)
                        
                        if overlap_start < overlap_end:
                            # Potential collision in time
                            # Do detailed check: interpolate positions
                            # Use fine-grained stepping for robustness
                            # Step size: 0.1s (sufficient for 5m/s speed and 3m safety dist)
                            # Max movement in 0.1s is 0.5m. Relative max speed 10m/s -> 1m.
                            # Safety dist 3m -> 0.1s is safe.
                            
                            t = overlap_start
                            while t <= overlap_end:
                                # Pos i
                                ratio_i = (t - t_start_i) / (t_end_i - t_start_i) if t_end_i > t_start_i else 0
                                seg_i = static_segments[i][idx_i]
                                pos_i = seg_i['start'] + ratio_i * (seg_i['end'] - seg_i['start'])
                                
                                # Pos j
                                ratio_j = (t - t_start_j) / (t_end_j - t_start_j) if t_end_j > t_start_j else 0
                                seg_j = static_segments[j][idx_j]
                                pos_j = seg_j['start'] + ratio_j * (seg_j['end'] - seg_j['start'])
                                
                                # Check Depot Exception
                                if np.linalg.norm(pos_i) < 5.0 and np.linalg.norm(pos_j) < 5.0:
                                    t += 0.1
                                    continue # Ignore depot

                                if np.linalg.norm(pos_i - pos_j) < safety_dist:
                                    # Collision!
                                    # Estimate duration of this collision for better branching
                                    # Heuristic: overlap_end - overlap_start + margin
                                    duration = overlap_end - overlap_start + 1.0 # 1s margin
                                    return (i, j, t, duration)
                                
                                t += 0.1
            
            return None

        # CBS Priority Queue
        # Item: (cost, node_id, delays)
        # delays: dict {uav_id: {0: start_delay}}
        
        root_delays = {i: {0: 0.0} for i in range(len(initial_paths))}
        root_cost = 0 # Could be max time
        
        queue = []
        heapq.heappush(queue, (root_cost, 0, root_delays))
        
        node_id = 0
        max_iter = 1000 # Increased iteration limit as checks are faster
        
        while queue and node_id < max_iter:
            cost, _, current_delays = heapq.heappop(queue)
            node_id += 1
            
            conflict = detect_conflict(current_delays)
            
            if conflict is None:
                print("CBS: Solution found!")
                final_paths = []
                for i in range(len(initial_paths)):
                    d = current_delays[i].get(0, 0.0)
                    print(f"UAV {i+1} Start Delay: {d:.2f}s")
                    
                    path = initial_paths[i]
                    if d > 0.1 and path:
                        # Prepend wait points (assuming 5 pts/sec for visibility/simulation)
                        # This helps visualize the delay in some viewers, though geometrically it's a static point
                        n_wait = int(d * 5)
                        wait_points = [path[0]] * n_wait
                        final_paths.append(wait_points + path)
                    else:
                        final_paths.append(path)
                return final_paths
            
            uav1, uav2, t_conf, duration = conflict
            # print(f"Conflict: UAV {uav1+1} & {uav2+1} at t={t_conf:.1f}, dur={duration:.1f}")
            
            # Branching
            # Branch 1: Delay UAV 1
            new_delays1 = {u: d.copy() for u, d in current_delays.items()}
            new_delays1[uav1][0] = new_delays1[uav1].get(0, 0.0) + duration
            
            # Re-calc cost (makespan)
            # Max end time
            max_time1 = 0
            for i in range(len(initial_paths)):
                path_time = sum(s['duration'] for s in static_segments[i]) + new_delays1[i].get(0, 0.0)
                if path_time > max_time1: max_time1 = path_time
            
            heapq.heappush(queue, (max_time1, node_id * 2 + 1, new_delays1))
            
            # Branch 2: Delay UAV 2
            new_delays2 = {u: d.copy() for u, d in current_delays.items()}
            new_delays2[uav2][0] = new_delays2[uav2].get(0, 0.0) + duration
            
            max_time2 = 0
            for i in range(len(initial_paths)):
                path_time = sum(s['duration'] for s in static_segments[i]) + new_delays2[i].get(0, 0.0)
                if path_time > max_time2: max_time2 = path_time
                
            heapq.heappush(queue, (max_time2, node_id * 2 + 2, new_delays2))
            
        print("CBS: Max iterations reached or no solution.")
        return initial_paths

    def plan(self):
        return self.solve_global_optimization((0, 0))
