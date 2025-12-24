import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely import affinity
from shapely.ops import unary_union

def decompose_polygon_trapezoidal(polygon, angle):
    """
    Decompose a polygon (possibly with holes) into simpler convex/monotone parts
    using a scan-line approach aligned with the given angle.
    Returns a list of Polygons.
    """
    # 1. Rotate polygon to align with Y-axis (scan direction)
    rotated_poly = affinity.rotate(polygon, -angle, origin='centroid')
    
    # 2. Get all vertices from exterior and interiors
    vertices = []
    if rotated_poly.geom_type == 'Polygon':
        polys = [rotated_poly]
    elif rotated_poly.geom_type == 'MultiPolygon':
        polys = rotated_poly.geoms
    else:
        return [rotated_poly] # Should not happen

    all_coords = []
    for p in polys:
        all_coords.extend(list(p.exterior.coords))
        for interior in p.interiors:
            all_coords.extend(list(interior.coords))
            
    # Sort unique Y-coordinates (critical events)
    ys = sorted(list(set([pt[1] for pt in all_coords])))
    
    parts = []
    # Slice the polygon at each Y coordinate (and midpoints for robustness)
    # A robust approach is to slice at midpoints between critical Ys.
    
    if len(ys) < 2:
        return polys
        
    for i in range(len(ys) - 1):
        y_start = ys[i]
        y_end = ys[i+1]
        y_mid = (y_start + y_end) / 2
        
        # Define a horizontal strip
        # We can just intersect the polygon with a box covering this strip
        min_x, _, max_x, _ = rotated_poly.bounds
        strip_box = Polygon([
            (min_x - 1, y_start), (max_x + 1, y_start),
            (max_x + 1, y_end), (min_x - 1, y_end)
        ])
        
        # Intersection
        strip_part = rotated_poly.intersection(strip_box)
        
        if strip_part.is_empty:
            continue
            
        if strip_part.geom_type == 'Polygon':
            parts.append(strip_part)
        elif strip_part.geom_type == 'MultiPolygon':
            for g in strip_part.geoms:
                parts.append(g)
        elif strip_part.geom_type == 'GeometryCollection':
             for g in strip_part.geoms:
                if g.geom_type == 'Polygon':
                    parts.append(g)

    # 3. Rotate parts back? No, generate path on rotated parts then rotate points.
    # Calculate cut lines in rotated frame (horizontal lines at split Ys)
    # We can infer them from shared boundaries of parts, or just collect them here.
    # But doing it here requires tracking intersections.
    # Simpler: Return parts, and let caller figure out boundaries or we just plot parts.
    return parts, rotated_poly.centroid

def merge_parts(parts):
    """
    Merges vertically adjacent parts if they have a 1-to-1 relationship.
    This removes unnecessary horizontal cuts while preserving splits around obstacles.
    """
    if not parts:
        return []
        
    # Iteratively merge until no more merges can be done
    while True:
        adj = build_adjacency_graph(parts)
        merged_pair = None
        
        # Find a mergeable pair
        # A pair (i, j) is mergeable if:
        # 1. They are adjacent
        # 2. i has only j as neighbor on one side (e.g. bottom)
        # 3. j has only i as neighbor on the other side (e.g. top)
        # BUT 'side' is hard to distinguish in generic adjacency.
        # Simplified heuristic: Merge if i and j are each other's ONLY neighbor? 
        # No, i might have other neighbors on the OTHER side.
        # Correct logic:
        # Check shared boundary. If the shared boundary covers the full 'interface' 
        # and there are no other parts sharing that specific interface line.
        
        # Let's use the '1-to-1' heuristic based on degree?
        # No, degree includes neighbors on both sides.
        # We need to know 'above' and 'below' neighbors.
        # Since we are in rotated frame, 'above' means higher Y.
        
        # Build directed graph? or just classify neighbors by Y.
        
        # For each part, classify neighbors as 'above' or 'below' or 'side' (shouldn't be side in slab decomp)
        # Slab decomp slices horizontally. Neighbors are strictly above/below.
        
        neighbors_info = []
        for i, p in enumerate(parts):
            min_y, max_y = p.bounds[1], p.bounds[3]
            above = []
            below = []
            for neighbor_idx in adj[i]:
                np_bounds = parts[neighbor_idx].bounds
                n_min_y, n_max_y = np_bounds[1], np_bounds[3]
                
                # Check relative position
                # If neighbor is above
                if n_min_y >= max_y - 1e-5:
                    above.append(neighbor_idx)
                # If neighbor is below
                elif n_max_y <= min_y + 1e-5:
                    below.append(neighbor_idx)
            
            neighbors_info.append({'above': above, 'below': below})
            
        # Find a pair (i, j) where i is below j, and:
        # i's ONLY above neighbor is j
        # j's ONLY below neighbor is i
        
        for i in range(len(parts)):
            info_i = neighbors_info[i]
            # Check 'above' neighbors
            if len(info_i['above']) == 1:
                j = info_i['above'][0]
                info_j = neighbors_info[j]
                
                # Check if j has i as its ONLY 'below' neighbor
                if len(info_j['below']) == 1 and info_j['below'][0] == i:
                    merged_pair = (i, j)
                    break
        
        if merged_pair:
            i, j = merged_pair
            # Merge i and j
            # Use unary_union to handle potential precision issues
            new_poly = unary_union([parts[i], parts[j]])
            
            # Simplify to remove the internal seam if possible
            # new_poly = new_poly.simplify(1e-6) 
            
            # Construct new list
            new_parts = []
            for k, p in enumerate(parts):
                if k != i and k != j:
                    new_parts.append(p)
            new_parts.append(new_poly)
            
            parts = new_parts
        else:
            break
            
    return parts

def get_decomposition_lines(parts, origin, angle):
    """
    Identify shared boundaries (cut lines) between parts.
    Returns a list of LineStrings in the ORIGINAL coordinate frame.
    """
    lines = []
    # Check all pairs
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1 = parts[i]
            p2 = parts[j]
            if p1.distance(p2) < 1e-5:
                intersection = p1.intersection(p2)
                # Intersection of two touching polygons should be a LineString or MultiLineString (the boundary)
                if intersection.geom_type in ['LineString', 'MultiLineString']:
                    # Rotate back
                    if intersection.geom_type == 'LineString':
                        rotated_geom = affinity.rotate(intersection, angle, origin=origin)
                        lines.append(rotated_geom)
                    elif intersection.geom_type == 'MultiLineString':
                        for geom in intersection.geoms:
                            rotated_geom = affinity.rotate(geom, angle, origin=origin)
                            lines.append(rotated_geom)
                elif intersection.geom_type == 'GeometryCollection':
                     for geom in intersection.geoms:
                        if geom.geom_type in ['LineString', 'MultiLineString']:
                            rotated_geom = affinity.rotate(geom, angle, origin=origin)
                            lines.append(rotated_geom)
    return lines

def build_adjacency_graph(parts):
    """
    Builds an adjacency graph where nodes are indices of parts,
    and edges represent spatial adjacency (touching).
    """
    adj = {i: [] for i in range(len(parts))}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            # Check if parts touch.
            # Using a small buffer or distance check is robust.
            if parts[i].distance(parts[j]) < 1e-5:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def traverse_graph(adj, parts):
    """
    Traverse the adjacency graph to determine a visitation order (DFS).
    Starts from the part with the minimum Y coordinate (bottom-most).
    """
    if not parts:
        return []
        
    # Find start node (lowest min_y)
    start_node = 0
    min_y = float('inf')
    for i, p in enumerate(parts):
        _, y_min, _, _ = p.bounds
        if y_min < min_y:
            min_y = y_min
            start_node = i
            
    visited = set()
    path_indices = []
    
    # DFS stack
    stack = [start_node]
    
    # To ensure we visit neighbors in a spatial order (e.g. left-to-right),
    # we can sort adjacency lists.
    
    while stack:
        u = stack.pop()
        if u in visited:
            continue
            
        visited.add(u)
        path_indices.append(u)
        
        # Get neighbors
        neighbors = adj[u]
        
        # Sort neighbors to prioritize "closest" or specific direction?
        # Simple heuristic: prioritize neighbors with similar Y, then increasing Y.
        # For a stack, we push neighbors in REVERSE order of desired visit.
        # We want to visit "closest" first.
        
        # Let's just push them.
        for v in neighbors:
            if v not in visited:
                stack.append(v)
                
    # Handle disconnected components (if any)
    for i in range(len(parts)):
        if i not in visited:
            # Start a new DFS from this unvisited node
            stack = [i]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                path_indices.append(u)
                for v in adj[u]:
                    if v not in visited:
                        stack.append(v)
                        
    return path_indices

def generate_simple_scan(polygon, uav_width):
    """
    Generates a simple back-and-forth scan for a convex/monotone polygon.
    Assumes polygon is already rotated to align with axes.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    path_points = []
    y = min_y + uav_width / 2
    direction = 1 # 1: left->right, -1: right->left
    
    while y < max_y:
        line = LineString([(min_x - 10, y), (max_x + 10, y)])
        intersection = polygon.intersection(line)
        
        if intersection.is_empty:
            y += uav_width
            continue
            
        points = []
        if intersection.geom_type == 'LineString':
            points.extend(list(intersection.coords))
        elif intersection.geom_type == 'MultiLineString':
            # This shouldn't happen for convex/monotone parts if decomposed correctly
            # But if it does, handle it
            for geom in intersection.geoms:
                points.extend(list(geom.coords))
        elif intersection.geom_type == 'Point':
             points.append(intersection)
             
        # Sort points by X
        points.sort(key=lambda p: p[0])
        
        # If we have multiple segments (e.g. decomposition wasn't perfect), 
        # just connect them left-to-right for now
        if direction == 1:
            path_points.extend(points)
        else:
            path_points.extend(points[::-1])
            
        direction *= -1
        y += uav_width
        
    return path_points

def generate_coverage_path(polygon, uav_width, angle, obstacles=None):
    """
    Generates coverage path using Boustrophedon Cellular Decomposition if obstacles exist.
    Includes a Headland Path (inner buffer) for turning.
    """
    # 1. Subtract obstacles from patch
    effective_area = polygon
    if obstacles:
        # Filter obstacles that intersect with the patch
        relevant_obs = [obs for obs in obstacles if polygon.intersects(obs)]
        if relevant_obs:
            obs_union = unary_union(relevant_obs)
            effective_area = polygon.difference(obs_union)
    
    if effective_area.is_empty:
        return []

    # 2. Generate Headland Path (Inner Buffer)
    # The headland path is the boundary of the inner buffer.
    # The scanning is done on the area INSIDE the headland path.
    headland_width = uav_width / 2
    headland_poly = effective_area.buffer(-headland_width)
    
    use_headland = False
    scan_area = effective_area
    headland_path_points = []
    
    if not headland_poly.is_empty:
        use_headland = True
        scan_area = headland_poly
        
        # Extract headland path (boundary of headland_poly)
        if headland_poly.geom_type == 'Polygon':
            headland_path_points.extend(list(headland_poly.exterior.coords))
            # Handle interiors (holes) if any? Usually headland is just the outer frame.
            # But if there are holes (obstacles), we might want to circle them too?
            # For simplicity, let's stick to the exterior for the main headland.
        elif headland_poly.geom_type == 'MultiPolygon':
            for geom in headland_poly.geoms:
                headland_path_points.extend(list(geom.exterior.coords))
    
    # 3. Decompose scan_area into monotone/convex parts
    # Note: decompose_polygon_trapezoidal returns parts in rotated frame
    parts, origin = decompose_polygon_trapezoidal(scan_area, angle)
    
    # 3.1 Merge compatible parts to reduce fragmentation
    parts = merge_parts(parts)
    
    # 3.2 Build Adjacency Graph and Traverse to determine optimal order
    adj = build_adjacency_graph(parts)
    ordered_indices = traverse_graph(adj, parts)
    
    # 4. Generate path for each part
    all_subpaths = []
    for idx in ordered_indices:
        part = parts[idx]
        subpath = generate_simple_scan(part, uav_width)
        if subpath:
            all_subpaths.append(subpath)
            
    if not all_subpaths:
        # If no scan lines (area too small), but we have headland, return headland
        if use_headland:
            final_path_points = headland_path_points
            return {
                'path': final_path_points,
                'scan_path': [],
                'headland_path': headland_path_points,
                'headland_poly': headland_poly
            }
        else:
            return {'path': [], 'scan_path': [], 'headland_path': [], 'headland_poly': None}
        
    # 5. Connect subpaths (Greedy TSP / Nearest Neighbor)
    # All subpaths are in ROTATED frame.
    
    connected_scan_path_rotated = []
    current_pos = None
    
    for subpath in all_subpaths:
        if not connected_scan_path_rotated:
            connected_scan_path_rotated.extend(subpath)
            current_pos = subpath[-1]
            continue
            
        # Check two options: Append subpath as is, or reversed
        p_start = np.array(subpath[0])
        p_end = np.array(subpath[-1])
        curr = np.array(current_pos)
        
        dist_direct = np.linalg.norm(curr - p_start)
        dist_reverse = np.linalg.norm(curr - p_end)
        
        if dist_direct <= dist_reverse:
            connected_scan_path_rotated.extend(subpath)
            current_pos = subpath[-1]
        else:
            connected_scan_path_rotated.extend(subpath[::-1])
            current_pos = subpath[0]

    # 6. Rotate scan path back to original frame
    scan_path_original = []
    for p in connected_scan_path_rotated:
        pt = Point(p)
        rotated_pt = affinity.rotate(pt, angle, origin=origin)
        scan_path_original.append((rotated_pt.x, rotated_pt.y))
        
    # Get decomposition lines for visualization
    decomposition_lines = get_decomposition_lines(parts, origin, angle)
        
    # 7. Combine Headland Path and Scan Path
    # Order: Headland -> Scan
    final_path = []
    if use_headland:
        final_path.extend(headland_path_points)
        
    final_path.extend(scan_path_original)
        
    return {
        'path': final_path,
        'scan_path': scan_path_original,
        'headland_path': headland_path_points if use_headland else [],
        'headland_poly': headland_poly if use_headland else None,
        'decomposition_lines': decomposition_lines
    }

def optimize_scan_angle(patch, uav_width, obstacles=None):
    best_result = None
    min_cost = float('inf')
    # Try fewer angles for speed if obstacles are present
    angles = range(0, 180, 30) if obstacles else range(0, 180, 15)
    
    for angle in angles:
        result = generate_coverage_path(patch, uav_width, angle, obstacles)
        if not result or not result['path']:
            continue
        path = result['path']
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
        if length < min_cost:
            min_cost = length
            best_result = result
            
    # Return the dictionary structure, or a minimal one if no path found
    if best_result is None:
        return {'path': [patch.centroid.coords[0], patch.centroid.coords[0]], 'scan_path': [], 'headland_path': [], 'headland_poly': None, 'decomposition_lines': []}
    return best_result

