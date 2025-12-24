import math
import networkx as nx
from shapely.geometry import LineString
from shapely.prepared import prep
from shapely.ops import unary_union

def build_visibility_graph(obstacles, uav_width, points_of_interest):
    """
    Builds a visibility graph for path planning.
    obstacles: List of Shapely Polygons
    uav_width: UAV width (for inflation)
    points_of_interest: List of (x, y) tuples (start, end points)
    """
    G = nx.Graph()
    
    # 1. Inflate obstacles
    # Add a safety margin of 1.0 meters to ensure we don't graze the obstacle
    inflation_radius = uav_width / 2.0 + 1.0
    inflated_obstacles = [obs.buffer(inflation_radius, cap_style=3, join_style=2) for obs in obstacles]
    # Union to handle overlapping inflated obstacles
    if inflated_obstacles:
        combined_obstacles = unary_union(inflated_obstacles)
        if combined_obstacles.geom_type == 'Polygon':
            inflated_obstacles = [combined_obstacles]
        elif combined_obstacles.geom_type == 'MultiPolygon':
            inflated_obstacles = list(combined_obstacles.geoms)
        else:
            inflated_obstacles = [] # Should not happen if obstacles exist
    else:
        inflated_obstacles = []
    
    # 2. Extract vertices from inflated obstacles
    nodes = []
    # Add points of interest (start/end)
    for p in points_of_interest:
        nodes.append(tuple(p))
        
    for obs in inflated_obstacles:
        # Simplify slightly to reduce vertex count if needed
        # obs = obs.simplify(0.5)
        coords = list(obs.exterior.coords)[:-1] # Remove duplicate end point
        nodes.extend(coords)
        # Handle holes if any? (Not doing for now)

    # Remove duplicates
    nodes = list(set(nodes))
    
    # Add nodes to graph
    for node in nodes:
        G.add_node(node)
        
    # 3. Check visibility between all pairs of nodes
    # Optimization: Spatial index could be used, but for N < ~200, O(N^2) is fine
    # Use prepared geometry for faster intersection checks
    prepared_obstacles = [prep(obs) for obs in inflated_obstacles]
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u = nodes[i]
            v = nodes[j]
            
            # Check if line segment u-v intersects any obstacle interior
            line = LineString([u, v])
            intersects = False
            
            # First, quick bounding box check or simple distance check?
            # Directly check intersection with prepared obstacles
            for k, obs in enumerate(inflated_obstacles):
                # We want to check if the line goes THROUGH the obstacle.
                # Touching the boundary is fine (grazing).
                # But prepared geometry 'intersects' includes boundary.
                
                if prepared_obstacles[k].intersects(line):
                    # Check if it's just touching the boundary or actually crossing
                    # A robust way: check if line.intersection(obs) is a LineString with length > epsilon
                    # Or simpler: check if the midpoint is inside the obstacle
                    # But the line could be a chord.
                    
                    intersection = obs.intersection(line)
                    if intersection.is_empty:
                         continue
                         
                    # If intersection is just points (touching vertices), it's fine
                    if intersection.geom_type in ['Point', 'MultiPoint']:
                        continue
                        
                    # If intersection is a LineString, check its length
                    # If length is tiny, it's just a touch
                    if intersection.length > 1e-3:
                         # It goes through the interior
                         intersects = True
                         break
            
            if not intersects:
                dist = math.hypot(u[0]-v[0], u[1]-v[1])
                G.add_edge(u, v, weight=dist)
    
    return G

