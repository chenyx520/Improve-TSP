import numpy as np
import random
from shapely.geometry import Polygon

def generate_random_polygon(center, avg_radius, irregularity, spikeyness, num_verts):
    irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / num_verts
    spikeyness = np.clip(spikeyness, 0, 1) * avg_radius
    angle_steps = []
    lower = (2 * np.pi / num_verts) - irregularity
    upper = (2 * np.pi / num_verts) + irregularity
    sum_steps = 0
    for _ in range(num_verts):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        sum_steps += tmp
    k = sum_steps / (2 * np.pi)
    angle_steps = [x / k for x in angle_steps]
    points = []
    angle = random.uniform(0, 2 * np.pi)
    for _ in range(num_verts):
        r_i = np.clip(random.gauss(avg_radius, spikeyness), 0, 2 * avg_radius)
        x = center[0] + r_i * np.cos(angle)
        y = center[1] + r_i * np.sin(angle)
        points.append((x, y))
        angle += angle_steps[len(points)-1]
    return Polygon(points)

