import requests
import re
import os
import math
import json
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import env_utils as utils
import parser_utils as parse_osm_mapping
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely import geometry, ops, wkb
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from overpass_queries import QueryMap
import networkx as nx
import collections
import operator

NUM_POLYS_THRESHOLD = 150 #Note: This is an upper-bound. Due to voronoi clipping, a few less polygons can be generated.


def bounding(points):
    x_coords, y_coords = zip(*points)
    return [(min(x_coords), min(y_coords)), (max(x_coords), max(y_coords))]


def ring_coding(ob):
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes


#From https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def pathify(poly):
    verts = np.concatenate(
        [np.asarray(poly.exterior)] + [np.asarray(r) for r in poly.interiors])
    codes = np.concatenate(
        [ring_coding(poly.exterior)] + [ring_coding(r) for r in poly.interiors])
    return Path(verts, codes)


def plot_poly_with_holes(polyhole, ax, boundary, facecolor, edgecolor):
    path = pathify(polyhole)
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor)

    ax.add_patch(patch)
    ax.set_xlim(boundary[0][0], boundary[1][0])
    ax.set_ylim(boundary[0][1], boundary[1][1])
    ax.set_aspect(1.0)


def random_points_within(poly, num):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    counter = 0
    while len(points) < num:
        randpoint = geometry.Point([np.random.uniform(min_x, max_x),
                                    np.random.uniform(min_y, max_y)])
        if randpoint.within(poly):
            points.append(randpoint)
        if counter > num * 3:
            print("WARNING: random point generation taking long time")
        counter += 1
    return points


def descretize_voronoi_to_polys(vor, bbox):
    points = vor.points
    lines = [geometry.LineString(
        vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

    result = geometry.MultiPolygon(
        [poly.intersection(bbox) for poly in ops.polygonize(lines)])
    result = geometry.MultiPolygon(
        [p for p in result] + [p for p in bbox.difference(
            ops.unary_union(result))])

    return result


def clip_voronoi_cells(cells, clip):
    clipped = []
    for cell in cells:
        for clp in clip:
            if cell.intersects(clp):
                cell = cell.difference(clp)
        if cell.type == 'Polygon':
            clipped.append(cell)
        elif cell.type == 'MultiPolygon':
            for poly in cell:
                clipped.append(poly)
    return clipped


def map_voronoi_points(cells, points):
    point_to_cell = {}
    cell_to_point = {}
    for cell in cells:
        for p in points:
            if p.within(cell):
                point_to_cell[(p.x, p.y)] = cell
                cell_to_point[cell.wkb] = p
                break

    return point_to_cell, cell_to_point


def graph_from_voronoi(points_to_cells, cell_to_point, landmarks_to_point,
                       points_to_landmark):
    graph = {}

    points = list(points_to_cells.keys()) + list(points_to_landmark.keys())
    merged_cells = {**points_to_cells, **points_to_landmark}

    landmarks_aspoly_to_point = {
        value.wkb: key for key, value in points_to_landmark.items()}
    merged_points = {**cell_to_point, **landmarks_aspoly_to_point}

    for p in merged_cells.keys():
        graph[p] = []
        curr_cell = merged_cells[p]
        for c in merged_points.keys():
            if curr_cell.intersects(wkb.loads(c)):
                if isinstance(merged_points[c], tuple):
                    graph[p].append(merged_points[c])
                else:
                    graph[p].append((merged_points[c].x, merged_points[c].y))
    return merged_cells, graph


def plot_points(points):
    for p in points:
        if isinstance(p, geometry.point.Point):
            plt.plot(p.x, p.y, marker='o', markersize=1.5)
        else:
            plt.plot(p[0], p[1], marker='o', markersize=1.5)


def plot_polys(polys):
    for poly in polys:
        plt.plot(*poly.exterior.xy, linewidth=0.5)


def plot_voronoi(v, ax):
    voronoi_plot_2d(v, ax, show_points=False,
                    show_vertices=False, line_width=0.5)


def plot_points_and_names(items):
    ### Font for text for landmark names
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 7,}
    ### Matplotlib markers
    markers = {'point': '.',
               'circle': 'o',
                'triangle_down': 'v',
                'triangle_up': '^',
                'square': 's',
                'pentagon': 'p',
                'diamond': 'D',
                'x_filled': 'X',
                'star': '*',}

    ### Colours
    colours = ['darkolivegreen', 'darkgreen', 'darkslategray', 'blue',
               'darkred', 'darkcyan', 'black', 'darkorange', 'indigo',
               'crimson', 'darkslateblue']

    # make text the same colour as the point?
    for item in items:
        p, landmark_name = item
        r_colour = random.choice(colours)
        plt.plot(p[0], p[1], marker=markers['triangle_down'], markersize=1.9,
                 color=r_colour)
        font['color'] = r_colour
        plt.text(p[0], p[1], landmark_name, fontdict=font)


def node_distance(prev_node, current_node):
    x_distance = current_node[0] - prev_node[0]
    y_distance = current_node[1] - prev_node[1]
    return math.sqrt(math.pow(x_distance, 2) + math.pow(y_distance, 2))


def build_graph(ways, nodes):
    plot_fig = False
    ### Plot dimensions
    if plot_fig:
        fig, ax = plt.subplots(figsize=(10, 8))


    ### Creating convex hulls from OSM way data
    hulls = []
    landmarks_to_point = {}
    points_to_landmark = {}

    for way in ways.keys():
        points = np.array(ways[way])
        if len(points) <= 2:
            continue # print("cannot convex")
        else:
            hull = ConvexHull(points)
            hulls.append(hull)

            pt = (hull.points[hull.vertices[0]][0],
                  hull.points[hull.vertices[0]][1])
            landmarks_to_point[way] = pt

            verticies = [hull.points[v] for v in hull.vertices]
            as_poly = geometry.Polygon(verticies)
            points_to_landmark[(pt[0], pt[1])] = as_poly

    ### Converting convex hulls to polygons
    hpolys, hverts, hpoly_navigation_points = [], [], []
    for h in hulls:
        verticies = [h.points[v] for v in h.vertices]
        hverts.extend(verticies)
        hpoly_navigation_points.append(geometry.Point(verticies[0][0],
                                                      verticies[0][1]))
        hpolys.append(geometry.Polygon(verticies))

    if plot_fig:
        plot_polys(hpolys)
    ### Creating polygons for landmarks
    lpolys, lpoly_navigation_points = [], []

    center_points, center_point_names = [], []
    for n in nodes.keys():
        centerpt = nodes[n]
        d1, d2 = utils.gdiag(utils.meters_to_miles(3), centerpt)
        points = np.array([(d1[0], d1[1]), (d2[0], d2[1]), (d1[0], d2[1]),
                           (d2[0], d1[1])])

        bbox = ConvexHull(points)
        bbox_aspoly = geometry.Polygon([bbox.points[v] for v in bbox.vertices])
        lpolys.append(bbox_aspoly)

        pt = (bbox.points[bbox.vertices[0]][0],
              bbox.points[bbox.vertices[0]][1])
        lpoly_navigation_points.append(geometry.Point(pt[0], pt[1]))
        landmarks_to_point[n] = pt
        points_to_landmark[pt] = bbox_aspoly
        center_points.append(centerpt )
        center_point_names.append(n)

    # merging overlapping polygons
    merged = ops.cascaded_union(hpolys + lpolys)

    # creating map boundary polygon
    boundary = bounding(hverts)
    bbox = geometry.box(boundary[0][0], boundary[0][1], boundary[1][0],
                        boundary[1][1])

    # symmetric difference polygon, between bounding box and ways
    polyhole = geometry.Polygon(bbox.exterior.coords,
                                [m.exterior.coords for m in merged])
    if plot_fig:
        plot_poly_with_holes(polyhole, ax, boundary, facecolor='#cccccc',
                             edgecolor='#999999')

    # get NUMPOLY number of points in symmetric difference polygon
    randpoints = random_points_within(polyhole, NUM_POLYS_THRESHOLD)

    # get voronoi diagram of points
    vor = Voronoi(np.array([[r.x, r.y] for r in randpoints]))
    vor_p = descretize_voronoi_to_polys(vor, bbox)

    # clip voronoi cells that overlap way polys
    clipped = clip_voronoi_cells(cells=vor_p, clip=merged)

    mapping_points = randpoints + hpoly_navigation_points + \
                     lpoly_navigation_points
    point_to_cell, cell_to_point = map_voronoi_points(
        cells=clipped + [m for m in merged], points=mapping_points)

    # plot names
    plot_names = False
    center_items = [(center_points[i], center_point_names[i]) for i in range(len(center_points))]
    if plot_names and plot_fig:
        plot_landmark_names(center_items)

    if plot_fig:
        plot_points_and_names(center_items)


    # build graph from cells
    merged_cells, graph = graph_from_voronoi(
        point_to_cell, cell_to_point, landmarks_to_point, points_to_landmark)

    return center_points, center_point_names, graph


def get_map_dictionary(map_name):
    """Takes in a map name and returns an overpass query to load the map."""
    map_class = QueryMap()
    return map_class.get_map_query(map_name)
    # return map_class._MAP_NAME_REGISTRY[map_name]

def load_map(overpass_url, overpass_query):
    """Takes in an overpass query and returns map data."""
    print("Overpass url: ", overpass_url)
    print("Overpass query: ", overpass_query)

    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    data = response.json()
    return data


def get_map_from_osm_url(map_name):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = get_map_dictionary(map_name)
    map = load_map(overpass_url, query)
    return map


def filter_names(landmark_names, num_landmarks):
    sorted_names = [name for name in landmark_names if re.search(
        '\',.\(\)/', name) is None]
    sorted_names = [name for name in sorted_names if len(name.split(' ')) < 4]
    sorted_names = [name for name in sorted_names if len(name) > 2]
    sorted_names = [name for name in sorted_names if '(' not in name]
    sorted_names = [name for name in sorted_names if '/' not in name]
    sorted_names = [name for name in sorted_names if '#' not in name]
    sorted_names = sorted_names[:num_landmarks]
    return sorted_names


def convert_osm_to_voronoi(map_data):
    parse_osm_mapping.build_ways(map_data)
    ways = parse_osm_mapping.get_ways()
    nodes = parse_osm_mapping.get_nodes()
    landmark_centers, landmark_names, graph = build_graph(ways, nodes)

    # sort and return landmark names and centers
    num_landmarks = random.choice([10, 11, 12, 13, 14, 15])
    sorted_names = filter_names(landmark_names, num_landmarks)

    new_centers, new_names = [], []
    for i in range(len(landmark_centers)):
        name, center = landmark_names[i], landmark_centers[i]
        if name in sorted_names:
            new_names.append(name)
            new_centers.append(center)

    return new_centers, new_names, graph


def create_graph_from_landmarks(center_points, center_point_names):
    print("Inside create graph from landmarks")
    center_items = [(center_points[i], center_point_names[i]) for i in range(
        len(center_points))]

    grid_size, spacing, noise, connect_th = 15, 1.0, 0.3, 1.4
    random_seed = random.randint(0, 100000)
    random.seed(random_seed)

    G = nx.Graph()
    # TODO: creates a graph by displacing input center points
    min_x = min([point[0] for point in center_points])
    min_y = min([point[1] for point in center_points])

    center_points = [
        (item[0] - min_x, item[1] - min_y) for item in center_points]
    center_points = [(item[0] * 100, item[1] * 100) for item in center_points]

    for point in center_points:
        G.add_node(len(G), pos=[point[0], point[1]])

    node_distances = []
    for i in range(len(center_points)):
        for j in range(len(center_points)):
            node_distances.append(node_distance(center_points[i],
                                                center_points[j]))

    coords = nx.get_node_attributes(G, 'pos')
    node_connections = {}
    for n1 in G.nodes:
        node_connections[n1] = 0
        for n2 in G.nodes:
            dist = np.linalg.norm(np.subtract(coords[n1], coords[n2]))
            if dist <= connect_th and n1 != n2:
                G.add_edge(n1, n2, weight=dist)
                node_connections[n1] += 1

    for node in node_connections:
        if node_connections[node] == 0:
            G.remove_node(node)


    avg_edge_len = np.mean(list(nx.get_edge_attributes(G, 'weight').values()))
    pos = nx.get_node_attributes(G, 'pos')

    node_labels = {node : center_point_names[node] for node in G.nodes}
    return G, pos, node_labels


def _sample_path_in_graph(graph, s=None, t=None, k=None, p=0.2, weight='weight'):
    """K-th shortest path sampling, where K ~ Geometric(p)."""
    k = np.random.geometric(p) if k is None else k
    s = np.random.choice(list(graph.nodes.keys())) if s is None else s
    t = np.random.choice(list(graph.nodes.keys())) if t is None else t
    paths = nx.shortest_simple_paths(graph, s, t, weight=weight)

    return list(itertools.islice(paths, k))[-2:]


def sample_paths_in_graph(G, pos, node_labels, n_paths, dirpath,
                          draw_graph=False, save_graph=True):
    kwargs = {'arrowsize': 20, 'width': 2.5, 'node_size': 30}

    path_data = []

    for path_idx in range(n_paths):
        fig_name = "path_%s.png" % str(path_idx)
        start_node = random.choice(list(node_labels.keys()))
        end_nodes = [node for node in node_labels.keys() if node is not start_node]
        end_node = random.choice(end_nodes)

        paths = _sample_path_in_graph(G, s=start_node, t=end_node)

        if len(paths) < 2:
            continue

        path, alt_path = paths

        path_nodes = list(set(alt_path + path))
        path_node_labels = {idx : node_labels[idx] for idx in path_nodes}
        landmark_names = [path_node_labels[idx] for idx in path_node_labels]

        print("Path idx: ", path_idx, ", Path length: ", len(path))

        path_data.append({
            "path_idx": path_idx,
            "start_node": start_node,
            "end_node": end_node,
            "path": path,
            "alt_path": alt_path,
            "fig_name": fig_name,
        })

        if save_graph or draw_graph:
            fig=plt.figure(figsize=(15, 8))

            fig = plt.subplot(121)
            nx.draw(G, pos, labels=path_node_labels, edge_color='lightgrey',
                    font_size=8, font_family='serif', node_color='lightgrey',
                    node_size=30, width=1.0, font_weight='semibold',
                    cmap=plt.get_cmap('viridis'))


        color, alt_color = 'dodgerblue', 'orange' # 'dodgerblue'
        R = nx.DiGraph()
        P = nx.DiGraph()

        for idx in range(len(alt_path)-1):
            R.add_edge(alt_path[idx], alt_path[idx+1])
        if save_graph or draw_graph:
            nx.draw_networkx_nodes(R, pos, edge_color=alt_color,
                                   node_color=alt_color, **kwargs)

        if save_graph or draw_graph:
            nx.draw_networkx_nodes(P, pos, edge_color=color, node_color=color,
                                       **kwargs)

        edges = collections.defaultdict(list)

        for idx in range(len(alt_path) - 1):
            start = min(alt_path[idx], alt_path[idx+1])
            end = max(alt_path[idx], alt_path[idx+1])
            edges[(start, end)].append((alt_path[idx], alt_path[idx+1],
                                        alt_color))

        for idx in range(len(path) - 1):
            start = min(path[idx], path[idx+1])
            end = max(path[idx], path[idx+1])
            edges[(start, end)].append((path[idx], path[idx+1], color))

        C = nx.compose(R, P)
        for (start, end), edge_list in edges.items():
            for edge_idx, (s, e, color) in enumerate(edge_list):
                shifted_pos = {key: (val[0], val[1]) for key, val in pos.items()}
                if save_graph or draw_graph:
                    nx.draw_networkx_edges(C, shifted_pos, edgelist=[(s, e)],
                                           edge_color=color, **kwargs)

        if save_graph:
            fig = plt.subplot(2,2,2)
            fig.axis("off")
            landmark_str = "\n".join(item for item in landmark_names)
            plt.text(0.3, 0.3, "List of Landmarks:\n\n%s" % landmark_str,
                     fontfamily='serif', fontweight='bold')
            plt.tight_layout()

            # picture of map
            fig = plt.subplot(2, 2, 4)
            fig.axis("off")
            map_name = dirpath.split('/')[-3]
            file_name = '/'.join(item for item in dirpath.split('/')[:-2]) + '/%s.png' % map_name
            im = plt.imread(file_name)
            plt.imshow(im)

            plt.tight_layout()
            plt.savefig(dirpath + fig_name)


    if draw_graph:
        plt.show()
        plt.clf()

    return G, path_data


def check_modules():
    print("Testing modules")
    metadata, metadata_keys = {}, ["map_name", "start_node", "end_node",
                                   "path_idx", "path", "alt_path", "graph_obj",
                                   "node_labels", "pos"]

    finished = ["brown", "harvard", "mit", "upenn", "yale",
                "austin", "ann-arbor", "atlanta", "baltimore", "berkeley"]

    map_names = ["brown", "harvard", "mit", "upenn", "yale", "austin",
                 "ann-arbor", "atlanta", "baltimore", "berkeley", "stanford"]

    map_name = map_names[-1]

    n_paths = 5
    draw_graph, save_graph = False, True

    if os.path.isdir("../../data/paths/%s/" % map_name) is False:
        os.mkdir("../../data/paths/%s/" % map_name)

    dirpath = "../../data/paths/%s/figs/" % map_name

    if os.path.isdir(dirpath) is False:
        os.mkdir(dirpath)

    map_data = get_map_from_osm_url(map_name)
    landmark_centers, landmark_names, graph = convert_osm_to_voronoi(map_data)

    G, pos, node_labels = create_graph_from_landmarks(landmark_centers,
                                                 landmark_names)

    G, path_data = sample_paths_in_graph(G, pos, node_labels, n_paths, dirpath,
                                         draw_graph, save_graph)

    for sample in path_data:
        metadata[sample["path_idx"]] = {
            "map_name": map_name,
            "start_node": sample["start_node"],
            "end_node": sample["end_node"],
            "path": sample["path"],
            "alt_path": sample["alt_path"],
            "landmark_centers": landmark_centers,
            "landmark_names": landmark_names,
            "node_labels": node_labels,
            "fig_name": sample["fig_name"],
        }

    print(metadata)

    ## save metadata
    with open("../../data/paths/%s/%s.json" % (map_name, map_name), "w+") as f:
        f.write(json.dumps(metadata, indent=3))

    ### save amt file
    with open("../../data/paths/%s/%s.csv" % (map_name, map_name), "w+") as f:
        f.write("cu\n")
        for sample in path_data:
            f.write("http://cs.brown.edu/people/rpatel59/figures/amt/%s/figs/%s\n" %
                    (map_name, sample["fig_name"]))


def check_map_loading():
    map_names = ["brown", "harvard"]
    map_names = ["temp"]
    for map_name in map_names:
        print("Map name: ", map_name)
        map_data = get_map_from_osm_url(map_name)
        print("Map data: ", map_data)


if __name__ == "__main__":
    check_modules()
