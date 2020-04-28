import curses
import numpy as np
import pygame
import json, re, os, sys
from environment import Environment
import math

def metrics(gold, path):
    #ground truth
    gold = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 6), \
            (6, 6), (6, 7), (6, 8)]
    path = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), \
            (4, 8), (5, 8), (6, 8), (6, 9), (6, 10)]

    gold_mask = np.zeros((12, 12))
    path_mask = np.zeros((12, 12))

    for loc in gold:
        gold_mask[loc[0]][loc[1]] = 1
    for loc in path:
        path_mask[loc[0]][loc[1]] = 1


    gold_mask = np.ndarray.flatten(gold_mask)
    path_mask = np.ndarray.flatten(path_mask)

    acc = [1 if gold_mask[i] == path_mask[i] else 0 for i in range(len(gold_mask))]
    tp = [1 if ((gold_mask[i] == path_mask[i]) and gold_mask[i] == 1) else 0 for i in range(len(gold_mask))]
    tn = [1 if ((gold_mask[i] == path_mask[i]) and gold_mask[i] == 0) else 0 for i in range(len(gold_mask))]

    fp = [1 if ((gold_mask[i] != path_mask[i]) and path_mask[i] == 1) else 0 for i in range(len(gold_mask))]
    fn = [1 if ((gold_mask[i] != path_mask[i]) and path_mask[i] == 0) else 0 for i in range(len(gold_mask))]

    print('Acc: ', float(sum(acc))/len(acc))
    print('Prec: ', float(sum(tp))/(sum(tp) + sum(fp)))
    print('Recl: ', float(sum(tp))/(sum(tp) + sum(fn)))



def sail_data(verbose, map_name):
    env = Environment(map_name)

    states, propositions = env.get_state(), env.get_propositions()

    lpopl_props = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u'.split(',')
    prop2char = {
        propositions[i]:lpopl_props[i] for i in range(len(propositions))
    }

    grid = [
        [['-', '-', '-', '-'] for i in range(len(states))
         ] for i in range(len(states))
    ]

    return prop2char, states, env

def print_sail_map():
    map_name = 'map-grid'
    prop2char, states, env = sail_data(True, map_name)

    map = {x_coord: {} for x_coord in states}
    width, height = len(states[0]), len(states)
    max_props = 0
    for x_coord in states:
        for y_coord in states[x_coord]:
            map[x_coord][y_coord] = [
                prop2char[prop] for prop in states[x_coord][y_coord]]
            if len(map[x_coord][y_coord]) > max_props:
                max_props = len(map[x_coord][y_coord])

    grid_map = []
    for _ in range(height):
        grid_map.append([['-']*max_props for _ in range(width)])

    for x_coord in map:
        for y_coord in map[x_coord]:
            props = sorted(map[x_coord][y_coord])
            for i in range(len(props)):
                grid_map[x_coord][y_coord][i] = props[i]
            grid_map[x_coord][y_coord] = ''.join(
                item for item in grid_map[x_coord][y_coord])

    map_str = ''
    for x in range(len(grid_map)):
        for y in range(len(grid_map[x])):
            map_str += grid_map[x][y] + '\t'
        map_str += '\n'


def get_lat(p):
    assert(isinstance(p, tuple)), ("%s datatype not supported as lat/long datatype" % type(p))
    return p[0]

def get_lon(p):
    assert(isinstance(p, tuple)), ("%s datatype not supported as lat/long datatype" % type(p))
    return p[1]

# Calculate great-circle distance between two points
# Credit: https://www.movable-type.co.uk/scripts/latlong.html
def haversine_distance(p1, p2):
    r = 6371e3
    print("p1", p1)
    print("p2", p2)
    del_lat = abs(math.radians(get_lat(p2) - get_lat(p1)))
    del_lon = abs(math.radians(get_lon(p2) - get_lon(p1)))

    a = math.sin(del_lat / 2) * math.sin(del_lat / 2) + math.cos(get_lat(p1)) * math.cos(get_lat(p2)) * math.sin(del_lon / 2) * math.sin(del_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = r * c

    return d

# Returns SW, NE lat/long points x meters from p
# Credit: https://gis.stackexchange.com/questions/15545/calculating-coordinates-of-square-x-miles-from-center-point
def gdiag(r, p):
    NS = r / 69
    EW = NS / math.cos(get_lat(p))

    d1 = (get_lat(p) - NS, get_lon(p) + EW)
    d2 = (get_lat(p) + NS, get_lon(p) - EW)

    return d1, d2
#    return (round(d1[0], 6), round(d1[1], 6)), ((round(d2[0], 6)), round(d2[1], 6))

# Scrape 5m box of landmarks around drone pos
def create_map_from_osm(p, radius):
    overpass_url = "http://overpass-api.de/api/interpreter"

    left_bound, right_bound = gdiag(meters_to_miles(radius), p)
    print("left bound:", left_bound)
    print("right bound:", right_bound)
    # Overpass query gets all named nodes and named ways that aren't too large
    # [!] tags exclude large ways
    overpass_query = """
    [out:json];
    (node["name"]( {}, {}, {}, {});
    way["name"!="Brown University"]["name"][!"place"][!"highway"][!"railway"]({}, {}, {}, {});
    );
    (._;>;);
    out body;
    """.format(left_bound[0], left_bound[1], right_bound[0], right_bound[1], left_bound[0], left_bound[1], right_bound[0], right_bound[1])

    print("QUERY:", overpass_query)

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data

def meters_to_miles(mt):
    return mt / 1609.344

if __name__ == '__main__':
    print_sail_map()

