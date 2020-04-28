import xml.etree.ElementTree as ET
import json
import sympy
from sympy import *

class Environment(object):
    def __init__(self, map_name):
        self.map_name = map_name
        self.propositions, self.states = self.from_xml(map_name)

    def get_state(self):
        return self.states

    def get_propositions(self):
        return self.propositions

    def get_symbol2prop(self):
        lpopl_props = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u'.split(',')
        symbol2prop = {lpopl_props[i]:self.propositions[i] for i in range(
            len(self.propositions))}
        return symbol2prop

    def get_prop2symbol(self):
        lpopl_props = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u'.split(',')
        prop2symbol = {self.propositions[i]:lpopl_props[i] for i in range(
            len(self.propositions))}
        return prop2symbol

    def get_seq_from_trajectory(self, trajectory):
        trajectory = [(3, 5, -1), (3, 5, 180)]
        trajectory =  [(3, 7, 180), (3, 7, 270), (2, 7, 270), (1, 7, 270), (1, 7, 180), (1, 8, 180), (1, 9, 180)]

        seq = [self.states[item[0]][item[1]] for item in trajectory]
        return seq

    def from_xml(self, map_name):
        print("Inside from xml")

        propositions, states = [], {}
        dirpath = '/Users/romapatel/github/lang2ltl/navigation-corpus/sail/maps/'
        f = open(dirpath + self.map_name + '.xml', 'r')
        s = '\n'.join(item for item in f.readlines())

        nodes, edges, n_cells = {}, {}, 0
        for child_1 in ET.fromstring(s):
            for child_2 in child_1:
                name, items = child_2.tag, child_2.attrib
                if name == 'node':
                    nodes[len(nodes)] = items
                elif name == 'edge':
                    edges[len(edges)] = items

        objects, floors, walls = [], [], []
        for idx in nodes:
            if len(nodes[idx]['item']) > 1:
                objects.append(nodes[idx]['item'])

        row_idxs, col_idxs = [int(nodes[idx]['y']) for idx in nodes], \
                             [int(nodes[idx]['x']) for idx in nodes]
        n_rows, n_cols = max(row_idxs), max(col_idxs)

        for idx in edges:
            floors.append(edges[idx]['floor'])
            walls.append(edges[idx]['wall'])

        for x in range(n_rows+1):
            states[x] = {}
            for y in range(n_cols+1):
                states[x][y] = []

        objects, floors, walls = list(set(objects)), list(set(floors)), \
                                 list(set(walls))
        propositions = ['at_' + item for item in objects] + \
            ['is_floor_' + item for item in floors] + \
            ['is_wall_' + item for item in walls] + \
            ['move_forward', 'move_backward', 'move_left', 'move_right']

        for idx in nodes:
            if len(nodes[idx]['item']) > 1:
                x, y = int(nodes[idx]['y']), int(nodes[idx]['x'])
                states[x][y].append('at_' + nodes[idx]['item'])

        for idx in edges:
            p1, p2 = edges[idx]['node1'].split(','), \
                     edges[idx]['node2'].split(',')
            wall, floor = edges[idx]['wall'], edges[idx]['floor']
            states[int(p1[1])][int(p1[0])].extend(
                ['is_floor_' + floor, 'is_wall_' + wall])

        def count(items, prop_type):
            items = [item for item in items if prop_type in item]
            return len(items)

        for x in states:
            for y in states[x]:
                states[x][y] = list(set(states[x][y]))
                states[x][y] = sorted(states[x][y])

        return propositions, states


if __name__=='__main__':
    print("Running environment")


