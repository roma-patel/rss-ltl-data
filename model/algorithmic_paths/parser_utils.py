import typing
from enum import Enum
from collections import Iterable, namedtuple
import json
import yaml

TokenTypes = namedtuple('TokenTypes', ['return_type', 'args_types', 'necessity'])

# constants:
ALMOST_TOUCHING_MARGIN = 5


class Item:
    """represents the basic building blocks of the structured representations - objects that have shape, color and
    spatial location, each  belongs to a certain 'box'.
    """

    def __init__(self, dic):
        assert isinstance(dic, dict)
        self.__y_loc = dic['y_loc']
        self.__x_loc = dic['x_loc']
        self.color = Color(dic['color'])
        self.size = Size(dic['size'])
        self.shape = Shape(dic['type'])
        self.box = None # a pointer to the containing box (List of Items). Added when the box constructor is done.

    def __repr__(self):
        return "{0} {1} {2} at x: ({3}-{4}) y: ({5},{6})".format(
            self.size.name, self.color.name, self.shape.name, self.left, self.right, self.bottom, self.top).lower()

    def __copy__(self):
        return Item({'y_loc': self.__y_loc,
                     'x_loc': self.__x_loc,
                     'color':self.color.name,
                     'size':self.size.name ,
                     'type':self.shape.name })

    @property
    def right(self):
        return self.__x_loc + self.size.value

    @property
    def top(self):
        return 100 - self.__y_loc

    @property
    def bottom(self):
        return 100 - (self.__y_loc + self.size.value)

    # note: in the original representation the value of y_loc is bigger when the item is closer to the bottom.
    # here it is switched in order to be more intuitive and match the regular notion of x,y axes.

    @property
    def left(self):
        return self.__x_loc

    # information concerning the spatial location of an item is provided to outside modules
    # through these methods alone. Logical forms cannot relate directly to an item's (x,y) coordinates

    def touching_right(self, use_margin = False):
        margin = ALMOST_TOUCHING_MARGIN if use_margin else 0
        return self.right >= 100 - margin

    def touching_left(self, use_margin = False):
        margin = ALMOST_TOUCHING_MARGIN if use_margin else 0
        return self.left <= margin

    def touching_bottom(self, use_margin = False):
        margin = ALMOST_TOUCHING_MARGIN if use_margin else 0
        return self.bottom <= margin

    def touching_top(self, use_margin = False):
        margin = ALMOST_TOUCHING_MARGIN if use_margin else 0
        return self.top >= 100 - margin

    def touching_wall(self, use_margin = False):
        return self.touching_right(use_margin) or self.touching_left(use_margin) \
               or self.touching_bottom(use_margin) or self.touching_top(use_margin)

    def touching_corner(self, use_margin = False):
        return sum([self.touching_right(use_margin), self.touching_left(use_margin),
                    self.touching_bottom(use_margin), self.touching_top(use_margin)])==2

    def __distance(self, other):
        # in test assert that always >= 0 as items never overlap
        if not isinstance(other, Item):
            raise TypeError
        if self is other:
            return 0
        return max(self.left - other.right,
               other.left - self.right,
               self.bottom - other.top,
               other.bottom - self.top)

    def is_touching(self, other, use_margin=False):
        margin = ALMOST_TOUCHING_MARGIN if use_margin else 1
        return self is not other and self.box is other.box and self.__distance(other) <= margin

    def is_top(self):
        return self.top == max(item.top for item in self.box)

    def is_bottom(self):
        return self.top == min(item.top for item in self.box)

    def is_second(self):
        result = self.box.is_tower() and self.bottom==21
        return result

    def is_third(self):
        return self.box.is_tower() and self.bottom==42


class Box:

    def __init__(self, items_as_dicts : typing.List[dict]):
        self.items = [Item(d) for d in items_as_dicts]
        for item in self.items:
            item.box = self

    def __repr__(self):
        return "Box({})".format(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        return self.items[key]

    def __iter__(self):
        for s in self.items:
            yield  s

    def __contains__(self, item):
        return item in self.items

    def __copy__(self):
        return Box([item.__copy__() for item in self.items])

    def is_tower(self):

        return all(s.shape == Shape.SQUARE for s in self.items) and \
               all(s.right == self.items[0].right for s in self.items)


class Image:
    '''
    represent the structured representation of an image
    '''

    def __init__(self, structured_rep : typing.List[typing.List[dict]]):
        self.boxes = [Box(items_as_dicts) for items_as_dicts in structured_rep]

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, key):
        return self.boxes[key]

    def __iter__(self):
        for s in self.boxes:
            yield s

    def __copy__(self):
        return [box.__copy__() for box in self.boxes ]

    def get_all_boxes(self):
        return self.boxes

    def get_all_items(self):
        return [item for box in self.boxes for item in box]

    # whether this is an image of towers of blocks or a regular image with objects of various shapes
    def is_tower(self):
        return all([b.is_tower() for b in self.boxes])


class Sample:
    '''represents one line (sample) in the data set, which is composed of an image, a sentence, and a label
    which the truth value of the sentence w.r.t the image
    '''
    def __init__(self, line):
        self.evals = line["evals"]
        self.identifier = line["identifier"]
        self.sentence = line["sentence"]
        self.label = line["label"] == 'true'
        self.structured_rep = Image(line["structured_rep"])
        self.abstraction_dict = {}



class Size(Enum):
    SMALL = 10
    MEDIUM = 20
    BIG = 30


class Color(Enum):
    YELLOW = 'Yellow'
    BLACK = 'Black'
    BLUE = '#0099ff'


class Shape(Enum):
    CIRCLE = 'circle'
    SQUARE = 'square'
    TRIANGLE = 'triangle'


class Location(Enum):
    TOP = 'top'
    SECOND = 'second'
    BOTTOM = 'bottom'


class Relation(Enum):
    ABOVE = 'above'
    BELOW = 'below'
    TOUCH = 'touch'
    CLOSELY_TOUCH = 'closely touch'


class Side(Enum):
    RIGHT = 'right',
    LEFT = 'left',
    TOP = 'top',
    BOTTOM = 'bottom'
    ANY = 'any'


class TemporalOp(Enum):
    UNTIL = 'until'
    FINALLY = 'finally'
    NEXT = 'next'
    GLOBALLY = 'globally'


class LogicalOp(Enum):
    AND = 'and'
    OR = 'or'
    NOT = 'not'


class Direction(Enum):
    ABOVE = 'left'
    BELOW = 'right'
    UP = 'up'
    DOWN = 'down'


def is_operand(x):
    ltl_operators = ["F", "U", "N", "G", "&", "|", "~"]
    return ((x >= 'a' and x <= 'z') or
            (x >= 'A' and x <= 'Z') and x not in ltl_operators)

def convert_from_postfix(ltl_task):
    s = []
    print(ltl_task)
    for i in ltl_task:
        if (is_operand(i)):
            s.insert(0, i)
        else:
            op1 = s[0]
            s.pop(0)
            op2 = s[0]
            s.pop(0)
            s.insert(0, "(" + op2 + i + op1 + ")")
    return s[0]

def convert_from_symbols(ltl_task, symbol2prop):
    ltl_task = [item for item in ltl_task]
    for idx in range(len(ltl_task)):
        symbol = ltl_task[idx]
        if symbol in symbol2prop:
            ltl_task[idx] = symbol2prop[symbol]
    return " ".join(item for item in ltl_task)


def pretty_print_translations(instructions, ltl_tasks, symbol2prop):
    for idx in range(len(instructions)):
        print("Natural language instruction:\t", instructions[idx])
        ltl_task = convert_from_postfix(ltl_tasks[idx])
        ltl_task = convert_from_symbols(ltl_task, symbol2prop)
        print("LTL task specification:\t", ltl_task)

def get_grammar(type):
    pass

def load_functions(filename):
    functions_dict = {}
    with open(filename) as functions_file:
        for i, line in enumerate(functions_file):
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            entry = line.split()

            split_idx = entry.index(':') if ':' in entry else len(entry)
            entry, necessary_words = entry[:split_idx], entry[split_idx:]

            if len(entry) < 3 or not entry[1].isdigit() or int(entry[1]) != len(entry) - 3:
                print("could not parse function in line  {0}: {1}".format(i, line))
                # should use Warning instead
                continue
            token, return_type, args_types = entry[0], entry[-1], entry[2:-1]
            functions_dict[token] = TokenTypes(return_type=return_type, args_types=args_types,
                                               necessity=necessary_words)
        functions_dict['1'] = TokenTypes(return_type='int', args_types=[], necessity=['1', 'one', 'a'])
        functions_dict.update(
            {str(i): TokenTypes(return_type='int', args_types=[], necessity=[str(i)]) for i in range(2, 10)})

    return functions_dict

def parse_json(js):
    data = json.dumps(js['elements'])
    y = yaml.load(data)
    return y

nodes = {}
named_nodes = {}
ways = {}

def build_ways(y):
    for landmark in y["elements"]:
        # Creating dict of node id's to lat/long
        if landmark["type"] == "node":
            nodes[landmark["id"]] = (landmark["lat"], landmark["lon"])

        # Processing named landmarks
        if "tags" in landmark and "name" in landmark["tags"]:

            # If landmark is way, use one of its bounding nodes for lat/long
            if landmark["type"] == "way":
                l_name = landmark["tags"]["name"]
                ways[l_name] = []

                for node_id in landmark["nodes"]:
                    ways[l_name].append((nodes[node_id][0], nodes[node_id][1]))

            if landmark["type"] == "node":
                n_name = landmark["tags"]["name"]
                named_nodes[n_name] = (landmark["lat"], landmark["lon"])

def get_ways():
    global ways
    return ways

def get_nodes():
    global named_nodes
    return named_nodes
