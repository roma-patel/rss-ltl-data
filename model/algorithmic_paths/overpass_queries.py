import json
import os

class QueryMap(object):
    overpass_url = "http://overpass-api.de/api/interpreter"
    _MAP_NAMES = ["brown", "harvard", "upenn", "mit", "nyu", "columbia", "umass",
                  "tufts"]
    _NAME_REGISTRY = {
        "brown": "Brown University",
        "harvard": "Harvard University",
        "mit": "Massachusetts Institute of Technology", # cambridge
        "upenn": "University of Pennsylvania", # philadelphia
        "yale": "Yale University", # new haven
        "austin": "University of Austin at Texas",
        "ann-arbor": "University of Michigan",
        "atlanta": "Georgia Insititute of Technology",
        "baltimore": "Johns Hopkins University",
        "berkeley": "Berkeley Art Museum",
        "seattle": "University of Washington, Seattle",
        "temp": "Brown University",
    }
    _COORD_REGISTRY = {
        "test-case": (39.9512,-75.1991, 39.9555,-75.1889),
        "brown": (41.825117, -71.405640, 41.827596, -71.399009),
        "harvard": (42.3669, -71.1313, 42.3748,-71.1125),
        "mit": (42.3556,-71.1060, 42.3684,-71.0921),
        "upenn": (39.9512,-75.1991, 39.9555,-75.1889),
        "yale": (41.30872,-72.93117, 41.31052,-72.92312), # 77
        "austin": (30.28539,-97.74083, 30.28994,-97.73106), # 82
        "ann-arbor": (42.2732,-83.7455, 42.2801,-83.7315), # 201
        "atlanta": (33.77331,-84.40129, 33.77746,-84.39121),
        "baltimore": (39.32466,-76.62322, 39.32937,-76.61593),
        "berkeley": (37.87051,-122.26683, 37.87514,-122.25768),
    }

    def get_map_query(self, map_name):
        return """
        [out:json];
        (node["name"]%s;
        way["name"!="%s"]["name"][!"highway"][!"railway"]%s;
        );
        (._;>;);
        out body;""" % (str(self._COORD_REGISTRY[map_name]),
                        str(self._NAME_REGISTRY[map_name]),
                        str(self._COORD_REGISTRY[map_name]))


    def save_dictionary(self, dirpath):
        f = open(dirpath + "map_query_dictionary.json")
        f.write(json.dumps(self._MAP_NAME_REGISTRY))



if __name__ == '__main__':
    map = QueryMap()
    print(map._MAP_NAME_REGISTRY)
