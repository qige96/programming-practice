import uuid

class Vertex:
    def __init__(self, vid, vname, vdata):
        self.vid = vid
        self.vname = vname
        self.vdata = vdata


class Graph:
    def __init__(self, use_uuid=False):
        self.use_uuid = use_uuid
        self.vertices = {}
        self.adjlist = {}
        self.v_size = 0
        self.e_size = 0
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'vertices: ' + str(list(self.vertices.keys()))

    def display_graph(self):
        output = ''
        output += '{:<10}  {:<100}\n'.format('Vertices', 'Edges(out)')
        for v, e in self.adjlist.items():
            output += '{:<10}  {:<100}\n'.format(str(v), str(e))
        print(output)
        # return output

    def is_empty(self):
        return self.v_size == 0

    def size(self):
        return (self.v_size, self.e_size)

    def has_vertex(self, vid):
        return vid in self.vertices

    def has_edge(self, vfrom_id, vto_id):
        return vto_id in self.adjlist[vfrom_id]

    def add_vertex(self, vid=None, vname=None, vdata=None):
        if not vid:
            if self.use_uuid:
                vid = uuid.uuid4().hex
            else:
                vid = str(self.v_size)
        v = Vertex(vid, vname, vdata)
        self.vertices[v.vid] = v
        self.adjlist[v.vid] = []
        self.v_size += 1
        return v

    def add_edge(self, vfrom_id, vto_id):
        self.adjlist[vfrom_id].append(vto_id)
        self.e_size += 1

    def remove_vertex(self, vid):
        for k in self.vertices:
            if self.has_edge(k, vid):
                self.remove_edge(k, vid)
            if self.has_edge(vid, k):
                self.remove_edge(vid, k)
        del self.vertices[vid]
        self.v_size -= 1
        del self.adjlist[vid]

    def remove_edge(self, vfrom_id, vto_id):
        self.adjlist[vfrom_id].remove(vto_id)
        self.e_size -= 1

    def get_vertex(self, vid):
        return self.vertices[vid]

    def get_vertex_by_name(self, vname):
        for v in self.vertices.values():
            if v.vname == vname:
                return v

    def to_json(self):
        json_graph = {}
        json_graph['adjlist'] = self.adjlist
        json_graph['vertices'] = { 
                vid: v.__dict__ \
                        for vid, v in self.vertices.items() 
                }
        return json_graph


def dump_graph(g, filename='graph_data.json'):
    import json
    json.dump(g.to_json(), open(filename, 'w'), indent=4)

def load_graph(filename):
    import json
    json_graph = json.load(open(filename, 'r'))
    g = Graph()
    for v in json_graph['vertices'].values():
        g.add_vertex(v['vid'], v['vname'], v['vdata'])
    for vfrom_id, e in json_graph['adjlist'].items():
        for vto_id in e:
            g.add_edge(vfrom_id, vto_id)
    return g


if __name__ == '__main__':
    g = Graph()
    
    va = g.add_vertex(vname='a')
    vb = g.add_vertex(vname='b')
    vc = g.add_vertex(vname='c')
    vd = g.add_vertex(vname='d')

    g.add_edge(va.vid, vb.vid)
    g.add_edge(vb.vid, va.vid)
    g.add_edge(va.vid, vc.vid)
    g.add_edge(vb.vid, vd.vid)

    g.display_graph()
    dump_graph(g)
    print(g.size())

    print('-------------------------------------------')

    g.remove_vertex(va.vid)
    g.display_graph()
    print(g.size())
