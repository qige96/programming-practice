
class Vertex:
    def __init__(self, vid, vname, vdata):
        self.vid = vid
        self.vname = vname
        self.vdata = vdata


class Graph:
    def __init__(self):
        self.vertices = {}
        self.adjlist = {}
        self.v_size = 0
        self.e_size = 0
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.__repr__())

    def display_graph(self):
        output = ''
        output += '{:<10}  {:<100}\n'.format('Vertices', 'Edges(out)')
        for v, e in self.adjlist.items():
            output += '{:<10}  {:<100}\n'.format(str(v), str(e))
        print(output)
        return output

    def is_empty(self):
        return self.v_size == 0

    def size(self):
        return (self.v_size, self.e_size)

    def has_vertex(self, vid):
        return vid in self.vertices

    def has_edge(self, vfrom_id, vto_id):
        return vto_id in self.adjlist[vfrom_id]

    def add_vertex(self, vname=None, vdata=None):
        v = Vertex(self.v_size, vname, vdata)
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

if __name__ == '__main__':
    g = Graph()
    
    va = g.add_vertex('a')
    vb = g.add_vertex('b')
    vc = g.add_vertex('c')
    vd = g.add_vertex('d')

    g.add_edge(va.vid, vb.vid)
    g.add_edge(vb.vid, va.vid)
    g.add_edge(va.vid, vc.vid)
    g.add_edge(vb.vid, vd.vid)

    g.display_graph()
    print(g.size())

    print('-------------------------------------------')

    g.remove_vertex(va.vid)
    g.display_graph()
    print(g.size())
