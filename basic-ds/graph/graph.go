package main

import (
    "fmt"
    "strconv"
    "strings"
)

type any interface{}

type Vertex struct {
    vid string
    vdata map[string]any
}

type Graph struct {
    vertices map[string][]Vertex
    adjlist map[string][]string
    vSize int
    eSize int
}

func (g *Graph) IsEmpty() (bool) {
    return g.vSize == 0
}

func (g *Graph) Size() (int, int) {
    return g.vSize, g.eSize
}

func (g *Graph) AddVertex(vdata map[string]any) (Vertex, error) {
    vid := strconv.Itoa(g.vSize)
    v := Vertex{vid, vdata}
    g.vertices[vid] = append(g.vertices[vid], v)
    g.adjlist[vid] = make([]string, 0)
    g.vSize++
    return v, nil
}

func (g *Graph) AddEdge(vFromId string, vToId string) {
    g.adjlist[vFromId] = append(g.adjlist[vFromId], vToId)
    g.eSize++
}

func (g *Graph) RemoveVertex(vid string) {
    _, ok := g.vertices[vid]
    if ok {
        for k := range g.vertices {
            g.RemoveEdge(vid, k)
            g.RemoveEdge(k, vid)
        }
        delete(g.vertices, vid)
        delete(g.adjlist, vid)
        g.vSize--
    }
}

func (g *Graph) RemoveEdge(vFromId string, vToId string) {
    for i := 0; i < len(g.adjlist[vFromId]); {
        if g.adjlist[vFromId][i] == vToId {
            copy(g.adjlist[vFromId][i:], g.adjlist[vFromId][i+1:])
            g.adjlist[vFromId] = g.adjlist[vFromId][:len(g.adjlist[vFromId])-1]
            g.eSize--
            break
        }
        i++
    }
}

func DisplayGraph (g Graph) {
    fmt.Printf("%-10s  %s\n", "Vertex", "Edges(out)")
    fmt.Println(strings.Repeat("-", 22))
    for vFromId, vToList := range g.adjlist {
        fmt.Printf("%-10s  %v\n", vFromId, vToList)
    }
    fmt.Println(strings.Repeat("-", 22))
    fmt.Printf("%-10d  %-10d\n\n", g.vSize, g.eSize)
}

func NewGraph() (Graph, error) {
    return Graph{make(map[string][]Vertex), make(map[string][]string), 0, 0}, nil
}

func main() {
    g, _ := NewGraph()

    v1, _ := g.AddVertex(nil)
    v2data := map[string]any{"name": "v2", "weight": 10}
    v2, _ := g.AddVertex(v2data)
    v3, _ := g.AddVertex(nil)

    g.AddEdge(v3.vid, v1.vid)
    g.AddEdge(v1.vid, v2.vid)
    g.AddEdge(v1.vid, v3.vid)
    DisplayGraph(g)

    g.RemoveVertex(v3.vid)
    DisplayGraph(g)

    fmt.Println(g)
}
