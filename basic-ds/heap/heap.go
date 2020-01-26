

type interface{} any

type MinHeap {
    list []any
    size int
}

func (h *MinHeap) Size() int {
    return h.size
}

func (h *MinHeap) IsEmpty() bool {
    return h.Size() == 0
}

func (h *MinHeap) Push(item any) {
    return nil
}

func (h *MinHeap) Pop() (any, error) {
    return nil, nil
}

func (h*MinHeap) shitfUp(idx int) {
    return nil
}

func (h *MinHeap) shftDown(idx int) {
    return nil
}

func NewMinHeap() MinHeap {
    return nil
}

func main(){}
