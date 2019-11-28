package main

import (
    "fmt"
    "errors"
)

type any interface{}

type MyQueue struct {
    list []any
    size int
}

func (q *MyQueue) Size() int {
    return q.size
}

func (q *MyQueue) IsEmpty() bool {
    return q.Size() == 0
}

func (q *MyQueue) Enqueue(item any) {
    q.list = append(q.list, item)
    q.size++
}

func (q *MyQueue) Dequeue() (any, error) {
    if q.IsEmpty() {
        return 0, errors.New("Dequeue from empty queue!")
    }
    item := q.list[0]
    q.list = q.list[1:]
    q.size--
    return item, nil
}

func NewQueue() MyQueue {
    return MyQueue{make([]any, 0), 0}
}

func main() {
    q1 := NewQueue()
    fmt.Println(q1.Size())
    q1.Enqueue(1024)
    q1.Enqueue("hello world!")
    countryCapitalMap := make(map[string]string)
    countryCapitalMap [ "France"  ] = "巴黎"
    q1.Enqueue(countryCapitalMap)
    fmt.Println(q1.Dequeue())
    fmt.Println(q1.Dequeue())
    fmt.Println(q1.Dequeue())
    fmt.Println(q1.IsEmpty())
}
