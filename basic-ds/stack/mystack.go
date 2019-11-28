package main

import (
    "fmt"
    "errors"
)

type any interface{}

type MyStack struct {
    list []any
    size int
}

func (s *MyStack) IsEmpty() bool {
    return s.size == 0
}

func (s *MyStack) Size() int {
    return s.size
}

func (s *MyStack) Push(item any) {
    s.list = append(s.list, item)
    s.size++
}

func (s *MyStack) Pop() (any, error) {
    if s.IsEmpty() {
        return nil, errors.New("Pop from empty stack!")
    }
    item := s.list[len(s.list) - 1]
    s.list = s.list[:len(s.list)-1]
    s.size--
    return item, nil
}

func NewMyStack() MyStack {
    stack := MyStack{make([]any, 0), 0}
    return stack
}

func main() {
    // s1 := new(MyStack)
    s1 := NewMyStack()
    s1.Push(1024)
    s1.Push("hello world!")
    s1.Push(true)
    fmt.Println(s1.IsEmpty())
    fmt.Println("------------------------------")
    fmt.Println(s1.Pop())
    fmt.Println(s1.Pop())
    fmt.Println(s1.Pop())
    fmt.Println(s1.IsEmpty())
}
