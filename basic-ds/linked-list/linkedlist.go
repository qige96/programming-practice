package main

import (
    "fmt"
    "errors"
)

type any interface{}

type Node struct {
    data any
    prev *Node
    next *Node
}

func NewNode(data any) Node {
    return Node{data, nil, nil}
}

type LinkedList struct {
    head *Node
    tail *Node
    size int
}

func (l *LinkedList) Size() int {
    return l.size
}

func (l *LinkedList) IsEmpty() bool {
    return l.size == 0
}

func (l *LinkedList) AddLast(data any) (error) {
    temp := NewNode(data)
    temp.prev = l.tail
    if l.IsEmpty(){
        l.tail = &temp
        l.head = &temp
    } else {
        l.tail.next = &temp
        l.tail = &temp
    }
    l.size++
    return nil
}

func (l *LinkedList) AddFirst(data any) (error){
    temp := NewNode(data)
    temp.next = l.head
    if l.IsEmpty() {
        l.head = &temp
        l.tail = &temp
    } else {
        l.head.prev = &temp
        l.head = &temp
    }
    l.size++
    return nil
}

func (l *LinkedList) InsertAfter(n *Node, data any) (error) {
    temp := NewNode(data)
    temp.prev = n
    temp.next = n.next
    n.next.prev = &temp // this line must come first
    n.next = &temp
    l.size++

    return nil
}


func (l *LinkedList) InsertBefore(n *Node, data any) (error) {
    temp := NewNode(data)
    temp.next = n
    temp.prev = n.prev
    n.prev.next = &temp // this line must come first
    n.prev = &temp
    l.size++
    return nil
}

func (l *LinkedList) find(data any) (int, error) {
    ptr := l.head
    i := 0
    for ptr != nil {
        if ptr.data == data {
            return i, nil
        }
        ptr = ptr.next
        i++
    }
    return -1, nil
}

func (l *LinkedList) Remove(data any) (Node, error) {
    ptr := l.head
    temp := &Node{nil, nil, nil}
    for ptr != nil {
        if ptr.data == data {
            temp = ptr
            ptr.next = ptr.next.next
            l.size--
            break
        }
    }
    return *temp, nil
}

func (l *LinkedList) get(index int) (any, error) {
    if index < 0 && index >= l.Size() {
        return nil, errors.New("Index out of range!")
    }
    ptr := l.head
    i := 0
    for i < index {
        ptr = ptr.next
        i++
    }
    return ptr.data, nil
}

func NewList() LinkedList {
    return LinkedList{nil, nil, 0}
}

func main(){
    l1 := NewList()
    l1.AddLast(1)
    l1.AddFirst(0)
    fmt.Println(l1.IsEmpty())
    fmt.Println(l1.Size())
    fmt.Println(l1.find(1))
    fmt.Println(l1.get(1))
    l1.Remove(0)
    fmt.Println(l1.Size())
}
