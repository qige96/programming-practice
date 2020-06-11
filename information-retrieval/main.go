package main

import (
    "fmt"
    "strings"
)

func main() {
    docs := []string{
        "hello world",
        "hello golang",
        "golang is made by Google",
        "Google might be the most powerful company in this world",
    }
    doc_dict := make(map[int]string)
    tDocs := make([][]string, 0, 4)
    for i, s := range docs {
        doc_dict[i] = s
        tDocs = append(tDocs, strings.Split(s, " "))
    }

    index := make(map[string][]int)
    for i, ts := range tDocs {
        for _, term := range ts {
            index[term] = append(index[term], i)
        }
    }


    // index := map[string][]int{
    //     "hello": []int{0,1,2},
    //     "world": []int{0, 3, 4},
    // }
    fmt.Println(index)
}
