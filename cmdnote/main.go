package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"time"
)

type Configuration struct {
	Editor        string
	Browser       string
	LocalRepoDir  string
	RemoteRepoDir string
}

var conf Configuration = Configuration{
	"nvim",
	"less",
	"",
	"",
}

// load configuration from conf file
func GetConf(confFilePath string) Configuration {
	content, err := ioutil.ReadFile(confFilePath)
	if err != nil {
		log.Fatal(err)
	}
	err = json.Unmarshal(content, &conf)
	if err != nil {
		log.Fatal(err)
	}
	return conf
}

// get a time-formated filename
func getTimeFileName() string {
	t := time.Now()
	filename := t.Format("2006-01-02_15-04-05") + ".txt"
	return filename
}

// invoke external commands
func invoke(prog string, args []string) {
	cmd := exec.Command(prog, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
}

// read a note
func readNote(filePath string) {
	invoke(conf.Browser, []string{filePath})
}

//write a note
func writeNote(filePath string) {
	invoke(conf.Editor, []string{filePath})
}

func main() {
	conf := GetConf("conf.json")
	fmt.Println(conf.Editor, " ", conf.Browser)
	fmt.Println(getTimeFileName())
	// writeNote(getTimeFileName())
	// readNote("2020-07-19_11-49-29.txt")
}
