package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"time"

	"github.com/blevesearch/bleve"
)

const (
	BleveFolder = "notes.bleve"
)

type Configuration struct {
	Editor        string `json:"editor"`
	Browser       string `json:"browser"`
	LocalRepoDir  string `json:"localRepoDir,omitempty"`
	RemoteRepoDir string `json:"remoteRepoDir,mitempty"`
}

var CONF Configuration = Configuration{
	"nvim",
	"less",
	GetDefaultLocalRepoDir(),
	"",
}

func init(repoPath string) {
	if !os.IsExist(repoPath) {
		os.MkdirAll(repoPath, os.ModePerm)
	}
	if !os.IsExist(os.Join(repoPath, BleveFolder)) {
		mapping := bleve.NewIndexMapping()
		index, err := bleve.New(BleveFolder, mapping)
		if err != nil {
			log.Fatal(err)
		}
	}
	if !os.IsExist(os.Join(repoPath, ".git")) {
		// git init
	}
}

func GetDefaultLocalRepoDir() string {
	fName, err := filepath.Abs(os.Args[0])
	if err != nil {
		log.Fatal(err)
	}
	return path.Join(path.Dir(fName), "cmd_notes")
}

// load configuration from CONF file
func LoadConf(confFilePath string) Configuration {
	content, err := ioutil.ReadFile(confFilePath)
	if err != nil {
		log.Fatal(err)
	}
	err = json.Unmarshal(content, &CONF)
	if err != nil {
		log.Fatal(err)
	}
	return CONF
}

// dump confFilePath to CONF file
func DumpConf(confFilePath string) {
	content, err := json.MarshalIndent(CONF, "", "\t")
	if err != nil {
		log.Fatal(err)
	}
	err = ioutil.WriteFile(confFilePath, content, 0644)
	if err != nil {
		log.Println(err)
	}
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
func readNote(prog, notePath string) {
	invoke(prog, []string{notePath})
}

//write a note
func writeNote(prog, notePath string) {
	fileDir := path.Dir(notePath)
	fmt.Println(fileDir)
	err := os.MkdirAll(fileDir, os.ModePerm)
	if err != nil {
		log.Fatal(err)
	}
	invoke(prog, []string{notePath})
}

func searchNote(keywords string) []string {
	return nil
}

// recursively list all files under a directory
func AllFilePaths(dir string) []string {
	allFiles := []string{}
	err := filepath.Walk(dir,
		func(p string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			// fmt.Println(p, info.Size())
			if s, _ := os.Stat(p); !s.IsDir() {
				allFiles = append(allFiles, p)
			}
			return nil
		})

	if err != nil {
		log.Println(err)
	}

	return allFiles
}

func AllNoteNames() []string {
	allFilePaths := AllFilePaths(CONF.LocalRepoDir)
	allNotes := []string{}
	for _, fpath := range allFilePaths {
		noteName, _ := filepath.Rel(CONF.LocalRepoDir, fpath)
		allNotes = append(allNotes, noteName)
	}
	return allNotes
}

// list all notes in local repository
func listNotes() {
	allNotes := AllNoteNames()
	for _, noteName := range allNotes {
		fmt.Println(noteName)
	}
}

// list all notes in local repository, and provide interactive inspection
func listNotesInteractive() {
	allNotes := AllNoteNames()
	for i, noteName := range allNotes {
		fmt.Printf("%5d) %s\n", i, noteName)
	}

	interactiveSession(allNotes)
}

func interactiveSession(noteNames []string) {
	var noteId int
	var prog string
	var err error

	for {
		fmt.Print("Which note would you like to check? ")
		_, err = fmt.Scanln(&noteId)
		if err != nil {
			if err.Error() == "unexpected newline" {
				break
			} else {
				log.Fatal(err)
			}
		}

		// fmt.Println(noteId, int(noteId))
		fmt.Print("Which program would you like to use? ")
		_, err = fmt.Scanln(&prog)
		if err != nil {
			if err.Error() == "unexpected newline" {
				prog = CONF.Browser
			} else {
				log.Fatal(err)
			}
		}

		fmt.Println(prog, noteNames[noteId])
		notePath := path.Join(CONF.LocalRepoDir, noteNames[noteId])
		invoke(prog, []string{notePath})
	}
}

func main() {
	init(CONF.LocalRepoDir)
	// fmt.Println(CONF)
	DumpConf(path.Join(path.Dir(GetDefaultLocalRepoDir()), "conf.json"))
	// writeNote("nvim", path.Join(CONF.LocalRepoDir, getTimeFileName()))
	// readNote("less", "2020-07-19_11-49-29.txt")
	listNotesInteractive()
}
