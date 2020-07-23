import argparse
import datetime
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG = {
        'writer': 'nvim',
        'reader': 'less',
        'abstracter': 'grep',
        'localRepoDir': os.path.join(BASE_DIR, 'cmd_notes'),
        'remoteRepoURI': ''
        }

def create_and_write():
    d = datetime.datetime.now()
    filename = d.strftime('%Y-%m-%d_%H-%M-%S.txt')
    fullname = os.path.join(CONFIG['localRepoDir'], filename)
    os.makedirs(os.path.dirname(fullname), exist_ok=True)
    subprocess.call([CONFIG['writer'], fullname])

def writenote(filename):
    fullname = os.path.join(CONFIG['localRepoDir'], filename)
    os.makedirs(os.path.dirname(fullname), exist_ok=True)
    subprocess.call([CONFIG['writer'], fullname])

def readnote(filename):
    fullname = os.path.join(CONFIG['localRepoDir'], filename)
    subprocess.call([CONFIG['reader'], fullname])

def list_files_recursive(path):
    """
    Function that receives as a parameter a directory path

    :return list_: File List and Its Absolute Paths
    """
    files = []

    # r = root, d = directories, f = files
    for r, _, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    lst = [file for file in files]
    return lst

def interactive(results):
    print('')
    while True:
        noteno = input('Which note would you like to check? ')
        if noteno == '':
            break
        prog = input('Which programm would you like to use?(default: less) ')
        if prog == '':
            prog = 'less'
        subprocess.call([prog, results[int(noteno)][1]])

def listnotes(is_interactive=False):
    files = list_files_recursive(CONFIG['localRepoDir'])
    results = []
    for f in files:
        if not os.path.isdir(f):
            results.append([f.replace(CONFIG['localRepoDir'], '')[1:], f])
            
    for i, entry in enumerate(results):
        print('[{0}] {1}'.format(i, entry[0]))

    if is_interactive:
        interactive(results)

def search(keywords, is_interactive=False):
    files = list_files_recursive(CONFIG['localRepoDir'])
    results = []
    for f in files:
        if not os.path.isdir(f):
            results.append([f.replace(CONFIG['localRepoDir'], '')[1:], f])
            
    for i, entry in enumerate(results):
        print('[{0}] {1}'.format(i, entry[0]))

    if is_interactive: 
        interactive(results)


def main():
    parser = argparse.ArgumentParser(description='Command line note taking tool.')
    subparsers = parser.add_subparsers(dest='subparser_name')
    write_parser = subparsers.add_parser('w')
    write_parser.add_argument('filename', nargs='?')
    write_parser.add_argument('--writer')

    read_parser = subparsers.add_parser('r')
    read_parser.add_argument('filename')
    read_parser.add_argument('--reader')

    search_parser = subparsers.add_parser('s')
    search_parser.add_argument('keywords')
    search_parser.add_argument('-i', '--interactive', dest='is_interactive', action='store_true')

    ls_parser = subparsers.add_parser('ls')
    ls_parser.add_argument('-i', '--interactive', dest='is_interactive', action='store_true')

    args = parser.parse_args()

    if args.subparser_name == 'r':
        readnote(args.filename)
    elif args.subparser_name == 'w':
        writenote(args.filename)
    elif args.subparser_name == 'ls':
        listnotes(args.is_interactive)
    elif args.subparser_name == 's':
        search(args.keywords, args.is_interactive)
    elif args.subparser_name == None:
        create_and_write() 


if __name__ == "__main__":
    main()
