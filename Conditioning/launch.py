import os
import argparse

# Parsing arguments

parser = argparse.ArgumentParser(description='Launch multiple simple conditioning runs')

parser.add_argument("filename", help="input text file listing command lines to launch", 
                    type=str)

args = parser.parse_args()

if not (os.path.exists(args.filename)):
    raise ValueError("Invalid path : {}".format(args.filename))

toDoList = args.filename

with open(toDoList, 'r') as f:
    tasks = f.readlines()

try:
    for i in range(len(tasks)):
        if tasks[i].split()[0] == 'OK':
            pass
        else:
            try:
                return_code = os.system(tasks[i])
            except:
                continue
            if return_code == 0:
                tasks[i] = 'OK ' + tasks[i]
finally:
    with open(toDoList, 'w') as f:
        f.writelines(tasks)

