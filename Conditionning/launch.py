import os

toDoList = "C:/Users/Valentin/Desktop/testfile.txt"

with open(toDoList, 'r') as f:
    tasks = f.readlines()

try:
    for i in range(len(tasks)):
        if tasks[i].split()[0] == 'OK':
            pass
        else:
            try:
                os.system(tasks[i])
                tasks[i] = 'OK ' + tasks[i]
            except KeyboardInterrupt:
                break
finally:
    with open(toDoList, 'w') as f:
        f.writelines(tasks)
