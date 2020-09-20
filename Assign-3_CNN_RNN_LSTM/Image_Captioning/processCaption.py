import fileinput

filename = '.\Image_Caption_data\\6\\train_6_preprocessed.txt'

with fileinput.FileInput(filename, inplace=True) as file:
    for line in file:
        print(line.replace('.jpg| ', '.jpg#'), end='')

with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace('| ', '\t'), end='')

