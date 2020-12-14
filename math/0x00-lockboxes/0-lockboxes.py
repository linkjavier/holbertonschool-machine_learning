#!/usr/bin/python3
""" Module to search key """


def canUnlockAll(boxes):
    """ Function that determines if all the boxes can be opened"""

    keyStorage = []
    boxesLenght = len(boxes)

    for i in range(boxesLenght - 1):
        isKey = 0

        for j in range(len(boxes[i])):
            keyStorage.append(boxes[i][j])

            if boxes[i][j] == i + 1:
                isKey = 1

        if i + 1 in keyStorage:
            continue

        for k in range(i + 2, boxesLenght):
            if k in keyStorage:
                for l in range(len(boxes[k])):
                    keyStorage.append(boxes[k][l])
                    if boxes[k][l] == i + 1:
                        isKey = 1
        if isKey == 0:

            return False

    return True
