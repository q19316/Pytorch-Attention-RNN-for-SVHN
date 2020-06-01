import matplotlib.pyplot as plt
import numpy as np
import csv

SMOOTHING = 0.9
FILENAME = "logs/loss_history.csv"

def main():
    csv_file = open(FILENAME, 'r')
    read_lines = csv.reader(csv_file)

    steps = []
    losses = []
    for line in read_lines:
        steps.append(int(line[0]))
        losses.append(float(line[1]))

    h = losses[0]
    for i in range(len(losses)):
        h = SMOOTHING * h + (1-SMOOTHING) * losses[i]
        losses[i] = h

    plt.figure()
    plt.scatter(steps, losses)
    plt.ylim(0., 0.5)
    plt.show()


if __name__ == '__main__':
    main()
