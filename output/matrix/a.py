import matplotlib.pyplot as plt
import numpy as np

for i in range(45, 200):

    name = "matrix_" + str(i) + ".txt"
    ma = np.loadtxt(name,skiprows=1)
    frame = None
    with open(name) as file:
        frame = file.readline()
    frame = int(frame.split(':')[1])
    plt.figure(figsize=(30, 30))
    plt.matshow(ma,fignum=1)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(30)
    plt.title("matrix: "+name+ "  frame:"+str(frame),fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.show()