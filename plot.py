import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# get data
data = np.genfromtxt("build/data.csv", delimiter=",")

fig, ax = plt.subplots()

#x = np.arange(0, data.shape[0])
game, = ax.plot(data[0,1], data[0,2], 'o', c='b', label='agent')
ax.plot(0, 0, 'x', c='black', label='spawn')
ax.plot(data[0,3], data[0,4], 'o', c='r', label='goal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Agent in Test Environment")
ax.legend()
title = ax.text(0.15,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

def animate(i):
    game.set_data(data[i,1], data[i,2])
    title.set_text('Epoch {:1.0f}'.format(data[i,0]))
    return game, title


ani = animation.FuncAnimation(fig, animate, blit=True, interval=5)
plt.show()


