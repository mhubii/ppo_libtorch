import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# get data
data = np.genfromtxt("data/data.csv", delimiter=",")

fig, ax = plt.subplots()

# setup all plots
ax.plot(0, 0, 'x', c='black', label='Spawn')                      # spawn of the agent

# adding a circle around the goal that indicates maximum distance to goal before the environment gets reset
circle = plt.Circle((data[0,3], data[0,4]), 10, linestyle='--', color='gray', fill=False, label='Maximum Goal Distance')
ax.add_patch(circle)

agent, = ax.plot(data[0,1], data[0,2], 'o', c='b', label='Agent') # agent
agent_line, = ax.plot(data[0,1], data[0,2], '-', c='b')           # small tail following the agent
goal, = ax.plot(data[0,3], data[0,4], 'o', c='r', label='Goal')   # goal

# plot settings
ax.set_xlabel('x / a.u.')
ax.set_ylabel('y / a.u.')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Agent in Test Environment")
ax.legend()
title = ax.text(0.15,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

# plot everything
epochs = np.array([1,5,10,15,20])

for e in epochs:

    epoch_data = data[np.where(data[:,0]==e)]

    # tail for the agent
    tail = 0

    def animate(i):
        agent.set_data(epoch_data[i,1], epoch_data[i,2])
        global tail
        if (epoch_data[i,5] == 3): # AGENT enum in main.cpp, 3 = RESETTING
            tail = 0
        agent_line.set_data(epoch_data[i-tail:i,1], epoch_data[i-tail:i,2])
        if (tail <= 50):
            tail += 1
        goal.set_data(epoch_data[i,3], epoch_data[i,4])
        circle.center = (epoch_data[i,3], epoch_data[i,4])
        title.set_text('Epoch {:1.0f}'.format(epoch_data[i,0]))
        return agent, agent_line, goal, circle, title


    ani = animation.FuncAnimation(fig, animate, blit=True, interval=5, frames=1000)
    plt.show()
    #ani.save('img/epoch_{}.gif'.format(e), writer='imagemagick', fps=100)
