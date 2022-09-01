import argparse
import pathlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="data/data_test.csv",
                        help="Path to the generated trajectories.")
    parser.add_argument("--epochs", nargs="+", default="0",
                        help="Epochs to be plotted.")
    parser.add_argument("--online_view", action="store_true",
                        help="Whether to show online view or generate gif.")
    parser.add_argument("--output_path", type=str, default="img",
                        help="The path to write generated gifs to.")
    parser.add_argument("--output_file", type=str,
                        default="test", help="The prefix of the gif.")
    args = parser.parse_args()

    # create output path
    path = pathlib.Path(args.output_path)
    if not args.online_view:
        if not path.exists():
            path.mkdir(parents=True)

    # get data
    data = np.genfromtxt(args.csv_file, delimiter=",")

    fig, ax = plt.subplots()

    # setup all plots
    # spawn of the agent
    ax.plot(0, 0, "x", c="black", label="Spawn")

    # adding a circle around the goal that indicates maximum distance to goal before the environment gets reset
    circle = plt.Circle((data[0, 3], data[0, 4]), 10, linestyle="--",
                        color="gray", fill=False, label="Maximum Goal Distance")
    ax.add_patch(circle)

    agent, = ax.plot(data[0, 1], data[0, 2], "o",
                     c="b", label="Agent")  # agent
    # small tail following the agent
    agent_line, = ax.plot(data[0, 1], data[0, 2], "-", c="b")
    goal, = ax.plot(data[0, 3], data[0, 4], "o", c="r", label="Goal")   # goal

    # plot settings
    ax.set_xlabel("x / a.u.")
    ax.set_ylabel("y / a.u.")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Agent in Test Environment")
    ax.legend()
    title = ax.text(0.15, 0.85, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
                    transform=ax.transAxes, ha="center")

    # plot everything
    for e in args.epochs:
        e = int(e)

        epoch_data = data[np.where(data[:, 0] == e)]

        # tail for the agent
        global tail, frame
        tail, frame = 0, 0

        def animate(i):
            global tail, frame
            agent.set_data(epoch_data[frame, 1], epoch_data[frame, 2])
            # AGENT enum in main.cpp, 1, 2, 3 = WON, LOST, RESETTING
            if (epoch_data[frame, 5] in [1, 2, 3]):
                tail = 0
            agent_line.set_data(
                epoch_data[frame-tail:frame, 1], epoch_data[frame-tail:frame, 2])
            if (tail < 50):
                tail += 1
            goal.set_data(epoch_data[frame, 3], epoch_data[frame, 4])
            circle.center = (epoch_data[frame, 3], epoch_data[frame, 4])
            title.set_text("Epoch {:1.0f}".format(epoch_data[frame, 0]))
            frame += 1
            return agent, agent_line, goal, circle, title

        ani = animation.FuncAnimation(
            fig, animate, blit=True, interval=5, frames=1000)
        if args.online_view:
            plt.show()
        else:
            ani.save(f"{path.absolute()}/{args.output_file}_{e}.gif",
                     writer="imagemagick", fps=100)


if __name__ == "__main__":
    main()
