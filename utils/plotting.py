import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    # fig1 = plt.figure(figsize=(10,5))
    # plt.plot(stats.episode_lengths)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Length")
    # plt.title("Episode Length over Time")
    # if noshow:
    #     plt.close(fig1)
    # else:
    #     plt.show(fig1)

    # Plot the episode reward over time
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed, linewidth=3)
    plt.xlabel("Episode", fontsize=30)
    plt.ylabel("Cumulative Reward", fontsize=30)
    plt.grid()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if noshow:
        plt.close()
    else:
        plt.show()


    # Plot time steps and episode number
    # fig3 = plt.figure(figsize=(10,5))
    # plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    # plt.xlabel("Time Steps")
    # plt.ylabel("Episode")
    # plt.title("Episode per time step")
    # if noshow:
    #     plt.close(fig3)
    # else:
    #     plt.show(fig3)


def draw_reward_plot(rewards, smooth):
    rewards_smoothed = pd.Series(rewards).rolling(smooth, min_periods=smooth).mean()
    plt.plot(rewards_smoothed, linewidth=5, color='g')

    plt.ylabel('Cost', fontsize=40)
    plt.xlabel('Episodes', fontsize=40)
    plt.legend(loc='upper left')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(prop={'size': 33})
    plt.grid()
    plt.show()

    record_rewards(rewards)


def record_rewards(rewards):
    f = open("save/result.txt", "a")
    f.write("r1=" + str(rewards) + "\n")

    f.write("________________________END EXPERMINT____________________________" + "\n")


