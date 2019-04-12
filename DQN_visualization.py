import matplotlib
import matplotlib.pyplot as plt
from utils.plot import load_reward_data
import numpy as np


def plot_reward(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='results.png'):
    # matplotlib.rcParams.update({'font.size': 11})
    tx, ty = load_reward_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    # fig = plt.figure(figsize=(20, 5))
    plt.plot(tx, ty, label="{}".format(name))

    # tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-10:]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-10:]))))
        print(ty)
    plt.legend(loc=4)
    # plt.show()
    # plt.savefig(save_filename)
    # plt.clf()
    # plt.close()

    return np.round(np.mean(ty[-10]))

# env_ids = [env + 'NoFrameskip-v4' for env in ['AirRaid', 'Assault', 'Carnival', 'DemonAttack']]
env_ids = [env + 'NoFrameskip-v4' for env in ['Assault', 'AirRaid', 'DemonAttack', 'Carnival']]

plt.figure()
matplotlib.rcParams.update({'font.size': 13})
matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)
for idx, env_id in enumerate(env_ids):
    log_dir = 'log/' + env_id + '/'
    plt.subplot(2, 2, idx+1)
    plot_reward(log_dir, env_id[:-14], 'DQN', 1000000, bin_size=10, smooth=1)
plt.show()

