import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Set random seed for reproducibility
np.random.seed(1000)


width = 15
height = 5

y_final = width - 1
x_final = height - 1

y_wells = [0, 1, 3, 5, 5, 7, 9, 11, 12, 14]
x_wells = [3, 1, 2, 0, 4, 1, 3, 2, 4, 1]


# Initial tunnel rewards
standard_reward = -0.1
tunnel_rewards = np.ones(shape=(height, width)) * standard_reward

for x_well, y_well in zip(x_wells, y_wells):
    tunnel_rewards[x_well, y_well] = -5.0

tunnel_rewards[x_final, y_final] = 5.0


gamma = 0.9
nb_actions = 4

Q = np.zeros(shape=(height, width, nb_actions))

x_start = 0
y_start = 0

max_steps = 2000
alpha = 0.25
n_episodes = 20000
n_exploration = 15000


def show_values(t):
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.matshow(np.max(Q, axis=2), cmap=cm.Pastel1)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_title('Values (t={})'.format(t))

    for i in range(height):
        for j in range(width):
            if i == x_final and j == y_final:
                msg = 'E'
            elif (i, j) in zip(x_wells, y_wells):
                msg = r'$\otimes$'
            else:
                msg = '{:.2f}'.format(np.max(Q[i, j]))
            ax.text(x=j, y=i, s=msg, va='center', ha='center')

    plt.show()


def show_policy(t):
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.matshow(np.zeros_like(Q[:, :, 0]), cmap=cm.Pastel1)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_title('Policy (t={})'.format(t))

    for i in range(height):
        for j in range(width):
            action = np.argmax(Q[i, j])

            if i == x_final and j == y_final:
                msg = 'E'
            elif (i, j) in zip(x_wells, y_wells):
                msg = r'$\otimes$'
            else:
                if action == 0:
                    msg = r'$\uparrow$'
                elif action == 1:
                    msg = r'$\rightarrow$'
                elif action == 2:
                    msg = r'$\downarrow$'
                else:
                    msg = r'$\leftarrow$'

            ax.text(x=j, y=i, s=msg, va='center', ha='center')

    plt.show()


def is_final(x, y):
    if (x, y) in zip(x_wells, y_wells) or (x, y) == (x_final, y_final):
        return True
    return False


def select_action(epsilon, i, j):
    if np.random.uniform(0.0, 1.0) < epsilon:
        return np.random.randint(0, nb_actions)
    return np.argmax(Q[i, j])


def sarsa_step(epsilon):
    e = 0

    i = x_start
    j = y_start

    while e < max_steps:
        e += 1

        action = select_action(epsilon, i, j)

        if action == 0:
            if i == 0:
                x = 0
            else:
                x = i - 1
            y = j

        elif action == 1:
            if j == width - 1:
                y = width - 1
            else:
                y = j + 1
            x = i

        elif action == 2:
            if i == height - 1:
                x = height - 1
            else:
                x = i + 1
            y = j

        else:
            if j == 0:
                y = 0
            else:
                y = j - 1
            x = i

        action_n = select_action(epsilon, x, y)
        reward = tunnel_rewards[x, y]

        if is_final(x, y):
            Q[i, j, action] += alpha * (reward - Q[i, j, action])
            break

        else:
            Q[i, j, action] += alpha * (reward + (gamma * Q[x, y, action_n]) - Q[i, j, action])

            i = x
            j = y


if __name__ == '__main__':
    # Show tunnel rewards
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.matshow(tunnel_rewards, cmap=cm.Pastel1)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_title('Rewards')

    for i in range(height):
        for j in range(width):
            msg = '{:.2f}'.format(tunnel_rewards[i, j])
            ax.text(x=j, y=i, s=msg, va='center', ha='center')

    plt.show()

    # Show initial values
    show_values(t=0)

    # Show initial policy
    show_policy(t=0)

    # Train the model
    for t in range(n_episodes):
        epsilon = 0.0

        if t <= n_exploration:
            epsilon = 1.0 - (float(t) / float(n_exploration))

        sarsa_step(epsilon)

    # Show final values
    show_values(t=n_episodes)

    # Show final policy
    show_policy(t=n_episodes)