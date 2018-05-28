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

gamma = 0.99
alpha = 0.25
rho = 0.001
nb_actions = 4
max_steps = 1000
n_episodes = 50000
n_exploration = 30000


# Initial tunnel rewards
standard_reward = -0.1
tunnel_rewards = np.ones(shape=(height, width)) * standard_reward

for x_well, y_well in zip(x_wells, y_wells):
    tunnel_rewards[x_well, y_well] = -5.0

tunnel_rewards[x_final, y_final] = 5.0

# Setup allowed starting points
xy_grid = np.meshgrid(np.arange(0, height), np.arange(0, width), sparse=False)
xy_grid = np.array(xy_grid).T.reshape(-1, 2)

xy_final = list(zip(x_wells, y_wells))
xy_final.append([x_final, y_final])

xy_start = []

for x, y in xy_grid:
    if (x, y) not in xy_final:
        xy_start.append([x, y])

xy_start = np.array(xy_start)


def starting_point():
    xy = np.squeeze(xy_start[np.random.randint(0, xy_start.shape[0], size=1)])
    return xy[0], xy[1]


tunnel_values = np.zeros(shape=(height, width))
policy_importances = np.zeros(shape=(height, width, nb_actions))


def show_values(t):
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.matshow(tunnel_values, cmap=cm.Pastel1)
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
                msg = '{:.1f}'.format(tunnel_values[i, j])
            ax.text(x=j, y=i, s=msg, va='center', ha='center')

    plt.show()


def show_policy(t):
    policy = get_softmax_policy()

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.matshow(np.zeros_like(tunnel_values), cmap=cm.Pastel1)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_title('Policy (t={})'.format(t))

    for i in range(height):
        for j in range(width):
            action = np.argmax(policy[i, j])

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


def get_softmax_policy():
    softmax_policy = policy_importances - np.amax(policy_importances, axis=2, keepdims=True)
    return np.exp(softmax_policy) / np.sum(np.exp(softmax_policy), axis=2, keepdims=True)


def is_final(x, y):
    if (x, y) in zip(x_wells, y_wells) or (x, y) == (x_final, y_final):
        return True
    return False


def select_action(epsilon, i, j):
    if np.random.uniform(0.0, 1.0) < epsilon:
        return np.random.randint(0, nb_actions)

    policy = get_softmax_policy()
    return np.argmax(policy[i, j])


def action_critic_episode(epsilon):
    (i, j) = starting_point()
    x = y = 0

    e = 0

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

        reward = tunnel_rewards[x, y]
        td_error = reward + (gamma * tunnel_values[x, y]) - tunnel_values[i, j]

        tunnel_values[i, j] += (alpha * td_error)
        policy_importances[i, j, action] += (rho * td_error)

        if is_final(x, y):
            break
        else:
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

        action_critic_episode(epsilon)

    # Show final values
    show_values(t=n_episodes)

    # Show final policy
    show_policy(t=n_episodes)


