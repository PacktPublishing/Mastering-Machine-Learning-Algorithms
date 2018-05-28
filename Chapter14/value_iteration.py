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

gamma = 0.9
nb_max_epochs = 100000
tolerance = 1e-5
nb_actions = 4

# Initial tunnel rewards
standard_reward = -0.1
tunnel_rewards = np.ones(shape=(height, width)) * standard_reward

for x_well, y_well in zip(x_wells, y_wells):
    tunnel_rewards[x_well, y_well] = -5.0

tunnel_rewards[x_final, y_final] = 5.0

policy = np.random.randint(0, nb_actions, size=(height, width)).astype(np.uint8)
tunnel_values = np.zeros(shape=(height, width))


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
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.matshow(np.zeros_like(tunnel_values), cmap=cm.Pastel1)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_title('Policy (t={})'.format(t))

    for i in range(height):
        for j in range(width):
            action = policy[i, j]

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


def value_evaluation():
    old_tunnel_values = tunnel_values.copy()

    for i in range(height):
        for j in range(width):
            rewards = np.zeros(shape=(nb_actions,))
            old_values = np.zeros(shape=(nb_actions,))

            for k in range(nb_actions):
                if k == 0:
                    if i == 0:
                        x = 0
                    else:
                        x = i - 1
                    y = j

                elif k == 1:
                    if j == width - 1:
                        y = width - 1
                    else:
                        y = j + 1
                    x = i

                elif k == 2:
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

                rewards[k] = tunnel_rewards[x, y]
                old_values[k] = old_tunnel_values[x, y]

            new_values = np.zeros(shape=(nb_actions,))

            for k in range(nb_actions):
                new_values[k] = rewards[k] + (gamma * old_values[k])

            tunnel_values[i, j] = np.max(new_values)


def policy_selection():
    policy = np.zeros(shape=(height, width)).astype(np.uint8)

    for i in range(height):
        for j in range(width):
            if is_final(i, j):
                continue

            values = np.zeros(shape=(nb_actions,))

            values[0] = (tunnel_rewards[i - 1, j] + (gamma * tunnel_values[i - 1, j])) if i > 0 else -np.inf
            values[1] = (tunnel_rewards[i, j + 1] + (gamma * tunnel_values[i, j + 1])) if j < width - 1 else -np.inf
            values[2] = (tunnel_rewards[i + 1, j] + (gamma * tunnel_values[i + 1, j])) if i < height - 1 else -np.inf
            values[3] = (tunnel_rewards[i, j - 1] + (gamma * tunnel_values[i, j - 1])) if j > 0 else -np.inf

            policy[i, j] = np.argmax(values).astype(np.uint8)

    return policy


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
    e = 0

    policy = None

    while e < nb_max_epochs:
        e += 1
        old_tunnel_values = tunnel_values.copy()
        value_evaluation()

        if np.mean(np.abs(tunnel_values - old_tunnel_values)) < tolerance:
            policy = policy_selection()
            break

    # Show final values
    show_values(t=e)

    # Show final policy
    show_policy(t=e)

