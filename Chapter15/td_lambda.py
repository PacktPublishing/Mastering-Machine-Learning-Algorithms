import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Set random seed for reproducibility
np.random.seed(1000)


width = 15
height = 5

y_final = width - 1
x_final = height - 1

y_wells = [0, 1, 3, 5, 5, 6, 7, 9, 10, 11, 12, 14]
x_wells = [3, 1, 2, 0, 4, 3, 1, 3, 1, 2, 4, 1]

y_prizes = [0, 3, 4, 6, 7, 8, 9, 12]
x_prizes = [2, 4, 3, 2, 1, 4, 0, 2]

gamma = 0.95
tolerance = 1e-5
nb_actions = 4
max_steps = 1000
alpha = 0.25
lambd = 0.6
n_episodes = 5000

standard_reward = -0.1
tunnel_rewards = np.ones(shape=(height, width)) * standard_reward

tunnel_values = np.zeros(shape=(height, width))
eligibility_traces = np.zeros(shape=(height, width))
policy = np.random.randint(0, nb_actions, size=(height, width)).astype(np.uint8)

# Setup allowed starting points
xy_grid = np.meshgrid(np.arange(0, height), np.arange(0, width), sparse=False)
xy_grid = np.array(xy_grid).T.reshape(-1, 2)

xy_final = list(zip(x_wells, y_wells)) + list(zip(x_prizes, y_prizes))
xy_final.append([x_final, y_final])

xy_start = []

for x, y in xy_grid:
    if (x, y) not in xy_final:
        xy_start.append([x, y])

xy_start = np.array(xy_start)


def starting_point():
    xy = np.squeeze(xy_start[np.random.randint(0, xy_start.shape[0], size=1)])
    return xy[0], xy[1]


def init_tunnel_rewards():
    for x_well, y_well in zip(x_wells, y_wells):
        tunnel_rewards[x_well, y_well] = -5.0

    for x_prize, y_prize in zip(x_prizes, y_prizes):
        tunnel_rewards[x_prize, y_prize] = 1.0

    tunnel_rewards[x_final, y_final] = 5.0


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
            elif (i, j) in zip(x_prizes, y_prizes):
                msg = r'$\diamond$'
            else:
                msg = '{:.2f}'.format(tunnel_values[i, j])
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
                msg = ''
                if (i, j) in zip(x_prizes, y_prizes):
                    msg = r'$\diamond$ '

                if action == 0:
                    msg += r'$\uparrow$'
                elif action == 1:
                    msg += r'$\rightarrow$'
                elif action == 2:
                    msg += r'$\downarrow$'
                else:
                    msg += r'$\leftarrow$'

            ax.text(x=j, y=i, s=msg, va='center', ha='center')

    plt.show()


def is_final(x, y):
    if (x, y) in zip(x_wells, y_wells) or (x, y) == (x_final, y_final):
        return True
    return False


def episode():
    (i, j) = starting_point()
    x = y = 0

    e = 0

    state_history = [(i, j)]

    init_tunnel_rewards()
    total_reward = 0.0

    while e < max_steps:
        e += 1

        action = policy[i, j]

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
        total_reward += reward

        td_error = reward + (gamma * tunnel_values[x, y]) - tunnel_values[i, j]
        eligibility_traces[i, j] += 1.0

        for sx, sy in state_history:
            tunnel_values[sx, sy] += (alpha * td_error * eligibility_traces[sx, sy])
            eligibility_traces[sx, sy] *= (gamma * lambd)

        if is_final(x, y):
            break
        else:
            i = x
            j = y

            state_history.append([x, y])

            tunnel_rewards[x_prizes, y_prizes] *= 0.85

    return total_reward


def policy_selection():
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


if __name__ == '__main__':
    # Init tunnel rewards
    init_tunnel_rewards()

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
    total_rewards = []

    for _ in range(n_episodes):
        e_reward = episode()
        total_rewards.append(e_reward)
        policy_selection()

    # Plot the total rewards
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(total_rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Rewards (t={})'.format(n_episodes))
    ax.grid()
    plt.show()

    # Show final values
    show_values(t=n_episodes)

    # Show final policy
    show_policy(t=n_episodes)



