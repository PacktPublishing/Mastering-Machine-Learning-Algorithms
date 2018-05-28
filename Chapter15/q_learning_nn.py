import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


# Set random seed for reproducibility
np.random.seed(1000)


width = 5
height = 5

y_final = width - 1
x_final = height - 1

y_wells = [0, 1, 3, 4]
x_wells = [3, 1, 2, 0]

tunnel = np.zeros(shape=(height, width), dtype=np.float32)

max_steps = 150
gamma = 0.95
n_episodes = 10000
n_exploration = 7500


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


x_start = 0
y_start = 0
nb_actions = 4


# Neural model
model = Sequential()

model.add(Dense(8, input_dim=width * height))
model.add(Activation('tanh'))

model.add(Dense(4))
model.add(Activation('tanh'))

model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='mse')


def train(state, q_value):
    model.train_on_batch(np.expand_dims(state.flatten(), axis=0), np.expand_dims(q_value, axis=0))


def get_Q_value(state):
    return model.predict(np.expand_dims(state.flatten(), axis=0))[0]


def select_action_neural_network(epsilon, state):
    Q_value = get_Q_value(state)

    if np.random.uniform(0.0, 1.0) < epsilon:
        return Q_value, np.random.randint(0, nb_actions)

    return Q_value, np.argmax(Q_value)


def is_final(x, y):
    if (x, y) in zip(x_wells, y_wells) or (x, y) == (x_final, y_final):
        return True
    return False


def reset_tunnel():
    tunnel = np.zeros(shape=(height, width), dtype=np.float32)

    for x_well, y_well in zip(x_wells, y_wells):
        tunnel[x_well, y_well] = -1.0

    tunnel[x_final, y_final] = 0.5

    return tunnel


def q_step_neural_network(epsilon, initial_state):
    e = 0
    total_reward = 0.0

    (i, j) = starting_point()

    prev_value = 0.0
    tunnel = initial_state.copy()
    tunnel[i, j] = 1.0

    while e < max_steps:
        e += 1

        q_value, action = select_action_neural_network(epsilon, tunnel)

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

        tunnel_n = tunnel.copy()
        tunnel_n[i, j] = prev_value
        tunnel_n[x, y] = 1.0

        prev_value = tunnel[x, y]

        if is_final(x, y):
            q_value[action] = reward
            train(tunnel, q_value)
            break

        else:
            q_value[action] = reward + (gamma * np.max(get_Q_value(tunnel_n)))
            train(tunnel, q_value)

            i = x
            j = y

            tunnel = tunnel_n.copy()

    return total_reward


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

    # Train the model
    total_rewards = []

    for t in range(n_episodes):
        tunnel = reset_tunnel()

        epsilon = 0.0

        if t <= n_exploration:
            epsilon = 1.0 - (float(t) / float(n_exploration))

        t_reward = q_step_neural_network(epsilon, tunnel)
        total_rewards.append(t_reward)
        print('{} - {:.2f}'.format(t + 1, t_reward))

    # Show total rewards
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(total_rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Rewards (t={})'.format(n_episodes))
    ax.grid()
    plt.show()

    # Generate some trajectories
    trajectories = []
    tunnels_c = []

    for i, j in xy_start:
        tunnel = reset_tunnel()

        prev_value = 0.0

        trajectory = [[i, j, -1]]

        tunnel_c = tunnel.copy()
        tunnel[i, j] = 1.0
        tunnel_c[i, j] = 1.0

        final = False
        e = 0

        while not final and e < max_steps:
            e += 1

            q_value = get_Q_value(tunnel)
            action = np.argmax(q_value)

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

            trajectory[e - 1][2] = action
            trajectory.append([x, y, -1])

            tunnel[i, j] = prev_value

            prev_value = tunnel[x, y]

            tunnel[x, y] = 1.0
            tunnel_c[x, y] = 1.0

            i = x
            j = y

            final = is_final(x, y)

        trajectories.append(np.array(trajectory))
        tunnels_c.append(tunnel_c)

    trajectories = np.array(trajectories)

    # Show the trajectories
    fig, ax = plt.subplots(3, 5, figsize=(14, 8))

    for i in range(3):
        for j in range(5):
            ax[i, j].matshow(tunnels_c[(j * 4) + i], cmap=cm.Pastel1)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

            for x, y, action in trajectories[(j * 4) + i]:
                if x == x_final and y == y_final:
                    msg = 'E'
                else:
                    if action == -1:
                        msg = r'$\otimes$'
                    elif action == 0:
                        msg = r'$\uparrow$'
                    elif action == 1:
                        msg = r'$\rightarrow$'
                    elif action == 2:
                        msg = r'$\downarrow$'
                    else:
                        msg = r'$\leftarrow$'

                ax[i, j].text(x=y, y=x, s=msg, va='center', ha='center')

    plt.show()