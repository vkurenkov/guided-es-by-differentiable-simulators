from mass_spring_robot_config import robots
import random
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os
import wandb
import argparse

random.seed(0)
np.random.seed(0)

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda)

max_steps = 4096
vis_interval = 256
output_vis_interval = 8
steps = 2048 // 3
assert steps * 2 <= max_steps

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

x = vec()
v = vec()
v_inc = vec()

head_id = 0
goal = vec()

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -4.8
friction = 2.5

gradient_clip = 1
spring_omega = 10
damping = 15

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

center = vec()

act = scalar()


def n_input_states():
    return n_sin_waves + 4 * n_objects + 2


@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_inc)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(act)
    ti.root.dense(ti.i, max_steps).place(center)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()


dt = 0.004
learning_rate = 25


@ti.kernel
def compute_center(t: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, i]
        center[t] = (1.0 / n_objects) * c


@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt +
                                                 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - center[t]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 4 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i,
                                  j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 2] * v[t,
                                                                  j][0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 3] * v[t,
                                                                  j][1] * 0.05
        actuation += weights1[i, n_objects * 4 +
                              n_sin_waves] * (goal[None][0] - center[t][0])
        actuation += weights1[i, n_objects * 4 + n_sin_waves +
                              1] * (goal[None][1] - center[t][1])
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        act[t, i] = actuation


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        target_length = spring_length[i] * (1.0 +
                                            spring_actuation[i] * act[t, i])
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        ti.atomic_add(v_inc[t + 1, a], -impulse)
        ti.atomic_add(v_inc[t + 1, b], impulse)


use_toi = False


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Vector([0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_v = old_v
        depth = old_x[1] - ground_height
        if depth < 0 and new_v[1] < 0:
            # friction projection
            new_v[0] = 0
            new_v[1] = 0
        new_x = old_x + dt * new_v
        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = -x[t, head_id][0]


# gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF)


def forward(output=None, visualize=True):
    if random.random() > 0.5:
        goal[None] = [0.9, 0.2]
    else:
        goal[None] = [0.1, 0.2]
    goal[None] = [0.9, 0.2]

    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)

    total_steps = steps if not output else steps * 2

    for t in range(1, total_steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)

        if (t + 1) % interval == 0 and visualize:
            gui.line(begin=(0, ground_height),
                     end=(1, ground_height),
                     color=0x0,
                     radius=3)

            def circle(x, y, color):
                gui.circle((x, y), ti.rgb_to_hex(color), 7)

            for i in range(n_springs):

                def get_pt(x):
                    return (x[0], x[1])

                a = act[t - 1, i] * 0.5
                r = 2
                if spring_actuation[i] == 0:
                    a = 0
                    c = 0x222222
                else:
                    r = 4
                    c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                gui.line(begin=get_pt(x[t, spring_anchor_a[i]]),
                         end=get_pt(x[t, spring_anchor_b[i]]),
                         radius=r,
                         color=c)

            for i in range(n_objects):
                color = (0.4, 0.6, 0.6)
                if i == head_id:
                    color = (0.8, 0.2, 0.3)
                circle(x[t, i][0], x[t, i][1], color)
            # circle(goal[None][0], goal[None][1], (0.6, 0.2, 0.2))

            if output:
                gui.show('mass_spring/{}/{:04d}.png'.format(output, t))
            else:
                gui.show()

    loss[None] = 0
    compute_loss(steps - 1)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])


def clear():
    clear_states()


def setup_robot(objects, springs):
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = objects[i]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]

def initialize_weights():
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input_states())) * 2
        bias1[i] = 0.0

    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_springs)) * 3
        bias2[i] = 0.0

def update_weights_simple(lr):
    total_norm_sqr = 0
    for i in range(n_hidden):
        for j in range(n_input_states()):
            total_norm_sqr += weights1.grad[i, j]**2
        total_norm_sqr += bias1.grad[i]**2

    for i in range(n_springs):
        for j in range(n_hidden):
            total_norm_sqr += weights2.grad[i, j]**2
        total_norm_sqr += bias2.grad[i]**2

    print(total_norm_sqr)

    # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
    gradient_clip = 0.2
    scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] -= scale * weights1.grad[i, j]
        bias1[i] -= scale * bias1.grad[i]

    for i in range(n_springs):
        for j in range(n_hidden):
            weights2[i, j] -= scale * weights2.grad[i, j]
        bias2[i] -= scale * bias2.grad[i]

def optimize(toi, visualize):
    global use_toi
    use_toi = toi

    initialize_weights()

    losses = []
    # forward('initial{}'.format(robot_id), visualize=visualize)
    for iter in range(100):
        clear()
        # with ti.Tape(loss) automatically clears all gradients
        with ti.Tape(loss):
            forward(visualize=visualize)

        print('Iter=', iter, 'Loss=', loss[None])

        update_weights_simple(lr=None)

        losses.append(loss[None])

    return losses


def train_taichi(lr, seed, num_iterations, robot_id, toi):
    # Fix seeds
    np.random.seed(seed)
    random.seed(seed)

    # Load the robot and check whether to use toi
    setup_robot(*robots[robot_id]())
    global use_toi
    use_toi = toi

    # Initialize the system and policy
    initialize_weights()

    for epoch in range(num_iterations):
        clear()
        with ti.Tape(loss):
            forward(visualize=False)

        # Optimize
        update_weights_simple(lr)

        # Logging
        wandb.log({"Loss": loss[None], "Num Samples": epoch + 1})
        print(f"Epoch: {epoch}; Mean loss: {loss[None]}")

        # Reset
        clear_states()

        # Save model at some points
        if epoch % 10 == 0:
            np.save(f"models/policy-{wandb.run.id}", extract_current_policy())

    # Save the latest model
    np.save(f"models/policy-{wandb.run.id}", extract_current_policy())

# Parse training arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alg",  type=str, default="taichi", choices=["taichi", "vanilla-es", "guided-es"])
parser.add_argument("--robot-id", type=int, choices=[0, 1, 2], default=1, help="Type of the robot to optimize.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr",   type=float, default=1e-2)
parser.add_argument("--use-toi", action="store_true", default=False, help="Whether to account for collision with time of impact fix")
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--num-perturbs", type=int, default=10, help="Number of perturbations to estimate the gradient for vanilla and guided es.")
parser.add_argument("--num-prev-grads", type=int, default=4, help="Number of previous gradients to span subspace for guided es.")
parser.add_argument("--std", type=float, default=0.01, help="Standard deviation for perturbations (used for es and guided es).")
parser.add_argument("--alpha", default=0.5, type=float, help="Hyperparameter for Guided ES.")
parser.add_argument("--beta", type=float, default=2, help="Hyperparameter for Guided ES.")
args = parser.parse_args()

# Logging
wandb.init(project="ti-mass-spring")
wandb.config.update(args)

# Setup the robot
setup_robot(*robots[args.robot_id]())

### EVOLUTIONARY STRATEGIES
TI_WEIGHTS        = [weights1, bias1, weights2, bias2]
NUM_PARAMETERS = np.sum([np.prod(weights.shape) for weights in TI_WEIGHTS])

def extract_current_policy():
    np_weights = []
    for weights in TI_WEIGHTS:
        np_weights.append(weights.to_numpy().flatten())

    return np.hstack(np_weights)

def extract_current_policy_grads():
    np_grads = []
    for weights in TI_WEIGHTS:
        np_grads.append(weights.grad.to_numpy().flatten())

    return np.hstack(np_grads)

def init_zero_gradients():
    return np.zeros(NUM_PARAMETERS)

def sample_perturbation(std, guided_config=None):
    if guided_config is None:
        return np.random.randn(NUM_PARAMETERS) * std
    else:
        perturbation  = std * np.sqrt(guided_config["a"] / NUM_PARAMETERS) * np.random.randn(NUM_PARAMETERS)
        perturbation += std * np.sqrt((1 - guided_config["a"]) / guided_config["k"]) * np.matmul(guided_config["u"], np.random.randn(guided_config["k"]))
        return perturbation

def set_policy_weights_from_numpy(np_weights):
    shapes = [w.shape for w in TI_WEIGHTS]
    result = []
    prev_ind = 0
    for i in range(len(shapes)):
        np_i_weights = np_weights[prev_ind:prev_ind+np.prod(shapes[i])].reshape(shapes[i])
        prev_ind += np.prod(shapes[i])

        TI_WEIGHTS[i].from_numpy(np_i_weights)

def set_policy_grads_from_numpy(np_grads):
    shapes = [w.shape for w in TI_WEIGHTS]
    result = []
    prev_ind = 0
    for i in range(len(shapes)):
        np_i_grads = np_grads[prev_ind:prev_ind+np.prod(shapes[i])].reshape(shapes[i])
        prev_ind += np.prod(shapes[i])
        TI_WEIGHTS[i].grad.from_numpy(np_i_grads)

def estimate_policy(weights):
    set_policy_weights_from_numpy(weights)

    clear()
    forward(visualize=False)

    return loss[None]

def train_es(lr, seed, std, num_perturbations, num_iterations, robot_id, toi):
    # Fix seeds
    np.random.seed(seed)
    random.seed(seed)

    # Load the robot and check whether to use toi
    setup_robot(*robots[robot_id]())
    global use_toi
    use_toi = toi

    # Initialize the system and policy
    initialize_weights()

    for epoch in range(num_iterations):
        policy_weights  = extract_current_policy()
        policy_grads    = init_zero_gradients()
        policy_losses   = []
        for _ in range(num_perturbations):
            policy_perturb = sample_perturbation(std=std)

            # Positive perturbation
            pos_loss = estimate_policy(policy_weights + policy_perturb)

            # Negative perturbation
            neg_loss = estimate_policy(policy_weights - policy_perturb)

            # Compute grads
            policy_grads += policy_perturb * (pos_loss - neg_loss)

            # For logging
            policy_losses.extend([pos_loss, neg_loss])

        # Compute grads
        policy_grads = policy_grads / (2 * std**2)
        set_policy_weights_from_numpy(policy_weights)
        set_policy_grads_from_numpy(policy_grads)

        # Optimize
        update_weights_simple(lr)

        # Log current loss
        wandb.log({"Loss": np.mean(policy_losses), "Num Samples": (epoch + 1) * num_perturbations * 2})
        print(f"Epoch: {epoch}; Mean loss: {np.mean(policy_losses)}")

        # Save model at some points
        if epoch % 10 == 0:
            np.save(f"models/policy-{wandb.run.id}", policy_weights)

    # Save the latest model
    np.save(f"models/policy-{wandb.run.id}", policy_weights)


### Guided evolutionary strategies
class GuidingBuffer:
  def __init__(self, num_params, size):
    self._replay   = [np.zeros(num_params)] * size
    self._size     = size

    self._cur_size = 0
    self._index    = 0
    q,_ = np.linalg.qr(np.stack(self._replay).transpose())
    self.q = q.astype(np.float32)

  def append(self, memento):
    self._replay[self._index] = memento
    self._index    = (self._index + 1) % self._size
    self._cur_size = min(self._cur_size + 1, self._size)
 

  def update_orthogonal(self):
    q,_ = np.linalg.qr(np.stack(self._replay).transpose())
    self.q = q.astype(np.float32)

def train_guided_es(lr, seed, std, num_perturbations, k, alpha, beta, num_iterations, robot_id, toi):
    # Fix seeds
    np.random.seed(seed)
    random.seed(seed)

    # Load the robot and check whether to use toi
    setup_robot(*robots[robot_id]())
    global use_toi
    use_toi = toi

    # Initialize the system and policy
    initialize_weights()

    # Create buffer for guding distribution
    guiding_buffer = GuidingBuffer(num_params=NUM_PARAMETERS, size=k)

    # Guided ES
    for epoch in range(num_iterations):
        # Update guiding space with taichi gradient
        clear()
        with ti.Tape(loss):
            forward(visualize=False)
        guiding_buffer.append(extract_current_policy_grads())
        guiding_buffer.update_orthogonal()

        # Do ES
        policy_weights  = extract_current_policy()
        policy_grads    = init_zero_gradients()
        policy_losses   = []
        for _ in range(num_perturbations):
            policy_perturb = sample_perturbation(std=std, guided_config={"a": alpha, "k": k, "u": guiding_buffer.q})

            # Positive perturbation
            pos_loss = estimate_policy(policy_weights + policy_perturb)

            # Negative perturbation
            neg_loss = estimate_policy(policy_weights - policy_perturb)

            # Compute grads
            policy_grads += policy_perturb * (pos_loss - neg_loss)

            # For logging
            policy_losses.extend([pos_loss, neg_loss])

        # Compute grads
        policy_grads = (beta * policy_grads) / (2 * std**2)
        set_policy_weights_from_numpy(policy_weights)
        set_policy_grads_from_numpy(policy_grads)

        # Optimize
        update_weights_simple(lr)

        # Log current loss
        wandb.log({"Loss": np.mean(policy_losses), "Num Samples": (epoch + 1) * num_perturbations * 2 + 1})
        print(f"Epoch: {epoch}; Mean loss: {np.mean(policy_losses)}")

        # Save model at some points
        if epoch % 10 == 0:
            np.save(f"models/policy-{wandb.run.id}", policy_weights)

    # Save the latest model
    np.save(f"models/policy-{wandb.run.id}", policy_weights)


# Run specified algorithm
if args.alg == "taichi":
    train_taichi(lr=args.lr, seed=args.seed, num_iterations=args.iterations, robot_id=args.robot_id, toi=args.use_toi)
elif args.alg == "vanilla-es":
    train_es(lr=args.lr, seed=args.seed, std=args.std, num_perturbations=args.num_perturbs, num_iterations=args.iterations, robot_id=args.robot_id, toi=args.use_toi)
elif args.alg == "guided-es":
    train_guided_es(lr=args.lr, seed=args.seed, std=args.std, num_perturbations=args.num_perturbs, 
        k=args.num_prev_grads, alpha=args.alpha, beta=args.beta, num_iterations=args.iterations, robot_id=args.robot_id, toi=args.use_toi)