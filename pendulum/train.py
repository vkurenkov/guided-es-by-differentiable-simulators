import taichi as ti
import numpy as np
import random
import math
import wandb
import argparse
import time as ptime
import os
import cma

from gym.envs.classic_control import rendering
from utils import RunningStat

# Run taichi
ti.init(arch=ti.cpu)
ti.set_logging_level(level=ti.INFO)

#_____SIMULATION VARIABLES:_____
dt = 0.01
ACTION_FREQUENCY          = 100
N_SECONDS                 = 5
SIMULATION_TRAINING_STEPS = ACTION_FREQUENCY*N_SECONDS
SIMULATION_TRAINING_EPOCHS = 500

# mass of the link
m  = ti.var(ti.f32, shape=())
# length of the link
l  = ti.var(ti.f32, shape=())
# velocity coefficient
k_q = ti.var(ti.f32, shape=())
# control coefficient
k_j = ti.var(ti.f32, shape=())
# contact coefficient
k_s = ti.var(ti.f32, shape=())
# moment of intertia of the link
J   = ti.var(ti.f32, shape=())
# gravitation
g   = ti.var(ti.f32, shape=())

# angle between link and the ground
q_1 = ti.var(ti.f32,shape=(SIMULATION_TRAINING_STEPS))
# velocity
dq_1 = ti.var(ti.f32,shape=(SIMULATION_TRAINING_STEPS))
# acceleration
ddq_1 = ti.var(ti.f32,shape=(SIMULATION_TRAINING_STEPS))
# energy
energy = ti.var(ti.f32, shape=(SIMULATION_TRAINING_STEPS))

# target angle
target_q = ti.var(ti.f32, shape=())
# contact angle
contact_q = ti.var(ti.f32, shape=())
# save time...
time      = ti.var(ti.f32, shape=(SIMULATION_TRAINING_STEPS))

#_____NEURAL NETWORK :_____
INPUT_SHAPE = 8
INTERNAL_SHAPE = 64
OUTPUT_SHAPE = 1

# Input normalization
normalization_stat  = RunningStat(shape=INPUT_SHAPE, eps=1e-2)
normalized_inp_mean = ti.var(ti.f32, shape=(INPUT_SHAPE))
normalized_inp_std  = ti.var(ti.f32, shape=(INPUT_SHAPE))

weights_1 = ti.var(ti.f32, shape=(INTERNAL_SHAPE,INPUT_SHAPE))
weights_1_m = ti.var(ti.f32, shape=(INTERNAL_SHAPE,INPUT_SHAPE))
weights_1_v = ti.var(ti.f32, shape=(INTERNAL_SHAPE,INPUT_SHAPE))

weights_2 = ti.var(ti.f32, shape=(INTERNAL_SHAPE,INTERNAL_SHAPE))
weights_2_m = ti.var(ti.f32, shape=(INTERNAL_SHAPE,INTERNAL_SHAPE))
weights_2_v = ti.var(ti.f32, shape=(INTERNAL_SHAPE,INTERNAL_SHAPE))

weights_3 = ti.var(ti.f32, shape=(OUTPUT_SHAPE,INTERNAL_SHAPE))
weights_3_m = ti.var(ti.f32, shape=(OUTPUT_SHAPE,INTERNAL_SHAPE))
weights_3_v = ti.var(ti.f32, shape=(OUTPUT_SHAPE,INTERNAL_SHAPE))

bias_1 = ti.var(ti.f32, shape=(INTERNAL_SHAPE))
bias_1_m = ti.var(ti.f32, shape=(INTERNAL_SHAPE))
bias_1_v = ti.var(ti.f32, shape=(INTERNAL_SHAPE))

bias_2 = ti.var(ti.f32, shape=(INTERNAL_SHAPE))
bias_2_m = ti.var(ti.f32, shape=(INTERNAL_SHAPE))
bias_2_v = ti.var(ti.f32, shape=(INTERNAL_SHAPE))

bias_3 = ti.var(ti.f32, shape=(OUTPUT_SHAPE))
bias_3_m = ti.var(ti.f32, shape=(OUTPUT_SHAPE))
bias_3_v = ti.var(ti.f32, shape=(OUTPUT_SHAPE))

internal1 = ti.var(ti.f32, shape=(SIMULATION_TRAINING_STEPS,INTERNAL_SHAPE))
internal2 = ti.var(ti.f32, shape=(SIMULATION_TRAINING_STEPS,INTERNAL_SHAPE))

#output layer - voltage applied to a system
u = ti.var(ti.f32, shape=(SIMULATION_TRAINING_STEPS))
rand_force = ti.var(ti.f32, shape=(SIMULATION_TRAINING_STEPS))


#_____PLACEHOLDER FOR LOSS FUNCTION:_____
loss = ti.var(ti.f32, shape=())

#activating gradients calculation
ti.root.lazy_grad()


@ti.kernel
def initialize_system_params():
    # System
    g[None] = 9.8
    l[None] = 0.31 / 2.0#0.36
    m[None] = 0.125

    # Coefficients
    J[None] = 0.00475#0.0047#0.0045
    k_j[None] = 0.06499285#0.0306
    k_q[None] = 0.00109
    if args.env == "pendulum":
        k_s[None]      = 0.0
        target_q[None] = 0.0
    elif args.env == "jumper":
        k_s[None] = 2.0*100
        target_q[None]  = np.pi * (3.0/4.0)

    # Angles
    contact_q[None] = np.pi

def initialize_weights():
    for i in range(INTERNAL_SHAPE):
        # print(np.random.randn())
        for j in range(INPUT_SHAPE):
            std = np.sqrt(2/(INTERNAL_SHAPE+INPUT_SHAPE))
            # print(np.random.randn())

            weights_1[i,j] = np.random.randn() * std
            weights_1_m[i, j] = 0.0
            weights_1_v[i, j] = 0.0
        # for _ in range()
        bias_1[i] = np.random.randn()
        bias_1_m[i] = 0.0
        bias_1_v[i] = 0.0

    for i in range(INTERNAL_SHAPE):
        for j in range(INTERNAL_SHAPE):
            std = np.sqrt(2/(INTERNAL_SHAPE+INTERNAL_SHAPE))
            weights_2[i,j] = np.random.randn() * std
            weights_2_m[i, j] = 0.0
            weights_2_v[i, j] = 0.0
        bias_2[i] = np.random.randn()
        bias_2_m[i] = 0.0
        bias_2_v[i] = 0.0

    for i in range(OUTPUT_SHAPE):
        for j in range(INTERNAL_SHAPE):
            std = np.sqrt(2/(INTERNAL_SHAPE+OUTPUT_SHAPE))
            weights_3[i,j] = np.random.randn() * std
            weights_3_m[i, j] = 0.0
            weights_3_v[i, j] = 0.0
        bias_3[i] = 0.0#np.random.randn()
        bias_3_m[i] = 0.0
        bias_3_v[i] = 0.0


def initialize_normalization_buffers():
    global normalization_stat
    normalization_stat = RunningStat(shape=INPUT_SHAPE, eps=1e-2)
    mean, std = normalization_stat.mean, normalization_stat.std
    for i in range(INPUT_SHAPE):
        normalized_inp_mean[i] = mean[i]
        normalized_inp_std[i]  = std[i]
    
def update_normalization_buffers():
    if args.obs_normalize and not args.real:
        for t in range(3, SIMULATION_TRAINING_STEPS-1):
            inp = np.array([q_1[t], dq_1[t], q_1[t-1], dq_1[t-1], q_1[t-2], dq_1[t-2], q_1[t-3], dq_1[t-3]] )
            normalization_stat.increment(s=inp, ssq=inp**2, c=1)

        mean, std = normalization_stat.mean, normalization_stat.std
        for i in range(INPUT_SHAPE):
            normalized_inp_mean[i] = mean[i]
            normalized_inp_std[i]  = std[i]

@ti.func
def relu(x:ti.f32) -> ti.f32:
    # print(x)
    return ti.max(0,x)

@ti.func
def clamp(x, xmin:ti.f32, xmax=ti.f32):
    '''
    Constrain a value to lie between two further values.
    `clamp` returns the value of x constrained to the range xmin to xmax.
    :parameter x:
        Specify the value to constrain.
    :parameter xmin:
        Specify the lower end of the range into which to constrain x.
    :parameter xmax:
        Specify the upper end of the range into which to constrain x.
    :return:
        The returned value is computed as `min(xmax, max(xmin, x))`.
    '''
    return min(xmax, max(xmin, x))

@ti.func
def sigmoid(x: ti.f32) -> ti.f32:
    return ti.exp(x) / (ti.exp(x) + 1)

@ti.func
def randn(mean=0.0, std=1.0):
    pi = 3.1415926535
    n_distributed = ti.sqrt(-2 * ti.log(ti.random())) * ti.cos(2 * pi * ti.random())
    return mean + std * n_distributed

@ti.kernel
def layer1_forward(t:ti.i32):
    for i in range(INTERNAL_SHAPE):
        pi = 3.1415926535
        activation = 0.0
        activation += weights_1[i,0] * (q_1[t] - normalized_inp_mean[0]) / normalized_inp_std[0] 
        activation += weights_1[i,1] * (dq_1[t] - normalized_inp_mean[1]) / normalized_inp_std[1] 
        activation += weights_1[i,2] * (q_1[t-1] - normalized_inp_mean[2]) / normalized_inp_std[2] 
        activation += weights_1[i,3] * (dq_1[t-1] - normalized_inp_mean[3]) / normalized_inp_std[3] 
        activation += weights_1[i,4] * (q_1[t-2] - normalized_inp_mean[4]) / normalized_inp_std[4] 
        activation += weights_1[i,5] * (dq_1[t-2] - normalized_inp_mean[5]) / normalized_inp_std[5] 
        activation += weights_1[i,6] * (q_1[t-3] - normalized_inp_mean[6]) / normalized_inp_std[6] 
        activation += weights_1[i,7] * (dq_1[t-3] - normalized_inp_mean[7]) / normalized_inp_std[7] 

        activation += bias_1[i]
        activation = ti.tanh(activation)
        internal1[t,i] = activation

@ti.kernel
def layer2_forward(t:ti.i32):
    for i in range(INTERNAL_SHAPE):
        activation = 0.0
        for j in ti.static(range(INTERNAL_SHAPE)):
            activation += weights_2[i,j] * internal1[t,j]
        activation += bias_2[i]
        internal2[t,i] = ti.tanh(activation)

@ti.kernel
def layer3_forward(t:ti.i32):
    for i in range(OUTPUT_SHAPE):
        activation = 0.0
        for j in ti.static(range(INTERNAL_SHAPE)):
            activation += weights_3[i,j] * internal2[t,j]
        activation += bias_3[i]

        if args.env == "pendulum":
            if args.smooth:
                u[t] = ti.tanh(activation)
            else:
                u[t] = clamp(activation, -1, 1)
        elif args.env == "jumper":
            if args.smooth:
                u[t] = sigmoid(activation)
            else:
                u[t] = clamp(activation, 0, 1)

@ti.kernel
def cost_function_energy():
    for t in range(SIMULATION_TRAINING_STEPS-1):
        target_energy  = m * g * ti.sin(target_q) * l
        # current_energy = 0.5 * J * dq_1[t]**2 + 0.5 * m * g * ti.sin(q_1[t]) * l
        # print(current_energy)
        loss[None] += (energy[t] - target_energy) ** 2

@ti.kernel
def cost_function_position():
    for t in range(SIMULATION_TRAINING_STEPS-1):
        target_position  = target_q
        current_position = q_1[t]
        loss[None] += (current_position - target_position) ** 2

@ti.kernel
def compute_energy(t:ti.i32):
    # Compute energy
    energy[t] = 0.5 * J * dq_1[t]**2 + m * g * ti.sin(q_1[t]) * l

#_____SYSTEM DYNAMICS_____
@ti.kernel
def simulate_dynamics(t:ti.i32):
    # No contact
    if q_1[t] < contact_q:
        ddq_1[t] = (k_j * u[t] - m*g*l*ti.cos(q_1[t]) - k_q*dq_1[t]) / J
    # Contact
    else:
        ddq_1[t] = (k_j * u[t] - m*g*l*ti.cos(q_1[t]) - k_q*dq_1[t] - k_s*(q_1[t] - contact_q)) / J

    # Semi-implicit euler
    dq_1[t + 1] = dq_1[t] + ddq_1[t] * dt
    q_1[t + 1] = q_1[t] + dq_1[t + 1] * dt

@ti.kernel
def clear_states():
    for i in range(SIMULATION_TRAINING_STEPS):
        if args.env == "pendulum":
            q_1[i] = 1.5*np.pi
        elif args.env == "jumper":
            q_1[i] = np.pi/1.5

        dq_1[i] = 0.0
        ddq_1[i] = 0.0
        energy[i] = 0.0
        u[i] = 0.0
        time[i]  = i
        rand_force[i] = randn(0.0, 0.001)

def forward(loss_func,visualize=False):
    loss[None] = 0.0
    for t in range(SIMULATION_TRAINING_STEPS-1):
        compute_energy(t)
        if t > 3:
            layer1_forward(t)
            layer2_forward(t)
            layer3_forward(t)
        simulate_dynamics(t)

        if visualize:
            print(f"Timestep: {t}; U: {u[t]}; Q: {q_1[t]}; dQ: {dq_1[t]}")
            render(t)

    loss_func()

def forward_real():
    obs              = real_env.reset()
    all_observations = []
    done             = False
    t                = 0
    while not done:
        action    = real_policy.forward(obs)
        obs, done = real_env.step(action)
        all_observations.append(obs)

        # print(f"Timestep: {t}; U: {action}; Q: {obs[0]}; dQ: {obs[1]}")
        t += 1
    return real_env.cost(all_observations, cost=args.loss)

def update_layer_weights_fromage(weights, bias, lr):
    """
    https://jeremybernste.in/blog/getting-to-the-bottom
    https://github.com/jxbz/fromage/blob/master/fromage.py
    """
    # Compute layer norm
    layer_norm = 0.0
    layer_grad_norm = 0.0
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            layer_norm  += weights[i,j] ** 2
            layer_grad_norm += weights.grad[i, j] ** 2
        layer_norm += bias[i] ** 2
        layer_grad_norm += bias.grad[i] ** 2

    layer_norm = np.sqrt(layer_norm)
    layer_grad_norm = np.sqrt(layer_grad_norm)

    #print(f"Layer norm: {layer_norm}; Layer grad norm: {layer_grad_norm}")

    # Fromage update
    for i in range(weights.shape[0]):
        # Weights update
        for j in range(weights.shape[1]):
            if layer_norm > 0 and layer_grad_norm > 0:
                weights[i, j] += -lr * (weights.grad[i, j]) * (layer_norm) / (layer_grad_norm)
            else:
                weights[i, j] += -lr * (weights.grad[i, j])
            weights[i, j] *= 1.0 / np.sqrt(1 + lr**2)

        # Bias update
        if layer_norm > 0 and layer_grad_norm > 0:
            bias[i] += -lr * (bias.grad[i]) * (layer_norm) / (layer_grad_norm)
        else:
            bias[i] += -lr * (bias.grad[i])
        bias[i] *= 1.0 / np.sqrt(1 + lr ** 2)
    
def update_layer_weights_adam(weights, bias, m_w, m_b, v_w, v_b, epoch, lr, b1=0.9, b2=0.99):
    for i in range(weights.shape[0]):
        # Weights update
        for j in range(weights.shape[1]):
            # Compute first and second moment
            m_w[i, j] = b1 * m_w[i, j] + (1.0 - b1) * weights.grad[i, j]
            v_w[i, j] = b2 * v_w[i, j] + (1.0 - b2) * weights.grad[i, j] ** 2

            # Compute the gradient based on the first and second moments
            m_hat     = m_w[i, j] / (1 - pow(b1, epoch))
            v_hat     = v_w[i, j] / (1 - pow(b2, epoch))
            grad      = lr / np.sqrt(v_hat + 1e-6) * m_hat

            # Apply gradient
            weights[i, j] -= grad

        # Bias update
        # Compute first and second moment
        m_b[i] = b1 * m_b[i] + (1.0 - b1) * bias.grad[i]
        v_b[i] = b2 * v_b[i] + (1.0 - b2) * bias.grad[i] ** 2

        # Compute the gradient based on the first and second moments
        m_hat     = m_b[i] / (1 - pow(b1, epoch))
        v_hat     = v_b[i] / (1 - pow(b2, epoch))
        grad      = lr / np.sqrt(v_hat + 1e-6) * m_hat

        # Apply gradient
        bias[i] -= grad

def update_weights_simple(lr:ti.f32, epoch):
    if args.optimizer == "fromage":
        update_layer_weights_fromage(weights=weights_1, bias=bias_1, lr=lr)
        update_layer_weights_fromage(weights=weights_2, bias=bias_2, lr=lr)
        update_layer_weights_fromage(weights=weights_3, bias=bias_3, lr=lr)
    elif args.optimizer == "adam":
        update_layer_weights_adam(weights=weights_1, bias=bias_1, m_w=weights_1_m, m_b=bias_1_m, v_w=weights_1_v, v_b=bias_1_v, epoch=epoch+1, lr=lr)
        update_layer_weights_adam(weights=weights_2, bias=bias_2, m_w=weights_2_m, m_b=bias_2_m, v_w=weights_2_v, v_b=bias_2_v, epoch=epoch+1, lr=lr)
        update_layer_weights_adam(weights=weights_3, bias=bias_3, m_w=weights_3_m, m_b=bias_3_m, v_w=weights_3_v, v_b=bias_3_v, epoch=epoch+1, lr=lr)
    elif args.optimizer == "sgd":
        for i in range(INTERNAL_SHAPE):
            for j in range(INPUT_SHAPE):
                weights_1[i,j] -= lr *weights_1.grad[i,j]

            bias_1[i] -= lr * bias_1.grad[i]

        for i in range(INTERNAL_SHAPE):
            for j in range(INTERNAL_SHAPE):
                weights_2[i,j] -= lr *weights_2.grad[i,j]
            bias_2[i] -= lr * bias_2.grad[i]

        for i in range(OUTPUT_SHAPE):
            for j in range(INTERNAL_SHAPE):
                weights_3[i,j] -= lr *weights_3.grad[i,j]
            bias_3[i] -= lr * bias_3.grad[i]

def normalize_grads(weights, bias):
    """
    https://jeremybernste.in/blog/getting-to-the-bottom
    https://github.com/jxbz/fromage/blob/master/fromage.py
    """
    # Compute layer norm
    layer_norm = 0.0
    layer_grad_norm = 0.0
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            layer_norm  += weights[i,j] ** 2
            layer_grad_norm += weights.grad[i, j] ** 2
        layer_norm += bias[i] ** 2
        layer_grad_norm += bias.grad[i] ** 2

    layer_norm = np.sqrt(layer_norm)
    layer_grad_norm = np.sqrt(layer_grad_norm)

    #print(f"Layer norm: {layer_norm}; Layer grad norm: {layer_grad_norm}")

    # Fromage update
    for i in range(weights.shape[0]):
        # Weights update
        for j in range(weights.shape[1]):
            if layer_norm > 0 and layer_grad_norm > 0:
                weights.grad[i, j] = (weights.grad[i, j]) * (layer_norm) / (layer_grad_norm)
            else:
                weights.grad[i, j] = (weights.grad[i, j])

        # Bias update
        if layer_norm > 0 and layer_grad_norm > 0:
            bias.grad[i] = (bias.grad[i]) * (layer_norm) / (layer_grad_norm)
        else:
            bias.grad[i] = (bias.grad[i])

def normalize_all_grads():
    normalize_grads(weights_1, bias_1)
    normalize_grads(weights_2, bias_2)
    normalize_grads(weights_3, bias_3)

global viewer
viewer = None
def render(t,mode='human'):
    global viewer

    if viewer is None:
        viewer = rendering.Viewer(500,500)
        bound = l[None] * 1.4
        viewer.set_bounds(-bound,bound,-bound,bound)

    # Reference width
    WIDTH = 0.05

    # Link position in the frame
    origin_point = (0, 0)
    end_point    = (l[None] * np.cos(q_1[t]), l[None] * np.sin(q_1[t]))

    # Draw the link
    transform    = rendering.Transform(rotation=q_1[t], translation=origin_point)
    link         = viewer.draw_polygon([(0, -WIDTH), (0, WIDTH), (l[None], WIDTH), (l[None], -WIDTH)])
    link.add_attr(transform)
    link.set_color(0, 0.8, 0.8)

    # Draw a circle about the origin of the link
    circ = viewer.draw_circle(WIDTH)
    circ.set_color(0.8, 0.8, 0.0)
    circ.add_attr(transform)

    # Draw a circle about the target angle
    target_transform = rendering.Transform(translation=(l[None] * np.cos(target_q[None]), l[None] * np.sin(target_q[None])))
    target_circ = viewer.draw_circle(WIDTH / 1.25)
    target_circ.set_color(1.0, 0.0, 0.0)
    target_circ.add_attr(target_transform)

    # Draw a contact circle
    contact_transform = rendering.Transform(translation=(l[None] * np.cos(contact_q[None]), l[None] * np.sin(contact_q[None])))
    contact_circ = viewer.draw_circle(WIDTH / 1.25)
    contact_circ.set_color(0.0, 0.0, 1.0)
    contact_circ.add_attr(contact_transform)

    return  viewer.render(return_rgb_array = mode == 'rgb_array')


def train_taichi(lr, seed, loss_func, num_iterations, real_robot=False, resume=None):
    # Direct training on real robot with this method is impossible
    if real_robot:
        raise Exception("Direct training on real robot directly using differentiable robot simulator is not possible.")

    np.random.seed(seed)
    random.seed(seed)

    initialize_weights()
    initialize_system_params()
    initialize_normalization_buffers()
    clear_states()

    # Load policy to resume the training
    if resume is not None:
        set_policy_weights_from_numpy(np.load(resume))

    # Save best policies
    best_loss = 10000000

    for epoch in range(num_iterations):
        with ti.Tape(loss):
            forward(loss_func, visualize=False)

        # Update normalization buffers
        update_normalization_buffers()

        # Optimize
        update_weights_simple(lr, epoch=epoch)

        # Logging
        average_loss = loss[None]
        wandb.log({"Loss": average_loss, "Min Loss": average_loss, "Num Samples": (epoch + 1)})
        print(f"Epoch: {epoch}; Mean loss: {loss[None]}; Time: {ptime.asctime()}")

        # Reset
        clear_states()

        # Save best model
        if average_loss < best_loss and epoch % 1 == 0:
            best_loss = average_loss
            np.save(f"models/{wandb.run.id}/best-policy-{wandb.run.id}-{average_loss}", extract_current_policy())


    # Save the latest model
    np.save(f"models/{wandb.run.id}/latest-policy-{wandb.run.id}", extract_current_policy())

### EVOLUTIONARY STRATEGIES

### Guided evolutionary strategies
NUM_PARAMETERS = np.prod(weights_1.shape) + np.prod(weights_2.shape) + np.prod(weights_3.shape) + np.prod(bias_1.shape) + np.prod(bias_2.shape) + np.prod(bias_3.shape)

def extract_current_policy():
    w1 = weights_1.to_numpy()
    w2 = weights_2.to_numpy()
    w3 = weights_3.to_numpy()
    b1 = bias_1.to_numpy()
    b2 = bias_2.to_numpy()
    b3 = bias_3.to_numpy()

    return np.hstack([w1.flatten(), w2.flatten(), w3.flatten(), b1.flatten(), b2.flatten(), b3.flatten()])

def extract_current_policy_grads():
    w1g = weights_1.grad.to_numpy()
    w2g = weights_2.grad.to_numpy()
    w3g = weights_3.grad.to_numpy()
    b1g = bias_1.grad.to_numpy()
    b2g = bias_2.grad.to_numpy()
    b3g = bias_3.grad.to_numpy()

    return np.hstack([w1g.flatten(), w2g.flatten(), w3g.flatten(), b1g.flatten(), b2g.flatten(), b3g.flatten()])

def init_zero_gradients():
    return np.zeros(NUM_PARAMETERS)

def sample_perturbation(std, guided_config=None):
    if guided_config is None:
        return np.random.randn(NUM_PARAMETERS) * std
    else:
        perturbation  = std * np.sqrt(guided_config["a"] / NUM_PARAMETERS) * np.random.randn(NUM_PARAMETERS)
        perturbation += std * np.sqrt((1 - guided_config["a"]) / guided_config["k"]) * np.matmul(guided_config["u"], np.random.randn(guided_config["k"]))
        return perturbation

def set_policy_weights_from_numpy(weights):
    shapes = [weights_1.shape, weights_2.shape, weights_3.shape, bias_1.shape, bias_2.shape, bias_3.shape]
    result = []
    prev_ind = 0
    for i in range(len(shapes)):
        result.append(weights[prev_ind:prev_ind+np.prod(shapes[i])].reshape(shapes[i]))
        prev_ind += np.prod(shapes[i])

    weights_1.from_numpy(result[0])
    weights_2.from_numpy(result[1])
    weights_3.from_numpy(result[2])
    bias_1.from_numpy(result[3])
    bias_2.from_numpy(result[4])
    bias_3.from_numpy(result[5])

def set_policy_grads_from_numpy(grads):
    shapes = [weights_1.shape, weights_2.shape, weights_3.shape, bias_1.shape, bias_2.shape, bias_3.shape]
    result = []
    prev_ind = 0
    for i in range(len(shapes)):
        result.append(grads[prev_ind:prev_ind+np.prod(shapes[i])].reshape(shapes[i]))
        prev_ind += np.prod(shapes[i])

    weights_1.grad.from_numpy(result[0])
    weights_2.grad.from_numpy(result[1])
    weights_3.grad.from_numpy(result[2])
    bias_1.grad.from_numpy(result[3])
    bias_2.grad.from_numpy(result[4])
    bias_3.grad.from_numpy(result[5])

def estimate_policy(weights, loss_func, real_robot=False):
    if real_robot:
        real_policy.set_weights(weights)
        cost = forward_real()
        return cost
    else:
        set_policy_weights_from_numpy(weights)
        clear_states()
        forward(loss_func)

        return loss[None]

def train_es(lr, seed, loss_func, std, num_perturbations, num_iterations, real_robot=False, resume=None, num_evals=1):
    # Fix seeds
    np.random.seed(seed)
    random.seed(seed)

    # Log best policy
    best_loss = 1000000

    # Initialize the system and policy
    initialize_weights()
    initialize_system_params()
    initialize_normalization_buffers()

    # Load policy to resume the training
    if resume is not None:
        set_policy_weights_from_numpy(np.load(resume))

    for epoch in range(num_iterations):
        policy_weights  = extract_current_policy()
        policy_grads    = init_zero_gradients()
        policy_losses   = []

        # Evaluate current policy and save it
        cur_costs = []
        for _ in range(num_evals):
            cur_costs.append(estimate_policy(weights=policy_weights, loss_func=loss_func, real_robot=real_robot))
        np.save(f"models/{wandb.run.id}/epoch{epoch}-cur-{wandb.run.id}-{np.mean(cur_costs)}", policy_weights)

        # Find current smallest loss
        cur_best_loss    = np.mean(cur_costs)
        cur_best_weights = np.copy(policy_weights)

        # Compute perturbations
        for _ in range(num_perturbations):
            policy_perturb = sample_perturbation(std=std)

            # Positive perturbation
            pos_loss = estimate_policy(policy_weights + policy_perturb, loss_func=loss_func, real_robot=real_robot)
            # Save best model
            if pos_loss < cur_best_loss:
                cur_best_loss = pos_loss
                cur_best_weights = np.copy(policy_weights + policy_perturb)
            update_normalization_buffers()

            # Negative perturbation
            neg_loss = estimate_policy(policy_weights - policy_perturb, loss_func=loss_func, real_robot=real_robot)
            # Save best model
            if neg_loss < cur_best_loss:
                cur_best_loss = neg_loss
                cur_best_weights = np.copy(policy_weights - policy_perturb)
            update_normalization_buffers()

            # Compute grads
            policy_grads += policy_perturb * (pos_loss - neg_loss)
            # print(pos_loss - neg_loss)

            # For logging
            policy_losses.extend([pos_loss, neg_loss])

        # Evaluate current smallest loss and save it
        min_costs = []
        if np.mean(cur_costs) > cur_best_loss:
            for _ in range(num_evals):
                min_costs.append(estimate_policy(weights=cur_best_weights, loss_func=loss_func, real_robot=real_robot))
        else:
            min_costs = cur_costs
        np.save(f"models/{wandb.run.id}/epoch{epoch}-min-{wandb.run.id}-{np.mean(min_costs)}", cur_best_weights)

        # Compute grads
        policy_grads = policy_grads / (2 * std**2 * num_perturbations)
        set_policy_weights_from_numpy(policy_weights)
        set_policy_grads_from_numpy(policy_grads)

        # Optimize
        update_weights_simple(lr, epoch=epoch)

        # Log current loss
        average_loss = np.mean(policy_losses)
        wandb.log({
            "Loss": average_loss, 
            "Min Loss": np.mean(min_costs),
            "Min Loss Std": np.std(min_costs), 
            "Centroid Loss": np.mean(cur_costs),
            "Centroid Loss Std": np.std(cur_costs),
            "Num Samples": (epoch + 1) * num_perturbations * 2,
        })
        print(f"Epoch: {epoch}; Mean loss: {np.mean(policy_losses)}; Time: {ptime.asctime()}")

    # Save the latest model
    np.save(f"models/{wandb.run.id}/latest-policy-{wandb.run.id}", policy_weights)


### Guided evolutionary strategies
class GuidingBuffer:
  def __init__(self, num_params, size):
    self._replay   = [np.zeros(num_params)] * size
    self._size     = size
    self._num_params = num_params

    self._cur_size = 0
    self._index    = 0
    q,_ = np.linalg.qr(np.stack(self._replay).transpose())
    self.q = q.astype(np.float32)

  def reset(self):
      self._replay   = [np.zeros(self._num_params)] * self._size
      self._cur_size = 0
      self._index    = 0

  def append(self, memento):
    self._replay[self._index] = memento
    self._index    = (self._index + 1) % self._size
    self._cur_size = min(self._cur_size + 1, self._size)
 

  def update_orthogonal(self):
    q,_ = np.linalg.qr(np.stack(self._replay).transpose())
    self.q = q.astype(np.float32)

def compute_difsim_gradient(n_steps, loss_func, lr):
    """
    Side-effect: Does not restore the starting point of the shooting.
    So you need to do it after yourself.
    """
    if n_steps == 0:
        clear_states()
        with ti.Tape(loss):
            forward(loss_func, visualize=False)

        return extract_current_policy_grads()
    else:
        # Starting point of shooting
        start_weights = extract_current_policy()

        # Do n optimization steps
        for step in range(n_steps):
            clear_states()
            with ti.Tape(loss):
                forward(loss_func, visualize=False)

            # Update normalization buffers
            update_normalization_buffers()

            # Step gradient
            if args.optimizer == "adam":
                raise NotImplementedError("Adam is not supported in this setting.")
            else:
                update_weights_simple(lr, epoch=step)

        # Compute "gradient"
        final_weights = extract_current_policy()
        difsim_grad   = final_weights - start_weights
        
        return difsim_grad

def train_guided_es(lr, seed, loss_func, std, num_perturbations, k, alpha, 
    beta, num_iterations, real_robot=False, resume=None, num_evals=1):
    # Fix seeds
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the system and policy
    initialize_weights()
    initialize_system_params()
    initialize_normalization_buffers()

    # Load policy to resume the training
    if resume is not None:
        set_policy_weights_from_numpy(np.load(resume))

    # Create buffer for guding distribution
    guiding_buffer = GuidingBuffer(num_params=NUM_PARAMETERS, size=k)

    # Log best policies
    best_loss = 1000000

    # Guided ES
    for epoch in range(num_iterations):
        policy_weights  = extract_current_policy()
        policy_grads    = init_zero_gradients()
        policy_losses   = []

        # Update guiding space with taichi gradient
        difsim_grad = compute_difsim_gradient(n_steps=args.shooting_n, loss_func=loss_func, lr=lr)

        # Protect against nan values
        print(np.count_nonzero(difsim_grad))
        if np.any(np.isnan(difsim_grad)) or np.count_nonzero(difsim_grad) == 0:
            guiding_buffer.reset()
            guiding_buffer.update_orthogonal()
            print(f"Differentiable simulator produced nans or zeros :(")
        else:
            guiding_buffer.append(difsim_grad)
            guiding_buffer.update_orthogonal()

        # Evaluate current policy and save it
        cur_costs = []
        for _ in range(num_evals):
            cur_costs.append(estimate_policy(weights=policy_weights, loss_func=loss_func, real_robot=real_robot))
        np.save(f"models/{wandb.run.id}/epoch{epoch}-cur-{wandb.run.id}-{np.mean(cur_costs)}", policy_weights)

        # Find current smallest loss
        cur_best_loss    = np.mean(cur_costs)
        cur_best_weights = np.copy(policy_weights)

        # Compute perturbations
        for _ in range(num_perturbations):
            policy_perturb = sample_perturbation(std=std, guided_config={"a": alpha, "k": k, "u": guiding_buffer.q})

            # Positive perturbation
            pos_loss = estimate_policy(policy_weights + policy_perturb, loss_func=loss_func, real_robot=real_robot)
            # Update normalization buffers
            update_normalization_buffers()
            # Save best model
            if pos_loss < cur_best_loss:
                cur_best_loss = pos_loss
                cur_best_weights = np.copy(policy_weights + policy_perturb)

            # Negative perturbation
            neg_loss = estimate_policy(policy_weights - policy_perturb, loss_func=loss_func, real_robot=real_robot)
            # Update normalization buffers
            update_normalization_buffers()
            # Save best model
            if neg_loss < cur_best_loss:
                cur_best_loss = neg_loss
                cur_best_weights = np.copy(policy_weights - policy_perturb)

            # print(f"Pos loss: {pos_loss}; Neg loss: {neg_loss}")
            # Compute grads
            policy_grads += policy_perturb * (pos_loss - neg_loss)
            # print(policy_perturb * (pos_loss - neg_loss))

            # For logging
            policy_losses.extend([pos_loss, neg_loss])

        # Evaluate current smallest loss and save it
        min_costs = []
        if np.mean(cur_costs) > cur_best_loss:
            for _ in range(num_evals):
                min_costs.append(estimate_policy(weights=cur_best_weights, loss_func=loss_func, real_robot=real_robot))
        else:
            min_costs = cur_costs
        np.save(f"models/{wandb.run.id}/epoch{epoch}-min-{wandb.run.id}-{np.mean(min_costs)}", cur_best_weights)

        # Compute grads
        policy_grads = (beta * policy_grads) / (2 * std**2 * num_perturbations)
        set_policy_weights_from_numpy(policy_weights)
        set_policy_grads_from_numpy(policy_grads)

        # Optimize
        update_weights_simple(lr, epoch=epoch)

        # Log current loss
        average_loss = np.mean(policy_losses)
        wandb.log({
            "Loss": average_loss, 
            "Min Loss": np.mean(min_costs),
            "Min Loss Std": np.std(min_costs), 
            "Centroid Loss": np.mean(cur_costs),
            "Centroid Loss Std": np.std(cur_costs),
            "Num Samples": (epoch + 1) * num_perturbations * 2,
        })
        print(f"Epoch: {epoch}; Mean loss: {np.mean(policy_losses)}; Time: {ptime.asctime()}")

    # Save the latest model
    np.save(f"models/{wandb.run.id}/latest-policy-{wandb.run.id}", policy_weights)

### Covariance Matrix Adaptation Evolutionary Strategies
def train_cma_es(lr, seed, loss_func, std, num_perturbations, num_iterations, real_robot=False, resume=None, num_evals=1):
    # Fix seeds
    np.random.seed(seed)
    random.seed(seed)

    # Log best policy
    best_loss = 1000000

    # Initialize the system and policy
    initialize_weights()
    initialize_system_params()
    initialize_normalization_buffers()

    # Load policy to resume the training
    if resume is not None:
        set_policy_weights_from_numpy(np.load(resume))

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(
        extract_current_policy(), 
        std, 
        {"popsize": num_perturbations*2, "seed": np.nan, "CMA_cmean": lr}
    )

    for epoch in range(num_iterations):
        policy_weights  = np.copy(es.mean)
        policy_losses   = []

        # Evaluate current policy and save it
        cur_costs = []
        for _ in range(num_evals):
            cur_costs.append(estimate_policy(weights=policy_weights, loss_func=loss_func, real_robot=real_robot))
        np.save(f"models/{wandb.run.id}/epoch{epoch}-cur-{wandb.run.id}-{np.mean(cur_costs)}", policy_weights)

        # Find current smallest loss
        cur_best_loss    = np.mean(cur_costs)
        cur_best_weights = np.copy(policy_weights)

        # Estimate solutions
        solutions = es.ask(num_perturbations*2)
        for solution in solutions:
            # Positive perturbation
            loss = estimate_policy(solution, loss_func=loss_func, real_robot=real_robot)
            # update_normalization_buffers()
            # Save best model
            if loss < cur_best_loss:
                cur_best_loss = loss
                cur_best_weights = np.copy(solution)

            # For logging
            policy_losses.extend([loss])
        
        # Make a step with CMA-ES
        es.tell(solutions, policy_losses)

        # Evaluate current smallest loss and save it
        min_costs = []
        if np.mean(cur_costs) > cur_best_loss:
            for _ in range(num_evals):
                min_costs.append(estimate_policy(weights=cur_best_weights, loss_func=loss_func, real_robot=real_robot))
        else:
            min_costs = cur_costs
        np.save(f"models/{wandb.run.id}/epoch{epoch}-min-{wandb.run.id}-{np.mean(min_costs)}", cur_best_weights)

        # Log current loss
        average_loss = np.mean(policy_losses)
        wandb.log({
            "Loss": average_loss, 
            "Min Loss": np.mean(min_costs),
            "Min Loss Std": np.std(min_costs), 
            "Centroid Loss": np.mean(cur_costs),
            "Centroid Loss Std": np.std(cur_costs),
            "Num Samples": (epoch + 1) * num_perturbations * 2,
        })
        print(f"Epoch: {epoch}; Mean loss: {np.mean(policy_losses)}; Time: {ptime.asctime()}")

    # Save the latest model
    np.save(f"models/{wandb.run.id}/latest-policy-{wandb.run.id}", policy_weights)

def visualize_policy(path=None, real_robot=False):
    initialize_weights()
    initialize_system_params()
    clear_states()
    # Visualize loaded policy, otherwise vis random
    if path is not None:
        set_policy_weights_from_numpy(np.load(path))

    # Real robot vs simulation
    if real_robot:
        real_policy.set_weights(extract_current_policy())
        cost = forward_real()
        print(f"Real robot cost: {cost}")
    else:
        forward(cost_function_energy, visualize=True)

# Parse training arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alg",  type=str, default="taichi", choices=["taichi", "vanilla-es", "guided-es", "cma-es"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr",   type=float, default=1e-3)
parser.add_argument("--loss",  type=str, default="total-energy", choices=["total-energy", "position"])
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--num-perturbs", type=int, default=20, help="Number of perturbations to estimate the gradient for vanilla and guided es.")
parser.add_argument("--num-prev-grads", type=int, default=4, help="Number of previous gradients to span subspace for guided es.")
parser.add_argument("--std", type=float, default=0.01, help="Standard deviation for perturbations (used for es and guided es).")
parser.add_argument("--alpha", default=0.5, type=float, help="Hyperparameter for Guided ES.")
parser.add_argument("--beta", type=float, default=2, help="Hyperparameter for Guided ES.")
parser.add_argument("--random-forces", action="store_true", default=False, help="Whether to add random forces")
parser.add_argument("--real", action="store_true", default=False, help="Whether we should use real robot.")
parser.add_argument("--resume", type=str, default=None, help="Path to the policy to continue training (default -- None)")
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "fromage", "sgd"])
parser.add_argument("--visualize", action="store_true", help="Visualize a policy.")
parser.add_argument("--env",  type=str, default="pendulum", choices=["pendulum", "jumper"])
parser.add_argument("--obs-normalize", action="store_true", default=False)
parser.add_argument("--smooth", action="store_true", default=False)
parser.add_argument("--codename", type=str)
parser.add_argument("--shooting-n", type=int, default=10, help="How many steps to optimize with differentiable simulator.")
args = parser.parse_args()

# Create real robot environment if specified
if args.real:
    from real.env import RealPendulum
    from real.env import RealPolicy

    real_env    = RealPendulum(target_q=0.0, n_seconds=N_SECONDS, action_frequency=ACTION_FREQUENCY)
    real_policy = RealPolicy(tanh=args.smooth)

# Visualization only
if args.visualize:
    for _ in range(10):
        visualize_policy(path=args.resume, real_robot=args.real)
else:   
    # Logging
    wandb.init(project="pendulum")
    wandb.config.update(args)

    # Choose loss based on the task
    if args.loss == "total-energy":
        target_loss = cost_function_energy
    elif args.loss == "position":
        target_loss = cost_function_position
    else:
        raise NotImplementedError()

    # Create a folder for the run
    os.mkdir(f"models/{wandb.run.id}")

    # Run specified algorithm
    if args.alg == "taichi":
        train_taichi(lr=args.lr, seed=args.seed, loss_func=target_loss, num_iterations=args.iterations, 
            real_robot=args.real, resume=args.resume)
    elif args.alg == "vanilla-es":
        train_es(lr=args.lr, seed=args.seed, loss_func=target_loss, std=args.std, 
            num_perturbations=args.num_perturbs, num_iterations=args.iterations, real_robot=args.real,
            resume=args.resume)
    elif args.alg == "guided-es":
        train_guided_es(lr=args.lr, seed=args.seed, loss_func=target_loss, std=args.std, 
            num_perturbations=args.num_perturbs, k=args.num_prev_grads, alpha=args.alpha, beta=args.beta,
            num_iterations=args.iterations, real_robot=args.real, resume=args.resume)
    elif args.alg == "cma-es":
        train_cma_es(lr=args.lr, seed=args.seed, loss_func=target_loss, std=args.std, 
            num_perturbations=args.num_perturbs, num_iterations=args.iterations, real_robot=args.real,
            resume=args.resume)

# Clean up
if args.real:
    real_env.close()