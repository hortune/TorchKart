import numpy as np


def process_frame(frame):
    #frame = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
    frame = frame.astype(np.uint8)[::2, ::2,:]
    return frame


def discrete_action(action):
    d_action = [0, 0, 0]
    if action[0] > 0:
        d_action[0] = 1
    elif action[0] < 0:
        d_action[1] = 1
    elif action[2] > 0:
        d_action[2] = 1
    return d_action
    
def actionTransform(action):
    continuous_action = [0, 0, 0, 0, 0]
    if action[0] > 0:
        continuous_action[0] = 60
        continuous_action[2] = 1
    if action[1] > 0:
        continuous_action[0] = -60
        continuous_action[2] = 1
    if action[2] > 0:
        continuous_action[2] = 1
    return continuous_action
