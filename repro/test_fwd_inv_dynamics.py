import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import argparse

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="cpu or cuda",
    )

  args = parser.parse_args()

  _device = args.device
  urdf_fname="assets/factory_franka.urdf"
  urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), urdf_fname)

  model = DifferentiableRobotModel(
      urdf_path, name="panda",  device=_device
  )

  mse = nn.MSELoss()

  T = 1000
  dt = 1/60
  t = np.arange(T)*dt
  q = torch.from_numpy(np.tile(-np.cos(2*np.pi*dt*t), [7,1]).T).to(_device).to(torch.float)
  qd = torch.from_numpy(np.tile(2*np.pi*dt*np.sin(2*np.pi*dt*t), [7,1]).T).to(_device).to(torch.float)
  qdd = torch.from_numpy(np.tile((2*np.pi*dt)**2*np.cos(2*np.pi*dt*t), [7,1]).T).to(_device).to(torch.float)

  for i in range(T):
    # step through each timestep

    # inverse dynamics to get tau
    tau_pred = model.compute_inverse_dynamics(q[i,:], qd[i,:], qdd[i,:], include_gravity=True, use_damping=False)
    # forward dynamics to get qdd
    qdd_pred = model.compute_forward_dynamics(q[i,:], qd[i,:], tau_pred, include_gravity=True, use_damping=False)
    output = torch.norm(qdd_pred - qdd[i,:])

    print(f"L2 loss: {output}")