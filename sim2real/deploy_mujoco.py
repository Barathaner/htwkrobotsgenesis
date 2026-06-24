# load model inference

#create mujoco scene
# load robot urdf in mujoco
# get robot data observations
# transform everything into model observation
# send to model
# get actions
# send to robot
# kill switches
# add joint limit termination
import os
import mujoco as mj
import mujoco.viewer
import math
import torch
import torch.nn as nn
import os
import yaml

with open("deploy.yaml", "r") as f:
    config = yaml.safe_load(f)

class _RNN(nn.Module):
    """Wrapper whose attribute name (`rnn`) makes the LSTM weights load under the checkpoint's
    `rnn.rnn.*` keys (rsl-rl RNNModel nests an RNN module that itself holds the nn.LSTM)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)


class ActorLSTM(nn.Module):
    """Onboard recurrent actor matching the trained rsl-rl RNNModel actor.

    Flow per step: obs(obs_dim) -> LSTM(obs_dim -> rnn_hidden_dim) carrying hidden state across
    steps -> MLP head -> action(num_actions). The LSTM hidden state IS the policy's memory (it
    replaces the training-time proprio frame-stack), so we feed a single proprio frame each step
    and keep (h, c) between steps. Call reset() once before handing control to the policy.
    """

    def __init__(self, obs_dim: int, rnn_hidden_dim: int, head_dims: tuple[int, int],
                 num_actions: int, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = _RNN(obs_dim, rnn_hidden_dim, num_layers)
        h0, h1 = head_dims
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, h0), nn.ELU(),
            nn.Linear(h0, h1),             nn.ELU(),
            nn.Linear(h1, num_actions),
        )
        self._hidden = None  # (h, c), each (num_layers, 1, rnn_hidden_dim)

    def reset(self) -> None:
        """Clear the LSTM memory (call at bring-up before the policy takes over)."""
        self._hidden = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (obs_dim,) for a single robot -> (seq=1, batch=1, obs_dim)
        x = obs.reshape(1, 1, -1)
        out, self._hidden = self.rnn.rnn(x, self._hidden)
        return self.mlp(out.reshape(-1))


def load_model(checkpoint_path: str) -> ActorLSTM:
    """Load the trained recurrent actor for inference.

    All layer sizes are inferred from the checkpoint weights, so the deploy network always matches
    the trained one regardless of config. Loads `actor_state_dict` (direct RL training); falls back
    to `student_state_dict` / `actor` for older checkpoints.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_state_dict") or ckpt.get("student_state_dict") or ckpt.get("actor")
    if sd is None:
        raise KeyError(f"No actor weights in {checkpoint_path} (keys: {list(ckpt.keys())})")
    if "rnn.rnn.weight_ih_l0" not in sd:
        raise KeyError(
            "Checkpoint has no LSTM weights ('rnn.rnn.weight_ih_l0'); this deploy script expects the "
            "recurrent RNNModel actor. For a feedforward MLP actor, use the older ActorMLP path."
        )
    # Infer dims from the weights: LSTM weight_ih_l0 is (4*hidden, obs_dim); head from mlp.*.weight.
    w_ih = sd["rnn.rnn.weight_ih_l0"]
    obs_dim = w_ih.shape[1]
    rnn_hidden_dim = w_ih.shape[0] // 4
    num_layers = sum(1 for k in sd if k.startswith("rnn.rnn.weight_ih_l"))
    head_dims = (sd["mlp.0.weight"].shape[0], sd["mlp.2.weight"].shape[0])
    num_actions = sd["mlp.4.weight"].shape[0]

    actor = ActorLSTM(obs_dim, rnn_hidden_dim, head_dims, num_actions, num_layers)
    # strict=False: the checkpoint also holds distribution.std_param, unused for deterministic inference.
    missing, unexpected = actor.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("distribution.")]
    if missing or unexpected:
        print(f"[load_model] missing={missing}  unexpected={unexpected}")
    actor.eval()
    print(f"[load_model] recurrent actor: obs_dim={obs_dim} rnn_hidden={rnn_hidden_dim} "
          f"layers={num_layers} head={head_dims} actions={num_actions}")
    return actor



model = mj.MjModel.from_xml_path(config["policy"]["urdf_path"])
data = mj.MjData(model)
model.opt.gravity[:] = 0.0
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.geomgroup[0] = 0  # hide collision geoms (group 0), show visual meshes (group 1)
    while viewer.is_running():
        print(f"model: {model.njnt} joints, {model.nu} actuators, "
        f"nq={model.nq} (qpos size), nv={model.nv} (qvel size)\n")
        mj.mj_step(model, data)
        viewer.sync()
