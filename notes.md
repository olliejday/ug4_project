
Installation required:

gtimer
torchvision (Cuda 10.2)
tkinter
pygame
inputs
pip install -e . // in CQL/d4rl and in d4rl repo itself

______


Train with

python examples/cql_antmaze_new.py --env=antmaze-medium-play-v0 --policy_lr=1e-4 --seed=10 --lagrange_thresh=5.0 --min_q_weight=5.0 --gpu=0 --min_q_version=3

Eval with

python scripts/run_policy.py data/<path>/params.pkl

______

TODO:
    - Work on env eg. reward function
    - Work on human control to gather data
    - Get urdf for Franka + Barret
        (bh_alone.urdf problems with pybullet? Perhaps bc no inertial)