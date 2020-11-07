
Installation required:

gtimer
torchvision (Cuda 10.2)
pip install -e . // in CQL/d4rl and in d4rl repo itself


Note that not sure the current CQL loads and saves models, but this
shouldn't be too hard using the rllkit code
https://github.com/vitchyr/rlkit/blob/5274672e9ff6481def0ffed61cd1b1c52210a840/rlkit/core/rl_algorithm.py#L55
If an issue with snapshot and envs then comment out part of get_snapshot

Note sure how to evaluate a saved model either