
Install requirements.txt

pip install -e . // in gym-panda, CQL/d4rl and in d4rl repo itself

______


Train with

python examples/cql_antmaze_new.py --env=antmaze-medium-play-v0 --policy_lr=1e-4 --seed=10 --lagrange_thresh=5.0 --min_q_weight=5.0 --gpu=0 --min_q_version=3

Eval with

python scripts/run_policy.py data/<path>/params.pkl

______

Panda gym 
obs (8): (3) cartesian end eff pos, (2) finger joint pos,
acs(4): (3) target Cartesian pos end-eff, (1) joint variable for both fingers  
rew(1): 1 if obj z > 0.45
(render gives camera view)
info: obj pos
done: when obj z > 0.45

My version

Based on these assumptions:
1. better match adriot, esp. hand shaped reward
2. assume some sensor system extracted object position
3. task of lifting off table not move from a to b (possible extension)
obs (5): (3) cartesian end eff pos, (2) finger joint pos, (3) end eff pos - obj pos
acs(4): (3) target Cartesian pos end-eff, (1) joint variable for both fingers  
reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
       if obj_pos[2]>0.45:
            reward += 10.0                                          # task complete - object lifted 
note that sparse reward corresponds to the "done" signal here
info: obj pos     
done: when obj z > 0.45

*Note: not using Barrett hand at the moment

______

Penetration
Fixed by:
    Think changing sim timestep
    and setting forces in setJointMotorControlArray() in step()
    
Tried some other sim params didn't seem to help

Be careful setting actions (dx,dy,dz) too high as this can lead to penetration
Set action bounds: currently -5 to 5 as found this ok on the joystick controller 
but if have issues change sim params or reduce these bounds.

Bug seems to return if load env using 
 data['evaluation/env'] 
But not if make new env with gym.make("panda-v0")
____

- For more alt panda gym see https://github.com/quenting44/panda-gym

____

Epoch                                        38
---------------------------------------  ----------------
/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/autograd/__init__.py:132: UserWarning: Error detected in AddmmBackward. Traceback of forward call that caused the error:
  File "/snap/pycharm-professional/226/plugins/python/helpers/pydev/pydevd.py", line 2167, in <module>
    main()
  File "/snap/pycharm-professional/226/plugins/python/helpers/pydev/pydevd.py", line 2158, in main
    globals = debugger.run(setup['file'], None, None, is_module)
  File "/snap/pycharm-professional/226/plugins/python/helpers/pydev/pydevd.py", line 1470, in run
    return self._exec(is_module, entry_point_fn, module_name, file, globals, locals)
  File "/snap/pycharm-professional/226/plugins/python/helpers/pydev/pydevd.py", line 1477, in _exec
    pydev_imports.execfile(file, globals, locals)  # execute the script
  File "/snap/pycharm-professional/226/plugins/python/helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/home/ollie/University/y4/ug4_project/cql/panda_env_dev/cql_panda_train.py", line 234, in <module>
    experiment(variant)
  File "/home/ollie/University/y4/ug4_project/cql/panda_env_dev/cql_panda_train.py", line 139, in experiment
    algorithm.train()
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/core/rl_algorithm.py", line 46, in train
    self._train()
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/core/batch_rl_algorithm.py", line 170, in _train
    self.trainer.train(train_data)
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/torch/torch_rl_algorithm.py", line 40, in train
    self.train_from_torch(batch)
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/torch/sac/cql.py", line 179, in train_from_torch
    self.qf2(obs, new_obs_actions),
  File "/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/torch/networks.py", line 88, in forward
    return super().forward(flat_inputs, **kwargs)
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/torch/networks.py", line 73, in forward
    preactivation = self.last_fc(h)
  File "/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 93, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/nn/functional.py", line 1690, in linear
    ret = torch.addmm(bias, input, weight.t())
 (Triggered internally at  /pytorch/torch/csrc/autograd/python_anomaly_mode.cpp:104.)
  allow_unreachable=True)  # allow_unreachable flag
Traceback (most recent call last):
  File "/home/ollie/University/y4/ug4_project/cql/panda_env_dev/cql_panda_train.py", line 139, in experiment
    algorithm.train()
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/core/rl_algorithm.py", line 46, in train
    self._train()
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/core/batch_rl_algorithm.py", line 170, in _train
    self.trainer.train(train_data)
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/torch/torch_rl_algorithm.py", line 40, in train
    self.train_from_torch(batch)
  File "/home/ollie/University/y4/ug4_project/cql/CQL/d4rl/rlkit/torch/sac/cql.py", line 299, in train_from_torch
    policy_loss.backward(retain_graph=False)
  File "/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/tensor.py", line 221, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/ollie/.virtualenvs/ug4_project/lib/python3.6/site-packages/torch/autograd/__init__.py", line 132, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [256, 1]], which is output 0 of TBackward, is at version 40001; expected version 40000 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
python-BaseException


Occurs on server and on PC at epoch 38.
But on PyCharm debugger PC the data seems ok at this iteration.
qf2.fcs.2.bias is on v 40001 as is qf1.fcs.1.weight
whereas policy is on v 40000
40000 comes form the pre-training before "policy eval start"
So not been fully training up until this point
Setting policy_eval_start to 0 (in args not dict) then issue raises immediately
See https://github.com/aviralkumar2907/CQL/issues/5 

Solution: (?)
X My line 182 in cql.py adding detach on the q fn part
  No, see the issue, this breaks part of the gradient for the policy

* Tried moving the Q function use for the policy until after the Q function update
not sure if this is how it should work but seems to match the paper
* Otherwise revert to torch 1.4

______

Not training well

______

TODO:
Have regathered data to try again since changed the env.
If not try torch 1.4
- Train CQL longer - ideally 1M training steps
- More complex env - randomness, peturbation
- Human data?

Keep an eye on:
    - Panda Gym allows intersection of gripper with tray
    - Check action bounds
