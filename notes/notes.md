
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
Doing for longer now - 3k epochs

DONE
Regathered data with a slightly better agent (3M examples)
Changed params based on CQL paper - adroit task params
______

	Running torch 1.4 
	Much slower bc not using GPU as CUDA version doesn't match
	Got "killed" at 320 epochs - not sure why
	Compare results
	Behaviour wise -does manage to get a grasp on some runs but seems to just be redoing the same initial movement regardless of where the object is placed, though the closing fingers seems to only happen when object close
	Does seem better esp than other experiment at that stage did also change the parameters
	Checked the object obsv and it changes each time the object moves to seems ok

torch 1.5 get same autograd in place error

1.7 quicker than 1.4
	Ran with no_gpu arg bc it complains "RuntimeError: All input tensors must be on the same device. Received cuda:0 and cpu"
	But since CUDA version is correct for this I think it still uses GPU
	
1.4 behaviour qualitatively / visually appears better but could not be as fundamental, since other epochs models not as good

Using conda for cuda 10.1 with torch 1.4

Ran a 1.4 version
Didn't solve it but looking good and close
Not sure if I noted but also have changed params to the ones paper quotes for using with adroit


GPU
Average epoch speed on v1.7 is about 150
Avg epoch speed on v1.4 is about 250
.
So v1.4 still much slower
.
v1.4 recognises the gpu but while running the gpu is not being used (by looking at memory usage)
.
so look at v1.4 gpu pushing to gpu
want the training and policy on gpu
https://discuss.pytorch.org/t/pytorch-is-not-using-gpu/57358

Think works on GPU now for train
Note had to change the uniform sampling so small chance that could be a bug (for future reference) https://github.com/pytorch/pytorch/issues/24162
Much quicker now :)

Eval and exploration are a small amount of epoch time so could increase?

____

1. Try wenbin's reward and obs

Handy note for mujoco to pybullet
Assuming the body id is n, the 3D position is at mjData.xpos+3n and the quaternion orientation is at mjData.xquat+4*n. You can also get the orientation as a 3x3 orthonormal matrix at mjData.xmat+9*n.

/home/ollie/.virtualenvs/ug4_project1_4/lib/python3.6/site-packages/pybullet_data/franka_panda/panda.urdf

photo of frames from rviz

Obs

relative_obj_pos = obj_pos - hand_pos

hand_yaw
        hand_mat = self.sim.data.get_geom_xmat('palm2')
        hand_quat = Rotation.from_dcm(hand_mat)
        hand_euler = hand_quat.as_euler('zyx', degrees=True)
        hand_yaw = hand_euler[0] * math.pi / 180
        hand_yaw = -np.sign(hand_yaw) * (math.pi - abs(hand_yaw))

joint_qpos = np.array([self.sim.data.get_joint_qpos(name) for name in names])
# fingers only

distance_finger1 = np.sqrt(np.square(obj_pos[0] - finger_pos1[0]) + np.square(obj_pos[1] - finger_pos1[1])
                           + np.square(obj_pos[2] - finger_pos1[2])) - self.object_radius
        finger_pos1 = self.sim.data.get_geom_xpos("finger1_dist").copy()
        finger_pos2 = self.sim.data.get_geom_xpos("finger2_dist").copy()
        finger_pos3 = self.sim.data.get_geom_xpos("finger3_dist").copy()


contactForce = self.sim.data.sensordata[0:7]
# Torque in pybullet is 6Dof 

!Think matches obs now

Reward

dist tips + divergence + dist centre + contact + penalty collision + topological

NOTE
Ignoring penalty obj vel as was zerod in wenbins original code
Also ignoring vector reward (to point hand to obj) from paper as not in wenbins original code
Object keypoints are from axis aligned bounding box may want to change to not axis aligned
    Don't think can get bounding box from aabb and ctr of rot so revert back to using aabb,
Not using topological at the moment as not working
    BC all points are planar or even on a line? so not able to find convex hull
    But seems like an important part of reward and was highest weighted
*Figure out way to find hull for gripper and bounding box (not axis aligned) for object to make topological reward
    and improve object kps

Initial reward weights taken from wenbin's code
Added a completion reward and then debug a bit and give it a try

____

Ran the new reward
It did't work - recorded better performance around 200 iters but on the eval wasnt better
Most seemed to move upwards away from object
Think last experiment (old reward) was better it went towards on some eval iters but no good at grasping

---

For next reward iteration

reward made up of:
distance to obj
contact
- penalty
z height obj 
  
checked and this has nice graph progressing through going to obj then gripping then lifting
same obs as wenbin here except hand yaw (83 vector) and acs (x y z and finger)

Results: Didn't work out

___

Next env iter

For next acs: affine transf on acs so NN can output 0-1 range (which I believe it does anyway based on SAC anyway)
For next obs: normalise inputs to 0-1
    Adroit (qpos for hand, xpos for object, xpos target) = qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos
    So trying obs [25]:
            "dist_fingers": dist_fingers, [6]
            "obj_z": [obj_pos[2]], [1] 
            "palm_pos": palm_pos, [3]
            "qpos_joints": qpos_joints, [12]
            "rel_pos": rel_pos, [3]

Normalising using fixed bounds that are based on pd agent max and min of obs and acs in 500 episodes (+ 30% leeway)

UPDATE: 
Acs less leeway bc it scales the NN outputs too much 
ie. NN [0, 1] if actions scaled to 50% then NN could do an output for 1.5 which may not want
But kept obs scale high incase we get extreme inputs
Changed from ceil/floor to round as it makes more sense
Overridden acs[-1] (fingers) since only part of the space does anything and had issues with grasping behaviour

ACS
# array of max and min acs found in 500 eps pd agent scaled to leeway
mx = np.array([18.79076577, 13.78390522, 32.46248846,  0.11      ]) * 1.1
mn = np.array([ -0.70994379, -13.88668208, -15.17208177,   -0.02      ]) * 1.1  # since all negative
scale = np.round(mx, decimals=3) - np.round(mn, decimals=3) # range
array([21.451, 30.437, 52.398,  0.1  ])
offset = np.round(mn, decimals=3) # min
array([ -0.781, -15.275, -16.689,   0.   ])
acs = acs * scale + offset

OBS
# array of max and min acs found in 500 eps pd agent scaled to 30% leeway
mx = np.array([ 0.35819622,  0.19836367,  0.45274739,  0.1863834 ,  0.19839868, 0.45282429,
        0.40217545,  
        0.7392161 ,  0.18227398,  0.52348188,
        0.27002175,  1.00071738,  0.1655928 , 
            -1.24123825,  0.1390429 , 2.82369482,  
            2.75088341,  0.        ,  0.        ,  
            0.07600003,  0.07599997,  0.        ,  
        0.00983092,  0.19838117,  0.4911857 ]) * 1.3
mnn = np.array([ 1.71960650e-04,  8.04060100e-08,  2.81048371e-05,  5.18226627e-08,
        1.20426205e-07,  8.89086052e-05, -1.47817961e-04,  4.31449456e-01,
       -1.79290969e-01,  8.28353383e-02, -1.91106306e-01, -2.14915632e-01,
       -1.72868083e-01, -2.56998225e+00, -1.81334319e-01,  1.60523483e+00,
        1.94219945e+00,  0.00000000e+00,  0.00000000e+00,  5.13511843e-05,
       -4.88971220e-06,  0.00000000e+00, -2.68499252e-01, -1.96912932e-01,
        3.50155939e-02]) # since mixed signs
mn = mnn - abs(mnn * 0.3) 
scale = np.maximum(np.round(mx, decimals=3) - np.round(mn, decimals=3), 0.01) # range, scale up and to avoid 0 div use 1 as min scale
array([0.466, 0.258, 0.589, 0.242, 0.258, 0.589, 0.523, 0.659, 0.47 ,
       0.623, 0.599, 1.58 , 0.44 , 1.727, 0.417, 2.547, 2.216, 0.01 ,
       0.01 , 0.099, 0.099, 0.01 , 0.362, 0.514, 0.614])
offset = np.round(mn, decimals=3) # min
array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   ,  0.302,
       -0.233,  0.058, -0.248, -0.279, -0.225, -3.341, -0.236,  1.124,
        1.36 ,  0.   ,  0.   ,  0.   , -0.   ,  0.   , -0.349, -0.256,
        0.025])

obs = (obs - offset) / scale

In a test on 500 pd agent episodes with a different seed to one used before it maintains the action and obs range

Results on new env good!
some good progress here seems to be training much better: more clearly goes towards the object now but it tends to stay there and not grip
x cap the closeness reward as it encourages pushing to tray too much
x make reward for having fingers closed around object
x multiply the actions by less -  too extreme movement
x add completed task to info and print if it happens!

Changed action z min offset to -15 (this was min from 500 pd agent runs) as otherwise it tries to intersect the tray
    also scaled the scale factor appropriately (still 1.3 x max)

----

Made above changes to env and reward and have changed reward weights

Run again 
lower action scale a bit - still weird pushing behaviour where the grippers come a bit unattached
some really good reaching behaviour - quickly learned as well ~ 30 epochs
starts to press it up the sloped sides - to get some z?
but no grasping
think the print of completion was broken
  
changed ac scale
changed reward and obs from palm to finger tip (grasp target) to try and prevent overreaching
doubled the weight of contact, grasp and z to encourage this behaviour

Results: no better 

changed back to old reward weights (but included the grasping reward)

eval performance doesn't seem to match during training eval
    could be path length less?
changed (increased) eval steps during training as could just be the random episode that the eval gets during training 

changed ac bounds back to original (see above) it's not these bounds for speed
    changed the forces in the p.setJointMotorControlArray to 100 (from 5 x 240) seemed to improve will see next run

____

Results good reaching again but no grasping, solved the breaking sim with the action forces

x doing 2 eval eps and it doesnt seem like it's just chance that don't get comparable env performance
x changed ac bounds
x change grasp action is it 0 or 1? make it treshold

changed eval seed to match training one, also changed code to use the same classes for eval so it should be
close now
#SEED HAS A BIG IMPACT
Though ideally want seed invariant behaviour not chancing it

Changed ac bounds to 10% as otherwise get a scaling effect
Obs bounds still 30%
For both I also removed the floor / ceil instead round to a few dp

Added an eps minimum scale for obs to avoid division by 0,
test new ac and obs bounds for 100 pd agent eps - OK!

Currently gets more reward for staying near object than lifting it - adding a boost to each stage 
so get near is good, grip/contact is ~10x better and lifting is ~5x better than that 
____

TODO:
Work on params / reward function / env
    Topology and obj kps    
    Reward weights
    Params - NN params

- Work on robustness under perturbed params eg. gravity, robot shape, object etc. (2nd marker)
    - Both in gathered data and in perturbed after learning
- Human data?
- Compare to other methods

Keep an eye on:
    - Panda Gym allows intersection of gripper with tray
    - Check action bounds

________

TIMELINE

- implementation - by week 2/3 sem 2;
- experimentation and evaluation - by week 6 sem 2;
