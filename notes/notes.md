
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

Itr 50 - recorded video
IT DID IT!!!!!!!! First pickup!! 
Needs work but have our first completion :)
With different seed gets even more pickups - careful tho seed shouldn't be a hparam!

Itr 20 - recorded video
Flipping it up as a hack 
Gets more reward than picking it up bc object z but doesn't pass threshold for completion!

Still issues with it pushing object through tray (though hand holds together now) - less force still?

xDebug object penetrating tray

Fixed bug in it penetrating tray by lowering the simulation param 'fixedTimeStep' to 0.005 (was 0.01666)
Now it picks up much more consistently and actually pretty well

x Update li &c

____

x Changed camera to close on Li's feedback

x Changed force params to https://github.com/qgallouedec/panda-gym/blob/master/panda_gym/envs/panda_env.py
.
They have pybullet do multiple sim steps per env step but this seemed to go crzy fast then and doesn't seem to have been done
in other cases eg. https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/loadpanda.py
so I left it
.
x Added action clip in env Limit pd to -1, 1

Adroit centring
  self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

#SIGNIFICANT CHANGE TO PD AGENT ACTION -> JOINT POSITION CONTROL DIRECTLY NOT IK
x changed pd agent to output 0, 1 then to -1 to 1 after process, most seem ok with current pd agent except scale fingers
x Trying with IK in the pd agent and the env using joint positions directly
x set to 0, 1 inputs ie calc new scale and offset for joint position control

xtry a test with env.action_space.sample() -> OK

#Current action and obs: Range of 500 pd agent runs, scaled for 25 % leeway (scale only, offset same)
#also removed obs that output always 0s

Tried using mean and std and it led to agent getting stuck at 1 or -1 action not moving
So not using scaled std

These are stats for 500 eps of PD agent with joint position control to use for scaling actions
action
_mean = array([ 0.02008761,  0.25486009, -0.01521539, -2.08170228,  0.00326891,
        2.33839231,  2.35234778,  0.0397479 ,  0.0397479 ])
_std = array([0.1882528 , 0.30223908, 0.1093747 , 0.26050974, 0.05065062,
       0.21928752, 0.29453576, 0.03993519, 0.03993519])
_min = array([-0.41109789, -1.11157729, -0.32441981, -2.87685238, -0.3       ,
        1.57872595,  1.63177117,  0.        ,  0.        ])
_max = array([ 0.98      ,  1.00411438,  0.42858269, -1.00333763,  0.26396   ,
        2.94853813,  3.09542839,  0.08      ,  0.08      ])
obs
_mean = array([0.06948607920799513, 0.01064014045877533, 0.09525471551881459,
       0.03932066822522797, 0.010644994960273341, 0.09547623670666466,
       0.11729979856542928, 0.5434034595684055, 0.00044514429090377434,
       0.18606473090696793, 0.015090418513382066, 0.248192467157062,
       -0.015294364772409075, -2.105414401533755, 0.003735502292605632,
       2.355187771295793, 2.3521831777351863, 0.029676513100094094,
       0.027724920054124354, -0.02947791052178796, 0.0003754021206685438,
       0.06876493234153894])
_std = array([0.06509164, 0.02973735, 0.09385008, 0.02916187, 0.02982581,
       0.09411771, 0.11683397, 0.07144959, 0.14798064, 0.09347103,
       0.16901112, 0.29806916, 0.10117057, 0.26681981, 0.04518137,
       0.19360735, 0.28129071, 0.0126499 , 0.01449605, 0.05605754,
       0.03160347, 0.09398505])
_min = array([0.02199213036774106, 2.802437896326504e-09, 0.0070921621972258875,
       3.486307248556919e-06, 2.4263667351351326e-09,
       0.006485759822393425, 0.030161234592936122, 0.21574750983654462,
       -0.36244449829691683, 0.019308383740818072, -0.40682897694619446,
       -1.0546190481233837, -0.31640120747741457, -2.8718891524392007,
       -0.158176452138274, 1.663980899376096, 1.6419835706670085,
       0.009704195538826956, 0.00016580998873661233, -0.26809356175399957,
       -0.19556410123267962, -0.01965750572605808])
_max = array([0.3581201150473813, 0.19892358616449943, 0.2330780511400498,
       0.19305533592376078, 0.19958718642409035, 0.2334841796629028,
       0.39998960232702613, 0.7007092450142101, 0.3347592996051957,
       0.39064084188968184, 0.4685819506071406, 0.9423653675063055,
       0.4049743490961882, -1.040374840980018, 0.2478303317381189,
       2.975077800832962, 2.9655006823335808, 0.074, 0.07400003525670308,
       0.0156394452436161, 0.19894109028121762, 0.20668229345009592])

x Added action scaling above and checked new outputs - OK

xChange max steps to 1500

xChange to pick and place? -> Li says pick and reach OK atm

___

Next run with new joint control:

it's clear this is harder env 
behaviour much worse

itr 420 manages to pickup well once it doesn't fully pickup but this seems to give it better reward for object z rather than completion
but other episodes it doesn't

itr 390 does the same trying to stand the object on end

Lots of the time it stops near the object
print actions - does seem to be that the policy outputs settle during the episode - why is it learning this?

I suspect it was reward that was making it stay there as it learned that staying near was better than missing and going 
further away so trying with different reward
If not can look at changing env / pd agent

xShorten eval max path length -> 1200
xChanged reward more sparse -> similar to adroit has only distance and then completion

xCHange completion to log not print
____

Next run (sparse reward):
had some completions (few) during training
couldn't find any completions during eval
If still stopping short need look into it 

still stops short of the object most times - why?? 
    plotting actions in eval looks as action bounds making it stop as action outputs saturate to 1 and 0 and it freezes
x -> changed action bounds to scaled std for more leeway, also added leeway on obs as will go off pd path



___

Next run - new action bounds
Better - learns to get v close to the object
and in v few epochs - 10-20!
But still not grasping just stops by object
Some completions in logs
Check actions again
Now does seem to be a learned thing as it's not saturating to +/- 1 just stopping

---

Trying fully sparse (ie only reward completion) as think the distance might make it stop short?
Lowered scale_std to 7

Running
Had some completions, 
But on eval the behavior is a bit janky and didn't see any completions
Doesn't seem to stop as much 


---

Trying on dense reward with new ac/obs space and new action inputs

Running
save more models at the start, sometimes completes quite early

actions seem strange sometimes it flops and still pausing
this could just be early learning behaviour

give it multiple tries!

itr10: messes up 1st and 3rd time but ok on 2nd

xlower max eval path len in _train.py

x increased joint forces -> solved flopping behaviour think some were too low
think this was also perhaps causing the stopping - could have been an avoidance behaviour for parts where it would flop?

---

Trying on dense reward with new ac/obs space and new action inputs
Now fixed a bug in joint forces

Running
Two pickups (but not completion) and one throw out the tray ! on itr 30

x Record iter 30
x Do action and reward plots on 30 and nearby iters
x Changed rewards weights
    30 - reward z too much compared to reward completion, reward grasp not coming in early enough
    10 and 40 high rwd without pickup - contact etc.
        Think it's learned as not near action bounds
        Mostly seems to just be better on the reward dist with a bit of contacts too
    20 doesnt go near - reward seems sensible
x increased threshold for grasp
x More freq saves (at least at start), can do less total iters if need
    changed to save every 3
x Lower learning rate? 
    policy_lr=3e-5, -> 1e-5
    qf_lr=3e-4, -> 1e-4


Q: Is it still stopping?
    Why? Bullet dynamics?
    Learned behaviour?
    Actions / obs limits?
    Is it when some obs goes too big / small?


----

Was worse with the changes

x Changed LR back 
Reward weights, changing z to be higher again and higher reward completion so it steps better

x Try again with min / max scaling and offset

_max = _max + abs(_max * 0.3)
_min = _min - abs(_min * 0.3) 
scale = np.maximum(np.round(_max, decimals=3) - np.round(_min, decimals=3), 0.01) # range, scale up and to avoid 0 div use 1 as min scale
    acs array([1.808, 2.75 , 0.979, 3.038, 0.733, 2.728, 2.882, 0.104, 0.104])  
    obs array([0.451, 0.259, 0.298, 0.251, 0.259, 0.299, 0.499, 0.76 , 0.906,
           0.494, 1.138, 2.596, 0.937, 3.005, 0.528, 2.703, 2.706, 0.089,
           0.096, 0.369, 0.513, 0.295])
offset = np.round(_min, decimals=3) # min
  acs array([-0.534, -1.445, -0.422, -3.74 , -0.39 ,  1.105,  1.142,  0.   ,
        0.   ])
  obs array([ 0.015,  0.   ,  0.005,  0.   ,  0.   ,  0.005,  0.021,  0.151,
       -0.471,  0.014, -0.529, -1.371, -0.411, -3.733, -0.206,  1.165,
        1.149,  0.007,  0.   , -0.349, -0.254, -0.026])
obs = (obs - offset) / scale


Running 
No good, going back to mean, std

* Note this is [0, 1] but I think [-1, 1] should be ok so worth trying that too?
  
----

x Try back to mean / std acs and obs scaling as have changed eg. reward weights

Running

Best model yet, 3 really good pickups and 2 where it goes near then sort of feints the pickup but doesn't grasp

----

Based on last run:
x zero weight reward closeness and halved rewards weight for grasp
x Plot csv to see how mean eval looks over time
x Change params for better stability
    Changed Tau to 5, increased policy eval start, temp to 1.0, changed eval and explr steps to be 2 x max path len
    Increased batch size to 352

Running:
21/01/20
#epoch 21
#Really good model :)) learns to pickup VERY well and consistently looks as good as the pd agent 
#also training much more stable, has a flat area where there's a few good models

x Use --plot arg to see best epochs and plot training

moved experiment 4 to "dense_reward"
____

Trying with sparse reward ie. only reward completion 
Running
sparse reward worked v well
saved as folder "sparse_reward", itrs 21 and 24 v good
this should outperform SAC now on sparse?

x try with moving object, adding forces to current model
the model actually looks v good with forces after grapsing for regrasping
and on moving objects

forces for gripper are quite high need to lower to better test regrasping
set to 60, think pybullet in N presumably(?) and 60N seems reasonable https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf
* reload data with new setting

----

#some significant env changes
Running with sparse for 
x new force setting
x Also added some object rotation in the env
x removed EE rel pos obs as captured in the finger rel pos

Running:

Working well, slower than before but good completion rate on eval
itr 15 good (also 12)

___

x Add z-rotation of hand to match obj to pd agent
x Add to obs: hand and obj orientation
x Redo ac and obs bounds

action
_mean = array([-0.11200337,  0.47604913, -0.11255514, -1.72322873,  0.05802695,
        2.20177014,  2.14862945,  0.04538093,  0.04538093])
_std = array([0.0751389 , 0.28780316, 0.03562647, 0.33271765, 0.04047524,
       0.18582715, 0.36296299, 0.03957525, 0.03957525])
_min = array([-0.17262171, -0.13898705, -0.20030639, -2.5081992 , -0.3       ,
        1.7024556 ,  1.40370972,  0.        ,  0.        ])
_max = array([ 0.98      ,  0.98307127,  0.31      , -1.34497701,  0.11655295,
        2.66      ,  2.95723005,  0.08      ,  0.08      ])
obs
_mean = array([ 1.09038504e-01,  2.43092468e-02,  1.14092714e-01,  5.87583031e-02,
        2.59921405e-02,  1.14554145e-01, -3.01276719e-03,  1.07389465e-02,
       -2.03404409e-02,  9.82628258e-01,  6.94992045e-01, -1.49608278e-01,
        1.04590331e-01,  3.79365430e-02,  1.73891483e-01,  2.02375957e-04,
        5.93251903e-04,  6.26668490e-01, -1.33427533e-01,  1.92303545e-01,
       -1.11894676e-01,  4.53734420e-01, -1.08550465e-01, -1.76779071e+00,
        5.61063733e-02,  2.22304690e+00,  2.15376396e+00,  3.28458920e-02,
        2.90117830e-02])
_std = array([0.09425424, 0.03453499, 0.09675853, 0.05580793, 0.03554957,
       0.09721908, 0.00554906, 0.02974946, 0.18101611, 0.01486331,
       0.00686845, 0.00318824, 0.11282189, 0.69398994, 0.69761722,
       0.00488094, 0.00255604, 0.08149176, 0.0372437 , 0.0881075 ,
       0.03549766, 0.31418283, 0.02916145, 0.35910005, 0.03699991,
       0.16163746, 0.3542222 , 0.01205007, 0.01599951])
_min = array([ 1.92500184e-02,  6.32373424e-08,  1.30999407e-02,  2.15915869e-06,
        1.14616768e-07,  1.25570414e-02, -2.83966747e-02, -2.97758005e-02,
       -4.17617907e-01,  9.08572268e-01,  6.70920824e-01, -1.62917261e-01,
        3.05185746e-02, -7.07103802e-01, -7.07096325e-01, -1.54908202e-02,
       -1.21148832e-02,  4.31514456e-01, -1.55106014e-01,  2.08437720e-02,
       -1.52984620e-01, -2.09558345e-01, -1.93352990e-01, -2.56789067e+00,
       -3.65522370e-02,  1.76241342e+00,  1.45049050e+00,  3.71973469e-05,
        7.44541225e-04])
_max = array([ 3.58512098e-01,  1.54126248e-01,  2.31075382e-01,  1.94226004e-01,
        1.61858430e-01,  2.32605271e-01,  3.40211551e-03,  1.17414884e-01,
        3.05386805e-01,  9.99999573e-01,  7.10365369e-01, -1.32971311e-01,
        3.99996360e-01,  8.89163902e-01,  9.34227788e-01,  1.71245804e-02,
        8.97560622e-03,  7.03969001e-01, -5.49391682e-12,  3.89069215e-01,
        2.54002901e-02,  9.29571496e-01, -2.41149798e-03, -1.36814082e+00,
        1.14056369e-01,  2.39719090e+00,  2.93964330e+00,  7.39999915e-02,
        7.40000473e-02])

Running:
Ok but stops again
Forgot to add scale_std - doh!

x added scale

Running:
Nails the rotation and good completions now
itr 15 and 21 of sparse_reward
save as sparse_reward

----

Evaluating with perturbation

Making perturbed physics / dynamics params env
        print(p.getDynamicsInfo(self.objectUid, -1))  # -1 for base
mass 0.1
lateral friction: 1.0
local inertia diagonal (0.0004338002836244407, 0.00026767564485615316, 0.0006307959751799897) 
local inertia pos (0.0, 0.0, 0.0)
local inertial orn (0.0, 0.0, 0.0, 1.0)
restitution 0.0
rolling friction 0.0
spinning friction 0.0
contact damping -1.0
contact stiffness -1.0
??? 2
??? 0.001

Findings (itr 24 vs pd):
- Mass : Both work well within a decent range (tested up to 5kg), 10kg both cant do
- Lateral friction : 0.01, both mostly ok, pd agent slightly better, 0.0 both cant do
- Restitution : seemingly no impact
- Rolling and spinning friction :  increased to 1, seemingly no impact
- contact damping, contact stiffness : both 3 it melts through the tray which acts like a time constraint
        pd ok, rl didn't seem to complete, not sure if bc too slow?
  
- gravity: 0 to -14 tested all ok on both, 
    positive eg. 1 or 2 doesn't work as object floats,
    though smaller eg. 0.2 and 0.4 RL agent grasps, and in larger
    upward gravity Rl agent does sensible track the object under this new dynamic 
    which is noteworthy behaviour completely untrained

-> So think best to perturb mass and lateral friction, with maybe gravity too

Also have force env that pushes it out of hand for regrasp
PD: gets rattled and goes a bit  crazy, doesn't regrasp, but this is cos of the way it's coded, I think this could be improved
RL: seems much more sensible in going toward object and trying to regrasp, didnt see any regrasps though


Trying with different random object each time
Both rather struggle, most of the objects are quite different to the training one and they don't really
model any idea of object shape just location
A few are similar enough to be picked up by PD agent but RL fared less well

-----

Re-running last, had bug in env random object placement was fixed 
Much worse again
Action and obs bounds are wrong bc evaluated on the old env so updated them on full random dropping object
Again, 500 eps of pd agent
_mean = array([ 0.02806388,  0.30300509, -0.02472393, -2.02323566,  0.00893459,
        2.32934239,  2.39454994,  0.03706155,  0.03706155])
_std = array([0.11699859, 0.2507982 , 0.07120614, 0.25733758, 0.03952605,
       0.23106606, 0.38370898, 0.0398212 , 0.0398212 ])
_min = array([-0.1950696 , -0.20995172, -0.40307912, -2.53196562, -0.3       ,
        1.7181369 ,  1.43635216,  0.        ,  0.        ])
_max = array([ 0.98      ,  0.96787091,  0.31      , -1.37680093,  0.16840063,
        2.86766564,  3.24907259,  0.08      ,  0.08      ])
max 3.2490725948033923 min -2.5319656163048476
obs
_mean = array([ 7.01500698e-02,  1.48965938e-02,  8.78227272e-02,  3.81298514e-02,
        1.69972892e-02,  8.81188941e-02, -3.50820470e-04,  2.73789175e-03,
       -2.36838933e-02,  9.83214924e-01,  5.99429983e-01, -2.00033211e-04,
        1.23243757e-01,  3.04535736e-02,  1.75282018e-01,  5.16394522e-04,
        4.81298620e-04,  5.70159169e-01, -5.37340672e-04,  1.84614199e-01,
        2.21636672e-02,  2.89742493e-01, -2.39614319e-02, -2.05760948e+00,
        8.76474704e-03,  2.34946610e+00,  2.38633066e+00,  3.28678610e-02,
        2.46175433e-02])
_std = array([0.06687247, 0.02015878, 0.08868218, 0.03144816, 0.02078771,
       0.08918064, 0.00506174, 0.0256444 , 0.17838172, 0.01471636,
       0.05853876, 0.08816896, 0.11898107, 0.69388889, 0.69773689,
       0.00432013, 0.00332711, 0.06590971, 0.08246218, 0.09432814,
       0.09399189, 0.26254034, 0.06454262, 0.2696445 , 0.03331196,
       0.20274376, 0.36500204, 0.01071287, 0.01769749])
_min = array([ 1.97172655e-02,  1.37670716e-07,  1.23376957e-02,  5.84826441e-07,
        5.74946320e-07,  1.17838912e-02, -2.72782970e-02, -3.53199986e-02,
       -4.37049335e-01,  8.98455386e-01,  4.89134296e-01, -1.67290472e-01,
        3.05185746e-02, -7.07105350e-01, -7.07092193e-01, -1.70136602e-02,
       -1.51571669e-02,  4.31514456e-01, -1.57026618e-01,  2.08887685e-02,
       -1.83305954e-01, -2.09558345e-01, -4.00651881e-01, -2.56789067e+00,
       -8.24572553e-02,  1.77881613e+00,  1.46723514e+00, -5.06052358e-06,
        1.57424662e-04])
_max = array([ 0.35833499,  0.15268328,  0.23109125,  0.19128437,  0.15839742,
        0.23262836,  0.04093635,  0.16055864,  0.34813955,  0.99999939,
        0.70465086,  0.1589483 ,  0.39999396,  0.90704618,  0.94062738,
        0.01869532,  0.01305024,  0.70677753,  0.15525477,  0.39585465,
        0.4521573 ,  0.91052161,  0.14276864, -1.40041846,  0.16113165,
        2.81671535,  2.96722784,  0.07399999,  0.07400005])





-----

Running (full random drop with rotation and updated bounds)
:: Not aaas good as when it dropped same spot but pretty good again on itr 27

Running initial tests:

For 25 episodes with max path length 700
#Note only one random seed in train and test here so not ideal for reporting

PD
    Normal
        Completed: 25 out of 25
        Mean return 2000.0
        Std return 0.0
        Max return 2000
        Min return 2000
        Mean episode length 215.6
        Std episode length 15.84929020492716
        Mean episode length 244
        Mean episode length 194
    Perturbed
        Completed: 10 out of 25
        Mean return 800.0
        Std return 979.7958971132713
        Max return 2000
        Min return 0
        Mean episode length 389.56
        Std episode length 135.24616963152783
        Mean episode length 499
        Mean episode length 194
    Object
        Completed: 11 out of 25
        Mean return 880.0
        Std return 992.7738916792686
        Max return 2000
        Min return 0
        Mean episode length 395.52
        Std episode length 133.09338676282906
        Mean episode length 499
        Mean episode length 198
    Force
        Completed: 0 out of 25
        Mean return 0.0
        Std return 0.0
        Max return 0
        Min return 0
        Mean episode length 499.0
        Std episode length 0.0
        Mean episode length 499
        Mean episode length 499
RL
    Normal
        Completed 12 out of 25
         | --------------------------  ------------
         | evaluation/Rewards Mean        1.8828
         | evaluation/Rewards Std        61.3355
         | evaluation/Rewards Max      2000
         | evaluation/Rewards Min         0
         | evaluation/Returns Mean      960
         | evaluation/Returns Std       999.2
         | evaluation/Returns Max      2000
         | evaluation/Returns Min         0
         | evaluation/Actions Mean        0.0659089
         | evaluation/Actions Std         0.163724
         | evaluation/Actions Max         0.658218
         | evaluation/Actions Min        -0.484057
         | evaluation/Num Paths          25
         | evaluation/Average Returns   960
         | --------------------------  ------------
    Perturbed
        Completed 2 out of 25
         | --------------------------  ------------
         | evaluation/Rewards Mean        0.236728
         | evaluation/Rewards Std        21.7578
         | evaluation/Rewards Max      2000
         | evaluation/Rewards Min         0
         | evaluation/Returns Mean      160
         | evaluation/Returns Std       542.586
         | evaluation/Returns Max      2000
         | evaluation/Returns Min         0
         | evaluation/Actions Mean        0.0575934
         | evaluation/Actions Std         0.173857
         | evaluation/Actions Max         0.683536
         | evaluation/Actions Min        -0.635698
         | evaluation/Num Paths          25
         | evaluation/Average Returns   160
    Object
        Completed 1 out of 25
         | --------------------------  ------------
         | evaluation/Rewards Mean        0.117144
         | evaluation/Rewards Std        15.306
         | evaluation/Rewards Max      2000
         | evaluation/Rewards Min         0
         | evaluation/Returns Mean       80
         | evaluation/Returns Std       391.918
         | evaluation/Returns Max      2000
         | evaluation/Returns Min         0
         | evaluation/Actions Mean        0.0239796
         | evaluation/Actions Std         0.178087
         | evaluation/Actions Max         0.733802
         | evaluation/Actions Min        -0.660681
         | evaluation/Num Paths          25
         | evaluation/Average Returns    80
         | --------------------------  ------------
    Force
        Completed 0 out of 25
         | --------------------------  ----------
         | evaluation/Rewards Mean      0
         | evaluation/Rewards Std       0
         | evaluation/Rewards Max       0
         | evaluation/Rewards Min       0
         | evaluation/Returns Mean      0
         | evaluation/Returns Std       0
         | evaluation/Returns Max       0
         | evaluation/Returns Min       0
         | evaluation/Actions Mean      0.0238002
         | evaluation/Actions Std       0.198284
         | evaluation/Actions Max       0.736551
         | evaluation/Actions Min      -0.646731
         | evaluation/Num Paths        25
         | evaluation/Average Returns   0
         | --------------------------  ----------
----

With some tweaks now get

PD Agent to on pandaForce

Completed: 21 out of 25
Mean return 1680.0
Std return 733.2121111929343
Max return 2000
Min return 0
Mean episode length 403.8
Std episode length 61.6869516186365
Mean episode length 499
Mean episode length 313

----

Since new PD agent wasn't working with RL after merging branches etc, I reverted to 

FROM
Commits on Feb 15, 2021
97d3c3616c4f2a1f2be6d739afc51654cbd2c16f
updated ac bounds, notes for another run

TO
Commits on Jan 29, 2021
6a294759763e7b0dd34cd9e08734f165e5570cb8 
evaluating current model

---

x Try reintegrating PD agent to make it work with RL (see github commits)
x Running
Works now and gets some completions but seems the rotation EE to match the object
is v hard to learn.
But need rotation in the PD regrasp
Some ideas
-> have less rotation in the object drop?
OR use separate PD agents for normal and regrasp

---

One issue was that the breakpoint was near the start so if it was slighlty +ve rotated the 
PD would rotate the EE all the way around to get to that angle
Have tweaked the code so it no longer does this
Gets 65/100 completions on pandaForce-v0 which matches performance of before
But seems more sensible behaviour so may be easier to learn

Updated ac and obs bounds for new pd agent

Data ranges
action
_mean = array([ 0.02375603,  0.43158534, -0.01576008, -1.81430351, -0.01987865,
        2.31259164,  2.39138202,  0.03699802,  0.03699802])
_std = array([0.08918371, 0.24551347, 0.06217003, 0.40963803, 0.03699512,
       0.27017929, 0.3119481 , 0.03981999, 0.03981999])
_min = array([-0.17270465, -0.15191363, -0.17637853, -2.52307641, -0.3       ,
        1.75827745,  1.56620033,  0.        ,  0.        ])
_max = array([ 0.98      ,  0.92762658,  0.31      , -0.90382799,  0.09314318,
        2.91113192,  3.0803401 ,  0.08      ,  0.08      ])
max 3.080340103171927 min -2.523076409553068
obs
_mean = array([ 6.81661063e-02,  1.69559452e-02,  9.05953964e-02,  3.64948460e-02,
        1.69536628e-02,  9.48007320e-02, -6.80628974e-04,  5.93325256e-03,
       -1.53475883e-02,  9.83968291e-01,  6.53715835e-01,  5.66967211e-04,
        1.30326718e-01,  5.37286309e-02,  1.15512157e-01,  2.80328803e-03,
        3.65818206e-03,  6.22721664e-01,  5.34093075e-04,  1.96366396e-01,
        1.87000435e-02,  4.00800272e-01, -1.59148788e-02, -1.87210578e+00,
       -1.94565248e-02,  2.33326391e+00,  2.38916427e+00,  2.67157787e-02,
        3.00689736e-02])
_std = array([0.06444425, 0.01921252, 0.09458704, 0.03013393, 0.01919489,
       0.09534901, 0.00841659, 0.03049017, 0.17414172, 0.0144692 ,
       0.07282482, 0.06575449, 0.12457435, 0.69967324, 0.70203714,
       0.03160367, 0.0187445 , 0.09453994, 0.05748417, 0.09949193,
       0.05852193, 0.27437893, 0.0562487 , 0.40918357, 0.03083807,
       0.2440567 , 0.30082499, 0.01626229, 0.01353412])
_min = array([ 1.45713888e-02,  2.01577933e-08,  1.18646123e-02,  5.85616189e-07,
        1.89041832e-08,  1.41603095e-02, -4.78759787e-02, -6.62406513e-02,
       -4.17377958e-01,  9.07120772e-01,  4.99203121e-01, -1.49373161e-01,
        3.05185746e-02, -7.07044443e-01, -7.07088056e-01, -7.58179027e-02,
       -6.03738066e-02,  4.31514456e-01, -1.53019824e-01,  2.19410784e-02,
       -1.66572799e-01, -2.09558345e-01, -1.57835754e-01, -2.56789067e+00,
       -1.37890175e-01,  1.79548625e+00,  1.57403389e+00, -1.16635344e-05,
       -4.73344713e-06])
_max = array([ 0.35833499,  0.15268328,  0.28293931,  0.18886651,  0.15252286,
        0.28898519,  0.06433838,  0.16056216,  0.35455735,  0.99999976,
        0.79144941,  0.14983867,  0.39999238,  0.89648331,  0.89292554,
        0.09381397,  0.06986657,  0.78004359,  0.15301469,  0.40104617,
        0.2165153 ,  0.87685478,  0.14217607, -0.94338807,  0.08231745,
        2.86370173,  2.96710144,  0.07399999,  0.07400005])
max 2.9671014401357163 min -2.5678906650756663


Running with new PD agent on normal

Goes towards object but then stops short - have had this behaviour 
before I think it's when the ac/obs bounds are off?

x So try again with larger scale_std?
No joy

Running with dense reward else revert

---

Reverted PD agent

Gets 100/100 completions on panda-v0
65/100 on pandaForce and think this one trained ok

But ac & obs ranges for pandaForce are way off 
For panda-v0 seems ok (maybe a little high on some acs)

x running on panda-v0 to validate reverted
Better looking training but no completions in eval

Running with dense reward
Got a few trained models with 2/11 completions
But not great

____

Have changed action bounds to account for no rotation

action
_mean = array([ 0.01759628,  0.44644278, -0.01332948, -1.76721219,  0.00689955,
        2.22169319,  2.35222207,  0.03636888,  0.03636888])
_std = array([0.08910328, 0.24961502, 0.05902243, 0.42637104, 0.03796177,
       0.29737726, 0.11606062, 0.03976852, 0.03976852])
_min = array([-0.18773279, -0.15149321, -0.17361837, -2.52238139, -0.3       ,
        1.55014257,  1.98497388,  0.        ,  0.        ])
_max = array([ 0.98      ,  0.96057964,  0.31      , -0.84494189,  0.12294165,
        2.86849496,  2.71602508,  0.08      ,  0.08      ])
max 2.868494956884263 min -2.522381387235749
obs
_mean = array([ 7.05726294e-02,  1.11063072e-02,  9.03907967e-02,  3.92186416e-02,
        1.11486849e-02,  9.09569763e-02,  5.73747229e-04,  9.62643615e-03,
       -1.27009801e-02,  9.88232349e-01,  6.55480897e-01,  5.35728616e-04,
        1.34567148e-01, -3.04876965e-02,  3.34614101e-02,  2.19435289e-04,
        1.11297427e-04,  6.19808108e-01,  5.31469275e-04,  1.98568119e-01,
        1.30319187e-02,  4.15571409e-01, -1.37453773e-02, -1.82622626e+00,
        6.46600091e-03,  2.24825844e+00,  2.35241086e+00,  2.43670527e-02,
        3.37543558e-02])
_std = array([0.06478054, 0.01975523, 0.09470092, 0.02952193, 0.01985626,
       0.09510141, 0.00729976, 0.04085118, 0.14592017, 0.01130091,
       0.07178027, 0.0648878 , 0.12652778, 0.70696367, 0.70576696,
       0.00525459, 0.00430436, 0.09146027, 0.0569693 , 0.0994747 ,
       0.05882229, 0.27903955, 0.05304272, 0.42876033, 0.0317729 ,
       0.26973813, 0.11131445, 0.01844993, 0.01123   ])
_min = array([ 2.35634314e-02,  2.51378378e-08,  1.21206231e-02,  3.19653345e-07,
        2.26232270e-08,  1.13801358e-02, -2.83363191e-02, -2.24550474e-01,
       -3.05079850e-01,  9.51017571e-01,  4.96863093e-01, -1.50679429e-01,
        3.05185746e-02, -7.07106707e-01, -7.07106606e-01, -1.32092636e-02,
       -1.50177399e-02,  4.31514456e-01, -1.52640907e-01,  2.04077666e-02,
       -1.77161241e-01, -2.09558345e-01, -1.54571999e-01, -2.56789067e+00,
       -8.40807204e-02,  1.59200494e+00,  1.99320908e+00, -1.63400331e-04,
        1.66778772e-03])
_max = array([ 0.35833499,  0.15268328,  0.27806447,  0.18955161,  0.15172839,
        0.27895782,  0.05044451,  0.21392182,  0.30515558,  0.99999954,
        0.78032017,  0.14983458,  0.39999861,  0.71984097,  0.71871467,
        0.0176924 ,  0.01655006,  0.76027894,  0.15314571,  0.39952235,
        0.20254016,  0.91731292,  0.12812112, -0.87801445,  0.11187831,
        2.81732884,  2.69423665,  0.07399999,  0.07400005])
max 2.817328844166045 min -2.5678906650756663

:::: Running without rotation on pd agent with sparse reward 
Had one completion in one trained model
Seems like it might be acs/obs bounds again limiting it
Another thought is if something I added to the env is limiting it eg. more rotation in object placement or the obj rotation obs?

---

Reverted to a pd agent (still with rotation and got better training)

It gets 100/100 completions on panda-v0
but 0/100 completions on pandaForce-v0

RL got on panda-v0 
Completed 44 out of 52
2021-02-26 11:34:55.139999 GMT | --------------------------  ------------
2021-02-26 11:34:55.140132 GMT | evaluation/Rewards Mean        5.13209
2021-02-26 11:34:55.140204 GMT | evaluation/Rewards Std       101.182
2021-02-26 11:34:55.140283 GMT | evaluation/Rewards Max      2000
2021-02-26 11:34:55.140354 GMT | evaluation/Rewards Min         0
2021-02-26 11:34:55.140423 GMT | evaluation/Returns Mean     1692.31
2021-02-26 11:34:55.140492 GMT | evaluation/Returns Std       721.602
2021-02-26 11:34:55.140560 GMT | evaluation/Returns Max      2000
2021-02-26 11:34:55.140628 GMT | evaluation/Returns Min         0
2021-02-26 11:34:55.140697 GMT | evaluation/Actions Mean        0.0332776
2021-02-26 11:34:55.140765 GMT | evaluation/Actions Std         0.158182
2021-02-26 11:34:55.140834 GMT | evaluation/Actions Max         0.620397
2021-02-26 11:34:55.140902 GMT | evaluation/Actions Min        -0.500996
2021-02-26 11:34:55.140970 GMT | evaluation/Num Paths          52
2021-02-26 11:34:55.141038 GMT | evaluation/Average Returns  1692.31
2021-02-26 11:34:55.141106 GMT | --------------------------  ------------

And on pandaForce-v0
Completed 1 out of 25
2021-02-26 11:44:04.845347 GMT | --------------------------  ------------
2021-02-26 11:44:04.845490 GMT | evaluation/Rewards Mean        0.1155
2021-02-26 11:44:04.845565 GMT | evaluation/Rewards Std        15.1983
2021-02-26 11:44:04.845635 GMT | evaluation/Rewards Max      2000
2021-02-26 11:44:04.845703 GMT | evaluation/Rewards Min         0
2021-02-26 11:44:04.845770 GMT | evaluation/Returns Mean       80
2021-02-26 11:44:04.845837 GMT | evaluation/Returns Std       391.918
2021-02-26 11:44:04.845903 GMT | evaluation/Returns Max      2000
2021-02-26 11:44:04.845983 GMT | evaluation/Returns Min         0
2021-02-26 11:44:04.846049 GMT | evaluation/Actions Mean       -0.0329539
2021-02-26 11:44:04.846115 GMT | evaluation/Actions Std         0.22997
2021-02-26 11:44:04.846181 GMT | evaluation/Actions Max         0.664788
2021-02-26 11:44:04.846247 GMT | evaluation/Actions Min        -0.670215
2021-02-26 11:44:04.846313 GMT | evaluation/Num Paths          25
2021-02-26 11:44:04.846379 GMT | evaluation/Average Returns    80
2021-02-26 11:44:04.846444 GMT | --------------------------  ------------

---

Trying to reintegrate some of the improved PD rotation (will I ever learn?)

Data ranges
action
_mean = array([ 0.02375603,  0.43158534, -0.01576008, -1.81430351, -0.01987865,
        2.31259164,  2.39138202,  0.03699802,  0.03699802])
_std = array([0.08918371, 0.24551347, 0.06217003, 0.40963803, 0.03699512,
       0.27017929, 0.3119481 , 0.03981999, 0.03981999])
_min = array([-0.17270465, -0.15191363, -0.17637853, -2.52307641, -0.3       ,
        1.75827745,  1.56620033,  0.        ,  0.        ])
_max = array([ 0.98      ,  0.92762658,  0.31      , -0.90382799,  0.09314318,
        2.91113192,  3.0803401 ,  0.08      ,  0.08      ])
max 3.080340103171927 min -2.523076409553068
obs
_mean = array([ 6.81661063e-02,  1.69559452e-02,  9.05953964e-02,  3.64948460e-02,
        1.69536628e-02,  9.48007320e-02, -6.80628974e-04,  5.93325256e-03,
       -1.53475883e-02,  9.83968291e-01,  6.53715835e-01,  5.66967211e-04,
        1.30326718e-01,  5.37286309e-02,  1.15512157e-01,  2.80328803e-03,
        3.65818206e-03,  6.22721664e-01,  5.34093075e-04,  1.96366396e-01,
        1.87000435e-02,  4.00800272e-01, -1.59148788e-02, -1.87210578e+00,
       -1.94565248e-02,  2.33326391e+00,  2.38916427e+00,  2.67157787e-02,
        3.00689736e-02])
_std = array([0.06444425, 0.01921252, 0.09458704, 0.03013393, 0.01919489,
       0.09534901, 0.00841659, 0.03049017, 0.17414172, 0.0144692 ,
       0.07282482, 0.06575449, 0.12457435, 0.69967324, 0.70203714,
       0.03160367, 0.0187445 , 0.09453994, 0.05748417, 0.09949193,
       0.05852193, 0.27437893, 0.0562487 , 0.40918357, 0.03083807,
       0.2440567 , 0.30082499, 0.01626229, 0.01353412])
_min = array([ 1.45713888e-02,  2.01577933e-08,  1.18646123e-02,  5.85616189e-07,
        1.89041832e-08,  1.41603095e-02, -4.78759787e-02, -6.62406513e-02,
       -4.17377958e-01,  9.07120772e-01,  4.99203121e-01, -1.49373161e-01,
        3.05185746e-02, -7.07044443e-01, -7.07088056e-01, -7.58179027e-02,
       -6.03738066e-02,  4.31514456e-01, -1.53019824e-01,  2.19410784e-02,
       -1.66572799e-01, -2.09558345e-01, -1.57835754e-01, -2.56789067e+00,
       -1.37890175e-01,  1.79548625e+00,  1.57403389e+00, -1.16635344e-05,
       -4.73344713e-06])
_max = array([ 0.35833499,  0.15268328,  0.28293931,  0.18886651,  0.15252286,
        0.28898519,  0.06433838,  0.16056216,  0.35455735,  0.99999976,
        0.79144941,  0.14983867,  0.39999238,  0.89648331,  0.89292554,
        0.09381397,  0.06986657,  0.78004359,  0.15301469,  0.40104617,
        0.2165153 ,  0.87685478,  0.14217607, -0.94338807,  0.08231745,
        2.86370173,  2.96710144,  0.07399999,  0.07400005])
max 2.9671014401357163 min -2.5678906650756663

Didn't work so reverting all other files (except this) to last push

---

Ran final experiments for CQL and PD agent
Made plotting code

Set up for SAC to run now as a comparison
Going to try using same initial replay buffer as CQL ie. loaded with PD agent data

First SAC run
Eval curves look sensible and behaviour isn't crazy so def think it's training
Couldn't see any completions on eval tho it seems to have some in training?

x Will need debug it a few times
x  and then do some hparam tweaks 
x   then run on all 5 seeds

---


Final experiments

Seeds = [117, 12321, 7456, 3426, 573]

* git add the variant.json and the progress.csv for       

x 1. Gathering PD agent
x 2. Running 5 random seeds training CQL models (cite "RL that matters" paper) for min 150 epochs
   Done: CQL: [117, 7456, 3426], running: [12321]
x 3. Train 5 random seeds onf SAC on same env
    x Done: seed [117, 12321, 7456, 3426, 573]
4. Evaluating all on all environments as well as PD
    x 1. Write a plot that does error bars across random seeds
    x 2. Run PD for 5 random seeds and report mean and std in 100 episodes
       x Wrote code for this
       * Run in all envs
    3. Eval all CQL trained models and report mean and std in 100 episodes
        x Pick best based on training logs
        x Run each in own seed in all envs
        Plot
5. Call and report / discuss final findings with Li, add to overleaf     


Generate tile images with 

`montage frame_{0..999..32}.png -geometry +0+0 -tile 10x tiled_image.jpg`

---

TODO 

* Perhaps worth giving SAC some hparam etc tuning
    or try without preloading buffer?
* Train in perturbed/force to comapre regrasping / robustness (new obs/acs bounds)
    * Set different acs and obs bounds for pandaForce env if gathering data
    * Compare on different benchmarks how robust? how does training carry over?
  
* Make env and data for only regrasp part ie. not like pandaForce
    * Train on std then on regrasp and see if splices into full regrasp env
  
If time 

* gather some human data to try? - in base env for now

---------

Keep an eye on:
    - Panda Gym allows intersection of gripper with tray
    - Check action bounds

________

