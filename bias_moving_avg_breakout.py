'''
Moving averaged reward back propagation example code
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import reinforceflow
from reinforceflow.agents import DeepQ
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.models import DeepQModel
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.envs import AtariWrapper
from reinforceflow.trainers.replay_trainer import ReplayTrainer
from reinforceflow.core.replay import BackPropagationReplayOH

reinforceflow.set_random_seed(555)


def moving_average_accumulator(accum, value):
    return accum * 0.7 + abs(value) * 0.3


backPropagationReplay = BackPropagationReplayOH(
    200000,
    32,
    0.,
    moving_average_accumulator,
    32,
    accum_bias=0,
    alpha = 0.6,
    beta=10)

env_name = "Breakout-v0"
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   to_gray=True,
                   noop_action=[1, 0, 0, 0],
                   start_action=[0, 1, 0, 0],
                   clip_rewards=True)

test_env = AtariWrapper(env_name,
                        action_repeat=4,
                        obs_stack=4,
                        new_width=84,
                        new_height=84,
                        to_gray=True,
                        start_action=[0, 1, 0, 0])

agent = DeepQ(env=env,
              use_double=True,
              model=DeepQModel(nature_arch=True, dueling=False),
              optimizer=RMSProp(7e-4, decay=0.99, epsilon=0.1),
              policy=EGreedyPolicy(1.0, 0.1, 4000000),
              targetfreq=10000)

trainer = ReplayTrainer(env=env,
                       agent=agent,
                       maxsteps=80000000,
                       replay = backPropagationReplay,
                       logdir='tmp/%s/moving_average_bias' % env_name,
                       logfreq=900,
                       test_env = test_env,
                       test_render = False 
                       )
trainer.train()
