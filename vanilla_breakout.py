'''
Vanilla Double-DQN 
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
from reinforceflow.core.replay import ExperienceReplay

reinforceflow.set_random_seed(555)

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
              model=DeepQModel(nature_arch=True, dueling=False),
              use_double = True,
              restore_from = None,
              optimizer = RMSProp(7e-4, decay=0.99, epsilon=0.1),
              policy = EGreedyPolicy(1.0, 0.1, 4000000),
              targetfreq = 10000,
              )

trainer = ReplayTrainer(env=env,
                       agent=agent,
                       maxsteps=80000000,
                       replay = ExperienceReplay(400000,32,32),
                       logdir='tmp/%s/vanilla' % env_name,
                       logfreq=1800,
                       render = False,
                       test_env = test_env,
                       test_render = False,
                       )
trainer.train()
