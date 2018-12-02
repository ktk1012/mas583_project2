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
from reinforceflow.envs import Vectorize
from reinforceflow.core import EGreedyPolicy, ProportionalReplay, BackPropagationReplay
from reinforceflow.core import WindowedBackPropagationReplay
from reinforceflow.core import Adam
from reinforceflow.models import FullyConnected
from reinforceflow.trainers.replay_trainer import ReplayTrainer
reinforceflow.set_random_seed(555)


def moving_average_accumulator(accum, value):
    return accum * 0.7 + abs(value) * 0.3


env_name = 'CartPole-v0'
env = Vectorize(env_name)
policy = EGreedyPolicy(eps_start=1.0, eps_final=0.2, anneal_steps=300000)

agent = DeepQ(env,
              device='/cpu:0',
              model=FullyConnected(),
              optimizer=Adam(0.0001),
              targetfreq=10000,
              policy=EGreedyPolicy(1, 0.4, 300000))

backPropagationReplay = BackPropagationReplay(
    30000,
    32,
    0.,
    moving_average_accumulator,
    32,
    beta=20)

windowedPropagationReplay = WindowedBackPropagationReplay(
    30000,
    32,
    100,
    32)

trainer = ReplayTrainer(env=env,
                        agent=agent,
                        maxsteps=300000,
                        replay=windowedPropagationReplay,
                        logdir='tmp/rf/DeepQ/%s' % env_name,
                        logfreq=10)

if __name__ == '__main__':
    trainer.train()
