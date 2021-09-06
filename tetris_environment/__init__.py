from gym.envs.registration import registry, register, make, spec
# Pygame
# ----------------------------------------
for game in ['Tetris']:
    nondeterministic = False
    register(
        id='{}-v0'.format(game),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={},
        nondeterministic=nondeterministic,
    )

