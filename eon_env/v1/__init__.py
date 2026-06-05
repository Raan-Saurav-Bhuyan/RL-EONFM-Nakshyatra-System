from gymnasium.envs.registration import register

register(
    id = 'EON-v0',
    entry_point = 'eon_env.v1.environment:EONEnv',
)
