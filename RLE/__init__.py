from gymnasium.envs.registration import register

register(
    id="RLE-v0",
    entry_point="RLE.env:RLEEnv"
)
register(
    id="Bayes_RLE",
    entry_point="RLE.env.Bayes_RLEEnv:Bayes_RLEEnv"
)
register(
    id="RLEEnv_expectation",
    entry_point="RLE.env.RLEEnv_expectation:RLEEnv_expectation"
)
# "RLEEnv4AlphaZero"

# register(
#     id="RLEEnv4AlphaZero",
#     entry_point="RLE.wrapper:RLEEnv4AlphaZero"
# )