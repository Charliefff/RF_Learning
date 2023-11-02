from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

    config = {
        "gpu": True,
        "training_steps": 1e8,
        "update_sample_count": 40000,
        "discount_factor_gamma": 0.99,
        "discount_factor_lambda": 0.95,
        "clip_epsilon": 0.2,
        "max_gradient_norm": 0.5,
        "batch_size": 512,
        "logdir": 'log/Enduro/',
        "update_ppo_epoch": 5,
        "learning_rate": 2.5e-4,
        "value_coefficient": 0.5,
        "initial_entropy_coefficient": 0.05,
        "final_entropy_coefficient": 0.01,
        "entropy_coefficient": 0.01,
        "horizon": 128,
        "env_id": "ALE/Breakout-v5",
        "eval_interval": 100,
        "eval_episode": 1,
    }
    agent = AtariPPOAgent(config)
    agent.train()
