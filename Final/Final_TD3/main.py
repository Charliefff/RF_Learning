from td3_agent_CarRacing import CarRacingTD3Agent
import sys

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    remote = sys.argv[1] if len(sys.argv) > 1 else False

    config = {
        # "map_name": "circle_cw_competition_collisionStop",
        'map_name': 'austria_competition_collisionStop',
        "render_mode": "rgb_array_birds_eye",
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.8,
        "tau": 0.005,
        "batch_size": 128,
        "warmup_steps": 1000,
        "total_episode": 100000,
        "lra": 1e-5,
        "lrc": 1e-5,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/CarRacing/TD3_/',
        # "logdir": './log/TD3_austria_clip/',
        "update_freq": 2,
        "eval_interval": 10,
        "eval_episode": 1,
        "remote": remote,
        "obs_space": (128, 128, 3),
        "action_space": (2,)

    }

    if config["remote"]:
        config["logdir"] = 'log/CarRacing/evaluation/'
        agent = CarRacingTD3Agent(config)

    else:
        agent = CarRacingTD3Agent(config)
        agent.load('/data/zhengyutong/raw_Reinforce_Learning/Final/final_project_env/log/austria_competition_collisionStop/model_397089_0.pth')
        agent.train()
