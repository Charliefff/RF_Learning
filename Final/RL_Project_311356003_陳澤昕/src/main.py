from car_racing_agent import CarRacingTD3Agent
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "warmup_steps": 500,
        "total_episode": 1000000,
        "lra": 2.5e-5,
        "lrc": 2.5e-5,
        "replay_buffer_capacity": 10000,
        # "logdir": 'log/CarRacing/circle_v2/',
        "logdir": 'log/CarRacing/austria_LSTM/',
        # "logdir": 'log/CarRacing/debug',
        "update_freq": 2,
        "eval_interval": 5,
        "eval_episode": 1,
        "server_url": 'http://127.0.0.1:1223',
        "observation_space_shape": (128, 128, 3),
        "action_space_shape": (2,),
        # "scenario": 'circle_cw_competition_collisionStop',
        "scenario": 'austria_competition_collisionStop',
        "env_remote": False,
    }

    if config['env_remote']:
        config['logdir'] = 'log/CarRacing/remote/'
        agent = CarRacingTD3Agent(config)
        agent.load_and_evaluate(
            '/data/zhengyutong/raw_Reinforce_Learning/Final/final_project_env/log/austria_competition_collisionStop/model_397089_0.pth')

    else:
        agent = CarRacingTD3Agent(config)
        agent.load('/data/zhengyutong/raw_Reinforce_Learning/Final/final_project_env/log/austria_competition_collisionStop/model_397089_0.pth')
        agent.train()
