"""
Entry point: run one episode with random agents and write a JSON log.

Usage:
    python -m unknown_room.run
    python -m unknown_room.run --ticks 40 --seed 42 --log runs/ep0.json
    python -m unknown_room.run --ticks 40 --seed 42 --reward misspecified --log runs/ep_mis.json
"""
from __future__ import annotations
import argparse
import random as _random
from pathlib import Path

from unknown_room.environment import UnknownRoomEnv
from unknown_room.policies.random_agent import RandomAgent
from unknown_room import rewards as reward_module

REWARD_FNS = {
    "individual":   reward_module.reward_individual,
    "collective":   reward_module.reward_collective,
    "misspecified": reward_module.reward_misspecified,
    "mixed_05":     reward_module.reward_mixed(0.5),
}


def run_episode(
    ticks: int = 20,
    seed: int | None = None,
    reward: str = "individual",
    log_path: str | None = None,
    verbose: bool = True,
) -> dict:
    reward_fn = REWARD_FNS.get(reward, reward_module.reward_individual)

    env = UnknownRoomEnv(
        ticks_per_phase=ticks,
        reward_fn=reward_fn,
        seed=seed,
        log_path=log_path,
    )

    agent_rng = _random.Random(seed)
    agents = {
        aid: RandomAgent(agent_id=aid, seed=agent_rng.randint(0, 2**32))
        for aid in env.possible_agents
    }

    observations = env.reset()

    while not env.done:
        actions = {}
        for aid in env.agents:
            obs = observations[aid]
            valid = env.valid_actions(aid)
            actions[aid] = agents[aid].act(obs, valid)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if verbose:
            print(
                f"Tick {env.tick - 1:3d} | "
                f"living={len(env.agents):2d} | "
                f"welfare={env.collective_welfare:.3f} | "
                f"pools={len(env.pools)}"
            )

    if log_path:
        env.logger.flush()
        print(f"\nLog written to {log_path}")

    summary = {
        "ticks_completed": env.tick,
        "final_living": len(env.agents),
        "final_collective_welfare": env.collective_welfare,
        "total_pools_created": env._next_pool_id[0],
    }
    if verbose:
        print("\n--- Episode summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run an Unknown Room episode.")
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reward", default="individual",
                        choices=list(REWARD_FNS), help="Reward function.")
    parser.add_argument("--log", type=str, default=None, dest="log_path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_episode(
        ticks=args.ticks,
        seed=args.seed,
        reward=args.reward,
        log_path=args.log_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
