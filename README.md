# Goals

The main goal here is practice. Because of that I'm making choices that might not be optimal in terms
of producing a desired output in the least amount of time. Also, some of these choices/tech stacks are new to me
and my implementations will likely be suboptimal. I'm also using this as an opportunity to experiment
with some coding practices.


1. Implementation in JAX and FLAX (these are new to me).
2. The RL policy should mix discrete and continuous actions.
3. The RL algorithm should be implemented by me - don't just use someone else's code here.
4. Choosing to implement SACv2 - or at least start there. The reason being that this has been a fundamental alg in the
work I've been doing for the past 3 years, but I don't know it well because someone else has always implemented it for
me.
5. A goal here is practice, not code reuse, so I'm not trying to write libraries.

## Completed

1. Here I have implemented an asynchronous version of SACv2 using JAX and FLAX. There is a single trainer process, and
multiple child processes (rollout workers). The rollout workers receive models from the trainer and use
those to collect rollout trajectories in the target environment. These are written into the trainer's replay 
buffer using the DeepMind Reverb library.
2. Successful training on baseline continuous action space environments including Pendulum and MountainCar.
3. Developed a gym environment that has a mixed action space with both discrete and continuous actions: MixedAction2D.
4. Successful training on MixedAction2D by mapping the discrete space to a continuous one.

## Future Goals

1. Successful training on MixedAction2D using a policy that works in the mixed action space such that it is aware of
the relationship between the discrete and continuous action choices.
2. Use transformers. When I created this project it was with the intention to use this code for training a particular
environment that has 1) mixed action space and 2) variable observation space. I still hope to complete that project.

## Coding principles

1. I like types and try to make sure they're included in all functions definitions. They serve two purposes: 1. they
provide documentation and 2. they can be used by my IDE and tools like mypy to detect errors in the code. The degree to
which I've used types here is probably a bit excessive for such a small piece of code. However, when the code base is 
smaller it is easier to get all the types right and build up, rather than going back and adding typing to a large 
codebase.
2. I dislike poorly defined datastructures - they are prone to error, they can be difficult to work with and they often
require developers to manually document/track the structure. More specifically, where possible I prefer to define
dataclasses that encode semantic information rather than using dictionaries. Again, I've probably overdone it here, but
I wanted to figure out how I would apply it to Jax based implementations, which tend to be very dynamic.
3. Code as configuration. I have worked with other systems that use a secondary system of configuration, ex. YAML files.
I find that to be problematic: it can be difficult to handle the conversion, it can produce a lot of boiler plate code,
it's yet another language/syntax to work with, and YAML, at least, is not a programmatic language - it doesn't support
calculations, for loops, etc. (please don't suggest JSONNET here - oh, the pain :( ). So here my experiment files,
scripts in `scripts/experiments`, are all python files, they define any configuration directly in python code.
4. I prefer kwargs. While it takes time and adds a lot of text, I prefer to explicitly use kwargs. I think this has
several advantages: it prevents errors from having args in the wrong order, it makes it easier to refactor code, it
makes it clear to the reader what data is going into the function call.
5. Comments. I try to follow the principle of making the code itself as easy for another human (who isn't me) to follow.
However, in practice, I found that it is difficult to communicate all that is needed by only relying on this concept.
Thus, I also use comments. In particular, I will add comment blocks to the tops of files to explain... more here


## TODO:

1. I'd still like to take things further with applying semantic data containers pretty much everywhere. In particular,
anywhere that I've returned a Tuple (like in the various loss functions), I cringe inside a little bit.
2. I don't like how there is a single ExpConfig that gets passed everywhere.
3. More unit tests would be good, particularly around the training functions and anything that's doing pytree ops.
4. While I haven't included them in the `main` branch, I have started working on models that use mixed action spaces. 
Because of this there is already code in sac.py that handles various nested structures, which might seem unnecessary
at this point.

## Installing Jax from Poetry

```
$ poetry source add jax https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$ poetry add --source jax jaxlib
$ poetry add jax
```

## Results

### Pendulum

```bash
python scripts/experiments/gym_envs/pendulum.py train
```

![pendulum eval return](https://github.com/csherstan/triangles/blob/main/img/gym_envs/pendulum/eval_return.png)

### Mountain Car

```bash
python scripts/experiments/gym_envs/mountain_car.py train
```

![mountain car eval return](https://github.com/csherstan/triangles/blob/main/img/gym_envs/mountain_car/eval_return.png)

### MixedAction2D

This is a simple environment that has a mixed action space: discrete and continuous actions.
The world is a bound 2D x-y arena. On each reset the agent is randomly placed in the environment and a new target
location is selected. The agent's goal is to move into the target zone (area around the target location) and signal
that it is done.

The agent can only move in either x or y on each turn and must select which axis to move on and by how much.
Finally it can signal that is done. See [triangles/env/mixed_action.py] for more 
info.

```bash
python scripts/experiments/mixed_action/continuous_action_terminating_env.py train
```

![continuous_action_terminating_env eval return](https://github.com/csherstan/triangles/blob/main/img/mixed_action/continuous_action_terminating_env/eval_return.png)









