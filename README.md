https://github.com/google/brain-tokyo-workshop/tree/master/es-clip

## Goals

The main goal here is practice, because of that I'm making choices that might not be optimal in terms
of producing a desired output in the least amount of time. Also, some of these choices/tech stacks are new to me
and my implementations will likely be suboptimal. I'm also using this as an opportunity to experiment
with some coding practices.

1. Implementation in JAX and FLAX.
2. Model must use a Transformer to handle variable number of triangles.
3. RL policy should mix discrete and continuous actions.
4. RL algorithm should be implemented by me. Too often I've simply relied on using someone elses implementation.
5. Choosing to implement SAC - or at least start there. The reason being that this has been a fundamental alg in the
work I've been doing for the past 3 years, but I don't know it well because someone else has always implemented it for
me.
6. Avoid writing libraries or code that gets reused: write it again. A goal here is practice, not code reuse.


## Installing Jax from Poetry

```
$ poetry source add jax https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$ poetry add --source jax jaxlib
$ poetry add jax
```
