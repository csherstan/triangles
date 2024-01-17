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