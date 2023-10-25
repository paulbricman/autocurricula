<div align="center">
    <h1 align="center">autocurricula</h1>
    <p>scalable population-based primitives for <a href="https://huggingface.co">the 🤗 ecosystem</a></p>
</div>

## Conceptual Guide

Human data has proven sufficient to <a href="https://www.noemamag.com/artificial-general-intelligence-is-already-here/">yield artificial general intelligence</a>. Future developments might increasingly rely on the design of multi-agent ecologies that enable models to further co-evolve. Such regimes involve entire populations of models that define each other's niche. Whenever individuals adapt to fit the demands of their optimization landscape (e.g. using a strong chess opening), they implicitly force others to react with novel adaptations of their own (e.g. an appropriate response to said opening), leading to a self-induced curriculum, an _autocurriculum_.

> _The solution of one social task often begets new social tasks, continually generating novel challenges, and thereby promoting innovation. Under certain conditions these challenges may become increasingly complex over time, demanding that agents accumulate ever more innovations._ ([Leibo et al., 2019](https://www.deepmind.com/publications/autocurricula-and-the-emergence-of-innovation-from-social-interaction-a-manifesto-for-multi-agent-intelligence-research))

### What's an example of an autocurriculum?

[AlphaZero](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go) has been trained to master several games using self-play. Due to playing against itself, each new strategic innovation it develops gets thrown back at it through a mirrored opponent. This forces it to continuously respond to its own innovations by producing novel innovations.

Training a model using self-play is pretty straightforward:

```python
from autocurricula import SelfPlayConfig, SelfPlayTrainer
from autocurricula.games.tictactoe import play

sp_config = SelfPlayConfig()
sp_trainer = SelfPlayTrainer(sp_config)
sp_trainer.train("facebook/opt-125m", play)
```

### What's a more advanced example of an autocurriculum?

Naive self-play can end up chasing cycles. Imagine an initial policy **_A_** that forces a counterpolicy **_B_** to emerge, the latter of which then forces a **_C_** policy. However, it might be the case that **_C_** is actually susceptible to **_A_**, incentivizing it again, and so chasing a cycle.

League training has been designed to alleviate this issue, yielding a [grandmaster StarCraft II bot](https://www.deepmind.com/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning). The setup involves agents playing against themselves (i.e. self-play), playing against their past selves (i.e. fictitious self-play), playing against tailored opponents (i.e. main exploiters), and playing against opponents targeting the systematic vulnerabilities of the whole population (i.e. league exploiters). Together, these interactions yield a never-ending stream of challenges for individuals to surpass without chasing cycles.

Training a model using league training is similarly straightforward:

```python
from autocurricula import LeagueConfig, LeagueTrainer
from autocurricula.games.tictactoe import play

lt_config = LeagueConfig()
lt_trainer = LeagueTrainer(lt_config)
lt_trainer.train("facebook/opt-125m", play)
```

### How could you possibly train more than a handful of models at once?

As seen in the above examples, `autocurricula` is fundamentally meant to help train multiple individuals at once. If one was to have each individual be a whole LLM that barely fits in memory on its own, working with an entire population of models might sound like an engineering nightmare.

However, this library capitalizes on a fruitful synergy between autocurricula and parameter-efficient fine-tuning methods that has previously been underexploited. By implementing individuals as mere parameter-efficient _adaptations_ of a single foundation model (e.g. LoRA weight diffs), the resulting population initially scales sublinearly in memory footprint with respect to the number of individuals. This optimization trick runs deep through the library, which ultimately relies on 🤗 `peft` for parameter-efficiency.

### How is this different from simply prompting a foundation model to assume different persona?

Statically prompting a backbone model to [assume different personas](https://microsoft.github.io/autogen/) means there's no actual optimization going on over time. It just happens to work in a certain way, and happens to yield a certain performance. In contrast, the whole point of the library is to _actually optimize_ these policies, rather than simply work with frozen models and circle back to the suboptimal handcrafted features of old.

That said, the two can be used together. For instance, you could have an autocurriculum that operates at the level of "teams" of personas that work together. In addition, some parameter-efficient fine-tuning methods directly optimize the prompt embeddings. If employing such methods as the learning substrate in autocurricula, one might imagine these to naturally yield the equivalent of: "YOU ARE THE GREATEST CHESS GRANDMASTER IN HISTORY AND THE WHOLE WORLD DEPENDS ON YOUR NEXT MOVE...". However, methods that actually tweak the model's parameters (e.g. LoRA) might be more expressive in directing the general cognition of the backbone model.

### How is this compatible with the 🤗 ecosystem?

First, `autocurricula` enables users to further train (and publish) 🤗 `transformers` models, accelerate multi-agent training schemes using 🤗 `accelerate`, optimize results for deployment using 🤗 `optimum`, etc. In addition, critical aspects of `autocurricula` can be configured using custom `peft.PeftConfig` and `trl.PPOConfig` objects, and the codebase follows similar patterns to the rest of the ecosystem.

Second, `autocurricula` itself builds on: 🤗 `transformers` for models, 🤗 `trl` for RL training, 🤗 `peft` for parameter-efficiency, 🤗 `datasets` for throughput, etc. In the future, it might also rely on 🤗 `diffusers` for multi-modal autocurricula.

## Usage

### Using a custom `play` method

```python
# The aim of this game is to produce a number that's larger than the one issued by the opponent.
```

### Implementing a custom trainer

```python
# GANTrainer
```
