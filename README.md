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

However, this library capitalizes on a fruitful synergy between autocurricula and parameter-efficient fine-tuning methods that has previously been underexploited. By implementing individuals as mere parameter-efficient _adaptations_ of a single foundation model (e.g. LoRA weight diffs), the resulting population initially scales sublinearly in memory footprint with respect to the number of individuals. This trick runs deep through the library, which ultimately relies on 🤗 `peft` for parameter-efficiency.

### How is this different from simply prompting a foundation model to assume different persona?

Statically prompting a backbone model to [assume different personas](https://microsoft.github.io/autogen/) means there's no actual optimization going on over time. It just happens to work in a certain way, and happens to yield a certain performance. In contrast, the whole point of the library is to _actually optimize_ these policies, rather than simply work with frozen models and circle back to the suboptimal handcrafted features of old.

That said, the two can be used together. For instance, you could have an autocurriculum that operates at the level of "teams" of personas that work together. In addition, some parameter-efficient fine-tuning methods directly optimize the prompt embeddings. If employing such methods as the learning substrate in `autocurricula`, one might imagine these to naturally yield the equivalent of: "YOU ARE THE GREATEST CHESS GRANDMASTER IN HISTORY AND THE WHOLE WORLD DEPENDS ON YOUR NEXT MOVE...". However, methods that actually tweak the model's parameters (e.g. LoRA) might be more expressive in directing the general cognition of the backbone model.

### How is this compatible with the 🤗 ecosystem?

First, `autocurricula` enables users to further train (and publish) 🤗 `transformers` models, accelerate multi-agent training schemes using 🤗 `accelerate`, optimize results for deployment using 🤗 `optimum`, etc. In addition, critical aspects of `autocurricula` can be configured using custom `peft.PeftConfig` and `trl.PPOConfig` objects, and the codebase follows similar patterns to the rest of the ecosystem.

Second, `autocurricula` itself builds on: 🤗 `transformers` for models, 🤗 `trl` for RL training, 🤗 `peft` for parameter-efficiency, 🤗 `datasets` for throughput, etc. In the future, it might also rely on 🤗 `diffusers` for multi-modal autocurricula.

## Usage

Using `autocurricula` beyond toy examples typically involves working with three functions:

1. **`play`** _Who wins?_ This function maps players, the backbone model, and its tokenizer to a batch of rewards. `play` essentially abstracts away from the specifics of the multi-player game at hand, allowing users to bring in their own (chess, Go, debate, prove/verify theorem, generate/discriminate story, [MakeMeSay](https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say), etc.).

2. **`match`** _Who plays who?_ Uses the player pool to schedule multi-player matchups. The `match` function, together with the `entry` function below, abstract away from the specifics of different autocurricula.

3. **`entry`** _Who is there to play?_ Yields new players based on the current generation. The only thing that separates `SelfPlayTrainer` and `LeagueTrainer` are different `match` and `entry` methods. In fact, both inherit from the abstract `AutocurriculumTrainer` and simply override these.

### Using a custom `play` method

The aim of this toy game is to produce a string that's lexicographically larger than the one produced by the opponent. Fun times.

```python
from autocurricula import SelfPlayConfig, SelfPlayTrainer
from autocurricula.games.utils import set_player


def play(players, model, tokenizer):
    prompt = " "
    prompt_ids = tokenizer(prompt, return_tensors="pt")

    # Have each player produce a string.
    # This game is symmetric, we treat players identically.
    strs = []
    for player in players:
        # Use `set_player` to activate the right player adapter.
        set_player(model, player)
        # Ideally you'd run multiple games in parallel by batch generation.
        str_ids = model.generate(**prompt_ids, do_sample=True, max_new_tokens=3)[0]
        strs += [tokenizer.decode(str_ids)]

    # The rest of this function is just basic data manipulation.
    # Package experiences as subsequent training data.
    experiences = [
        {
            # CoT-friendly. Learn to think games through!
            "thoughts": [{"context": prompt, "behavior": str}],
            "action": str,
        }
        for str in strs
    ]

    rewards = (strs[0] > strs[1], strs[0] < strs[1])

    return [rewards], [experiences]


sp_config = SelfPlayConfig()
sp_trainer = SelfPlayTrainer(sp_config)
sp_trainer.train("facebook/opt-125m", play)

# Access the final list of players.
print(sp_trainer.players)

# Access the final ELO leaderboard.
print(sp_trainer.leaderboard)
```

### Using custom `match` and `entry` methods

Across the `play`, `match`, and `entry` functions, players are simply represented as custom dicts, with `autocurricula` handling the model adapters under the hood. For instance, the `lt_trainer.players` pool might contain the following entry:

```JSON
{
    "role": "league_exploiter",
    "gen": 3,
}
```

These player "specs" can contain any kinds of fields to be employed by custom methods. The only requirement is for the `"gen"` field to be populated accordingly during `entry`. This helps identify the latest generation of players so as to only spend compute on training these.

The sketch below implements an autocurriculum with the following logic. Each generation, we get a new generator and a new discriminator. The latest generator (discriminator) then goes to play against all past discriminators (generators). The autocurriculum could then be used in tandem with custom `play` functions to train models to generate (and inspect) solutions to math problems, solutions to coding puzzles, engaging stories, etc.

```python
from autocurricula import AutocurriculumConfig, AutocurriculumTrainer


class FictitiousGANConfig(AutocurriculumConfig):
    def __init__(self, generations: int = 4, rounds: int = 2):
        super().__init__(generations, rounds)


class FictitiousGANTrainer(AutocurriculumTrainer):
    def __init__(self, ac_config):
        assert isinstance(ac_config, FictitiousGANConfig)
        super().__init__(ac_config)

    def entry(self):
        # We get one new G and one new D each generation.
        return [
            {"role": "generator", "gen": self.current_gen},
            {"role": "discriminator", "gen": self.current_gen},
        ]

    def match(self):
        gs = [e for e in self.players if e["role"] == "generator"]
        latest_g = sorted(gs, key=lambda x: x["gen"])[-1]

        ds = [e for e in self.players if e["role"] == "discriminator"]
        latest_d = sorted(ds, key=lambda x: x["gen"])[-1]

        # Every round, latest players play all compatible past players.
        return [(latest_g, d) for d in ds] + [(latest_d, g) for g in gs]

```

If you implement a new such autocurriculum, make sure to contribute it back to the library for others to use with their own games.
