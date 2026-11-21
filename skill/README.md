# AIUQ agent skill

`aiuq/` is a Claude Code skill that lets an agent on your laptop operate AIUQ: list and follow
experiments, create and launch new ones through the dashboard Launcher, debug failed jobs, stop runs,
and analyse or download outputs. Everything goes over key-based ssh to the Autosubmit host and the HPC of
your installation; the skill itself names no machine.

```
aiuq/
├── SKILL.md                         entry point: topology, rules, workflows
├── config.example                   settings of your installation  → ~/.aiuq/config
├── site.example.md                  what the agent must know about your machines → ~/.aiuq/site.md
├── scripts/
│   ├── aiuq                         CLI used by the agent (bash 3.2 compatible)
│   ├── dash_client.py               runs on the Autosubmit host, talks to the dashboard socket
│   └── analysis/ensemble_box.py     ensemble mean/spread over a box or point, runs on the HPC
└── references/                      config, raw Autosubmit commands, outputs, troubleshooting
```

## Install

```bash
mkdir -p ~/.claude/skills
ln -s /path/to/AIUQ/skill/aiuq ~/.claude/skills/aiuq         # all sessions
# or, only inside this repo:
mkdir -p .claude/skills && ln -s ../../skill/aiuq .claude/skills/aiuq
```

## Configure

```bash
mkdir -p ~/.aiuq
cp skill/aiuq/config.example  ~/.aiuq/config      # the two ssh aliases and the experiments folder
cp skill/aiuq/site.example.md ~/.aiuq/site.md     # optional: limits and habits of your machines
skill/aiuq/scripts/aiuq site                             # shows what is in use
skill/aiuq/scripts/aiuq health                           # ssh to both hosts, dashboard freshness
```

Requirements: key-based ssh aliases for the Autosubmit host and the HPC (`~/.ssh/config`) and the AIUQ
dashboard running on the Autosubmit host (`tools/dashboard`). `aiuq` has no built-in host: without the
config it stops and says which setting is missing. The site notes are free text for the agent (number of
GPUs, queue limits, ssh rate limits, where the stores are); the skill tells it to read them first.

The CLI can also be used by hand: `skill/aiuq/scripts/aiuq help`.

## Safety

Read-only commands run freely. `create`, `run`, `stop` and `as` refuse to run without `--yes`; the skill
tells the agent to pass it only after you confirmed that specific action. Deletion is left to the
dashboard's Delete button (`aiuq delete-plan` only shows what would be removed).
