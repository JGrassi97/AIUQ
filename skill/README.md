# AIUQ agent skill

`aiuq/` is a Claude Code skill that lets an agent on your laptop operate AIUQ: list and follow
experiments, create and launch new ones through the dashboard Launcher, debug failed jobs, stop runs,
and analyse or download outputs. Everything goes over key-based ssh to the Autosubmit host (`mafalda`)
and the HPC (`felipe`).

```
aiuq/
├── SKILL.md                         entry point: topology, rules, workflows
├── scripts/
│   ├── aiuq                         CLI used by the agent (bash 3.2 compatible)
│   ├── dash_client.py               runs on mafalda, talks to the dashboard socket
│   └── analysis/ensemble_box.py     ensemble mean/spread over a box or point, runs on felipe
└── references/                      config, raw Autosubmit commands, outputs, troubleshooting
```

## Install

```bash
mkdir -p ~/.claude/skills
ln -s ~/Documents/AIUQ/skill/aiuq ~/.claude/skills/aiuq      # all sessions
# or, only inside this repo:
mkdir -p .claude/skills && ln -s ../../skill/aiuq .claude/skills/aiuq
```

Requirements: ssh aliases `mafalda` and `felipe` with key-based login (`~/.ssh/config`), the AIUQ
dashboard running on mafalda (`tools/dashboard`). Other hosts or users: set `AIUQ_AS_HOST`,
`AIUQ_HPC_HOST`, `AIUQ_HPC_USER`, `AIUQ_HPC_BASE`.

The CLI can also be used by hand: `skill/aiuq/scripts/aiuq help`.

## Safety

Read-only commands run freely. `create`, `run`, `stop` and `as` refuse to run without `--yes`; the skill
tells the agent to pass it only after you confirmed that specific action. Deletion is left to the
dashboard's Delete button (`aiuq delete-plan` only shows what would be removed).
