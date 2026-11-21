# Site notes

Copy to `~/.aiuq/site.md` and describe your installation. `aiuq site` prints this file, and the
skill tells the agent to read it before acting: write here what the agent must know about *your* machines
and cannot find in the repository. Keep it short and factual; no credentials.

## Hosts
- Autosubmit host: `<alias>` — <what else runs there, limits (e.g. ssh rate limits), services>.
- HPC: `<alias>` — <scheduler, number and type of GPUs, who shares them, typical queue waits>.
- Name of the HPC as seen from the Autosubmit host, if different from the laptop alias.

## Folders
- Support folder on the HPC (checkpoints, images, static files): `<path>`.
- Output stores on the Autosubmit host and which experiments go where: `<path>`.
- Is the store readable from the HPC at the same path?

## Scheduler
- Partitions and their limits; where CPU analysis jobs should go.
- Memory rules (default per CPU, hard limits), `sacct` available or not.

## Rules of this site
- <e.g. no persistent services on the login node; heavy analysis only through Slurm>.

## Known quirks
- <anything that surprised you once and would surprise the agent again>.
