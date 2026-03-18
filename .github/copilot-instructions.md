---
name: "Commit Review and Notion Automation"
description: "Use when: committing changes, reviewing pre-commit code, or requesting pre-push checks. Automatically performs code review, creates Notion summaries, suggests improvements, and asks for approval before pushing."
applyTo: " "
---

# Pre-Commit Automation Workflow

Whenever you mention **commit**, **push**, **review**, or **pre-commit** in the context of this AIUQ repository, follow this workflow:

## Automatic Workflow Trigger

### Phase 1: Pre-Commit Check
1. Run `git status --short` to detect modified files
2. For each unstaged file, generate `git --no-pager diff --` to analyze changes
3. For each staged file, generate `git --no-pager diff --staged --` for review
4. Run error/linting checks on touched files to catch syntax issues

### Phase 2: Code Review
- **High Priority**: Flag breaking changes, unsafe patterns, or logic regressions
- **Medium Priority**: Note style inconsistencies, performance concerns, missing tests
- **Info**: Document new features, refactorings, or dependency changes
- **Edge Cases**: Identify potential issues with restart/resume logic, empty states, race conditions

### Phase 3: Notion Update
- Identify the **current branch** (e.g., `features/amip-simulations`)
- Search Notion for a page matching the branch name or context (e.g., "AMIP Simulation")
- If found: create a sub-page titled `Commit Check - [DATE] ([BRANCH])`
- Document:
  - Files changed (count + paths)
  - Main changes summary
  - Pre-commit findings (risks, suggestions, info)
  - Suggested commit message
  - Follow-up items

### Phase 4: Suggestions & Approval
Before the user pushes:
1. Present a summary of findings
2. Suggest fixes or improvements with concrete locations (file + line range)
3. **Ask for explicit confirmation** before allowing Push:
   - "Ready to push? (yes/no)"
   - If fixes suggested: "Apply fixes? (yes/no)"
4. Only proceed with `git push` after user confirms

---

## Details

### Code Review Focus
- **Deterministic vs Non-deterministic code**: Check for hardcoded paths, missing env vars, or unsafe defaults
- **State management**: Verify checkpoint resume paths, cleanup logic, and edge cases
- **API changes**: Surface breaking changes to function signatures, config keys
- **External dependencies**: Flag new imports and check for version constraints

### Notion Structure
- **Parent**: Auto-detect from current branch (e.g., "AMIP Simulation" for `features/amip-simulations`)
- **Format**: Markdown with code blocks, bullet lists, and links to files/lines
- **Auto-fix notes**: Mark sections as "[DONE - automatic fix]" if fixes were already applied

### Git Commands Used
```bash
git status --short
git --no-pager diff --stat
git --no-pager diff [--staged] -- <file>
git --no-pager log -n 5 --oneline
git push origin $(current_branch)
```

### Skip Conditions
- If no files are staged or unstaged: report "No changes detected"
- If `.gitignore` is the only change: optionally skip Notion update (lightweight)
- If user explicitly says "skip review": proceed to push without workflow

---

## Example Interaction

**User**: "I'm ready to commit the AMIP changes"

**Agent** (automatically):
1. ✓ Detects 3 files modified (runscripts/sim_neuralgcm.py, templates/config.yml, .gitignore)
2. ✓ Analyzes diffs → finds edge case in checkpoint resume logic
3. ✓ Creates page "Commit Check - 2026-03-18 (features/amip-simulations)" in Notion under AMIP Simulation
4. ✓ Suggests fix with line numbers
5. 📋 Asks: "**Apply fix to checkpoint edge case?** (yes/no)"
6. 📋 Asks: "**Ready to stage & push?** (yes/no)"

---

## Behavior Notes

- Always operate within the AIUQ workspace repository
- Use Notion integration for all documentation (never suggest external tools)
- If Notion page not found: create a new one under "Commit Logs" or ask user where to put it
- Keep notes concise: focus on actionable items, not verbose prose
- Mark manual user decisions clearly (e.g., "User chose not to apply fix")
