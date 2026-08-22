# agents/ — skills for third-party AI harnesses

Two self-contained skill files that make an AI coding assistant an expert
on fastgrab. They are written for **people and tools outside this repo**;
the maintainer's own Claude Code config lives in `.claude/` and is
separate.

| file | audience | use it when |
|---|---|---|
| [`fastgrab-user.md`](fastgrab-user.md) | developers **using** fastgrab in their own projects | installing, calling `Screenshot().capture()`, converting BGRA, recording with `fastgrab-record`, debugging install/display errors |
| [`fastgrab-dev.md`](fastgrab-dev.md) | **contributors** to this repository | changing backends, the C extension, the build, tests, CI; reviewing PRs |

Each file starts with a `name:` / `description:` YAML front-matter block
and is plain Markdown after that, so it drops into most skill / rules
systems unchanged.

## Loading

**Claude Code** — copy the file to a skill directory, either per project or
per user:

```bash
mkdir -p .claude/skills/fastgrab-user && cp agents/fastgrab-user.md .claude/skills/fastgrab-user/SKILL.md
# or globally:
mkdir -p ~/.claude/skills/fastgrab-user && cp agents/fastgrab-user.md ~/.claude/skills/fastgrab-user/SKILL.md
```

Then `/fastgrab-user` in a session, or let it auto-trigger from the
description. Same recipe for `fastgrab-dev`.

**Cursor / Windsurf / Continue / Aider** — add the file as a rule or
context file (e.g. `.cursor/rules/fastgrab-user.mdc`, or reference it from
`.cursorrules` / `.windsurfrules` / `.aider.conf.yml`'s `read:` list).

**Any other agent / API call** — paste the file body into the system
prompt, or fetch it raw:

```
https://raw.githubusercontent.com/mherkazandjian/fastgrab/main/agents/fastgrab-user.md
https://raw.githubusercontent.com/mherkazandjian/fastgrab/main/agents/fastgrab-dev.md
```

## Keeping them accurate

The files restate the public API, CLI flags, extras, test layout and CI
matrix. When any of those change in the repo, update the matching
section here in the same PR.
