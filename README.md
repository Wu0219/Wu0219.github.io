# yuhangwu.com

Personal site of **Yuhang (Johann) Wu** — MSc student in Sound & Music Computing
at Universitat Pompeu Fabra, Barcelona.

Hand-built from scratch: custom design system, no theme, no template.
Served by GitHub Pages (native Jekyll pipeline).

## Writing a new post

Drop a Markdown file into `_posts/` named `YYYY-MM-DD-slug.md`:

```markdown
---
layout: post
title: "Post title"
subtitle: "Optional subtitle"
date: 2026-07-20
tags: [music, machine-learning]
math: true   # only if the post uses LaTeX
lang: zh     # only for Chinese posts (defaults to en)
---

Content here. `$inline$` and `$$display$$` math both work when math: true.
```

Push to `master` — GitHub Pages rebuilds and deploys automatically in ~1 minute.

## Structure

```
_config.yml        site settings
_layouts/          default (shell) + post
assets/css/        the whole design system, hand-written
index.html         single-page home: about / publications / experience / …
blog/              post archive
_posts/            markdown posts
```
