# tiferet-dl

**tiferet-dl** is a fact-finding companion to [Tiferet](https://github.com/greatstrength/tiferet):
a Domain-Driven Design framework whose Abstract Core
(`DomainObject`, `DomainEvent`, `Aggregate`, `Service`) this collection does
not yet touch. Instead, this repository asks a narrower question first: which
deep learning principles, taught with a real citable source, are durable
enough to be worth re-expressing as a Tiferet-family capability later? It is
built alongside a graduate course, ECE 425/525 *Introduction to Deep
Learning: An Engineering Perspective* (Fall 2026, Dr. Jyotikrishna Dass,
University of Arizona), but is not bound to its grading calendar.

The domain is not implemented yet. What exists today is the study surface:
the domain vision statement and the core domain distillation. Any pattern
this collection eventually proposes for the official Tiferet ecosystem will
be specified as a **Request for Prototype (RFP)** — a public issue that
states one domain theory, its motivation, and the acceptance criteria for
testing that theory on a prototype branch — authored with
`tiferet-author-rfp` once a technique note actually grounds it. None have
been written yet; §10 of the distillation is where the candidates are
tracked.

## The claim

Most people learn deep learning the way a semester forces them to: a
notebook per assignment, a model that runs once for a grade, then forgotten.
`tiferet-dl` bets that every technique worth learning this way is worth
re-deriving into something that survives the semester — structured, named,
traced to a real source, and organized the same disciplined way the rest of
the Tiferet ecosystem is built. Homework proves a model can run once. This
collection proves the principle behind it was understood well enough to
explain, structure, and reuse.

## Documents

- [`docs/domain-vision.md`](docs/domain-vision.md) — value proposition (the
  bet, the benefits, the non-goals)
- [`docs/core-domain-distillation.md`](docs/core-domain-distillation.md) —
  ubiquitous language, the confirmed-spine behaviors, the agnostic/variable
  seam, and the candidate Tiferet-ecosystem principles

Read the vision first. The distillation depends on the framing it
establishes.

## The core of the work

The collection follows the same four-part shape as the course it's drawn
from:

1. **Fundamentals and implementation** — how a model turns input into a
   prediction, how a loss function scores that prediction, and how
   backpropagation turns that score into a correction.
2. **Modern architectures and efficient techniques** — convolutional
   networks and attention-based Transformers, plus the techniques that make
   training and running them affordable.
3. **Applications and computational performance** — what these techniques
   are for, and what they cost to run.
4. **Ethics and responsible AI** — whether a technique should be used at
   all, not just whether it works.

## v1 confirmed spine

The distillation gives full behavior treatment only to course units with a
fixed, citable *Dive into Deep Learning* (d2l.ai) chapter. Everything else
is tracked as a named backlog, not drafted, until the course supplies a
citable source (see [§10a](docs/core-domain-distillation.md#10a-v1-backlog--promote-only-once-citable)).

1. Preliminaries and foundations (d2l Ch. 1–2)
2. Linear models, loss functions, and softmax (d2l Ch. 3–4)
3. Multilayer perceptrons and backpropagation (d2l Ch. 5)
4. Classic and modern convolutional networks (d2l Ch. 7–8)
5. Attention, Transformers, and large-scale pretraining (d2l Ch. 11, in full)
6. Optimization algorithms (d2l Ch. 12)

## Status

| Piece | State |
| --- | --- |
| Study documents | Draft on `master` |
| RFPs | None authored yet — candidates tracked in distillation §10 |
| Technique notes | Not started |
| Tiferet-ecosystem proposals | Not started |

## Setup (when technique notes start landing)

This collection is documentation-only today — there is no package to
install. Once the first technique note needs runnable code alongside it:

```bash
git clone https://github.com/Ashatz/tiferet-dl.git
cd tiferet-dl
python -m venv .venv
source .venv/bin/activate
```

## Sources this study answers to

- ECE 425/525 *Introduction to Deep Learning: An Engineering Perspective*
  syllabus (Fall 2026, Dr. Jyotikrishna Dass, University of Arizona)
- Zhang, A., Lipton, Z. C., Li, M., and Smola, A. J. *Dive into Deep
  Learning*. Cambridge University Press, 2023. https://d2l.ai
- Bishop, C. M., and Bishop, H. *Deep Learning: Foundations and Concepts*.
  2024.
- Prince, S. J. D. *Understanding Deep Learning*. 2023.

## License

[MIT](LICENSE) © 2026 Andrew
