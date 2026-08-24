**Status:** Draft · **Domain:** `tiferet-dl` · **Code:** (not yet implemented) · **Branch:** `docs-core-domain-vision-and-distillation`

# tiferet-dl: Domain Vision Statement

## The bet
Most people learn deep learning the way a semester forces them to: a notebook per assignment, a model that runs once for a grade, and then nothing. The notebook gets zipped up and forgotten the day after it's submitted. The bet behind `tiferet-dl` is the opposite: every technique worth learning this way is worth re-deriving and re-expressing in a form built to last — structured, named, explained, and organized the same disciplined way the rest of the Tiferet ecosystem is built. Homework proves you can make a model run once. This collection proves you understood the principle well enough to explain it, structure it, and reuse it.

## What this domain makes real
`tiferet-dl` is a durable, growing reference collection of deep learning techniques, built alongside a graduate course (ECE 425/525, *Introduction to Deep Learning: An Engineering Perspective*) but not bound to its grading calendar. It takes each technique the course covers — from the basics of a single artificial neuron up through the attention mechanism that powers today's large language models — and turns it into something that survives the semester: a written explanation of the principle, traced back to a real source, organized so that someone who has never taken the course could still learn from it.

## What we get for it
**A vetted map of what actually matters.** Deep learning research produces new techniques faster than anyone can evaluate them. A course taught by someone whose job is to separate durable engineering principles from this week's trend is a filter worth keeping. `tiferet-dl` preserves that filter instead of letting it evaporate at the end of the term.

**A worked-example bank for the course itself.** The course requires a team project and four graded assignments that mix written and programming work. A running collection of clearly explained techniques is something to build on for those deliverables, instead of starting from a blank notebook each time.

**An early inventory of patterns worth stealing.** Deep learning systems solve a problem the Tiferet framework also cares about: how do you keep a pipeline's fixed structure separate from the pieces that change per use case? Techniques like configurable training loops, swappable efficiency optimizations, and strict train/evaluation separation are examples of that problem solved in a different domain. Collecting them here, honestly, is how we find out which ones are worth adapting later — without committing to adapt any of them yet.

## The core of the work
The collection follows the same four-part shape as the course it's drawn from:

1. **Fundamentals and implementation** — the mechanics every later technique depends on: how a model turns input into a prediction, how a loss function scores that prediction, and how backpropagation turns that score into a correction.
2. **Modern architectures and efficient techniques** — the two families of model structure in current use (convolutional networks and, especially, attention-based Transformers), plus the techniques that make training and running them affordable.
3. **Applications and computational performance** — what these techniques are for, and what they cost to run.
4. **Ethics and responsible AI** — the considerations that decide whether a technique should be used at all, not just whether it works.

The central commitment that holds this together: nothing gets written into this collection because it sounds important. Every technique captured here must trace to a specific, citable source — a textbook chapter, a named reference, or an assigned reading — before it counts as "worth learning." If a topic in the course doesn't have that kind of source yet, that gap gets written down honestly instead of papered over.

## What it deliberately does not do
`tiferet-dl` is not a production machine learning framework. It does not train models at scale, serve them, or manage infrastructure — if the Tiferet ecosystem ever wants that, it is a separate, much larger undertaking owned by its own future proposal, not by this collection.

It is not a restatement of the entire deep learning literature. It only captures techniques the course actually assigns or explicitly covers; broader research surveys are out of scope.

It is not a research-tracking feed for whatever is newest. A technique earns a place here by being taught with a citable source, not by being trending.

It is not a source of graded answers. Coursework — quizzes, assignments, and exams — stays the student's own independent work, consistent with the course's academic integrity and AI-use policies; this collection captures durable principles, not solutions to graded problems.

*Companion document:* `docs/core-domain-distillation.md` — the detailed technical walkthrough of the domain's vocabulary, behaviors (one per course unit), and the relationships between its parts.
