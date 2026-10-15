# Collaborate on Neural Net Lab

Neural Net Lab serves one problem family: opaque machine
intelligence. This page lists concrete, scoped ways an organization
can work with the project. Every task here is real and completable.
Nothing on this page implies an existing partnership; collaborations
are invitations until they actually happen, and no partner is named
publicly without their written agreement.

## Who this is for

AI labs, universities, and training providers.

## Sized tasks

1. **Extend the benchmark suite.** Pick one comparison that would
   teach something the current NumPy vs PyTorch table does not (a
   second model size, a memory measurement, a different optimizer)
   and run it under the fairness protocol documented in
   `scripts/benchmark_pytorch.py`. Results are published in
   `results/`, including the ones where the lab looks bad.
2. **Contribute an architecture with its gradient tests.** Add one
   component the lab lacks (an optimizer variant, an attention
   variant, an autograd-unified block) with finite difference
   gradient checks in the existing style. The bar is the test, not
   the novelty.
3. **Use the lab as an open teaching reference.** Teach ML
   fundamentals from it in a course or training program and publish
   your errata: where the explanations confused students, where the
   numbers surprised you. Teaching reports are the contribution this
   project values most, because its north star is students who can
   explain how the machine learns.

## What you get

A public, dated record in this repository: the benchmark artifact,
the merged component with its tests, or the teaching report, with
your organization's role stated as you approve it.

## Start

Open an issue in this repository describing the task you have in
mind. Small and specific beats grand and vague.
