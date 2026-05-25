# Contributing to tatva

We welcome your help to improve `tatva`. This guide explains how our team works together. Please follow these rules so we do not waste effort.

## Propose and Discuss First

Do not write code right away. We must understand the physics, the logic and the "why" of the problem before we start typing.

* Go to the Issues tab and create a new Issue.
* Use the Feature Request template.
* Explain the problem clearly.
* Discuss the plan with the team.
* Wait for approval before you start coding.

## Create a Branch

Once the team approves your Issue, create a new branch for your work. Use a clear name so everyone knows what the branch does.

* For new features: `feat/short-name` (Example: `feat/dg-elements`)
* For bug fixes: `bugfix/short-name` (Example: `bugfix/boundary-condition-error`)

## Write Your Code

Write clean and correct code. Focus on how to solve the problem accurately.

* If you write a complex function, include a pseudo-algorithm in the comments so other researchers can understand your logic.
* Make sure your code compiles and runs successfully on Linux.
* Add tests to prove your solution works correctly.

### Guidelines for AI Tools

You can use AI tools to help you write code, but you must keep this assistance under your own judgment and responsibility. 

* You must fully understand any code you submit. We will not merge a "vibe-coded" Pull Request if you cannot explain the underlying mechanics of the code.
* Do not use autonomous AI agents to write code and submit Pull Requests. We strictly prohibit autonomous AI submissions in tatva.

## Open a Pull Request

When you finish your work, submit a Pull Request (PR) to merge your branch into the main code.

* Use the Pull Request template.
* Link your PR to the approved Issue.
* Explain what changes you made.

## Review

A team member will review your PR. They will check your logic, your numerical approach, and your code structure. You might need to make changes based on their feedback. 

## The "pull-ready" Tag and Internal Testing

Once the reviewer approves your code, a maintainer will add a `pull-ready` label to your PR. 

This label triggers our heavy internal testing suite. These tests do not run automatically when you open a PR. They check for complex issues like performance drops on GPUs/CPUs and scaling problems with large simulations. 

If these internal tests fail, we will inform you in the PR comments so you can investigate. 

## Merge

Once your code passes the review and the `pull-ready` internal tests, a maintainer will merge your branch into the main code.

## Branch Maintenance and Cleanup

We never merge development branches directly into main. All code must pass through the Issue and Pull Request process.

Do not base your research or build new features on top of another contributor's development branch. We regularly delete stale and unmaintained branches to keep the repository clean. If you need a specific feature for your work, wait for the maintainers to merge it into the main branch.
