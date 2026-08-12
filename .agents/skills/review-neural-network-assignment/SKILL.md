---
name: review-neural-network-assignment
description: Review a neural-network course assignment for correctness and conformance by comparing its problem specification with its Python implementation, Typst report, data, figures, and observed results. Use when asked to check, audit, grade, or improve an assignment folder containing files such as specification.md, main.py, and report.typ; identify missing or incorrect requirements; validate reported experiments; or suggest code and report quality improvements.
---

# Review Neural Network Assignment

Perform an evidence-based, read-only review. Treat the assignment's own specification as the source of truth. Do not edit files unless the user explicitly asks for fixes.

## Select the assignment

1. Use the folder named by the user.
2. If several candidates exist and the intended one is not clear from context, ask the user to choose.
3. Read repository instructions such as `AGENTS.md`, then inspect the complete assignment folder. Include supporting data, images, dependency files, and generated artifacts when relevant.

## Build the rubric

Read `specification.md` completely before judging the implementation or report. Convert every testable statement into an atomic checklist, including:

- required model, algorithm, equations, labels, inputs, and outputs;
- dataset construction, minimum counts, noise, splits, and evaluation rules;
- required visualizations, interactions, files, archive layout, and naming;
- required report topics, results, length, and format;
- constraints implied by examples only when the specification presents them as mandatory.

Keep explicit requirements separate from reasonable quality expectations. Do not penalize an optional design choice as nonconformance.

## Inspect the implementation

Trace each applicable requirement to exact code or another artifact. Check both presence and behavior.

For neural-network code, inspect as applicable:

- feature shapes, label mapping, preprocessing, bias handling, activation, loss, gradients, and update signs;
- training/test independence, leakage, sample counts, stopping criteria, shuffling, random seeds, and metric calculations;
- whether the implemented learning rule actually matches the named model;
- whether visualizations and printed results satisfy the requested outputs;
- edge cases, input validation, type/runtime compatibility, naming, structure, comments, and needless duplication.

Distinguish a definite defect from a concern that needs runtime evidence.

## Inspect the report

Read `report.typ` completely and follow its references to code, figures, tables, and result images. Check that it:

- covers every report requirement from the rubric;
- accurately describes the implementation and experiment actually present;
- reports training and test results separately when required;
- defines important parameters, methods, metrics, and dataset sizes consistently;
- uses correct equations, notation, terminology, figure references, and conclusions;
- stays within page and format constraints after compilation;
- does not make reproducibility or performance claims contradicted by the code or observed run.

Treat a claim supported only by a screenshot or prose as weaker evidence than a reproducible calculation.

## Results

Not only specify which file are you refering too, but also the exact line numbers where the issue was found so i can easily click on it to locate it.
Provide a concise summary of the what you found in the code/report that is not meeting the requirements or it could be
better, do a check list using (X and empty) to mark whether each requirement is met or not.
