# Table 1: what each tool offers (draft, checked 2026-11-16; revised 2026-11-23 after review)

Written by hand from each tool's documentation; every cell has a source below. "Not
documented" means we found nothing in the documentation, not that the feature is known to
be absent. Versions: NONMEM 7.5.1/7.6.0, Monolix 2024R1, rxode2 5.1.8 and nlmixr2est 7.2.1,
Stan with Torsten 0.92, FiPy, py-pde, DifferentialEquations.jl with MethodOfLines.jl,
TenQSP 0.9.1 with tenqsp-fit 0.3.0.

| | NONMEM | Monolix | nlmixr2 / rxode2 | Stan / Torsten | FiPy | py-pde | DifferentialEquations.jl + MethodOfLines.jl | TenQSP + tenqsp-fit |
|---|---|---|---|---|---|---|---|---|
| Equations over an axis (age, maturation, depth, trait): PDEs | by hand: the user writes the method-of-lines ODEs; templates (DOPDE, DOEXPAND) help [1, 2] | by hand: the user writes the method-of-lines ODEs (no PDE construct documented) [5]; a research extension exists [8] | by hand (no PDE construct documented) [11] | by hand (no PDE construct documented) [16] | yes (finite volumes) [18] | yes (finite differences, method of lines) [20] | yes (symbolic PDEs, discretized automatically) [21] | yes (finite volumes, written as in a paper) |
| Delays | DDE solvers; distributed delays and transit chains by `ddexpand` [2, 3] | `delay()`, constant delays (estimable); transit absorption [5] | `delay()`, transit and lag functions [12] | no DDE solver documented [17] | not documented | not documented | DDE solvers [22] | as an axis (a chain of transit compartments, or transport with or without spread); dose lags; no DDE solver |
| Pharmacometric dosing and data | yes (the reference data format) [2] | yes [5] | yes, NONMEM conventions [13] | yes, NONMEM-style events (Torsten) [14] | no | no | through callbacks [23] | boluses, infusions, regimens, lags, bioavailability; NONMEM-style data in tenqsp-fit, which refuses records it would read differently (steady-state doses, resets) |
| Compartments and fields in one model | by hand [4] | by hand | by hand | by hand | not documented | not documented | partly (connectors join a discretized PDE to lumped parts) [24] | yes |
| Conservation ledger | not documented | not documented | not documented | not documented | not documented | not documented | not documented | yes, for every model |
| Exact derivatives of the solution | sensitivities to the random effects; finite differences for the fixed effects unless `FAST` (all MU-referenced) [2, 6, 7] | not documented [9, 10] | forward sensitivity equations; outer gradient by finite differences unless `fast=TRUE` [11, 15] | yes (automatic differentiation; forward and adjoint sensitivities) [16] | not documented | not documented | yes (forward, adjoint, automatic differentiation) [25] | yes (JAX: forward mode, or reverse by replay) |
| Population estimation | yes (FO, FOCE, Laplace, SAEM, IMP, Bayes) [2] | yes (SAEM) [9] | yes (FOCEi, SAEM and others) [15, 26] | yes (full Bayes) [14, 27] | no | no | not in these packages (Pumas is proprietary) [28] | yes (FOCE-I; Bayes by NUTS); no SAEM or importance sampling |
| Open source | no [29] | no (free academic license) [30] | yes, GPL-3 [31] | yes, BSD-3 [32] | yes, NIST terms [33] | yes, MIT [32] | yes, MIT [32] | yes, Apache-2.0 |
| Batches on a GPU | not documented (MPI across subjects) [2] | not documented (threads, clusters) [34] | not documented (threads across subjects) [11] | partly (OpenCL for some functions; MPI group solvers) [35, 36] | partly (GPU linear solver) [19] | not documented [20] | yes [37] | yes (JAX; Table S3) |

**Not in the table** (no column: they do not target the same problem, or document too little
of it). Pumas: proprietary; population estimation on DifferentialEquations.jl, documented for
ODE, DDE and SDE models; PDEs not documented [28, 38]. SimBiology (MATLAB): ODE models with
population estimation (`sbiofitmixed`: nlmefit, nlmefitsa); PDEs not documented [39]. PK-Sim
and MoBi (Open Systems Pharmacology): open-source PBPK, organs with their vascular,
interstitial and cellular spaces as a fixed structure, population simulation, and parameter
identification through an R package [40]. Virtual Cell: open-source reaction-diffusion in
cell geometries by finite volumes; its parameter estimation (COPASI) is for non-spatial
models [41]. PhysiCell (with BioFVM) and Chaste: open-source agent-based and continuum tissue
simulators; no pharmacometric data or population estimation documented [42, 43].

## Points to word carefully in the text

1. NONMEM's derivatives: the eta-sensitivities are integrated with the amounts; the THETA
   gradient uses forward finite differences by default, and `FAST` makes it analytic only
   when every THETA is MU-referenced [2, 6, 7]. Write it this way, not "finite differences".
2. nlmixr2est `fast=TRUE` (analytic outer gradient, Almquist et al. 2015) is in version 7.2.1
   (October 2026); cite the version.
3. Monolix does not say how the derivatives behind its linearized Fisher information are
   computed: "not documented", not "finite differences".
4. Stan: "no DDE solver" rests on the Stan forum, not the manual.
5. "Not documented" for the ledger: FiPy's finite volumes conserve by construction, but we
   found no ledger (an account of what entered, left and was made) in any of these tools.
6. NONMEM's PDE templates: the guide lists them under 7.5 and tags the utility NM74; Cheng
   and Li write "since NONMEM 7.4". Say "7.4/7.5".
7. Julia: dosing works through callbacks; no reader for NONMEM-style data outside Pumas.

## PDE models run in NONMEM or Monolix (for the introduction)

- Cheng Y, Li Y. Partial differential equation (PDE)-based spatial pharmacometrics in
  NONMEM: method of lines (MOL) implementation with AI-assisted model development. J Clin
  Pharmacol 2026;66(6):e70215. doi:10.1002/jcph.70215 (a 100-compartment grid and a plasma
  compartment, in NONMEM 7.5.1).
- NONMEM 7.5.1 introductory guide, section I.84, "The DOPDE method of modeling PDE's" [2].
- Grenier E, Louvet V, Vigneaux P. Parameter estimation in non-linear mixed effects models
  with SAEM algorithm: extension from ODE to PDE. ESAIM: M2AN 2014;48(5):1303-1329.
  doi:10.1051/m2an/2013140.
- Yan X, Bauer R, Koch G, et al. Delay differential equations based models in NONMEM.
  J Pharmacokinet Pharmacodyn 2021;48(6):763-802. doi:10.1007/s10928-021-09770-z.

## Sources

1. Cheng & Li 2026, J Clin Pharmacol, doi:10.1002/jcph.70215; preprint bioRxiv doi:10.64898/2026.03.01.708822
2. NONMEM Users Guide, Introduction to NONMEM 7.5.1 (Bauer, ICON), sections I.2, I.3, I.48, I.84-I.92; NONMEM Users Guide Part V, 7.6.0
3. Yan et al. 2021, doi:10.1007/s10928-021-09770-z
4. Cheng & Li 2026, full text (preprint)
5. Monolix 2024R1 documentation, "Delay differential equations"
6. NONMEM Users Guide Part VI (PREDPP), compact arrays appendix
7. NONMEM users' cookbook, "Accuracy control" (gradients)
8. Grenier et al. 2014, doi:10.1051/m2an/2013140; PAGE 2013 abstract "Extending Monolix to use models with partial differential equations"
9. Monolix 2024R1 documentation, "SAEM"
10. Monolix 2024R1 documentation, "Standard errors"
11. rxode2 documentation (home page)
12. rxode2 vignette, "Delay differential equations"
13. rxode2 vignette, "Dosing event types"
14. Torsten documentation (metrumresearchgroup.github.io/Torsten)
15. nlmixr2est 7.2.1, `foceiControl` reference; Almquist, Leander, Jirstrand, J Pharmacokinet Pharmacodyn 2015;42:191-209, doi:10.1007/s10928-015-9409-1
16. Stan User's Guide, "Ordinary differential equations" and "Adjoint ODE solver"
17. Stan forum, "Is it possible to solve delay differential equation?"
18. FiPy documentation (pages.nist.gov/fipy)
19. FiPy documentation, "Solvers"
20. py-pde documentation, "Getting started"
21. MethodOfLines.jl documentation
22. DifferentialEquations.jl documentation, "DDE problems"
23. DifferentialEquations.jl documentation, callback library
24. MethodOfLines.jl documentation, "Connectors"
25. SciMLSensitivity.jl documentation
26. Fidler et al. CPT:PSP 2019;8(9):621-633, doi:10.1002/psp4.12445; Schoemaker et al. CPT:PSP 2019;8(12):923-930, doi:10.1002/psp4.12471
27. Margossian, Zhang, Gillespie. CPT:PSP 2022;11(9):1151-1169, doi:10.1002/psp4.12812
28. Rackauckas et al., Pumas, bioRxiv doi:10.1101/2020.11.28.402297
29. ICON, NONMEM product page
30. Lixoft, licence types
31. CRAN pages of rxode2 5.1.8 and nlmixr2est 7.2.1 (GPL >= 3)
32. GitHub licence metadata: Stan and Torsten BSD-3-Clause; py-pde, DifferentialEquations.jl, MethodOfLines.jl, SciMLSensitivity.jl, DiffEqGPU.jl MIT
33. FiPy LICENSE.rst
34. Monolix 2024R1 documentation, "Running through the command line"
35. CmdStan guide, "Parallelization"
36. Torsten documentation, population ODE group solvers
37. DiffEqGPU.jl documentation
38. Pumas documentation (docs.pumas.ai), "Analytical solutions and differential equations"
39. SimBiology documentation, `sbiofitmixed` and "Nonlinear mixed-effects modeling" (MathWorks)
40. Open Systems Pharmacology Suite documentation and GitHub (Open-Systems-Pharmacology/Suite; OSPSuite.ParameterIdentification)
41. Virtual Cell help, "Parameter estimation overview" (vcell.org)
42. Ghaffarizadeh et al. PhysiCell. PLoS Comput Biol 2018;14(2):e1005991, doi:10.1371/journal.pcbi.1005991
43. Cooper et al. Chaste: Cancer, Heart and Soft Tissue Environment. J Open Source Softw 2020;5(47):1848, doi:10.21105/joss.01848
