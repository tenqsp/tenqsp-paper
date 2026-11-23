# Table 1: what each tool offers (draft, checked 2026-11-16; revised 2026-11-23 after review)

Written by hand from each tool's documentation; every cell has a source below. "Not
documented" means we found nothing in the documentation, not that the feature is known to
be absent. Versions: NONMEM 7.6.0, Monolix 2024R1, rxode2 5.1.8 and nlmixr2est 7.2.1,
Stan with Torsten 0.92, FiPy, py-pde, DifferentialEquations.jl with MethodOfLines.jl,
TenQSP 0.9.1 with tenqsp-fit 0.3.0.

| | NONMEM | Monolix | nlmixr2 / rxode2 | Stan / Torsten | FiPy | py-pde | DifferentialEquations.jl + MethodOfLines.jl | TenQSP + tenqsp-fit |
|---|---|---|---|---|---|---|---|---|
| Equations over an axis (age, maturation, depth, trait): PDEs | by hand: the user writes the method-of-lines ODEs; templates (DOPDE, DOEXPAND) help [1, 2] | by hand: the user writes the method-of-lines ODEs (no PDE construct documented) [4]; a research extension exists [7] | by hand (no PDE construct documented) [10] | by hand (no PDE construct documented) [15] | yes (finite volumes) [17] | yes (finite differences, method of lines) [19] | yes (symbolic PDEs, discretized automatically) [20] | yes (finite volumes, written as in a paper) |
| Delays | DDE solvers; distributed delays and transit chains by `ddexpand` [2, 3] | `delay()`, constant delays (estimable); transit absorption [4] | `delay()`, transit and lag functions [11] | no DDE solver documented [16] | not documented | not documented | DDE solvers [21] | as an axis (a chain of transit compartments, or transport with or without spread); dose lags; no DDE solver |
| Pharmacometric dosing and data | yes (the reference data format) [2] | yes [4] | yes, NONMEM conventions [12] | yes, NONMEM-style events (Torsten) [13] | no | no | through callbacks [22] | boluses, infusions, regimens, lags, bioavailability; NONMEM-style data in tenqsp-fit, which refuses records it would read differently (steady-state doses, resets) |
| Compartments and fields in one model | by hand [1] | by hand | by hand | by hand | not documented | not documented | partly (connectors join a discretized PDE to lumped parts) [23] | yes |
| Conservation ledger | not documented | not documented | not documented | not documented | not documented | not documented | not documented | yes, for every model |
| Exact derivatives of the solution | sensitivities to the random effects; finite differences for the fixed effects unless `FAST` (all MU-referenced) [2, 5, 6] | not documented [8, 9] | forward sensitivity equations; outer gradient by finite differences unless `fast=TRUE` [10, 14] | yes (automatic differentiation; forward and adjoint sensitivities) [15] | not documented | not documented | yes (forward, adjoint, automatic differentiation) [24] | yes (JAX: forward mode, or reverse by replay) |
| Population estimation | yes (FO, FOCE, Laplace, SAEM, IMP, Bayes) [2] | yes (SAEM) [8] | yes (FOCEi, SAEM and others) [14, 25] | yes (full Bayes) [13, 26] | no | no | not in these packages (Pumas is proprietary) [27] | yes (FOCE-I; Bayes by NUTS); no SAEM or importance sampling |
| Open source | no [28] | no (free academic license) [29] | yes, GPL-3 [30] | yes, BSD-3 [31] | yes, NIST terms [32] | yes, MIT [31] | yes, MIT [31] | yes, Apache-2.0 |
| Batches on a GPU | not documented (MPI across subjects) [2] | not documented (threads, clusters) [33] | not documented (threads across subjects) [10] | partly (OpenCL for some functions; MPI group solvers) [34, 35] | partly (GPU linear solver) [18] | not documented [19] | yes [36] | yes (JAX; Table S3) |

**Not in the table** (no column: they do not target the same problem, or document too little
of it). Pumas: proprietary; population estimation on DifferentialEquations.jl, documented for
ODE, DDE and SDE models; PDEs not documented [27, 37]. SimBiology (MATLAB): ODE models with
population estimation (`sbiofitmixed`: nlmefit, nlmefitsa); PDEs not documented [38]. PK-Sim
and MoBi (Open Systems Pharmacology): open-source PBPK, organs with their vascular,
interstitial and cellular spaces as a fixed structure, population simulation, and parameter
identification through an R package [39]. Virtual Cell: open-source reaction-diffusion in
cell geometries by finite volumes; its parameter estimation (COPASI) is for non-spatial
models [40]. PhysiCell (with BioFVM) and Chaste: open-source agent-based and continuum tissue
simulators; no pharmacometric data or population estimation documented [41, 42].

## Points to word carefully in the text

1. NONMEM's derivatives: the eta-sensitivities are integrated with the amounts; the THETA
   gradient uses forward finite differences by default, and `FAST` makes it analytic only
   when every THETA is MU-referenced [2, 5, 6]. Write it this way, not "finite differences".
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
- NONMEM 7.6.0 introductory guide, section I.86, "The DOPDE method of modeling PDE's" [2].
- Grenier E, Louvet V, Vigneaux P. Parameter estimation in non-linear mixed effects models
  with SAEM algorithm: extension from ODE to PDE. ESAIM: M2AN 2014;48(5):1303-1329.
  doi:10.1051/m2an/2013140.
- Yan X, Bauer R, Koch G, et al. Delay differential equations based models in NONMEM.
  J Pharmacokinet Pharmacodyn 2021;48(6):763-802. doi:10.1007/s10928-021-09770-z.

## Sources

1. Cheng Y, Li Y. J Clin Pharmacol 2026;66(6):e70215, doi:10.1002/jcph.70215 (published version; full text read)
2. NONMEM Users Guide, Introduction to NONMEM 7.6.0 (Bauer RJ, ICON Clinical Research, June 2025), sections I.28 (gradient accuracy), I.36-I.42 (EM, importance sampling, SAEM, Bayes, NUTS), I.46 and I.50 (MU referencing, FAST), I.75 (parallel computing), I.86 (DOPDE), I.87-I.94 (ddexpand: delays and transit compartments); NONMEM Users Guide Part V, Oct 2024 (data records)
3. Yan et al. 2021, doi:10.1007/s10928-021-09770-z
4. Monolix 2024R1 documentation, "Delay differential equations"
5. NONMEM Users Guide Part VI (PREDPP), compact arrays appendix
6. NONMEM users' cookbook, "Accuracy control" (gradients)
7. Grenier et al. 2014, doi:10.1051/m2an/2013140; PAGE 2013 abstract "Extending Monolix to use models with partial differential equations"
8. Monolix 2024R1 documentation, "SAEM"
9. Monolix 2024R1 documentation, "Standard errors"
10. rxode2 documentation (home page)
11. rxode2 vignette, "Delay differential equations"
12. rxode2 vignette, "Dosing event types"
13. Torsten documentation (metrumresearchgroup.github.io/Torsten)
14. nlmixr2est 7.2.1, `foceiControl` reference; Almquist, Leander, Jirstrand, J Pharmacokinet Pharmacodyn 2015;42:191-209, doi:10.1007/s10928-015-9409-1
15. Stan User's Guide, "Ordinary differential equations" and "Adjoint ODE solver"
16. Stan forum, "Is it possible to solve delay differential equation?"
17. FiPy documentation (pages.nist.gov/fipy)
18. FiPy documentation, "Solvers"
19. py-pde documentation, "Getting started"
20. MethodOfLines.jl documentation
21. DifferentialEquations.jl documentation, "DDE problems"
22. DifferentialEquations.jl documentation, callback library
23. MethodOfLines.jl documentation, "Connectors"
24. SciMLSensitivity.jl documentation
25. Fidler et al. CPT:PSP 2019;8(9):621-633, doi:10.1002/psp4.12445; Schoemaker et al. CPT:PSP 2019;8(12):923-930, doi:10.1002/psp4.12471
26. Margossian, Zhang, Gillespie. CPT:PSP 2022;11(9):1151-1169, doi:10.1002/psp4.12812
27. Rackauckas et al., Pumas, bioRxiv doi:10.1101/2020.11.28.402297
28. ICON, NONMEM product page
29. Lixoft, licence types
30. CRAN pages of rxode2 5.1.8 and nlmixr2est 7.2.1 (GPL >= 3)
31. GitHub licence metadata: Stan and Torsten BSD-3-Clause; py-pde, DifferentialEquations.jl, MethodOfLines.jl, SciMLSensitivity.jl, DiffEqGPU.jl MIT
32. FiPy LICENSE.rst
33. Monolix 2024R1 documentation, "Running through the command line"
34. CmdStan guide, "Parallelization"
35. Torsten documentation, population ODE group solvers
36. DiffEqGPU.jl documentation
37. Pumas documentation (docs.pumas.ai), "Analytical solutions and differential equations"
38. SimBiology documentation, `sbiofitmixed` and "Nonlinear mixed-effects modeling" (MathWorks)
39. Open Systems Pharmacology Suite documentation and GitHub (Open-Systems-Pharmacology/Suite; OSPSuite.ParameterIdentification)
40. Virtual Cell help, "Parameter estimation overview" (vcell.org)
41. Ghaffarizadeh et al. PhysiCell. PLoS Comput Biol 2018;14(2):e1005991, doi:10.1371/journal.pcbi.1005991
42. Cooper et al. Chaste: Cancer, Heart and Soft Tissue Environment. J Open Source Softw 2020;5(47):1848, doi:10.21105/joss.01848
