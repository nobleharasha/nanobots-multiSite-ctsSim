# Nanobot Algorithms for Multi-Site Cancer Detection and Treatment, Model/Simulator
A formal model, in 3-dimensional continuous space with discrete time, of feasible nanobots moving in a colloidal environment for the problem of multi-site cancer detection and treatment.
Each individual cancer site can have its own unique demand, or amount of drug needed.
Nanobot agents here have movement inspired by chemotactic nanoparticles which, when in the presence of an external chemical gradient, noisily follow the gradient by either ascending or descending it, depending on whether it is the case of positive or negative chemotaxis, respectively.

There are three distinct and increasingly sophisticated algorithms: KM, KMA, and KMAR.
KM involves agents following natural existing chemical "M" signals surrounding cancer sites.
KMA involves agents amplifying and boosting these natural signals by dropping payloads of chemical "A".
KMAR adds a repellent chemical payload "R" which is dropped at sites that are already sufficiently treated so as to encourage agents to search elsewhere.
(Chemical "K" is the cancer-treating drug.)
