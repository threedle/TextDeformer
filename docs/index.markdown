---
layout: default
---

<center>
<figure style="display:inline-block;margin:0;padding:0"><img src="figures/hand_to_octopus.gif" alt="octopus" width="200"/><figcaption style="text-align:center">Octopus</figcaption></figure>
<figure style="display:inline-block;margin:0;padding:0"><img src="figures/planck_to_einstein.gif" alt="einstein" width="200"/><figcaption style="text-align:center">Albert Einstein</figcaption></figure>
<figure style="display:inline-block;margin:0;padding:0"><img src="figures/spot_to_giraffe.gif" alt="giraffe" width="200"/><figcaption style="text-align:center">giraffe</figcaption></figure>

<p><em>
TextDeformer produces both global semantic changes and local detailing, driven by a target text prompt. Our method produces higher quality surfaces than previous, vertex-based methods.
</em></p>

<a href="https://github.com/threedle/TextDeformer" class="btn">Code</a>
<a href="" class="btn">Paper</a>

<!--
<a href="https://arxiv.org/abs/2112.03221" class="btn">Paper</a>
<a href="arxiv.com/supp" class="btn">Supplementary</a>
-->

</center>

* * *

## Abstract

 We present a technique for automatically producing a deformation of an input triangle mesh, guided solely by a text prompt. Our framework is capable of deformations that produce both large, low-frequency shape changes, and small high-frequency details. Our framework relies on differentiable rendering to connect geometry to powerful pre-trained image encoders, such as CLIP and DINO. Notably, updating mesh geometry by taking gradient steps through differentiable rendering is notoriously challenging, commonly resulting in deformed meshes with significant artifacts. These difficulties are amplified by noisy and inconsistent gradients from CLIP. To overcome this limitation, we opt to represent our mesh deformation through Jacobians, which updates deformations in a global, smooth manner (rather than locally-sub-optimal steps). Our key observation is that Jacobians are a representation that favors smoother, large deformations, leading to a global relation between vertices and pixels, and avoiding localized noisy gradients. Additionally, to ensure the resulting shape is coherent from all 3D viewpoints, we encourage the deep features computed on the 2D encoding of the rendering to be consistent for a given vertex from all viewpoints. We demonstrate that our method is capable of smoothly-deforming a wide variety of source mesh and target text prompts, achieving both large modifications to, e.g., body proportions of animals, as well as adding fine semantic details, such as shoe laces on an army boot and fine details of a face.
