---
layout: default
---

<center>
<figure style="display:inline-block;margin:0;padding:0">
    <video width='250' controls autoplay loop><source src="figures/a_bust_of_einstein_init.mp4" alt="einstein_init"/></video>
    <figcaption style="text-align:center">Source</figcaption>
</figure>
<figure style="display:inline-block;margin:0;padding:0">
    <video width='250' controls autoplay loop><source src="figures/a_bust_of_einstein_fin.mp4" alt="einstein_init"/></video>
    <figcaption style="text-align:center">Deformed</figcaption>
</figure>
<figure style="display:inline-block;margin:0;padding:0">
    <video width='250' controls autoplay loop><source src="figures/a_bust_of_einstein.mp4" alt="einstein"/></video>
    <figcaption style="text-align:center">Optimization process</figcaption>
</figure>
<p><em>
Target text: "a bust of Albert Einstein"
</em></p>
<p><em>
TextDeformer produces both global semantic changes and local detailing on an input mesh, driven by a target text prompt. Our method produces higher quality surfaces than previous, vertex-based methods.
</em></p>

<a href="https://github.com/threedle/TextDeformer" class="btn">Code</a>
<a href="https://arxiv.org/abs/2304.13348" class="btn">Paper</a>

<!--
<a href="https://arxiv.org/abs/2112.03221" class="btn">Paper</a>
<a href="arxiv.com/supp" class="btn">Supplementary</a>
-->

</center>

* * *

## Abstract

 We present a technique for automatically producing a deformation of an input triangle mesh, guided solely by a text prompt. Our framework is capable of deformations that produce both large, low-frequency shape changes, and small high-frequency details. Our framework relies on differentiable rendering to connect geometry to powerful pre-trained image encoders, such as CLIP and DINO. Notably, updating mesh geometry by taking gradient steps through differentiable rendering is notoriously challenging, commonly resulting in deformed meshes with significant artifacts. These difficulties are amplified by noisy and inconsistent gradients from CLIP. To overcome this limitation, we opt to represent our mesh deformation through Jacobians, which updates deformations in a global, smooth manner (rather than locally-sub-optimal steps). Our key observation is that Jacobians are a representation that favors smoother, large deformations, leading to a global relation between vertices and pixels, and avoiding localized noisy gradients. Additionally, to ensure the resulting shape is coherent from all 3D viewpoints, we encourage the deep features computed on the 2D encoding of the rendering to be consistent for a given vertex from all viewpoints. We demonstrate that our method is capable of smoothly-deforming a wide variety of source mesh and target text prompts, achieving both large modifications to, e.g., body proportions of animals, as well as adding fine semantic details, such as shoe laces on an army boot and fine details of a face.

## Overview
<img src="figures/overview.png" alt="Overview" width="1000"/>
<p><em>TextDeformer deforms a base mesh by optimizing per-triangle Jacobians using natural language as a guide. We optimize the deformation using three losses: a CLIP-based semantic loss drives the deformation toward the text prompt, a view-consistency loss matches multiple views of the same surface patch to ensure a coherent deformation, and our regularization on the Jacobians controls the fidelity to the base mesh</em></p>

## Results
<center>
<table>
<tr>
    <td style="border: 0;"> Source </td>
    <td style="border: 0;"> Deformed </td>
    <td style="border: 0;"> Optimization process </td>
</tr>
<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_camel_init.mp4" alt="camel init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_camel_fin.mp4" alt="camel fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_camel.mp4" alt="camel"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a camel" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_giraffe_init.mp4" alt="giraffe init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_giraffe_fin.mp4" alt="giraffe fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_giraffe.mp4" alt="giraffe"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a giraffe" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_ladybug_init.mp4" alt="ladybug init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_ladybug_fin.mp4" alt="ladybug fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_ladybug.mp4" alt="ladybug"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a ladybug" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/an_octopus_init.mp4" alt="octopus init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/an_octopus_fin.mp4" alt="octopus fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/an_octopus.mp4" alt="octopus"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "an octopus" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_submarine_init.mp4" alt="submarine init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_submarine_fin.mp4" alt="submarine fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_submarine.mp4" alt="submarine"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a submarine" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_heart_shaped_vase_init.mp4" alt="heart_shaped_vase init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_heart_shaped_vase_fin.mp4" alt="heart_shaped_vase fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_heart_shaped_vase.mp4" alt="heart_shaped_vase"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a heart shaped vase" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_royal_goblet_init.mp4" alt="a_royal_goblet init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_royal_goblet_fin.mp4" alt="a_royal_goblet fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_royal_goblet.mp4" alt="a_royal_goblet"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a royal goblet" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_mandolin_init.mp4" alt="mandolin init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_mandolin_fin.mp4" alt="mandolin fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_mandolin.mp4" alt="mandolin"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a mandolin" </td>
</tr>

<tr>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_bust_of_venus_init.mp4" alt="bust_of_venus init"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_bust_of_venus_fin.mp4" alt="bust_of_venus fin"/></video>
        </figure>
    </td>
    <td style="border: 0;">
        <figure style="display:inline-block;margin:0;padding:0">
            <video width='225' controls autoplay loop><source src="figures/a_bust_of_venus.mp4" alt="bust_of_venus"/></video>
        </figure>
    </td>
</tr>
<tr>
    <td colspan="3" style="border: 0;"> Target text: "a bust of venus" </td>
</tr>
</table>
</center>

## Citation
```
@InProceedings{Gao_2023_SIGGRAPH,
    author    = {Gao, William and Aigerman, Noam and Groueix Thibault and Kim, Vladimir and Hanocka, Rana},
    title     = {TextDeformer: Geometry Manipulation using Text Guidance},
    booktitle = {ACM Transactions on Graphics (SIGGRAPH)},
    year      = {2023},
}
```