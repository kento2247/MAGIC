## Method {#method}

### GENNAV

<!-- \input{tab/4-4-models} -->

We propose **GENNAV**, which integrates polygon-based segmentation with spatially grounded multimodal representations and landmark-distribution-driven patchification.  
GENNAV models both semantic and spatial characteristics with low computational overhead.  

It is inspired by polygon-based referring expression segmentation (RES) approaches \citep{zhu2022seqtr, liu2023polyformer, cheng2024parallel, nishimura24iros}, and is widely applicable to various RES and RNR models.

Fig.~\ref{fig:model} shows the structure of GENNAV.  
It consists of three modules:
- Existence Aware Polygon Segmentation module (**ExPo**)
- Landmark Distribution Patchification Module (**LDPM**)
- Visual-Linguistic Spatial Integration Module (**VLSiM**)

---

We define the input $\mathbf{x}$ as:

$$
\mathbf{x} = \{\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{inst}}\}
$$

where

- $\mathbf{x}_{\text{img}} \in \mathbb{R}^{3 \times W \times H}$: front-camera image  
- $\mathbf{x}_{\text{inst}} \in \{0,1\}^{V \times L}$: navigation instruction (one-hot sequence)

Here, $W$ and $H$ denote image width and height.

We preprocess $\mathbf{x}_{\text{inst}}$ using GISTEmbed \cite{solatorio2024gistembed}:

$$
\mathbf{h}_{\text{inst}} \in \mathbb{R}^{C_{\text{inst}}}
$$

---

### Landmark Distribution Patchification Module

This module models fine-grained visual representations related to potential landmarks by efficiently partitioning patches based on landmark distributions.

Such representations are important because $\mathbf{x}_{\text{inst}}$ may specify pedestrians or vehicles.  
However, many existing methods (e.g., \citep{rufus2021grounding, tnrsmral24}) downscale images, leading to insufficient resolution for distant landmarks.

The input is $\mathbf{x}_{\text{img}}$.

We introduce **landmark-distribution-based patchification**, which prioritizes regions with dense landmark distributions instead of uniform partitioning.

Each patch is encoded by DINOv2 \citep{oquab2023dinov2}, followed by a CNN.  
The final feature is:

$$
\mathbf{h}_{\text{ldp}}
$$

---

### Visual-Linguistic Spatial Integration Module

This module integrates:

- $\mathbf{x}_{\text{img}}$
- $\mathbf{h}_{\text{inst}}$
- pseudo-depth image
- road region

#### Visual feature extraction

$$
f_{\text{vis}}(\mathbf{x}_{\text{img}})
$$

obtained using frozen DINOv2 + CNN.

#### Depth feature

We generate pseudo-depth:

$$
\mathbf{x}_{\text{depth}} \in \mathbb{R}^{3 \times W \times H}
$$

using a monocular depth estimator (e.g., Depth Anything V2), and process it as:

$$
f_{\text{depth}}(\mathbf{x}_{\text{img}})
$$

#### Road feature

We extract road masks:

$$
\mathbf{x}_{\text{road}} \in \mathbb{R}^{3 \times W \times H}
$$

using PIDNet \citep{xu2023pidnet}, then:

$$
f_{\text{road}}(\mathbf{x}_{\text{img}})
$$

#### Multimodal fusion

$$
\mathbf{h}_{\text{mm}} =
\mathbf{h}_{\text{inst}} \odot
\left( f_{\text{vis}}(\mathbf{x}_{\text{img}}) + f_{\text{depth}}(\mathbf{x}_{\text{img}}) \right)
\odot
\left( f_{\text{road}}(\mathbf{x}_{\text{img}}) + f_{\text{depth}}(\mathbf{x}_{\text{img}}) \right)
$$

where $\odot$ denotes the Hadamard product.

---

### Existence Aware Polygon Segmentation Module

This module predicts:

- existence of target regions  
- polygon-based segmentation masks  

using:

- $\mathbf{h}_{\text{inst}}$
- $\mathbf{h}_{\text{mm}}$
- $f_{\text{road}}(\mathbf{x}_{\text{img}})$
- $\mathbf{h}_{\text{ldp}}$

#### Multimodal fusion

$$
\mathbf{h}_{\text{cmm}} =
(\mathbf{h}_{\text{ldp}} \odot \mathbf{h}_{\text{inst}}) + \mathbf{h}_{\text{mm}}
$$

#### Polygon regression

$$
\hat{\mathbf{c}}_i =
\text{MLP} \left(
\left[ \mathbf{h}_{\text{cmm}} ; f_{\text{road}}(\mathbf{x}_{\text{img}}) \right]
\right)
$$

- $\hat{\mathbf{c}}_i \in \mathbb{R}^2$: polygon vertices  
- ordered clockwise from top-left  

#### Classification

$$
p(\hat{\mathbf{y}}_{\text{ex}})
$$

over:

- no-target  
- single-target  
- multi-target  

#### Output

$$
\hat{\mathbf{y}} =
\{ p(\hat{\mathbf{y}}_{\text{ex}}), \hat{\mathbf{c}}_i \}
$$

---

### Loss Function

$$
\mathcal{L} =
\mathcal{L}_{\mathrm{CE}}\left(p(\hat{\mathbf{y}}_{\mathrm{ex}}), \mathbf{y}_{\mathrm{ex}}\right)
+
\lambda_{\mathrm{pt}}
\sum_{i=1}^{n_{\mathrm{pt}}}
\mathbb{I}\left[
\mathbf{y}_{\mathrm{ex}} \in \{\text{single-target}, \text{multi-target}\}
\right]
\cdot
\ell_1 \left(\hat{\mathbf{c}}_i, \mathbf{c}_i\right)
$$

where:

- $\lambda_{\text{pt}}$: loss weight  
- $\mathcal{L}_{\text{CE}}$: cross-entropy loss  
- $\ell_1$: L1 loss  
- $\mathbb{I}$: indicator function