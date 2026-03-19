## Method

We propose **Vela**, an automatic evaluation metric tailored for evaluating long and detailed image captions.
Fig.~\ref{fig:model} shows the architecture of **Vela**.
It consists of two main branches: the R2C-LLM branch and I2C-Align branch.

---

### R2C-LLM branch

This branch efficiently assesses the quality of $\bm{x}*\mathrm{cand}$ in relation to the corresponding $\bm{x}*\mathrm{ref}$ by employing a lightweight LLM in a non-autoregressive manner.

We adopt evaluation based on LLMs to take advantage of their extensive world knowledge and linguistic capability acquired through pretraining on broad-domain datasets.

To address the slow speed of autoregressive inference in LLM-as-a-Judge, we employ a non-autoregressive approach that significantly reduces the inference time.

Furthermore, although MLLMs typically perform early fusion of visual information, which results in increased computational costs and slow inference, we opt for a text-only LLM with a late fusion approach to mitigate these issues.

In the R2C-LLM branch, we first prepare a prompt $\bm{x}*\mathrm{prompt}$ using $\bm{x}*\mathrm{cand}$ and ${ \bm{x}*\mathrm{ref}^{(i)} }*{i=1}^N$.

Our evaluation prompt, designed based on previous work~\cite{flickr, summeval, tong2025gveval, umic}, is provided in Appendix~\ref{sec:prompts}.

We then feed $\bm{x}_\mathrm{prompt}$ into a text-only LLM (Qwen2.5-3B~\cite{qwen2.5}), and obtain the last hidden states in a non-autoregressive manner.

The sequence of hidden states is denoted as ${ \bm{h}*i }*{i=1}^M$, where $M$ denotes the sequence length.

Similar to previous works using the last hidden states (e.g., ~\cite{llm2vec, one, repet}), we compute $\bm{g}_{\text{r2c}}$, the output of the R2C-LLM branch, as follows:

[
\bm{g}*{\text{r2c}} = \left[ \frac{1}{M} \sum*{i=1}^{M} \bm{h}_i ; , ; \bm{h}_M \right].
]

---

### I2C-Align branch

This branch evaluates $\bm{x}*\mathrm{cand}$ with respect to $\bm{x}*\mathrm{img}$ using Long-CLIP~\cite{long-clip} without relying on MLLMs.

As previously mentioned, the early fusion of visual information in MLLM-based metrics results in high computational costs~\cite{chan2023clair, lee2024fleur, tong2025gveval}.

To avoid these costs, the I2C-Align branch does not employ MLLMs.

The I2C-Align branch uses Long-CLIP to extract $\bm{h}*\text{img}$ and $\bm{h}*\text{cand}$ from $\bm{x}*\text{img}$ and $\bm{x}*\text{cand}$, respectively.

Unlike existing metrics based on CLIP~\cite{clipscore, pac-s, pac-spp, bridge, polos, deneb}, the I2C-Align branch employs Long-CLIP to overcome the 77-token limit of the original CLIP model, which is insufficient for processing long captions that typically exceed 100 words.

The output of I2C-Align ($\bm{g}_\text{i2c}$) is then computed as follows:

[
\bm{g}*\text{i2c} = \left[ \left| \bm{h}*\text{img} - \bm{h}*\text{cand} \right| ; , ; \bm{h}*\text{img} \odot \bm{h}_\text{cand} \right],
]

where $\left| \bm{h}*\text{img} - \bm{h}*\text{cand} \right|$ and $\bm{h}*\text{img} \odot \bm{h}*\text{cand}$ denote the absolute element-wise difference and Hadamard product between $\bm{h}*\text{img}$ and $\bm{h}*\text{cand}$, respectively.

These operations have been shown to be effective in automatic evaluation across various text generation tasks, such as machine translation and image captioning~\cite{ruse, comet, polos, deneb}.

---

### Final scoring

The final scores $\hat{\bm{y}} \in \mathbb{R}^3$ are computed as follows:

[
\hat{\bm{y}} = (\hat{y}*{\text{desc}}, \hat{y}*{\text{rel}}, \hat{y}*{\text{flu}}) = \sigma\left( \bm{W} \left[ \bm{g}*{\text{r2c}} , \bm{g}_{\text{i2c}} \right] + \bm{b} \right),
]

where $\sigma$ denotes the sigmoid function, and $\bm{W}$ and $\bm{b}$ are trainable parameters.

Here, $\hat{y}*{\text{desc}}$, $\hat{y}*{\text{rel}}$, and $\hat{y}_{\text{flu}}$ denote the predicted scores for *Desc.*, *Rel.*, and *Flu.*, respectively.

We employed the mean squared error as our loss function.

---