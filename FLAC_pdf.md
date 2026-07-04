## Few-shot Acoustic Synthesis with Multimodal Flow Matching

### Amandine Brunetto

### Center for Robotics, Mines Paris - PSL University

```
https://amandinebtto.github.io/
```
```
Figure 1. Few-shot flow-matching acoustic synthesis (FLAC) and scene-consistency evaluation: Given a few-shot multimodal context
τ , including a depth map, an acoustic observation, and sensor poses, FLAC uses a diffusion transformer trained with flow matching to
generate room impulse responses (RIRs) in novel rooms. Unlike prior deterministic approaches, FLAC models the distribution of plausible
RIRs under sparse scene context, capturing acoustic uncertainty. Even with one shot, FLAC outperforms 8-shot state-of-the-art methods.
To assess generation quality, we introduce AGREE, a CLIP-style audio-geometry embedding that aligns both modalities in a shared latent
space, enabling scene-consistency evaluation through retrieval and distributional metrics.
```
### Abstract

```
Generating audio that is acoustically consistent with a
scene is essential for immersive virtual environments. Re-
cent neural acoustic field methods enable spatially con-
tinuous sound rendering but remain scene-specific, re-
quiring dense audio measurements and costly training for
each environment. Few-shot approaches improve scala-
bility across rooms but still rely on multiple recordings
and, being deterministic, fail to capture the inherent uncer-
tainty of scene acoustics under sparse context. We intro-
duce flow-matching acoustic generation (FLAC), a prob-
abilistic method for few-shot acoustic synthesis that mod-
els the distribution of plausible room impulse responses
(RIRs) given minimal scene context. FLAC leverages a
diffusion transformer trained with a flow-matching ob-
jective to generate RIRs at arbitrary positions in novel
scenes, conditioned on spatial, geometric, and acoustic
cues. FLAC outperforms state-of-the-art eight-shot base-
lines with one-shot on both the AcousticRooms and Hear-
ing Anything Anywhere datasets. To complement stan-
dard perceptual metrics, we further introduce AGREE, a
joint acoustic–geometry embedding, enabling geometry-
consistent evaluation of generated RIRs through retrieval
and distributional metrics. This work is the first to apply
generative flow matching to explicit RIR synthesis, estab-
lishing a new direction for robust and data-efficient acous-
tic synthesis. Project page: https://amandinebtto.
github.io/FLAC/
```
### 1. Introduction

```
Every room shapes the way we hear: a lecture hall amplifies
a speaker’s voice, while a cathedral envelops sound in lin-
gering reverberation. Reproducing these rich auditory ex-
periences is essential for creating virtual, immersive envi-
ronments, where users expect sound to reflect the space.
The acoustic properties of a room are encapsulated by
Room Impulse Responses (RIRs), which describe the sound
propagation between source-receiver pairs. RIRs allows for
auralization, i.e., transferring a room’s acoustic signature
onto any sound. However, accurately modeling RIRs is
challenging because they depend on complex interactions
between geometry, materials, and source-listener positions.
Recently, neural acoustic fields [2, 4, 10, 44, 54, 79, 83]
have enabled spatially continuous RIRs rendering in a
scene. However, they must be trained for each environment
using extensive RIR recordings. More scalable solutions re-
quire models that can generate RIRs in novel rooms, with
minimal data and without retraining.
A handful of works have explored few-shot acoustic syn-
thesis [34, 52, 57]. These methods generate RIRs in novel
environments using only a sparse set of information (e.g.,
depth maps, RGB images, sensor poses, and 8 to 20 RIR
recordings) without scene-specific retraining. With limited
knowledge about a new scene’s characteristics, there is no
single, deterministic possible RIR: few-shot generalization
is an inherently ambiguous problem. Yet, existing few-shot
methods overlook this uncertainty, producing only a unique
```
# arXiv:2603.19176v1 [cs.SD] 19 Mar 2026


deterministic prediction.
To address this, we propose FLAC, a conditional gener-
ative model for few-shot acoustic synthesis based on flow
matching [47]. This framework extends diffusion models
[33, 76] with increased performance and versatility and has
demonstrated strong performance in audio [36, 42, 49] and
images [18] generation. FLAC is, to the best of our knowl-
edge, the first application of generative flow matching to
explicit RIR synthesis. Rather than learning a determinis-
tic mapping, our model estimates a distribution of plausi-
ble RIRs given sparse scene context, explicitly capturing
the uncertainty inherent in few-shot scenarios. We condi-
tion the generation on multimodal context, including scene
geometry around the receiver, sensor poses, and a mini-
mal set of RIR recordings. By formulating few-shot acous-
tic synthesis as a conditional generative task, we enable
scene-consistent sound generation in novel environments
even from only one audio measurement.
To assess generation quality, we complement tradition-
ally used perceptual metrics by introducing a set of scene
consistency metrics that ensure the predicted RIR matches
the scene’s geometry. To this end, we introduce AGREE
(Acoustic-GeometRy EmbEdding), a CLIP-style [65] dual-
encoder network that aligns RIRs and scene geometry in
a shared latent space. This alignment enables zero-shot
audio and geometry retrieval. We leverage this shared
space to provide a geometry-consistent evaluation frame-
work through both retrieval-based scores and distributional
metrics.
We evaluate FLAC on the large-scale synthetic Acoustic-
Rooms [52] dataset. It achieves state-of-the-art RIR synthe-
sis performance, demonstrating generalization across novel
source-receiver positions within known rooms, as well as
in entirely unseen environments. We also validate our
model’s real-world capabilities through sim-to-real transfer
on the Hearing-Anything-Anywhere [83] dataset. On both
datasets, FLAC outperforms current state-of-the-art meth-
ods based on 8 audio recordings with a single one.
In summary, our main contributions are as follows:

- We propose FLAC the first conditional generative model
    for few-shot RIR synthesis based on flow matching. This
    approach accounts for the inherent uncertainty of acous-
    tics given sparse scene context, leading to more robust
    predictions.
- Our approach sets a new state-of-the-art on the Acousti-
    cRooms and Hearing-Anything-Anywhere datasets, gen-
    eralizing to both novel source-listener pairs and environ-
    ments. FLAC outperforms prior work with 8 × fewer RIR
    recordings.
- We introduce AGREE, a joint acoustic-geometry embed-
    ding space, and propose new scene-consistency metrics
    that evaluate how well predicted RIRs align with the
    scene geometry.

### 2. Related Work

```
Audio-visual learning. Audio-visual learning enhances
both acoustic and vision-related tasks, including audio spa-
tialization [22, 25, 39, 60, 82, 88], de-reverberation [9, 13],
RIR prediction [7, 44, 45, 52, 57, 74, 75], depth estima-
tion [3, 14, 62, 87, 90], navigation [6, 11, 21, 23, 24, 86],
floorplan reconstruction [58, 64], and pose estimation [11].
FLAC extends this line of work by leveraging depth infor-
mation for scene-consistent RIR generation.
```
```
Neural acoustic fields. Neural acoustic fields render
RIRs at novel poses by implicitly learning a mapping from
spatial coordinates to the room’s acoustic field. Some ap-
proaches incorporate physical acoustic models [79, 83],
others infer local geometry [54], exploit vision cues [10,
12, 44, 45] or use NeRF [59] and Gaussian splatting-based
[38] representations [2, 4]. However, these methods remain
scene-specific, requiring dense recordings and retraining for
each new environment.
```
```
Few-shot acoustic synthesis. Few-shot methods gen-
eralize across scenes using sparse observations. Few-
ShotRIR [57] uses 20 RGB, depth and binaural audio in-
puts. MAGIC [34] adds semantics by extracting features
with a segmentation-pretrained U-Net [70]. More recently,
xRIR [52] reduces inputs to eight audio recording and a
panoramic depth map, and introduces the AcousticRooms
dataset specifically designed for cross-room synthesis. All
prior methods treat few-shot RIR prediction as a determin-
istic mapping, overlooking the ambiguity of the task. By
using generative flow matching, FLAC captures the distri-
bution of plausible RIRs given sparse context, improving
generalization to new scenes even with one-shot.
```
```
Audio diffusion and flow matching. Diffusion-based
models have advanced text-to-audio generation across
speech, music, and general sound [19, 20, 26, 35, 48, 49,
56]. Flow matching further improves synthesis efficiency
[28, 36, 42]. [46] recently achieved speech binauralization
via flow matching. Building on these advances, we adapt
generative flow matching to RIR synthesis conditioned on
few-shot scene context.
```
```
Joint embedding models across modalities. Joint em-
bedding models align data from different modalities in a
shared representation space. CLIP [65] pioneered this for
image-text, later extended to audio-visual [29, 55, 61, 67],
audio-text [17, 29], and audio with diverse sensory modal-
ities [27], enabling zero-shot cross-modal retrieval. Stan-
dard audio embeddings cannot be applied directly to RIRs,
which differ substantially. We introduce AGREE, a joint
embedding space for RIRs and scene geometry, allowing
acoustic-geometry consistency evaluation.
```

Figure 2. Training and inference pipelines of FLAC: During training, a pre-trained VAE encodes ground-truth RIRs into latentsz 0.
Latents are linearly interpolated with noise to formzt. A DiT is trained to predict the velocitybvtthat transportszttoward the original data
distribution. At inference, RIRs are generated from random noise, guided by the few-shot spatial, geometric and acoustic context.

### 3. Method

```
FLAC is a conditional latent generative model [69] trained
with flow matching [1, 47] (Sec. 3.1) to synthesize RIRs
from few-shot scene information. It comprises: (i) a varia-
tional autoencoder (Sec. 3.2), (ii) a multimodal conditioner
(Sec. 3.3), and (iii) a diffusion transformer (Sec. 3.4). Fig. 2
provides an overview of the method.
```
#### 3.1. Latent Flow Matching

```
Ambiguity in few-shot synthesis. Estimating RIRs
across diverse environments and sensor poses is challeng-
ing, as they depend on many intertwined factors. With lim-
ited scene information, multiple RIRs can be equally plausi-
ble for the same source-receiver configuration. For instance,
even with precise geometry knowledge, missing material
properties introduces ambiguity: whether the floor is car-
peted or wooden alters the acoustics.
We address the inherently ambiguous problem of few-
shot RIR synthesis: Our goal is to predict monaural, om-
nidirectional RIRs at arbitrary source-receiver pairs in un-
seen environments, given minimal scene context. By using
a stochastic generative model, we aim to capture the uncer-
tainty inherent to RIR prediction under sparse observations.
```
```
Training. We train FLAC using the rectified flow match-
ing formulation [50, 51], which linearly interpolates data
and noise. This approach straightens the transport paths
between distributions, reducing the number of integration
steps at inference.
The goal is to capture the relationship between a RIR and
its spatial, geometric, and acoustic context. To this end, we
sample target RIRs with their associated context (AT,τ )
from the dataset. Each RIR is encoded into a latent repre-
sentation z 0 , which is linearly interpolated with Gaussian
```
```
noiseε∼N (0, I) to produce a noisy latent zt:
```
```
zt= (1− t) z 0 + tε, (1)
where the timestep t ∈ [0, 1] controls the noise level.
Timesteps are sampled by drawing α ∼ N (− 1. 2 , 22 ) and
mapping it to t using a sigmoid:
```
```
t = σ(−α) =
```
##### 1

```
1 + eα
```
##### . (2)

```
This schedule emphasizes on moderately noisy latents (t≈
0. 7 − 0. 8 ), which we found to improve performance. Com-
parisons of noise sampling strategies are provided in Ap-
pendix E.4.
The model u(zt,t,τ ) is trained to predict the velocity
field vt
vt=
```
```
dzt
dt
=ε− z 0 , (3)
using the following objective:
```
```
LRFM= Ez 0 ,ε,t,τ
```
```
h
∥u(zt,t,τ )− vt∥^2
```
```
i
```
. (4)

```
Inference. We employ classifier-free guidance [32], al-
lowing the model to learn both conditional and uncondi-
tional distributions by randomly dropping the conditioning
during training.
At inference, the guided velocity prediction is given by
```
```
ˆu(zt,t,τ ) = u(zt,t, ∅) + ω
```
##### 

```
u(zt,t,τ )− u(zt,t, ∅)
```
##### 

##### ,

##### (5)

```
where ω > 0 controls the conditioning strength, and
u(zt,t, ∅) denotes the unconditional prediction.
RIRs are generated by solving the ordinary differential
equation (ODE) backward, starting from Gaussian noiseε
and integrating the velocity field from t = 1 to t = 0:
zt−dt= zt+ ˆu(zt,t,τ )dt. (6)
```

```
Figure 3. FLAC diffusion transformer: The noise timestep t and
the target RIR pose are injected via AdaLN. Acoustic, spatial and
geometric context are provided through cross-attention.
```
#### 3.2. VAE

We train a variational autoencoder (VAE) to compress RIR
waveforms into z 0. The encoder consists of four convolu-
tional blocks, each performing downsampling and channel
expansion with strided convolutions. Before each down-
sampling block, we apply ResNet-style layers with dilated
convolutions and Snake activations [91]. The bottleneck has
a latent feature dimension of 32, and the decoder mirrors the
encoder. All convolutions are weight-normalized [72] and
the output passes through a tanh activation to match the
RIR amplitude range.
We found pre-trained audio embeddings unsuitable for
latent flow matching. Obtaining a compact RIR represen-
tation is challenging as it must preserve precise temporal
and spectral structure. To achieve this, we train the VAE
with complementary objectives: a multiresolution STFT
lossLMR[77, 84] combining spectral convergence, spectral
and energy decay terms; an adversarial hinge loss Ladv; a
feature-matching loss Lfeatusing Encodec [15] multi-scale
STFT discriminator; and a KL divergence lossLKLto regu-
larize the latent space. The final objective is:

```
L =LMR+Ladv+Lfeat+LKL (7)
```
```
Details on the implementation, individual loss terms, and
hyperparameters are provided in Appendix A.
```
#### 3.3. Multimodal Conditioning

```
FLAC generates RIRs at a target source-receiver pair
(PsT,PrT) based on multimodal scene contextτ :
```
- Acoustic: RIRs measured at the target receiver PrTfrom
    K different source positions, A = {A^1 ,...,AK}, cap-
    turing key room acoustic properties.
- Spatial: Corresponding source positions S =
    {Ps^1 ,...,PsK} and the target source position PsT.
- Geometric: A panoramic depth map GTrcaptured at the
    target receiver pose PrT, describing local room structure
    and surfaces.
Below, we detail how each modality is processed.

```
Acoustic. Similar to [52, 57] each of the K context RIRs
is transformed into a magnitude spectrogram and encoded
with a ResNet-18 backbone [30], trained jointly with the
rest of the model. The encoder outputs a 512-dimensional
embedding per RIR, capturing key acoustic properties.
```
```
Spatial. Since the receiver is shared between the con-
text and target RIRs, we express all source poses in the
receiver’s local coordinate frame and omit PrT (the ori-
gin). The resulting 3D coordinates are encoded with si-
nusoidal positional embeddings and projected into a high-
dimensional feature space through a linear layer.
```
```
Geometric. We condition on the geometry surrounding
the receiver to capture the location and shape of nearby sur-
faces. A panoramic depth map captured at PrTis converted
into an image containing 3D coordinates via equirectangu-
lar projection. Following [52], we compute reflection maps
by subtracting each source position (target and K context)
expressed in the receiver’s frame from these 3D coordinates.
DINOv3 [73] Vision Transformer (ViT) [16] S/16 is fine-
tuned to encode the reflection maps into compact features
capturing geometric structure and spatial relationships. An
overview of the geometry module is given in Appendix B.2.
```
#### 3.4. Diffusion Transformer

```
Inspired by recent advances in image and audio generation
[19, 20, 36, 43, 63], we parameterize the velocity fieldbvt
using a diffusion transformer (DiT) illustrated in Fig. 3. It
consists of a multi-layer Transformer architecture. A 1D
convolution followed by a linear layer maps between the
VAE latent space and the transformer embedding dimen-
sion. Each transformer block follows a fixed sequence: self-
attention with Rotary Positional Embedding (RoPE) [78],
followed by cross-attention over conditioning tokens and
a feedforward network (FNN), with residual connections
applied inside each sub-layer. We compute d-dimensional
Fourier features of the noise timestep t. The global con-
ditioning, containing the target pose and t, is incorporated
via Adaptive Layer Norm (AdaLN), where learned scale,
shift and gating parameters modulate both self-attention and
```

Figure 4. AGREE contrastive framework: Audio and geometry
inputs are encoded into a shared latent space, where a contrastive
objective maximizes similarity for matching pairs (diagonal en-
tries) and minimizes it for mismatched ones.

feedforward layers. Acoustic, spatial and geometric con-
text are incorporated through cross-attention. Finally, the
model consists of 12 transformer blocks with 8 heads and
a hidden width of 256. We train it with a learning rate of
5 × 10 −^5 , AdamW optimizer [53] and a batch size of 64 on
a single H100 GPU. We use an Exponential Moving Aver-
age (EMA) of the model weights during training and BF
precision. FLAC number of parameters and inference time
are reported in the Appendix H.

### 4. AGREE: Acoustic–Geometry Embedding

We introduce AGREE (Acoustic-GeometRy EmbEding), a
CLIP-style [65] multimodal embedding that aligns room
acoustics and geometry (see Fig. 4). The audio encoder
is fine-tuned from the pre-trained VAE encoder used in
FLAC (see Sec. 3.2). The geometry encoder is DINOv
ViT-S/16 fine-tuned on panoramic depth maps captured at
receiver positions. Following FLAC’s geometry pipeline,
depth values are projected into 3D coordinates and source
positions, expressed in the receiver frame, are subtracted.
Each encoder is followed by a linear projection, and both
are trained jointly with a contrastive objective to align
acoustic and geometric representations. The resulting em-
bedding space captures spatial-acoustic consistency, en-
abling zero-shot cross-modal retrieval and geometry-aware
evaluation. AGREE details are provided in Appendix C.

### 5. Experiments

#### 5.1. Datasets

AcousticRooms. We use the AcousticRooms (AR)
dataset [52], a large-scale simulated dataset of monaural
RIRs paired with equirectangular panoramic depth maps.
It spans 260 rooms across 10 categories with diverse ge-

```
ometries, sizes, and materials, totaling over 300k simu-
lated RIRs at 22,050 Hz. Generated with Treble Technol-
ogy’s wave-based simulation, it provides high simulation
accuracy beyond geometric or ray-tracing methods used in
[5, 8, 80]. Following [52], we split the dataset into 243 seen
and 17 unseen rooms to evaluate both in-room prediction
and generalization to new scenes. The unseen test set con-
tains 5,244 instances. A subset of the seen-room instances
is used for evaluation, it contains 6,217 instances across 131
rooms. In all our experiments, our VAE is pretrained on this
dataset.
```
```
Hearing-Anything-Anywhere. To evaluate generaliza-
tion to real-world environments, we use the Hearing-
Anything-Anywhere (HAA) dataset [83]. It provides
monaural RIRs recorded in four rooms, each with a fixed
source and multiple receiver positions. This setup is the
inverse of AcousticRooms, where the receiver is fixed and
the source varies. However, for single-channel RIRs, inter-
changing source and receiver is equivalent due to the sym-
metry of the wave equation [52]. All RIRs are sampled
to 22,050 Hz. Panoramic depth maps at each source pose
are derived from room meshes reconstructed using wall and
surface annotations. Appendix D.1 gives datasets details.
```
#### 5.2. Metrics

```
Perceptual metrics. We assess the perceptual quality of
generated RIRs using standard acoustic metrics [4, 44, 52,
57, 79] that correlate with human auditory perception. We
report the relative T60 error, normalized by the ground-
truth. T60 measures the reverberation time i.e., the dura-
tion for sound energy to decay by 60 dB. We also compute
the clarity error based on C50, the ratio of early-to-late en-
ergy, indicative of speech intelligibility and acoustic clar-
ity. Finally, we evaluate the Early Decay Time (EDT) error,
which capture early reflection characteristics by measuring
the time for an initial 5 dB energy decay.
```
```
Scene-consistency metrics. We introduce metrics based
on the AGREE embedding space to evaluate how well gen-
erated RIRs reflects the spatial characteristics of the envi-
ronment. We compute audio-to-audio recall (R@1/5/10),
quantifying how closely generated and ground-truth RIRs
align in this geometry-aware space. To capture overall
realism, we compute the Frechet distance FD ́ G, between
the distribution of generated and real audio embeddings in
AGREE space, analogous to the FID [31] used in image
generation. For reference, AGREE zero-shot retrieval re-
sults on the unseen AcousticRooms set are summarized in
Tab. 2. To maximize retrieval performance when evaluat-
ing RIR synthesis methods, we also train AGREE on the
entire AR dataset. Further details on the scene-consistency
metrics can be found in Appendix D.3.
```

Table 1. Performance on unseen AcousticRooms scenes: Results are shown for K ∈ { 8 , 1 ,✗} reference RIRs. For FLAC, we report
mean and standard deviation over 5 generations. FLAC outperforms all baselines even in the one-shot setting.ôdenotes ablations with
either geometry (G) or audio conditioning removed.

```
Method K G T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@1 (%)↑ R@5 (%)↑ R@10 (%)↑ FDG↓
Random Across Rooms ✗ ✗ 44.73 7.676 306.29 0.02 0.06 0.32 0.
Random Same Room ✗ ✗ 17.36 5.490 168.17 0.25 1.09 2.16 0.
FLACô ✗ ✓ 23.41± 0. 02 2.554± 0. 002 109.75± 0. 09 5.12± 0. 12 16.47± 0. 14 23.12± 0. 14 0.
Nearest Neighbor 1 ✗ 15.22 5.212 157.94 0.00 2.26 4.56 0.
Fast-RIR 1 ✓ 18.97 3.257 121.21 0.17 0.66 1.64 0.
xRIR 1 ✓ 14.47 1.961 74.45 0.28 1.36 2.59 0.
FLAC 1 ✓ 9.95± 0. 05 1.046± 0. 002 40.04± 0. 22 6.80± 0. 11 18.92± 0. 10 26.87± 0. 19 0.
Linear Interpolation 8 ✗ 14.45 3.503 114.27 0.41 2.30 4.02 0.
Nearest Neighbor 8 ✗ 10.91 2.792 90.08 0.00 10.26 17.28 0.
FLACô 8 ✗ 12.07± 0. 01 4.296± 0. 001 140.04± 0. 04 0.09± 0. 01 0.58± 0. 06 1.06± 0. 04 0.
Fast-RIR 8 ✓ 17.71 3.253 121.21 0.24 0.99 1.88 0.
xRIR 8 ✓ 9.98 1.354 49.40 0.54 2.00 3.38 0.
FLAC 8 ✓ 8.60± 0. 01 0.970± 0. 002 37.13± 0. 02 6.99± 0. 13 19.38± 0. 15 27.21± 0. 17 0.
```
Table 2. Zero-shot cross-modal retrieval on the unseen
AcousticRooms set: We report acoustic-to-geometry (A2G) and
geometry-to-acoustic (G2A) recall at 1, 5 and 10. † indicates train-
ing on the full dataset for benchmarking few-shot methods.

```
Method R@1↑ R@5A2G↑ R@10↑ R@1↑ R@5G2A↑ R@10↑
AGREE 59.78 83.53 89.35 59.10 85.56 91.
AGREE† 85.37 99.70 99.98 84.38 99.53 99.
```
#### 5.3. Baselines

We compare FLAC against several baselines:

- Random Across Rooms: randomly samples a RIR from
    the entire dataset.
- Random Same Room: randomly selects a RIR from the
    same room.
- Linear Interpolation: linearly interpolates K reference
    RIRs based on their distances to the target source.
- Nearest Neighbor (KNN): chooses the RIR closest in dis-
    tance to the target source among the K references.
- Fast-RIR [66]: generates RIRs with a GAN conditioned
    on T60 and scene size estimated from K RIRs and depth.
- xRIR [52]: combines acoustic and geometric features to
    weight K reference RIRs.

#### 5.4. Inference parameters

In all experiments, we use a guidance scale of 1 and perform
generation in a single inference step as it achieves the best
results on the perceptual metrics (T60, C50, EDT). These
metrics mainly capture global acoustic properties, such as
energy decay and clarity, but are insensitive to fine-grained
details or sample diversity. Thus, additional steps offer no
benefit. However, as shown in Fig. 5, increasing the guid-
ance weight or the number of steps improves FDG.

```
Figure 5. Impact of classifier-free guidance and inference
steps: Evolution of T60 and FDGas a function of the guidance
scale ω and the number of timesteps.
```
#### 5.5. Results

```
8-shot generation in novel environments. Quantitative
results on unseen scenes with K=8 are reported in Tab. 1.
FLAC consistently outperforms xRIR across all metrics,
reducing errors by 13 .8% T60, 28 .3% C50, and 24 .9%
EDT, and achieving higher audio-to-audio recall, indicat-
ing more geometry-consistent acoustic synthesis. For FDG,
FLAC slightly surpasses xRIR, reflecting improved dis-
tributional realism. Increasing the number of inference
steps or the classifier-free guidance weight further improves
FLAC FDG; for example, 20 steps reduce it to 0.280 (see
Fig. 5). KNN always achieves lower FDGas it simply re-
turns a reference RIR, which is already drawn from the true
distribution. For seen rooms, detailed results are provided
in Appendix E.1: FLAC reduces errors by 23 .9%, 29 .8%,
and 24 .8% for T60, C50, and EDT, respectively. These re-
sults demonstrate that FLAC not only improves RIR estima-
tion at new positions within seen spaces but also generalizes
more effectively to new environments.
Robustness under limited observations. We evaluate
methods robustness with fewer context RIRs, simulating
scenarios with limited recordings. For FLAC and xRIR,
models trained with K=8 and tested with fewer references.
```

Figure 6. Robustness to limited context RIRs in novel scenes:
Performance as the number of reference RIRs (K) decreases for
KNN, xRIR, and FLAC. FLAC remains the most stable and out-
performs state-of-the-art methods with K=8 even in one-shot.

Figure 7. Octave-band analysis of 4 RIRs in unseen rooms: 100
samples per instance are generated with FLAC. The mean and± 3
standard deviation (covering 99.7% of the distribution) are shown.
Std increases at low frequencies.

As shown in Tab. 1, FLAC maintains state-of-the-art per-
formance in the one-shot setting, surpassing prior methods
using eight recordings. Fig. 6 shows that FLAC remains sig-
nificantly more stable than KNN and xRIR as K decreases.
Recall metrics are little affected by reduced acoustic obser-
vations, indicating that geometry provides the dominant cue
for geometry-consistent RIR synthesis.

Capturing uncertainty of few-shot RIR synthesis. We
study variability by generating 100 RIRs per conditioning,
each produced with a different noise input. As shown by
the octave-band analysis in Fig. 7, samples standard devi-
ation increases at low frequencies. These bands also ex-
hibit longer uncertainty persistence time, defined as the time
until band-wise sample variance drops below the 75th per-
centile (see Fig. 8). This matches room acoustics theory:
low-frequency responses are governed by sparse, boundary-
dependent modes that are weakly constrained by limited
context, whereas above the Schroeder frequency dense
mode yields stable responses constrained by local geome-
try. This indicates that FLAC captures the inherent uncer-
tainty of underconstrained few-shot settings. A determin-
istic variant (fixed noise) degrades performance (+6% T60,

```
Table 3. Sim-to-real transfer to the Hearing-Anything-
Anywhere dataset: Few-shot methods are compared against
Diff-RIR and INRAS, which require per-scene training (†). For
FLAC, we report mean and standard deviation over 5 generations.
With K=8, FLAC matches or exceeds xRIR and Diff-RIR on per-
ceptual metrics, and with one-shot, it outperforms KNN and xRIR.
```
```
Method K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
Random Across Rooms ✗ 17.40 10.283 533.99 1.49 0.
Random Same Room ✗ 8.00 4.805 180.15 1.86 0.
Nearest Neighbor 1 8.19 5.000 187.55 1.20 0.
xRIR 1 8.63 4.862 183.27 14.85 0.
FLAC 1 3.45± 0. 02 2.170± 0. 014 90.02± 0. 24 17.94± 0. 62 0.
Linear Interpolation 8 4.12 2.695 88.19 3.62 0.
Nearest Neighbor 8 2.89 1.923 77.24 9.61 0.
xRIR 8 6.53 3.492 149.69 20.65 0.
FLAC 8 3.10± 0. 01 2.167± 0. 004 84.52± 0. 24 17.41± 0. 59 0.
INRAS† 12 6.61 3.966 158.07 2.27 0.
Diff-RIR† 12 3.74 2.067 88.09 26.97 0.
```
```
Figure 8. Uncertainty persistence time and band-wise energy
decay, averaged over 100 unseen samples. Uncertainty lasts longer
at low frequencies and decays faster at high frequencies.
```
```
+10% C50, -40% R@5), confirming that stochasticity is es-
sential. Quantitatively, FLAC’s intra-conditioning diversity
```
is 1. (^03) ± 0. 20 vs. 22. 96 between conditionings (a 4.5% ra-
tio), showing that FLAC introduces meaningful stochastic-
ity while remaining consistent with the context. See Ap-
pendix F for a t-SNE visualization.
Sim-to-real transfer. We evaluate real-world generaliza-
tion on the HAA dataset [83]. Baselines include Diff-RIR
[83], a physics-based differentiable renderer, and INRAS
[79], a novel-view acoustic synthesis method, both trained
with 12 references per room to predict RIRs at new lo-
cations. Unlike few-shot models, they must be retrained
separately for each room, requiring hours of training. Fol-
lowing [52], we fine-tune xRIR and FLAC. Note that we
do not fine-tune FLAC’s VAE. Few-shot models adapt to
all four rooms within minutes. For evaluation, AGREE is
also fine-tuned on HAA. As shown in Tab. 3, with eight
shots, FLAC outperforms xRIR and surpasses Diff-RIR on
most perceptual metrics, despite using fewer references and
no room-specific training. While 8-NN performs strongly,
it copies existing RIRs, lacking spatial continuity (audi-
ble ”jumps”). Remaining discrepancies likely stem from:
(i) HAA’s simplified geometry annotations (e.g., tables as
single planes); and (ii) the VAE not being fine-tuned on
real recordings, which may cause its latent representation
to miss certain acoustic phenomena. The small size of
HAA proved insufficient for stable adaptation of the VAE.


Table 4. Impact of geometry and acoustic encoders: Perfor-
mance on unseen AcousticRooms scenes using different configu-
rations of the geometry φGand acoustic φAencoders. We com-
pare xRIR’s ViT and DINOv3 ViT-S/16 with three initialization
strategies: trained from scratch, frozen, or fine-tuned (WDINO). For
φA, we evaluate the ResNet-18 and our frozen VAE encoder.

```
φG φA K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
ViT WDINO
[52]✗ ResNet 1 10.91 1.166 42.41 11.17 0.
S/16 [73]✗ ResNet 1 10.81 1.090 42.11 15.40 0.
S/16 [73]^ ResNet 1 10.42 1.427 51.79 5.29 0.
S/16 [73] ✓ ResNet 1 9.95 1.046 40.04 18.9 0.
S/16 [73]✓ VAE 1 9.40 1.057 39.31 17.11 0.
[52]✗ ResNet 8 9.46 1.063 39.57 11.80 0.
S/16 [73]✗ ResNet 8 9.29 0.994 38.61 16.24 0.
S/16 [73]^ ResNet 8 8.87 1.298 46.41 5.92 0.
S/16 [73] ✓ ResNet 8 8.60 0.970 37.13 19.38 0.
S/16 [73]✓ VAE 8 8.51 0.945 34.70 17.56 0.
```
Table 5. Impact of DiT variants: Performance on unseen Acous-
ticRooms scenes with In-Context, Cross-Attention (CA), and hy-
brid AdaLN+CA conditioning.

```
Method K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
In-Context 1 69.68 11.199 1236.98 0.06 1.
CA 1 15.68 1.750 85.98 6.10 0.
AdaLN+CA 1 9.95 1.046 40.04 18.92 0.
In-Context 8 8.12 1.081 41.97 0.194 0.
CA 8 9.31 1.234 45.81 11.93 0.
AdaLN+CA 8 8.60 0.970 37.13 19.38 0.
```
Yet, FLAC one-shot outperforms both KNN and eight-shot
xRIR, highlighting its advantage in data-scarce conditions.

Perceptual Evaluation. We conducted a listening study
with 46 participants on 14 unseen AR scenes. Participants
were presented with the ground-truth, audio generated by
FLAC (1-shot) and xRIR (8-shot), and were asked to select
which audio sounded closer to the GT. FLAC was preferred
in 93.01% of cases. Details are given in Appendix G.

#### 5.6. Ablation Study

Conditioning modalities. We analyze the impact of each
conditioning modality by removing either geometry or au-
dio (see Tab. 1). When conditioned only on geometry, the
model maintains strong audio-to-audio recall and outper-
forms random RIR prediction, confirming that geometric
cues provide rich information for RIR synthesis. In contrast,
using only audio leads to a drop in geometry-related met-
rics (recall and FDG). For perceptual metrics, geometry-
only achieves higher C50 and EDT but lower T60 compared
to the audio-only version. This aligns with their physical
meaning: C50 and EDT are influenced by early reflections
from nearby surfaces, while T60 captures global reverber-
ation that is harder to infer from local geometry. Overall,
combining both modalities through cross-attention yields
the best results, demonstrating the complementary nature
of geometric and acoustic conditioning.

```
Geometry conditioning encoder. In Tab. 4 we study how
the choice of geometry conditioning encoder affects perfor-
mance. We compare the ViT architecture from xRIR with
DINOv3 ViT-S/16, which have similar parameter counts
(19.8M vs. 21.7M). For DINOv3, we test three variants: (i)
trained from scratch, (ii) frozen pretrained weights, and (iii)
fine-tuned jointly with the model. Even when trained from
scratch, the ViT-S/16 outperforms xRIR’s ViT. As our in-
put differs substantially from RGB images, freezing DINO
weights degrades performance. Fine-tuning DINO yields
the best overall results. We report a similar analysis for the
AGREE geometric encoder in Appendix C.3, where fine-
tuning DINOv3 ViT-S/16 consistently improves zero-shot
retrieval. Note that even with the same conditioning archi-
tecture as xRIR, one-shot FLAC achieves comparable T60,
FDGand higher C50, EDT and R@5 than eight-shot xRIR.
```
```
Acoustic conditioning encoder. We evaluate replacing
the jointly trained ResNet-18 with our frozen, pretrained
VAE encoder (see Tab. 4). The VAE improves cross-room
generalization, though at higher computational cost. For ef-
ficiency, we use the ResNet-18 as the default encoder.
```
```
DiT variants. We study different DiT conditioning strate-
gies (see Tab. 5). In-Context concatenates all condition-
ing information with the input before self-attention. Cross-
Attention applies conditioning solely via cross-attention
layers. Our approach (see Fig. 3) injects target informa-
tion through AdaLN, and contextual information via cross-
attention. AdaLN+CA outperforms alternative designs. Il-
lustrations of variants are provided in Appendix B.3.
```
### 6. Conclusion

```
We introduced FLAC, a generative approach for few-shot
acoustic synthesis based on flow matching. By condition-
ing generation on multimodal few-shot context, FLAC can
synthesize RIRs at arbitrary sensor positions in novel en-
vironments. Our method captures the inherent ambiguity
of few-shot RIR synthesis, an aspect overlooked by exist-
ing deterministic methods. Experiments on two datasets
demonstrated state-of-the-art performance in novel environ-
ments, even with a single reference RIR. We also introduced
AGREE, a joint-embedding space between RIRs and ge-
ometry enabling both zero-shot cross-modal retrieval and
geometry-consistency evaluation. FLAC produces RIRs
that are both perceptually accurate and consistent with the
scene, an important aspect for immersive virtual experi-
ences. Future work may include supporting multiple sam-
ple rates in a single model, and collecting a larger, more di-
verse real-world audio-visual dataset to improve sim-to-real
transfer. The AGREE embedding could also benefit broader
audio-visual learning tasks.
```

### Acknowledgments

This work was supported by the French Agence Nationale
de la Recherche (ANR), under grant ANR22-CE94-
and was granted access to the HPC resources of IDRIS un-
der the allocation 2024-AD011015475R1 made by GENCI.
We would like to thank Simon de Moreau and the anony-
mous reviewers for their insightful comments and sugges-
tions.

### References

```
[1] Michael Samuel Albergo and Eric Vanden-Eijnden. Build-
ing normalizing flows with stochastic interpolants. In ICLR,
```
2023. 3
[2] Swapnil Bhosale, Haosen Yang, Diptesh Kanojia, Jiankang
Deng, and Xiatian Zhu. AV-GS: Learning material and ge-
ometry aware priors for novel view acoustic synthesis. In
NeurIPS, 2024. 1, 2
[3] Amandine Brunetto, Sascha Hornauer, X Yu Stella, and Fa-
bien Moutarde. The audio-visual batvision dataset for re-
search on sight and sound. In IROS, 2023. 2
[4] Amandine Brunetto, Sascha Hornauer, and Fabien
Moutarde. NeRAF: 3D scene infused neural radiance
and acoustic fields. In ICLR, 2025. 1, 2, 5
[5] Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vi-
cenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu,
Philip Robinson, and Kristen Grauman. Soundspaces:
Audio-visual navigaton in 3d environments. In ECCV, 2020.
5
[6] Changan Chen, Sagnik Majumder, Ziad Al-Halah, Ruohan
Gao, Santhosh Kumar Ramakrishnan, and Kristen Grauman.
Learning to set waypoints for audio-visual navigation. In
ICLR, 2021. 2
[7] Changan Chen, Ruohan Gao, Paul Calamia, and Kristen
Grauman. Visual acoustic matching. In CVPR, 2022. 2
[8] Changan Chen, Carl Schissler, Sanchit Garg, Philip
Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra,
Philip W Robinson, and Kristen Grauman. Soundspaces
2.0: A simulation platform for visual-acoustic learning. In
NeurIPS, 2022. 5
[9] Changan Chen, Wei Sun, David Harwath, and Kristen Grau-
man. Learning audio-visual dereverberation. In ICASSP,
2023. 2
[10] Mingfei Chen and Eli Shlizerman. Av-cloud: Spatial audio
rendering through audio-visual cloud splatting. In NeurIPS,
2024. 1, 2
[11] Ziyang Chen, Shengyi Qian, and Andrew Owens. Sound
localization from motion: Jointly learning sound direction
and camera rotation. In ICCV, 2023. 2
[12] Ziyang Chen, Israel D. Gebru, Christian Richardt, Anurag
Kumar, William Laney, Andrew Owens, and Alexander
Richard. Real acoustic fields: An audio-visual room acous-
tics dataset and benchmark. In CVPR, 2024. 2, 6
[13] Sanjoy Chowdhury, Sreyan Ghosh, Subhrajyoti Dasgupta,
Anton Ratnarajah, Utkarsh Tyagi, and Dinesh Manocha. Ad-
verb: Visually guided audio dereverberation. In ICCV, 2023.
2

```
[14] Jesper Haahr Christensen, Sascha Hornauer, and X Yu Stella.
Batvision: Learning to see 3d spatial layout with two ears. In
ICRA, 2020. 2
[15] Alexandre Defossez, Jade Copet, Gabriel Synnaeve, and ́
Yossi Adi. High fidelity neural audio compression. arXiv
preprint arXiv:2210.13438, 2022. 4, 1
[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale. In ICLR, 2020. 4
[17] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail,
and Huaming Wang. Clap learning audio concepts from nat-
ural language supervision. In ICASSP, 2023. 2
[18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M ̈uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-
fied flow transformers for high-resolution image synthesis.
In ICML, 2024. 2
[19] Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah
Taylor, and Jordi Pons. Long-form music generation with
latent diffusion. arXiv preprint arXiv:2404.10301, 2024. 2,
4
[20] Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah
Taylor, and Jordi Pons. Stable audio open. arXiv preprint
arXiv:2407.14358, 2024. 2, 4
[21] Chuang Gan, Yiwei Zhang, Jiajun Wu, Boqing Gong, and
Joshua B Tenenbaum. Look, listen, and act: Towards audio-
visual embodied navigation. In ICRA, 2020. 2
[22] Ruohan Gao and Kristen Grauman. 2.5 d visual sound. In
CVPR, 2019. 2
[23] Ruohan Gao, Changan Chen, Ziad Al-Halah, Carl Schissler,
and Kristen Grauman. Visualechoes: Spatial image repre-
sentation learning through echolocation. In ECCV, 2020. 2
[24] Ruohan Gao, Hao Li, Gokul Dharan, Zhuzhu Wang, Cheng-
shu Li, Fei Xia, Silvio Savarese, Li Fei-Fei, and Jiajun Wu.
Sonicverse: A multisensory simulation platform for embod-
ied household agents that see and hear. In ICRA, 2023. 2
[25] Rishabh Garg, Ruohan Gao, and Kristen Grauman. Visually-
guided audio spatialization in video with geometry-aware
multi-task learning. IJCV, 2023. 2
[26] Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish,
and Soujanya Poria. Text-to-audio generation using instruc-
tion tuned llm and latent diffusion model. arXiv preprint
arXiv:2304.13731, 2023. 2
[27] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat
Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan
Misra. Imagebind: One embedding space to bind them all.
In CVPR, 2023. 2
[28] Yiwei Guo, Chenpeng Du, Ziyang Ma, Xie Chen, and Kai
Yu. Voiceflow: Efficient text-to-speech with rectified flow
matching. In ICASSP, 2024. 2
[29] Andrey Guzhov, Federico Raue, Jorn Hees, and Andreas ̈
Dengel. Audioclip: Extending clip to image, text and au-
dio. In ICASSP, 2022. 2
[30] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In CVPR,
```
2016. 4


[31] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. Gans trained by a
two time-scale update rule converge to a local nash equilib-
rium. NeurIPS, 2017. 5
[32] Jonathan Ho and Tim Salimans. Classifier-free diffusion
guidance. In NeurIPSW, 2021. 3
[33] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models. In NeurIPS, 2020. 2
[34] Diwei Huang, Kunyang Lin, Peihao Chen, and Qing Du.
Map-guided few-shot audio-visual acoustics modeling. In
ICASSP, 2025. 1, 2
[35] Jiawei Huang, Yi Ren, Rongjie Huang, Dongchao Yang,
Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma,
and Zhou Zhao. Make-an-audio 2: Temporal-enhanced text-
to-audio generation. arXiv preprint arXiv:2305.18474, 2023.
2
[36] Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj
Mehrish, Amir Ali Bagherzadeh, Chuan Li, Rafael Valle,
Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super
fast and faithful text to audio generation with flow match-
ing and clap-ranked preference optimization. arXiv preprint
arXiv:2412.21037, 2024. 2, 4
[37] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade
Gordon, Nicholas Carlini, Rohan Taori, Achal Dave,
Vaishaal Shankar, Hongseok Namkoong, John Miller, Han-
naneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Open-
clip, 2021. 2, 3
[38] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, ̈
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM TOG, 42(4):139–1, 2023. 2
[39] Jaeyeon Kim, Heeseung Yun, and Gunhee Kim. Visage:
Video-to-spatial audio generation. In ICLR, 2025. 2
[40] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan:
Generative adversarial networks for efficient and high fi-
delity speech synthesis. In NeurIPS, 2020. 1
[41] Kundan Kumar, Rithesh Kumar, Thibault De Boissiere, Lu-
cas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre De Bre-
bisson, Yoshua Bengio, and Aaron C Courville. Melgan:
Generative adversarial networks for conditional waveform
synthesis. In NeurIPS, 2019. 1
[42] Sang-gil Lee, Zhifeng Kong, Arushi Goel, Sungwon Kim,
Rafael Valle, and Bryan Catanzaro. ETTA: Elucidating the
design space of text-to-audio models. In ICML, 2025. 2
[43] Mark Levy, Bruno Di Giorgi, Floris Weers, Angelos
Katharopoulos, and Tom Nickson. Controllable music pro-
duction with diffusion models and guidance gradients. arXiv
preprint arXiv:2311.00613, 2023. 4
[44] Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, and
Chenliang Xu. Av-nerf: Learning neural fields for real-world
audio-visual scene synthesis. In NeurIPS, 2023. 1, 2, 5
[45] Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, and
Chenliang Xu. Neural acoustic context field: Rendering re-
alistic room impulse response with neural fields. ICCVW,

2023. 2
[46] Susan Liang, Dejan Markovic, Israel D Gebru, Steven
Krenn, Todd Keebler, Jacob Sandakly, Frank Yu, Samuel
Hassel, Chenliang Xu, and Alexander Richard. Binau-
ralflow: A causal and streamable approach for high-quality

```
binaural speech synthesis with flow matching models. In
ICML, 2025. 2
[47] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maxim-
ilian Nickel, and Matthew Le. Flow matching for generative
modeling. In ICLR, 2023. 2, 3
[48] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu,
Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audi-
oLDM: Text-to-audio generation with latent diffusion mod-
els. In ICML, 2023. 2
[49] Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong,
Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and
Mark D Plumbley. Audioldm 2: Learning holistic audio gen-
eration with self-supervised pretraining. IEEE/ACM Trans-
actions on Audio, Speech, and Language Processing, 2024.
2
[50] Qiang Liu. Rectified flow: A marginal preserving approach
to optimal transport. arXiv preprint arXiv:2209.14577, 2022.
3
[51] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight
and fast: Learning to generate and transfer data with rectified
flow. In ICLR, 2023. 3
[52] Xiulong Liu, Anurag Kumar, Paul Calamia, Sebasti`a
V. Amengual Gar ́ı, Calvin Murdock, Ishwarya Ananthab-
hotla, Philip Robinson, Eli Shlizerman, Vamsi Krishna
Ithapu, and Ruohan Gao. Hearing anywhere in any envi-
ronment. In CVPR, 2025. 1, 2, 4, 5, 6, 7, 8, 3, 9
[53] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101, 2017. 5, 1,
2
[54] Andrew Luo, Yilun Du, Michael Tarr, Josh Tenenbaum, An-
tonio Torralba, and Chuang Gan. Learning neural acoustic
fields. In NeurIPS, 2022. 1, 2
[55] Tanvir Mahmud, Shentong Mo, Yapeng Tian, and Diana
Marculescu. Ma-avt: Modality alignment for parameter-
efficient audio-visual transformers. In CVPR, 2024. 2
[56] Navonil Majumder, Chia-Yu Hung, Deepanway Ghosal,
Wei-Ning Hsu, Rada Mihalcea, and Soujanya Poria.
Tango 2: Aligning diffusion-based text-to-audio genera-
tions through direct preference optimization. arXiv preprint
arXiv:2404.09956, 2024. 2
[57] Sagnik Majumder, Changan Chen, Ziad Al-Halah, and Kris-
ten Grauman. Few-shot audio-visual learning of environ-
ment acoustics. NeurIPS, 2022. 1, 2, 4, 5
[58] Sagnik Majumder, Hao Jiang, Pierre Moulon, Ethan Hen-
derson, Paul Calamia, Kristen Grauman, and Vamsi Krishna
Ithapu. Chat2map: Efficient scene mapping from multi-ego
conversations. In CVPR, 2023. 2
[59] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. CACM, 2021. 2
[60] Pedro Morgado, Nuno Nvasconcelos, Timothy Langlois, and
Oliver Wang. Self-supervised generation of spatial audio for
360 video. NeurIPS, 31, 2018. 2
[61] Pedro Morgado, Nuno Vasconcelos, and Ishan Misra. Audio-
visual instance discrimination with cross-modal agreement.
In CVPR, 2021. 2
```

[62] Kranti Kumar Parida, Siddharth Srivastava, and Gaurav
Sharma. Beyond image to depth: Improving depth predic-
tion using echoes. In CVPR, 2021. 2
[63] William Peebles and Saining Xie. Scalable diffusion models
with transformers. In ICCV, 2023. 4
[64] Senthil Purushwalkam, Sebastia Vicenc Amengual Gari,
Vamsi Krishna Ithapu, Carl Schissler, Philip Robinson, Ab-
hinav Gupta, and Kristen Grauman. Audio-visual floorplan
reconstruction. In ICCV, 2021. 2
[65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In ICML, 2021. 2, 5
[66] Anton Ratnarajah, Shi-Xiong Zhang, Meng Yu, Zhenyu
Tang, Dinesh Manocha, and Dong Yu. Fast-rir: Fast neural
diffuse room impulse response generator. In ICASSP, 2022.
6, 9
[67] Anton Ratnarajah, Sreyan Ghosh, Sonal Kumar, Purva
Chiniya, and Dinesh Manocha. Av-rir: Audio-visual room
impulse response estimation. In CVPR, 2024. 2, 3
[68] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker,
Bunlong Lay, Shinjii Watanabe, Alexander Richard, and
Timo Gerkmann. EARS: An anechoic fullband speech
dataset benchmarked for speech enhancement and derever-
beration. In Interspeech, 2024. 8
[69] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bjorn Ommer. High-resolution image syn- ̈
thesis with latent diffusion models. In CVPR, 2022. 3
[70] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net:
Convolutional networks for biomedical image segmentation.
In International Conference on Medical Image Computing
and Computer-Assisted Intervention. Springer, 2015. 2
[71] Mahnoor Fatima Saad and Ziad Al-Halah. How would it
sound? material-controlled multimodal acoustic profile gen-
eration for indoor scenes. In ICCV, 2025. 2
[72] Tim Salimans and Durk P Kingma. Weight normalization:
A simple reparameterization to accelerate training of deep
neural networks. In NeurIPS, 2016. 4
[73] Oriane Sim ́eoni, Huy V Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, ̈
et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 4,
8, 2, 3, 5
[74] Nikhil Singh, Jeff Mentch, Jerry Ng, Matthew Beveridge,
and Iddo Drori. Image2reverb: Cross-modal reverb impulse
response synthesis. In ICCV, 2021. 2
[75] Arjun Somayazulu, Changan Chen, and Kristen Grauman.
Self-supervised visual acoustic matching. NeurIPS, 2024. 2
[76] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equa-
tions. arXiv preprint arXiv:2011.13456, 2020. 2
[77] Christian J Steinmetz and Joshua D Reiss. auraloss: Audio
focused loss functions in pytorch. In Digital music research
network one-day workshop, 2020. 4, 1

```
[78] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen
Bo, and Yunfeng Liu. Roformer: Enhanced transformer with
rotary position embedding. Neurocomputing, 2024. 4
[79] Kun Su, Mingfei Chen, and Eli Shlizerman. Inras: Implicit
neural representation for audio scenes. In NeurIPS, 2022. 1,
2, 5, 7, 9
[80] Zhenyu Tang, Rohith Aralikatti, Anton Jeran Ratnarajah, and
Dinesh Manocha. Gwa: A large high-quality acoustic dataset
for audio processing. In SIGGRAPH, 2022. 5
[81] David Thery and Brian FG Katz. Anechoic audio and 3d-
video content database of small ensemble performances for
virtual concerts. In ICA, 2019. 8
[82] Arun Balajee Vasudevan, Dengxin Dai, and Luc Van Gool.
Semantic object prediction and spatial sound super-
resolution with binaural sounds. In ECCV, 2020. 2
[83] Mason Long Wang, Ryosuke Sawata, Samuel Clarke, Ruo-
han Gao, Shangzhe Wu, and Jiajun Wu. Hearing anything
anywhere. In CVPR, 2024. 1, 2, 5, 7
[84] Ryuichi Yamamoto, Eunwoo Song, and Jae-Min Kim. Par-
allel wavegan: A fast waveform generation model based on
generative adversarial networks with multi-resolution spec-
trogram. In ICASSP, 2020. 4, 1
[85] Jaeseong You, Dalhyun Kim, Gyuhyeon Nam, Geum-
byeol Hwang, and Gyeongsu Chae. Gan vocoder: Multi-
resolution discriminator is all you need. arXiv preprint
arXiv:2103.05236, 2021. 1
[86] Abdelrahman Younes, Daniel Honerkamp, Tim
Welschehold, and Abhinav Valada. Catch me if you
hear me: Audio-visual navigation in complex unmapped
environments with moving sounds. RA-L, 2023. 2
[87] Wenjie Zhang, Jun Yin, Long Ma, Peng Yu, Xiaoheng Jiang,
Zhen Tian, and Mingliang Xu. Echodiffusion: Waveform
conditioned diffusion models for echo-based depth estima-
tion. In AAAI, 2025. 2
[88] Hang Zhou, Xudong Xu, Dahua Lin, Xiaogang Wang, and
Ziwei Liu. Sep-stereo: Visually guided stereophonic audio
generation by associating source separation. In ECCV, 2020.
2
[89] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3D: A
modern library for 3D data processing. arXiv:1801.09847,
```
2018. 4
[90] Lingyu Zhu, Esa Rahtu, and Hang Zhao. Beyond visual field
of view: Perceiving 3d environment with echoes and vision.
CVPRW, 2022. 2
[91] Liu Ziyin, Tilman Hartwig, and Masahito Ueda. Neural net-
works fail to learn periodic functions and how to fix it. In
NeurIPS, 2020. 4


## Few-shot Acoustic Synthesis with Multimodal Flow Matching

## Supplementary Material

In the supplementary material, we first present details of
our VAE (Sec. A), FLAC (Sec. B), and AGREE (Sec. C).
We then provide additional information on our evaluation
setup, including datasets, metrics, and baselines (Sec. D).
Further results are reported in Sec. E, such as performances
on the seen set of AcousticRooms, on HAA with a reversed
setup, and complementary evaluation metrics. Additional
qualitative examples, including a video, are provided in
Sec. F. We give details about the perceptual evaluation setup
in Sec. G. Finally, we provide the number of parameters and
inference speed of models in Sec. H.

### A. VAE

#### A.1. Training objective

We provide details on each loss used to train the VAE. In
the following, let x andˆx denote the ground truth and pre-
dicted waveforms and X,Xb the magnitudes of their STFT
representations.
We employ a multiresolution STFT loss LMRinspired
by [77, 84]. This loss compares the spectrograms of the
ground-truth and predicted waveforms at m different reso-
lutions using the spectral convergence LSC, log-magnitude
LSLand energy-decay lossesLED:

```
LMR(x,xˆ) =
```
```
Xm
```
```
i=
```
```
LSC(Xi,Xbi)
```
```
+LSL(Xi,Xbi) +LED(Xi,Xbi), (8)
```
```
with
```
```
LSC(Xi,Xbi) =
```
```
||Xi−Xbi||F
||Xi||F
```
##### , (9)

```
LSL(Xi,Xbi) =|| log(Xi+ η)− log(Xbi+ η)|| 1 , (10)
LED(Xi,Xbi) =||10 log 10 E(Xi)− 10 log 10 E(Xbi)|| 1 ,
(11)
```
```
where
```
```
E(Xi) =
```
##### XT

```
k=d
```
##### XF

```
f=
```
```
(Xi(f,k))^2 , 1 ≤ d≤ T. (12)
```
To further improve the quality of the generated samples,
we employ an adversarial hinge loss Ladv and feature
matching loss Lfeat, based on the multi-scale STFT dis-
criminator from Encodec [15]. Multi-scale discriminators
are well suited for capturing structures in audio signals
[40, 41, 85]. The discriminator consists of multiple

```
identically structured networks operating on multi-scaled
complex-valued STFT, with the real and imaginary parts
concatenated. Each sub-network is composed of a 2D
convolutional layer, followed by 2D convolutions with
increasing dilation rates of 1, 2 and 4 in the time di-
mension and a stride of 2 along the frequency axis. A
final 2D convolution with kernel size 3 × 3 and stride
(1, 1) produces the output prediction. We use 5 scales
with STFT window lengths [2048, 1024 , 512 , 256 , 128],
hop lengths [512, 256 , 128 , 64 , 32] and FFT sizes
[2048, 1024 , 512 , 256 , 128].
The losses are expressed as:
```
```
Ladv(x,xˆ) =
```
##### XN

```
n=
```
```
max[0, 1 − Dn(x)] + max[0, 1 + Dn(ˆx)],
```
```
(13)
```
```
Lfeat(x,xˆ) =
```
##### 1

##### NL

##### XN

```
n=
```
##### XL

```
l=
```
```
||Dln(x)− Dln(xˆ)|| 1
mean(||Dln(x)|| 1 )
```
##### , (14)

```
where Dnlis the l-th layer of the n-th discriminator Dn.
Finally, we use a KL divergence loss LKLto regularize
the latent distribution. The KL loss is given by:
```
```
LKL(x) = DKL
```
##### 

```
qE(z|x)∥N (0, I)
```
##### 

##### , (15)

```
where qE(z|x) denotes the encoder’s approximate poste-
rior, μjand σjare the mean and standard deviation of the
j-th dimension of the latent variable z and d is the latent
dimensionality.
```
#### A.2. Implementation details

```
We train the VAE using the AdamW optimizer [53] with a
batch size of 64 on a single H100 GPU. The generator is
optimized with a learning rate of 1. 5 × 10 −^5 , the discrimi-
nator uses 3 × 10 −^5. For the loss terms, all components of
the multiresolution STFT loss are equally weighted, the KL
loss is weighted by 1 × 10 −^4 , the adversarial loss by 0. 1 ,
and the feature matching loss by 5. 0. The code is based on
the stable audio tools library^1.
```
### B. FLAC

#### B.1. Implementation details

```
When training on the synthetic dataset, we apply two forms
of data augmentation to improve robustness: (i) a small ran-
dom time shift of up to 10 samples in the time domain, and
```
(^1) https://github.com/Stability-AI/stable-audio-
tools


Figure A.1. Geometry module pipeline: A panoramic depth map captured at the receiver position is unprojected into a 3D point cloud
using the equirectangular projection P , then reprojected so that each pixel encodes its corresponding 3D coordinates. We subtract the
source pose associated with the RIR (previously projected in the receiver frame). The resulting representation is processed by a ViT
followed by a linear layer to produce geometry-aware features. In HAA, the source and receiver are interchanged.

(ii) the addition of pink noise with a randomly chosen SNR
between 40 and 60 dB.
Our DiT model consists of 12 transformer blocks with
8 heads and a hidden width of 256. We train it using a
flow matching objective using a learning rate of 5 × 10 −^5 ,
AdamW optimizer [53] and a batch size of 64 on a sin-
gle H100 GPU. We use an Exponential Moving Average
(EMA) of the model weights during training and BF16 pre-
cision.

#### B.2. Illustration of the geometry module

We illustrate the geometry module in Fig. A.1.

#### B.3. Illustration of the DiT variants

In Sec. 5.6, we compare our AdaLN+CA DiT architecture
(shown in Fig. 3) with two alternative designs: the In-
Context and the Cross-Attention (CA-only) variants. Both
alternative architectures are illustrated in Fig. A.2.

#### B.4. Conditioning: geometry and materials

Panoramic depth maps do not recover occluded surfaces,
creating ambiguity that directly motivates our stochastic
formulation. While richer geometry (e.g., meshes) could re-
duce this, it introduces heavier assumptions. FLAC (1-shot)
achieves strong performance, and depth can be estimated
from RGB using foundation models, making an RGB + 1-
RIR setup practical.
Similarly, while material-aware modeling have shown to
improve prediction [67, 71], explicit annotations are un-
available in real-world datasets like HAA. Instead, we rely
on implicit cues in conditioning RIRs, shown to capture
room-material properties [52].

### C. AGREE

#### C.1. Training objective

Given a batch B of geometry embeddings G ∈ RB×dand
acoustic embeddings A∈ RB×d, our model is trained using
a symmetric contrastive objective analogous to CLIP [65].
We first compute pairwise similarity logits

```
LG= λ GA⊤, LA= λ AG⊤, (16)
```
```
where λ is a learnable logit scaling parameter. Each row
of LG(resp. LA) contains the similarities between one ge-
ometry (resp. acoustic) embedding and all acoustic (resp.
geometry) embeddings in the same batch. The ground-
truth alignment corresponds to matching indices, y =
(1,...,B), and the loss is defined as
```
```
Lcontrast=
```
##### 1

##### 2

##### 

```
CE(LG, y) + CE(LA, y)
```
##### 

##### , (17)

```
where CE denotes cross-entropy loss. It encourages aligned
geometry-acoustic pairs to have high similarity while push-
ing apart mismatched pairs.
```
#### C.2. Implementation details

```
AGREE operates on waveforms sampled at 22.05 kHz of
length 10,240, matching the training data from Acoustic-
Rooms [52] and HAA [83].
For geometry, we use 256 × 512 panoramic depth maps
captured at each receiver location in AcousticRooms (and
at each source location in HAA). Following FLAC, depth
maps are unprojected via equirectangular projection to ob-
tain 3D points, and projected back in the image space so
each pixel contains 3D coordinates. Then, we subtract the
source pose expressed in the receiver frame (or the receiver
pose for HAA). This provides the geometric context around
the RIR’s recording configuration. Fig. A.1 illustrates this
process.
We jointly fine-tune DINOv3 ViT-S/16 [73] encoder and
our frozen VAE audio encoder pretrained on Acoustic-
Rooms. A linear layer maps each encoder’s output into a
shared 512-dimensional embedding space.
Training uses AdamW [53] with a learning rate of 1 e−^4 ,
cosine decay, 10,000 warm-up steps, and a weight decay
of 0.1. We employ a batch size of 128 and an embedding
dimensionality of 512. The model is trained for 100 epochs.
We base our implementations on OpenCLIP [37]^2.
```
#### C.3. Impact of the geometry encoder

```
We compare several geometry encoder setup. Specifically,
we compare four ViT variants: the encoder from [52], the
```
(^2) https://github.com/mlfoundations/open_clip


```
Figure A.2. DiT architecture variants: In-Context and Cross-Attention–only architectures. The In-Context variant concatenates all
conditioning information with the input before self-attention. In the Cross-Attention variant, conditioning is applied solely via cross-
attention layers.
```
ViT-S/16 implementations from OpenCLIP [37] and DI-
NOv3, and the larger DINOv3 ViT-S+/16. For DINOv3,
we assess the impact of using the pre-trained weights. Zero-
shot retrieval results reported in Tab. A.1.
With no specific weights initialization (WDINO=✗), us-
ing the ViT-S/16 from DINOv3 achieves the best perfor-
mance. Using frozen DINO weights is less effective, likely
because our geometric representation differs significantly
from the RGB images on which DINOv3 was pretrained.
The strongest results are obtained by fine-tuning DINOv
on our task, with ViT-S+/16 providing the best zero-shot
retrieval on most metrics.

Since in this work AGREE is primarily used as a shared
embedding space for evaluating the scene consistency of
generated RIRs, we also train it on the full AcousticRooms
dataset. In this setting, the performance gap between ViT-S
and ViT-S+ becomes negligible, and we therefore adopt the
smaller ViT-S/16 in the main experiments.

#### C.4. AGREE vs. CRIP

AGREE differs from CRIP [67] in three aspects: (i) Local
alignment: CRIP uses RGB images unaligned with the RIR
sensors, whereas AGREE uses pano depth captured at the
receiver location. (ii) Early reflections: AGREE encodes
local surfaces that govern early reflections (while also cap-

```
turing global structure), whereas CRIP primarily correlates
with late reverberation; ablating comparable geometry in
FLAC significantly degrades early-reflection metrics (C50,
EDT, see Tab. 1). (iii) Evaluation role: CRIP is an auxil-
iary training signal, whereas AGREE serves as a geometry-
aware evaluation framework.
```
```
Table A.1. Zero-shot retrieval on the unseen split of the Acous-
ticRooms dataset using different geometry encoders: We re-
port both acoustic-to-geometry (A2G) and geometry-to-acoustic
(G2A) results for several ViT variants. WDINOdenotes DINOv
pre-trained weights. Models marked with†are trained on the full
AcousticRooms dataset.
```
```
φG A2G G2A
ViT WDINO R@1↑ R@5↑ R@10↑ R@1↑ R@5↑ R@10↑
[52]✗ 6.26 17.82 25.69 6.63 19.20 27.
S/16 [37]✗ 26.53 54.96 67.19 27.47 55.78 67.
S/16 [73]✗ 42.26 72.08 81.11 42.32 71.28 80.
S/16 [73]^ 2.00 7.42 12.77 1.94 7.40 13.
S/16 [73] ✓ 59.78 83.53 89.35 59.10 85.56 91.
S+/16 [73]✓ 58.17 85.06 90.72 60.01 85.85 92.
S/16 [73]† ✓ 85.37 99.70 99.98 84.38 99.53 99.
S+/16 [73]† ✓ 85.26 99.68 99.97 84.28 99.78 99.
```

```
Table A.2. Performance on seen AcousticRooms scenes: Results are shown for K ∈ { 8 , 1 ,✗} reference RIRs. For FLAC, we report
mean and standard deviation over 5 generations. FLAC outperforms all baselines even in the one-shot setting.ôdenotes ablations with
either geometry (G) or audio conditioning removed.
```
```
Method K G T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@1 (%)↑ R@5 (%)↑ R@10 (%)↑ FDG↓
Random Across Rooms ✗ ✗ 44.02 6.415 274.43 0.02 0.10 0.21 0.
Random Same Room ✗ ✗ 13.47 3.582 126.28 1.03 3.12 4.49 0.
FLACô ✗ ✓ 24.56± 0. 03 2.464± 0. 003 114.84± 0. 11 4.97± 0. 14 15.63± 0. 17 22.52± 0. 13 0.
Nearest Neighbor 1 ✗ 11.91 3.406 120.72 0.00 6.79 9.96 0.
FastRIR 1 ✓ 15.88 2.152 92.44 0.23 0.89 1.64 0.
xRIR 1 ✓ 10.66 1.476 58.67 0.39 1.72 3.15 0.
FLAC 1 ✓ 6.46± 0. 07 0.692± 0. 004 28.11± 0. 08 8.48± 0. 12 21.54± 0. 20 28.77± 0. 13 0.
Linear Interpolation 8 ✗ 11.30 2.339 86.59 0.87 4.13 7.03 0.
Nearest Neighbor 8 ✗ 8.31 1.802 64.77 0.08 23.08 30.71 0.
FLACô 8 ✗ 10.47± 0. 01 3.500± 0. 001 125.99± 0. 07 0.10± 0. 01 0.41± 0. 04 0.86± 0. 04 0.
FastRIR 8 ✓ 15.19 2.063 86.46 0.14 0.79 1.61 0.
xRIR 8 ✓ 6.99 0.916 34.15 0.48 2.38 4.02 0.
FLAC 8 ✓ 5.32± 0. 01 0.643± 0. 001 25.69± 0. 05 8.52± 0. 09 21.89± 0. 20 29.28± 0. 16 0.
```
### D. Evaluation details

#### D.1. Datasets

```
AcousticRooms. Each RIR in AcousticRooms is sampled
at 22.05 kHz and truncated to 9,600 samples (≈0.435 s).
We compute the magnitude spectrograms of the contex-
tual RIRs before feeding them to the ResNet18 using an
FFT size of 124, a window length of 62, and a hop size of
```
31. The panoramic depth maps are provided at a resolution
of 256×512 and are projected into 3D point clouds using
equirectangular projection.
    The dataset includes randomized material assignments
drawn from 332 materials across 11 categories, ensuring
strong diversity in acoustic behavior even among similar
geometries. As the simulation meshes are untextured, no
RGB data are available.

HAA. RIRs in the HAA dataset are originally sampled at
48 kHz. We downsample them to 22.05 kHz using Librosa’s
resample and truncate them to 9,600 samples to match
our setup. Contextual RIRs are transformed using the same
FFT pipeline as for AcousticRooms. While the dataset does
not provide depth maps, simplified surface annotations are
available^3. We use these to reconstruct a mesh with Open3D
[89] and generate panoramic depth maps at each source po-
sition via Open3D raycasting. Note that, due to the simpli-
fied surface annotations, these depth maps differ substan-
tially from those in AcousticRooms, widening the domain
gap between the datasets. The test set comprises 1,282 in-
stances (Base rooms). To compute metrics, we first average
results within each room and then average across the four
rooms, preventing rooms with more data from dominating

(^3) https : / / github. com / maswang32 /
hearinganythinganywhere
the results. Following [52], we exclude the mean T60 for
the dampened room, as all methods report unusually high
values, likely due to its particular acoustic characteristics.

#### D.2. Perceptual metrics

```
Following [52], we compute metrics on waveform of length
8,000 on the AcousticRooms dataset and 9,600 on the HAA
dataset. We use T60, C50 and EDT errors obtained as fol-
lows, whereˆx is the synthesized RIR waveform:
```
```
T60(xˆ, x) =
```
```
|T60(ˆx)− T60(x)|
T60(x)
```
##### , (18)

```
C50(xˆ, x) =|C50(xˆ)− C50(x)|, (19)
EDT(ˆx, x) =|EDT(xˆ)− EDT(x)|. (20)
```
```
For the AcousticRooms dataset, T60 is estimated based on
T20, fitting the decay between -5 dB and -25 dB and lin-
early extrapolating to 60 dB. For HAA, it is based on T30,
following [52] implementation.
```
#### D.3. Scene-consistency metrics

```
To evaluate how well generated RIRs preserve geometry-
consistent acoustic behavior, we use AGREE, a CLIP-style
audio-geometry joint embedding network. Let φAand φG
denote its acoustic and geometry encoders, mapping audio
x and geometry g into a shared space. We propose met-
rics that both measure instance-level alignment (recall) and
global distributional consistency (Frechet Distance). ́
```
```
Audio-to-audio retrieval. For each generated RIRbxi, we
compute its normalized embeddingAbi= φA(ˆxi) and com-
pare it to the normalized embeddings Aj= φA(xj) of the
ground-truth RIRs. Similarity is measured as:
```
```
sij=Ab⊤iAj. (21)
```

```
The audio-to-audio recall at X (R@X) is then defined as:
```
##### R@X =

##### 1

##### N

##### XN

```
i=
```
##### 1

##### 

```
i∈ TopXj(sij)
```
##### 

##### , (22)

```
where TopXj(sij) returns the indices of the X most similar
ground-truth embeddings for each generated RIR. This met-
ric measures how often the ground-truth RIR corresponding
to a generated sample appears among the top-X most simi-
lar ground-truth embeddings.
```
Acoustic-to-geometry retrieval. We similarly evaluate
the alignment between generated audio and its correspond-
ing room geometry. For each generated RIR embedding
φA(ˆxj), we compute cosine similarities with all ground-
truth geometry embeddings φG(gi). The resulting recall
at X (R@X) measures how often the correct geometry is
retrieved among the top-X most similar embeddings.

Frechet Distance in AGREE space. ́ To assess distribu-
tional realism, we compute a Frechet Distance (FD) be- ́
tween the generated and ground-truth RIR embeddings in
the AGREE space. Let N (μr, Σr) and N (μg, Σg) denote
multivariate Gaussian approximations of the ground-truth
and generated audio embeddings, respectively:

```
μr=
```
##### 1

```
Nr
```
##### X

```
i
```
```
φA(xi),
```
```
Σr=
```
##### 1

```
Nr− 1
```
##### X

```
i
```
```
(φA(xi)− μr)(φA(xi)− μr)⊤.
```
##### (23)

and similarly for μg, Σg. The Fr ́echet Distance is then de-
fined as:

```
FDG=∥μr−μg∥^22 +Tr
```
##### 

```
Σr+ Σg− 2(ΣrΣg)^1 /^2
```
##### 

##### .(24)

```
Lower values indicate a closer match between the distribu-
tions of generated and real RIR embeddings, reflecting bet-
ter geometry-consistent realism.
```
#### D.4. Baselines implementation details

```
Random across rooms. This baseline randomly selects
one RIR from the entire dataset. In HAA, only training
samples are considered.
```
```
Random same room. This baseline randomly selects one
RIR from the same room. In HAA, only training samples
from the same room are considered.
```
Linear interpolation. This baseline interpolates the K
reference RIRs based on their distance to the target source.
For each reference k, the distance rkto the target source
PsTis computed, and the weight is set as wk= 1/(rk+ ε),
normalized to sum to 1. The final RIR is the weighted sum
of the K references.

```
Table A.3. Effect of the geometry and acoustic conditioning
encoders (seen): We evaluate different configurations of the ge-
ometry encoder φGand acoustic encoder φAon the seen set of
the AcousticRooms dataset. For φG, we compare the ViT from
xRIR and DINOv3 ViT-S/16 under various initialization strategies
(WDINO): trained from scratch, frozen, or initialized with DINO
weights. For φA, we experiment with the ResNet-18 used in xRIR
and our frozen VAE encoder.
```
```
φG φA K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
ViT WDINO
[52]✗ ResNet 1 6.77 0.741 30.15 12.9 0.
S/16 [73]✗ ResNet 1 6.83 0.740 30.23 17.65 0.
S/16 [73]^ ResNet 1 7.61 0.956 36.98 6.40 0.
S/16 [73] ✓ ResNet 1 6.46 0.692 28.11 21.54 0.
S/16 [73]✓ VAE 1 6.81 0.692 27.97 20.28 0.
[52]✗ ResNet 8 5.42 0.674 27.48 14.51 0.
S/16 [73]✗ ResNet 8 5.60 0.685 27.51 19.24 0.
S/16 [73]^ ResNet 8 6.30 0.858 32.42 7.62 0.
S/16 [73] ✓ ResNet 8 5.32 0.643 25.69 21.89 0.
S/16 [73]✓ VAE 8 5.74 0.619 24.42 21.12 0.
```
```
Table A.4. Effect of DiT architecture variants (seen): Per-
formance comparison of different conditioning strategies on the
seen set of the AcousticRooms dataset. We report results for In-
Context, Cross-Attention (CA), and our hybrid design combining
AdaLN for target information and CA for contextual cues.
```
```
Method K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
In-Context 1 72.37 11.299 1318.71 0.05 1.
CA 1 14.44 1.411 75.13 7.43 0.
AdaLN+CA 1 6.46 0.692 28.11 21.54 0.
In-Context 8 5.31 0.651 26.58 21.10 0.
CA 8 6.01 0.782 30.16 14.40 0.
AdaLN+CA 8 5.32 0.643 25.69 21.89 0.
```
```
Nearest neighbor (KNN). From the K reference RIRs,
this baseline returns the RIR whose source position is clos-
est (in Euclidean distance) to the target source position.
```
```
Fast-RIR. This baseline uses the authors’ original imple-
mentation. To align with our setup, we infer the room size
from the panoramic depth map captured at the receiver pose
and estimate T 60 using the K contextual RIRs. Specifically,
we compute T 60 for each of the K RIRs and average the re-
sults. Following prior work [57], we incorporate an energy
decay loss to improve performance.
```
```
xRIR. We use the implementation provided by the au-
thors. Following their supplementary material, the Vision
Transformer encoder uses 6 multi-head attention layers (
heads, hidden size 512) with a patch size of 16 × 32. Poses
are encoded using sinusoidal positional embeddings applied
to each 3D coordinate with 20 frequency bins. We set
λ = 0. 01 to balance the STFT and energy-decay losses. We
train xRIR on the AcousticRooms training split, excluding
both the seen and unseen test sets, and use the same trained
model for evaluating both splits. For HAA, we fine-tune the
AcousticRooms-pretrained model on the four HAA rooms
at the same time.
```

Table A.5. Acoustic-to-geometry retrieval on the AcousticRooms dataset: Results are shown for K ∈ { 8 , 1 ,✗} reference RIRs. For
FLAC, we report mean and standard deviation over 5 generations.ôdenotes ablations with either geometry (G) or audio conditioning
removed. FLAC achieves higher acoustic-to-geometry recall than the baselines, demonstrating higher scene consistency.

```
Method K G
Unseen Seen
R@1↑ R@5↑ R@10↑ R@1↑ R@5↑ R@10↑
Random Across Rooms ✗ ✗ 0.03 0.13 0.21 0.00 0.05 0.
Random Same Room ✗ ✗ 0.22 0.98 1.78 0.88 2.45 3.
FLACô ✗ ✓ 5.12± 0. 15 16.10± 0. 19 24.01± 0. 12 5.58± 0. 10 16.18± 0. 12 23.09± 0. 20
Nearest Neighbor 1 ✗ 0.05 2.04 4.09 0.21 5.65 8.
Fast-RIR 1 ✓ 0.16 0.79 1.52 0.15 0.82 1.
xRIR 1 ✓ 0.19 1.22 2.38 0.16 1.27 2.
FLAC 1 ✓ 5.60± 0. 06 16.54± 0. 24 24.32± 0. 37 6.37± 0. 08 17.71± 0. 21 25.05± 0. 10
```
```
Linear Interpolation 8 ✗ 0.28 1.40 2.87 0.64 2.85 5.
Nearest Neighbor 8 ✗ 0.16 8.66 15.20 0.68 19.21 25.
FLACô 8 ✗ 0.15± 0. 01 0.45± 0. 03 0.84± 0. 07 0.13± 0. 01 0.43± 0. 03 0.77± 0. 03
Fast-RIR 8 ✓ 0.17 1.09 1.82 0.16 0.77 1.
xRIR 8 ✓ 0.43 1.85 3.01 0.37 1.56 2.
FLAC 8 ✓ 5.79± 0. 07 16.84± 0. 08 24.62± 0. 17 6.53± 0. 14 18.10± 0. 15 25.58± 0. 15
```
INRAS. Similar to [12], we modify the original bounce
point sampling strategy, which only sampled points at
a fixed height. Instead, we construct scene meshes
from the HAA surface annotations and apply Pois-
son sampling to obtain 256 3D bounce points, pro-
viding a richer representation of the scene geometry.
As the training set contains only 12 RIRs per room,
we train per scene with a batch size of 12 for 5k
epochs. We found that adjusting the multi-resolution STFT
loss parameters improves performance, using FFT sizes
{ 128 , 512 , 1024 , 2048 }, hop lengths { 16 , 50 , 120 , 240 },
and window lengths{ 80 , 240 , 600 , 1200 }.

DiffRIR. We use the official implementation and adapt
it to 22,050 Hz. Specifically, we reduce the maximum
predicted audio length from 96,000 samples at 48 kHz to
44,100 samples at 22.05 kHz. We use the authors’ pre-
computed sound trajectories and rescale the delay values to
match the new sample rate.

### E. Additional results

#### E.1. Seen set of the AcousticRooms dataset

Comparison to the baselines. Tab. A.2 reports results
on the seen set of the AcousticRooms dataset, which con-
tains novel source-receiver positions within scenes ob-
served during training. With K=8, FLAC reduces er-
rors by 23 .9%, 29 .8%, and 24 .8% on T60, C50, and
EDT, respectively, compared to xRIR. It also substantially
improves scene-consistency metrics and slightly improves

```
FDG(− 8 .8%) over xRIR. Consistent with the unseen-set
results, FLAC with K=1 outperforms all other one-shot
methods and even surpasses xRIR with K=8.
```
```
Ablation of conditioning modalities. Tab. A.2 reports
FLAC variants with one conditioning modality removed (ô).
The trends mirror those observed on the unseen set. Re-
moving geometry leads to large drops in geometry-related
metrics (R@1-10, FDG) even with K=8 contextual RIRs.
Conversely, using geometry without contextual RIRs still
leads to satisfactory performance. C50 and EDT are bet-
ter with geometry-only than with context-only conditioning,
consistent with their dependence on early reflections, which
are closely tied to local scene geometry. T60 is better with
audio-only conditioning, aligning with its dependence on
global room characteristics that are difficult to infer from
local geometry alone.
```
```
Ablation on geometry and acoustic encoders. Tab. A.
compares FLAC with different geometry and acoustic
encoders on AcousticRooms’seen set. Consistent with
the unseen-set results, fine-tuning the DINOv3 ViT S/
achieves the best overall performance. Using our frozen
VAE as the acoustic encoder improves C50 and EDT but
slightly degrades other metrics, further motivating the use
of the simpler and more efficient ResNet-18 for acoustic
conditioning.
```
```
Influence of the DiT architecture. Consistent with re-
sults on the unseen set of the AcousticRooms dataset, the
```

Table A.6. Sim-to-real transfer on the Hearing-Anything-
Anywhere dataset with inverted receiver pose: Few-shot meth-
ods are compared to Diff-RIR and INRAS, which require per-
scene training (†). For FLAC, we report the mean and standard
deviation over 5 generations. Inverting the receiver pose improves
T60/C50/EDT but reduces geometry-consistency metrics.

```
Method K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
Random Across Rooms ✗ 17.40 10.283 533.99 2.02 0.
Random Same Room ✗ 8.00 4.805 180.15 1.51 0.
Nearest Neighbor 1 8.19 5.000 187.55 1.32 0.
xRIR 1 8.63 4.862 183.27 8.43 0.
FLAC 1 3.02± 0. 04 1.676± 0. 023 75.14± 0. 98 10.31± 1. 10 0.
Linear Interpolation 8 4.12 2.695 88.19 8.24 0.
Nearest Neighbor 8 2.89 1.923 77.24 12.64 0.
xRIR 8 6.53 3.492 149.69 10.93 0.
FLAC 8 2.87± 0. 04 1.595± 0. 017 69.14± 0. 56 10.31± 0. 75 0.
INRAS† 12 6.61 3.967 158.07 2.79 1.
Diff-RIR† 12 3.74 2.067 88.09 18.03 0.
```
combination of AdaLN for target-related conditioning and
cross-attention for context leads to the best performance
(see Tab. A.4).

#### E.2. Reversed setup in HAA

The HAA setup is reversed compared to AR. In HAA,
context references share the same source, whereas in AR
they share the same receiver. Consequently, the panoramic
depth is captured at the source pose in HAA and at the re-
ceiver pose in AR. In the main paper, we fine-tune FLAC
on HAA by simply swapping source and receiver poses
(same for AGREE). We also evaluated inverting the re-
ceiver projection. This modification significantly improves
T60/C50/EDT metrics but reduces scene consistency met-
rics. In this setting, AGREE used for evaluation is fine-
tuned with the same modification. Results are reported in
Tab. A.

#### E.3. Additional metrics

Acoustic-to-geometry retrieval. To complement the
audio-to-audio retrieval metrics presented in the main docu-
ment, we provide acoustic-to-geometry retrieval metrics in
Tab. A.5 for both the seen and unseen sets of the Acoustic-
Rooms dataset. FLAC produces more geometry-consistent
RIRs as demonstrated by its superior recall performances.

MAG and ENV on HAA. We report MAG and ENV met-
rics on the HAA dataset in Tab. A.7. Following [83], MAG
denotes the multiscale log-spectral L1 distance, which com-
pares generated and ground-truth waveforms in the time-
frequency domain across multiple resolutions. ENV is
the envelope distance, defined as the L1 distance between
the log-energy envelopes of the generated and ground-truth
waveforms. The energy decay envelope captures the decay
characteristics of a RIR, reflecting the reverberant proper-
ties of a room. For both metrics, we follow diffRIR im-

```
Table A.7. Additional metrics on the Hearing-Anything-
Anywhere dataset: Few-shot methods are compared against
Diff-RIR, which requires per-scene training (†) using MAG and
ENV error.
```
```
Method K MAG↓ ENV↓
Random Across Rooms ✗ 6.97 1.
Random Same Room ✗ 3.22 0.
Nearest Neighbor 1 3.24 0.
xRIR 1 3.97 0.
FLAC 1 3.16 0.
Linear Interpolation 8 3.63 0.
Nearest Neighbor 8 2.88 0.
xRIR 8 3.44 0.
FLAC 8 3.11 0.
INRAS† 12 3.30 0.
Diff-RIR† 12 2.53 0.
```
```
plementation. Consistent with other metrics, FLAC outper-
forms xRIR at K=8, and surpasses others at K=1.
```
#### E.4. Timestep sampling strategy

```
In Tab. A.8, we compare the effect of different training
timestep sampling strategies on performance: (i) LogSNR,
our baseline, which emphasizes higher t values; (ii) Uni-
form, sampling t ∼ U (0, 1); (iii) Ones, fixing t = 1
(fully noisy); and (iv) Logit-Normal, where t = σ(α) with
α∼N (0, 1), concentrating t in the mid-range.
```
```
Table A.8. Comparison of timestep sampling strategies during
training on the generation performance: LogSNR emphasizes
high t values, Logit-Normal concentrates on intermediate values,
Ones fixes t = 1 (i.e., full noise), and Uniform samples t uni-
formly. LogSNR provides the best overall trade-off between seen
and unseen performance.
```
```
Timestep Sampler T60 (%)↓ C50 (dB)Unseen↓ EDT (ms)↓ T60 (%)↓ C50 (dB)Seen↓ EDT (ms)↓
Uniform 8.97 0.943 38.18 5.22 0.636 25.
Ones 8.17 1.044 37.48 5.29 0.636 25.
Logit-Normal 9.77 1.068 42.65 5.80 0.681 27.
LogSNR 8.60 0.970 37.13 5.32 0.643 25.
```
### F. Qualitative results

```
t-SNE. Fig. A.3 provides a t-SNE visualization of multi-
ple generated samples across unseen rooms. Samples with
the same conditioning cluster tightly, while those from dif-
ferent rooms or conditionings are clearly separated, further
demonstrating that the model captures both consistency and
diversity in its generations. We observe that acoustically
similar scenes lies next to the other, e.g., Restaurants-Cafes,
Auditoriums-Listening Rooms, and Apartments (which in-
clude bathrooms)-standalone Bathrooms.
```
```
Octave-band analysis. We provide more visualization of
octave-band analysis in Fig. A.5, Fig. A.6.
```

Figure A.3. t-SNE visualization of generated RIRs across unseen rooms: Each color represents a different room category. Samples
cluster tightly within the same conditioning and remains well separated across conditionings and rooms, confirming consistent yet diverse
RIR generation.

Waveforms. We present in Fig. A.7 and Fig. A.8 gener-
ated waveforms with xRIR and FLAC against the ground
truth for different scenes of the AcousticRooms and HAA
datasets. The rank metric corresponds to the audio-to-audio
retrieval rank of each prediction against all the ground truth
RIRs of the evaluation set.

Video. We provide in the project page several examples
of audio generated with FLAC using K=8 and K=1 ref-
erence RIRs in unseen scenes from both simulated and real
environments. The anechoic speech samples used for aural-
ization come from the EARS dataset [68], and the music
samples are taken from AVAD-VR [81]. For audio ren-
dering along trajectories in real scenes, we convolve our
predicted single-channel RIR at each point of the trajectory
with a head-related impulse response derived from a prede-
fined head-related transfer function. This produces binaural
RIRs, which are then convolved with the source audio to
obtain the final binaural rendering. The rank metric shown
in the video corresponds to the audio-to-audio retrieval rank
of each prediction against all ground-truth RIRs in the full
unseen set, i.e., the position of the correct ground-truth RIR
in the similarity matrix.

### G. Perceptual evaluation

```
We conducted a listening study with 46 participants on
14 unseen AR scenes. Participants were presented with
the ground-truth (GT), audio generated by FLAC (1-shot)
and xRIR (8-shot), and were asked to select which audio
sounded closer to the GT. FLAC was preferred in 93.01% of
cases. The order of questions and the assignment of meth-
ods to “algorithm A” and “algorithm B” were randomized.
Participants were first shown a top view of the scene in-
cluding the microphone and source positions. They then lis-
tened to the GT audio (”true”), obtained by convolving the
ground-truth RIR with an anechoic signal. Next, they lis-
tened to the two generated samples (“algorithm A” and “al-
gorithm B”), which were produced by convolving the same
anechoic signal with RIRs generated by FLAC or xRIR.
Fig. A.4 shows an example of the user interface used.
For the audio content, we used anechoic speech from the
EARS dataset [68]. Different voices were used across ques-
tions, and some questions included music excerpts from
AVAD-VR [81] to evaluate additional use cases. To ensure
reliable listening conditions, participants were required to
complete the survey on a laptop or desktop computer while
```

wearing headphones. They were also asked to report their
background noise environment: 34 participants reported a
quiet environment and 12 reported some background noise
(possibilities were ”quiet”, ”some background noise”, ”very
noisy”).
A total of 49 participants initially completed the survey.
To screen for potential hearing issues or non-compliance
with headphone use, we asked participants whether they
had any known hearing impairments and whether they were
wearing headphones. Additionally, the first question was a
control trial in which participants had to choose the closer
audio to the GT audio between the same GT audio and
an audio recorded in a completely different scene. Three
participants failed this control question and were excluded
from the analysis, leaving 46 valid participants. Errors on
this control question were correlated with participants re-
porting hearing impairments.
We also collected demographic information. Among the
46 participants, 13 identified as female and 33 as male. The
age distribution was: 3 under 18, 6 between 18-24, 17 be-
tween 25-34, 5 between 35-44, 9 between 45-54, 4 between
55-64, and 2 aged 65 or older.

Figure A.4. User study interface. We created 2 synthetic audio
using FLAC one-shot and xRIR one-shot and ask people which
one matches closely with the ground truth.

### H. Number of parameters and inference speed

```
We report in Tab. A.9 the trainable and total inference pa-
rameter counts of our models, xRIR, Fast-RIR, and INRAS,
along with inference times measured on a single RTX 4090
GPU. Note that INRAS must be trained for each scene, this
results in 240M parameters for the full Acoustic Rooms
dataset. Results are shown for variants with the frozen
VAE encoder (replacing the jointly trained ResNet-18),
with xRIR’s ViT architecture instead of DINOV3 ViT-S/16,
with a DiT depth of 4 instead of 12 (FLAC-S), and for the
VAE used to obtain the latent z 0. Our models achieve real-
time performance, as inference is faster than the duration
of the generated audio. We also compare the performance
of FLAC-S against FLAC and xRIR in Tab. A.10. FLAC-S
(39M), obtained by reducing the DiT depth from 12 to 4,
has a similar number of parameters to xRIR (32M). Despite
this reduction, FLAC-S maintains performance comparable
to FLAC, outperforming xRIR. This indicates that the gains
of our method are not driven by model size.
```
```
Table A.9. Number of trainable parameters along with infer-
ence parameters and speed for our model, its variants, and state-
of-the-art methods
```
. M denotes million.
Model Trainable Param. (M) Inference Param. (M) Speed (ms)
INRAS [79] 1 ×Nscene 1 ×Nscene 1.
Fast-RIR [66] 116 116 0.
xRIR [52] 32.1 32.1 38.
VAE 14.6 14.2 8.
FLAC VAE 38.6 45.7 13.
FLAC ViT [52] 48.5 55.6 13.
FLAC-S 37.2 44.3 7.
FLAC 50.3 57.4 13.

```
Table A.10. Comparison between FLAC-S, FLAC, and xRIR
on the unseen AcousticRooms set: FLAC-S reduces the param-
eter count by 13M compared to FLAC. Despite having a similar
number of parameters to xRIR, it achieves performance compara-
ble to FLAC, outperforming xRIR.
```
```
Method K T60 (%)↓ C50 (dB)↓ EDT (ms)↓ R@5 (%)↑ FDG↓
xRIR 1 14.47 1.961 74.45 1.36 0.
FLAC-S 1 9.71± 0. 04 1.010± 0. 002 39.58± 0. 18 17.80± 0. 11 0.
FLAC 1 9.95± 0. 05 1.046± 0. 002 40.04± 0. 22 18.92± 0. 10 0.
xRIR 8 9.98 1.354 49.40 2.00 0.
FLAC-S 8 8.69± 0. 02 0.942± 0. 001 37.14± 0. 08 17.88± 0. 17 0.
FLAC 8 8.60± 0. 01 0.970± 0. 002 37.13± 0. 02 19.38± 0. 15 0.
```

Figure A.5. Additional octave band analysis on the AcousticRooms dataset: We generate 100 samples per instance in the unseen set
with FLAC under identical conditioning, and plot the mean along with a± 3 standard deviation interval (covering 99.7% of the distribution).
Ground truth and xRIR predictions are shown for comparison.

```
Figure A.6. Octave band analysis on the HAA dataset.
```

Figure A.7. Qualitative comparison of predicted RIR waveforms on the unseen set of the AcousticRooms dataset: We compare xRIR
(orange), FLAC (red), and FLAC with K = 1 (purple) against the ground truth (green).


Figure A.8. Qualitative comparison of predicted RIR waveforms on the HAA dataset: We compare xRIR (orange), FLAC (red), and
FLAC with K = 1 (purple) against the ground truth (green).


