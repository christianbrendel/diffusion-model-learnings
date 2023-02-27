# 📚 Diffusion Model Learnings 📚

Here I am documenting my learning notes and little experiments about diffusion models.


### 📒 Notes

In the `20230100_Notes` PDF I am documenting the basics necessary to understand the theory behind diffusion models. Moreover, you will find a lot of details on how the Vanilla diffusion model works (including all the mathematical derivations) as well as a historic overview of some of the most relevant papers that led to the success of diffusion models (probably not 100% complete though). In particular I am following these papers:
- 2015-03-12: Sohl-Dickstein et al., [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
- 2019-09-12: Song et al., [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)
- 2020-06-19: Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- 2020-10-06: Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- 2021-02-18: Nichol et al., [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- 2021-05-11: Dhariwal et al., [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- 2021-11-01: Ho et al., [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- 2021-12-20: Nichol et al., [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)
- 2021-12-20: Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- 2022-04-13: Ramesh et al., [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)
- 2023-02-10: Zhang et al., [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

Additionally I highly recommend these blog posts:
- Lilian Weng, [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- Lilian Weng, [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- Xitong Yang, [Understanding the Variational Lower Bound](https://xyang35.github.io/2017/04/14/variational-lower-bound/)
- Yunfan Jiang, [ELBO — What & Why](https://yunfanj.com/blog/2021/01/11/ELBO.html)
- Paul Rubenstain, [Deriving the variational lower bound](http://paulrubenstein.co.uk/deriving-the-variational-lower-bound/)

### 🧪 Experiments

In the notebooks folder you will find a few notebooks the following topics:

- Notebooks about some basics.
- Notebooks about Stable Diffusion and a copy of the corresponding [repo](https://github.com/Stability-AI/stablediffusion).
- Notebooks about the [diffusers library](https://github.com/huggingface/diffusers) from Hugging Face. 

To play around with the notebooks create the two conda environments from `environment_stable_diffusion.yaml` and `environment_diffusers.yaml` using `conda env create --file <path-to-yaml>`. Use the first environment for all notebooks about the basics and stable diffusion and the latter one for the notebooks on the diffusers library. For the notebooks about stable diffusion (without using diffusors) download the model checkpoint from [here](https://huggingface.co/stabilityai/stable-diffusion-2-1) and put it into the notebooks folder.
