## 🔭 Experience

### **Reality Defender** &nbsp;&nbsp;&nbsp;&nbsp; *Computer Vision Research Engineer*  
*Jan 2025 - Present*  
- **Deepfake Detection**: Developing and optimizing AI-driven security solutions to detect deepfakes and fraudulent media.

### **Freelance** &nbsp;&nbsp;&nbsp;&nbsp; *Computer Vision Research Engineer*  
*Dec 2024 - Present*  
- **Edge Vision Models**: Collaborated with [LuckyLab](https://www.luckylab.org/) to deploy cutting-edge CV models for edge devices.  
- **Real-Time Segmentation & Detection**: Focused on resource-constrained environments with efficient segmentation and detection algorithms.

### **NLPearl** &nbsp;&nbsp;&nbsp;&nbsp; *Deep Learning Research Engineer*  
*Jul 2024 - Jan 2025*  
- **Conversational AI Enhancements**: Developed real-time systems to detect conversational pauses and suggest optimal starter sentences for AI agents using fine-tuned LLMs with specialized prediction heads.  
- **Architectural Innovations**: Experimented with encoder-based and decoder-pretrained models, applying LoRA and multi-stage training to enhance prediction accuracy.  
- **Small Language Model Design**: Created an SLM to generate task-specific tokens, enabling multi-task outputs from a single fine-tuned model for efficient real-time inference.  
- **Audio Tokenization Solutions**: Designed solutions using pre-trained state-of-the-art audio tokenization models and LLMs tailored for audio-specific objectives.

### **Pashoot Robotics** &nbsp;&nbsp;&nbsp;&nbsp; *Computer Vision and Deep Learning Research Engineer*  
*May 2023 - Jul 2024*  
- **Vision Solutions for Automation**: Enhanced and innovated vision solutions critical to manufacturing automation.  
- **Applied Deep Learning**: Conducted research in 3D reconstruction, object detection, segmentation (few-shot, zero-shot), tracking, 6DOF estimation, and simulation using Blender.

## 💻 Projects

### [**PopYou2 - VAR Text**](https://github.com/amit154154/PopYou2)

[![GitHub Stars](https://img.shields.io/github/stars/amit154154/PopYou2?style=social)](https://github.com/amit154154/VAR_clip)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/AmitIsraeli/PopYou)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Report-yellow)](https://api.wandb.ai/links/amit154154/cqccmfsl)

- **Dataset Creation**: Generated ~100,000 Funko Pop! images with detailed prompts using [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo).
- **Model Fine-Tuning**: Adapted the [Visual AutoRegressive (VAR)](https://arxiv.org/abs/2404.02905) model for Funko Pop! generation by injecting a custom "doll" embedding.
- **Adapter Training**: Trained an adapter with a frozen [SigLIP image encoder](https://github.com/FoundationVision/VAR) and a lightweight LoRA module.
- **Text-to-Image Generation**: Enabled by replacing the SigLIP image encoder with its text encoder, retaining frozen components for efficiency and quality.

![VAR Explained](PopYou2/VAR_explained.png)

---

### [**Few-Shot Segmentation with SAM and LoRA**](https://github.com/amit154154/SAM_LORA)

[![GitHub Stars](https://img.shields.io/github/stars/amit154154/SAM_LORA?style=social)](https://github.com/amit154154/SAM_LORA)

- **LoRA Adaptation**: Employed LoRA to adapt SAM for few-shot segmentation with minimal images.
- **Prompt-Free Segmentation**: Eliminated reliance on external prompts or detection models like Grounding SAM or YOLO.
- **Performance Boost**: Outperformed prior methods (e.g., [PerSAM](https://arxiv.org/abs/2305.03048)) in class-specific segmentation quality and flexibility.
- **Diverse Training**: Trained on datasets like COCO, Soccer, and Cityscapes with varying sample sizes and class distributions.
- **Foundation Models Integration**: Explored enhancements using models with prior class-oriented knowledge (e.g., CLIP, SigLIP).

![Cityscapes](few_shot_sam_lora/cityscapes.png)

---

### [**CelebrityLook**](https://github.com/amit154154/CelebrityLook)

[![GitHub Stars](https://img.shields.io/github/stars/amit154154/CelebrityLook?style=social)](https://github.com/amit154154/CelebrityLook)

- **Real-Time Face Transformation**: Developed a mobile app utilizing advanced GAN technologies on-device.
- **High Performance**: Achieved 30fps on mobile devices with optimized CoreML models.
- **Award-Winning**: Won the MobileXGenAI hackathon hosted by Samsung Next.
- **Quality Enhancements**: Combined multiple losses from foundation models and facial feature extractors.
- **Research Implementation**: Implemented ["Bridging CLIP and StyleGAN through Latent Alignment for Image Editing"](https://arxiv.org/abs/2210.04506).
- **Advanced Experimentation**: Explored loss functions, latent spaces, and mapper architectures, introducing fine-tuning for robustness.

![Mapper Training](CelebryLook/mapper_training.png)


---

### [**PopYou - FastGAN CLIP**](https://github.com/amit154154/PopYou)

[![GitHub Stars](https://img.shields.io/github/stars/amit154154/PopYou?style=social)](https://github.com/amit154154/PopYou)

- **Semi-Synthetic Dataset**: Created using an image upscaling model and Deci Diffusion.
- **GAN Training**: Trained with FastGAN to generate high-quality Funko Pop designs.
- **Inversion Model Development**: Based on a frozen CLIP backbone for image generation from text and real-life images.
- **3D Generation Exploration**: Included mesh creation and multi-view rendering using diffusion models and textual inversion.
- **Benchmarking**: Evaluated against Deci Diffusion using CLIP similarity and FID scores.

<img src="PopYou!/Barack_Obama_fastgan.png" alt="Barack Obama FastGAN" width="400" />
---

### [**KoalaReadingAI**](#)

[![Spotify](https://img.shields.io/badge/Spotify-Podcast-green)](https://open.spotify.com/show/0fuZbZipy60VdRpkbIb9y1)
[![YouTube](https://img.shields.io/badge/YouTube-Channel-red)](https://www.youtube.com/channel/UCIbCIgJjIWmHyKC0Qc_C6FA)

- **AI Podcast Founder**: Automatically summarizes the latest AI research papers.
- **Tech Stack**: Utilizes ElevenLabs text-to-speech and ChatPDF APIs.
- **Automation**: Downloads latest papers from Hugging Face daily.
- **Free TTS Implementation**: Using Tortoise-TTS; working on Llama 2 summaries.
- **Platforms**: Available on [Spotify](https://open.spotify.com/show/0fuZbZipy60VdRpkbIb9y1) and [YouTube](https://www.youtube.com/channel/UCIbCIgJjIWmHyKC0Qc_C6FA).

<video controls autoplay muted>
  <source src="koala_reading_ai/koala_reading_ai_gif.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 📫 Contact Me

- **Email**: [amit1541541@gmail.com](mailto:amit1541541@gmail.com)
- **LinkedIn**: [linkedin.com/in/amit-israeli-aa4a30242](https://www.linkedin.com/in/amit-israeli-aa4a30242/)
- **GitHub**: [github.com/amit154154](https://github.com/amit154154)

---

&copy; 2024 Amit Israeli. All rights reserved.

---

Feel free to explore my repositories and get in touch if you're interested in collaboration or have any questions!
