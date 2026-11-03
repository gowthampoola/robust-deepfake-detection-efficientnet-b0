# Research Paper Documentation

This directory contains the final, authoritative publication documentation for the deepfake detection research study.

---

## Final Research Paper

- **File**: [`Deepfake_Detection_Final_Paper.pdf`](Deepfake_Detection_Final_Paper.pdf)
- **Title**: *Robust Deepfake Image Detection Under Real-World Post-Processing Using EfficientNet-B0*
- **Authors**: 
  - Hemanth Bolla (`hemanth.24bca8419@vitapstudent.ac.in`)
  - Poola Gowtham (`gowtham.24bce8237@vitapstudent.ac.in`)
  - Chilukuri Venkata Uday Adhitya (`adithya.24bca7974@vitapstudent.ac.in`)
  - Devarakonda Yaswanth (`yaswanth.24bca8001@vitapstudent.ac.in`)
- **Affiliation**: School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, Andhra Pradesh, India
- **Format**: Standard IEEE Two-Column Conference Format (11 pages, camera-ready)

---

## Abstract & Key Highlights

Deepfake images have become increasingly realistic as generative models (e.g., NVIDIA StyleGAN) have advanced. Conventional convolutional detectors trained on clean benchmark images develop brittle representations heavily reliant on high-frequency boundary cues, resulting in severe performance collapse under real-world transmission degradations (falling from 99.25% clean accuracy to 50.97% under Gaussian blur and ~58% under JPEG compression, resizing, and noise).

Through a two-stage degradation-aware robust fine-tuning protocol (5 clean epochs + 5 degradation-aware fine-tuning epochs), our robust EfficientNet-B0 model:
1. Recovers **+43.49 percentage points** under Gaussian blur (reaching **94.46%** accuracy).
2. Maintains $\ge \mathbf{93.22\%}$ accuracy across all evaluated distortions (JPEG $Q=50$, resizing $256 \to 112 \to 256$, blur $r=2.0$, noise $\sigma=0.05$, and sharpness/contrast enhancement).
3. Achieves **99.42%** accuracy on clean test images (+0.17 pp over baseline).
4. Eliminates up to **93.31%** of baseline errors without modifying model capacity (~5.3M parameters) or introducing additional inference-time architectural overhead.
