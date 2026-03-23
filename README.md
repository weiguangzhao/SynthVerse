# SynthVerse

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2602.04441)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/SynthVerse)
[![Benchmark](https://img.shields.io/badge/Benchmark-Hugging%20Face-FFD21E?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/SynthVerse-Benchmark)

**SynthVerse** is a large-scale, diverse **synthetic dataset for general point tracking**, with **5.8M+ training frames**, **59K test frames**, and **48K sequences** across **7+ scene types**. It provides high-quality dynamic motions, interactions, and annotations (including 3D point trajectories, depth, instance masks, and visibility) together with a **highly diverse benchmark** for evaluating methods under broad domain shift.

## Motivation and design

Point tracking has advanced quickly with foundation models, but progress toward **general** point tracking is still limited by scarce high-quality data: many datasets lack diversity and reliable trajectory annotations.

**SynthVerse** targets this gap with synthetic data that spans **new domains and object types** underrepresented in prior synthetic sets—for example **animated-film-style** content, **embodied manipulation**, **scene navigation**, and **articulated objects**. By covering a wider range of categories and motions, SynthVerse supports more robust training and evaluation. Experiments in our paper show consistent **generalization gains** when training with SynthVerse, and highlight **failure modes** of existing trackers under diverse settings.

## Dataset overview

| | |
| --- | --- |
| **Training frames** | 5,816K |
| **Test frames** | 59K |
| **Sequences** | 48K |
| **Scene types** | 7+ |

**Representative domains** include animation-style scenes, embodied interaction, navigation, hand interaction, and more. The project page on the `main` branch summarizes source-data provenance and annotation signals (RGB, depth, instance masks, camera poses, point trajectories, etc.).

## Benchmark

We release **SynthVerse-Benchmark** for systematic evaluation under diverse domain shifts. See the Hugging Face dataset card for splits, metrics, and usage:

- [InternRobotics/SynthVerse-Benchmark](https://huggingface.co/datasets/InternRobotics/SynthVerse-Benchmark)

Full training data:

- [InternRobotics/SynthVerse](https://huggingface.co/datasets/InternRobotics/SynthVerse)

## Getting started

**Project website (interactive demos & visualizations)** lives on the **`main`** branch of this repository.

```bash
git clone https://github.com/weiguangzhao/SynthVerse.git
cd SynthVerse
git checkout main
python server.py
# Open http://localhost:8000
```

**Paper (PDF / abstract):** [arXiv:2602.04441](https://arxiv.org/abs/2602.04441)

**Training & evaluation code:** *coming soon* (see also the Downloads section on the project page).

## Requirements (what SynthVerse is built for)

- Scalable **synthetic video** with controllable complexity and domain mix  
- **Rich ground truth** for point tracking and scene understanding (trajectories, depth, masks, visibility, cameras)  
- **Broad domain coverage** to stress-test generalization beyond narrow benchmarks  
- A **benchmark suite** aligned with the dataset for apples-to-apples comparison  

## Acknowledgements

The point cloud visualizer in the web demo is adapted from [TAPIP3D](https://github.com/zbw001/TAPIP3D).

## BibTeX

If you find SynthVerse useful, please cite:

```bibtex
@article{zhao2026SythnVerse,
  title={SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking},
  author={Weiguang Zhao and Haoran Xu and Xingyu Miao and Qin Zhao and Rui Zhang and Kaizhu Huang and Ning Gao and Peizhou Cao and Mingze Sun and Mulin Yu and Tao Lu and Linning Xu and Junting Dong and Jiangmiao Pang},
  journal={arXiv preprint arXiv:2602.04441},
  year={2026}
}
```

## Contact

**Email:** weiguang.zhao@liverpool.ac.uk

---

*This README summarizes content from the `main` branch project page (`index.html`). For data-pipeline experiments, see the `DataMaking` branch.*
