# Lessons Learned and Future Directions

This project went through a full end-to-end cycle, but the final results fell short of my expectations. Still, the process taught me a great deal about mixture-of-experts models, autonomous driving pipelines, and the practical realities of research.

This document records what I learned, where I fell short, and how I'd approach it differently next time.

*Note:* I remain committed to building autonomous driving models and will explore new ideas such as integrating language models into driving stacks. For now, I'm pausing this project to focus on other obligations. If you find this repository useful, please use it as inspiration or boilerplate for your own work, and feel free to contribute via pull requests.

## Table of Contents

- [Literature Gaps](#literature-gaps)
- [Data Collection Limitations](#data-collection-limitations)
- [Evaluation Shortcomings](#evaluation-shortcomings)
- [Dataset Exploration](#dataset-exploration)
- [Limited Hyperparameter and Model Search](#limited-hyperparameter-and-model-search)
- [Summary](#summary)

---

## Literature Gaps

I jumped into implementation too quickly without fully grounding myself in prior work. A few particularly relevant resources that I found later in the process:

* [CBDES MoE: Hierarchically Decoupled Mixture-of-Experts for Functional Modules in Autonomous Driving](https://arxiv.org/abs/2508.07838)
* [DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](https://arxiv.org/abs/2505.16278)
* [Safe Real-World Autonomous Driving by Learning to Predict and Plan with a Mixture of Experts](https://ieeexplore.ieee.org/document/10160992)
* [Generalizing Motion Planners with Mixture of Experts for Autonomous Driving](https://arxiv.org/abs/2410.15774)
* [Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture](https://ieeexplore.ieee.org/document/10323651)
* [Using Mixture of Expert Models to Gain Insights into Semantic Segmentation](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Pavlitskaya_Using_Mixture_of_Expert_Models_to_Gain_Insights_Into_Semantic_CVPRW_2020_paper.pdf)

---

## Data Collection Limitations

I collected a strong baseline with CARLA, but compute constraints forced me to leave out several potentially useful modalities (depth, radar, IMU, weather state, 3D bounding boxes, road geometry, traffic light states, NPC behaviors, lane topology, etc.). Future iterations should capture richer data streams to enable sensor fusion and harder scenario coverage.

---

## Evaluation Shortcomings

* **Test Sets:** I did not provision robust test datasets throughout development, which weakened my ability to track real progress.
* **Model Testing:** I often ran into tensor shape mismatches and patched them quickly rather than fixing root causes.
* **Metrics Tracking:** I treated this project more like a personal build than a formal research effort, so I didn’t consistently track training/evaluation metrics across runs. In hindsight, maintaining disciplined logs and dashboards would have made it much easier to compare approaches and debug issues.

---

## Dataset Exploration

* I should have spent more time inspecting my datasets to uncover optimizations or biases.
* I also could have incorporated more diverse datasets for expert models.
* On the positive side, I did produce and release two clean, reproducible datasets that may be useful for others:

  * [CARLA Autopilot Images](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-images)
  * [CARLA Autopilot Multimodal](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset)

---

## Limited Hyperparameter and Model Search

* I fixed hyperparameters (learning rate, weight decay, number of queries, etc.) instead of systematically tuning them.
* I avoided larger architectural changes because of pressure to move fast on rented GPUs.

---

## Summary

The main lesson: don't sprint straight to training. Better upfront literature review, more careful dataset design, and disciplined evaluation pipelines would have put this project on stronger footing. While AutoMoE didn't reach top-tier performance, the process left me with reusable datasets, working CARLA pipelines, and a clearer sense of what "serious" self-driving research demands.
