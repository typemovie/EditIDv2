<h3 align="center">
    EditIDv2: Editable ID Customization with Data-Lubricated ID Feature Integration for Text-to-Image Generation
</h3>
<p align="center"> 
<a href="https://arxiv.org/abs/2509.05659"><img alt="Build" src="https://img.shields.io/badge/Tech%20Report-EditIDv2-b31b1b.svg"></a>
<a href="https://typemovie.art/#/dashboard"><img src="https://img.shields.io/static/v1?label=Typemovie&message=application&color=green"></a>
</p>
<p align="center"> 
<span style="color:#137cf3; font-family: Gill Sans">Guandong Li,</span><sup></sup></a>  
<span style="color:#137cf3; font-family: Gill Sans">Zhaobin Chu</span></a> <br> 
<span style="font-size: 13.5px">iFlyTek Typemovie Research</span><br> 
<h3 align="center">
    EditID: Training-Free Editable ID Customization for Text-to-Image Generation
</h3>
<p align="center"> 
<a href="https://arxiv.org/abs/2503.12526"><img alt="Build" src="https://img.shields.io/badge/Tech%20Report-EditID-b31b1b.svg"></a>
<a href="https://typemovie.art/#/dashboard"><img src="https://img.shields.io/static/v1?label=Typemovie&message=application&color=green"></a>
</p>
<p align="center"> 
<span style="color:#137cf3; font-family: Gill Sans">Guandong Li,</span><sup></sup></a>  
<span style="color:#137cf3; font-family: Gill Sans">Zhaobin Chu</span></a> <br> 
<span style="font-size: 13.5px">iFlyTek Typemovie Research</span><br> 


### 🚩 更新日志

* **2025.09.09** 🔥 EditIDv2技术报告正式发布于arXiv。
* **2025.09.09** 🔥 EditIDv2项目代码正式开源，GitHub仓库现已开放，欢迎社区体验与反馈！
* **2025.08.21** 🎉 EditID论文被EMNLP 2025接收，喜大奔普！
* **2025.03.16** 🔥 EditID论文发布于arXiv。

我们将持续开源更多资源，包括训练代码、模型权重和数据集，敬请期待！🌟

## 📖 Abstract

We propose EditIDv2, a tuning-free solution specifically designed for high-complexity narrative scenes and long text inputs. Existing character editing methods perform well under simple prompts, but often suffer from degraded editing capabilities, semantic understanding biases, and identity consistency breakdowns when faced with long text narratives containing multiple semantic layers, temporal logic, and complex contextual relationships. In EditID, we analyzed the impact of the ID integration module on editability. In EditIDv2, we further explore and address the influence of the ID feature integration module. The core of EditIDv2 is to discuss the issue of editability injection under minimal data lubrication. Through a sophisticated decomposition of PerceiverAttention, the introduction of ID loss and joint dynamic training with the diffusion model, as well as an offline fusion strategy for the integration module, we achieve deep, multi-level semantic editing while maintaining identity consistency in complex narrative environments using only a small amount of data lubrication. This meets the demands of long prompts and high-quality image generation, and achieves excellent results in the IBench evaluation.

<p dir="auto" align="center">
    <img src="assets/editidv2.png" width="1024"/>
</p>

## ⚡️ 快速开始

##### 🔧 训练

训练EditIDv2权重使用以下命令，配置文件位于./train_configs/editid_insert.yaml：

```bash
python train_editid_loss.py --config ./train_configs/editid_insert.yaml
```

训练完成后，您将获得EditIDv2的模型权重，可用于后续推理。

#### 🚀 推理

EditIDv2的推理过程与PuLID完全兼容，您可以直接复用PuLID的推理代码或ComfyUI工作流，仅需替换为EditIDv2的训练权重。以下是快速上手步骤：

```bash
python infer.py --ckpt_path /path/to/editidv2_weights.pth
```

或通过PuLID的ComfyUI工作流，加载EditIDv2权重，参考PuLID ComfyUI教程。

## 🌈 More Examples

We provide sample prompts and results to showcase EditIDv2’s capabilities. For additional visualizations, check our [paper](https://arxiv.org/abs/2509.05659).

#### Complex Narrative Scenes

<details>
<summary>EditIDv2 excels at placing a subject into complex narrative scenes while maintaining identity consistency. </summary>
<p dir="auto" align="center">
<img src="assets/editidv21.png" width="1024"/>
</p>
</details>

## 📄 Disclaimer

This project is open-sourced for academic research. Images used are either generated or sourced from public datasets like MyStyle. If you have concerns, please contact us, and we will promptly address any issues. EditIDv2 is released under the Apache 2.0 License. When using other base models, ensure compliance with their licensing terms. This research advances personalized text-to-image generation. Users must comply with local laws and use the tool responsibly. The developers are not liable for misuse.

## 🚀 Updates

We aim to fully open-source EditIDv2, including training, inference, weights, and dataset, to support the research community. Thank you for your patience! 🌟

-  Release technical report.
-  Release GitHub repository.
-  Release inference code.
-  Release model checkpoints.
-  Release Hugging Face Space demo.
-  Release training code.
-  Release dataset.

## 📜 Citation

If EditIDv2 is helpful, please consider starring the repo. 🌟

For research purposes, cite our paper:

bibtex

```
@misc{li2025editidv2editableidcustomization,
      title={EditIDv2: Editable ID Customization with Data-Lubricated ID Feature Integration for Text-to-Image Generation}, 
      author={Guandong Li and Zhaobin Chu},
      year={2025},
      eprint={2509.05659},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.05659}, 
}

@article{li2025editid,
  title={EditID: Training-Free Editable ID Customization for Text-to-Image Generation},
  author={Li, Guandong and Chu, Zhaobin},
  journal={arXiv preprint arXiv:2503.12526},
  year={2025}
}
```
