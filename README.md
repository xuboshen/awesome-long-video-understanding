# 🎥 Awesome-Long-Video-Understanding

Update date: 7th Dec 2025

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of papers, datasets, and benchmarks for **Long Video Understanding**, with a special focus on but not limited to Multimodal Large Multimodal Models (LMMs).

Note: The *focus* of each paper is summarized from its arxiv abstract using DeepSeek-V3. Due to the large number of papers, there may be mistakes. If you find any, please open an issue to let us know. Thank you!

## 📋 Table of Contents
- [🎥 Awesome-Long-Video-Understanding](#-awesome-long-video-understanding)
  - [📋 Table of Contents](#-table-of-contents)
    - [Benchmark](#benchmark)
    - [Vision-Language Models](#vision-language-models)
    - [Subsampling methods](#subsampling-methods)
      - [RAG / Memory / Agentic / Language Repository / Frame Sampling Methods](#rag--memory--agentic--language-repository--frame-sampling-methods)
    - [Compression methods](#compression-methods)
      - [New LLM Architectures](#new-llm-architectures)
      - [Token Compression](#token-compression)
    - [Temporal Modeling (timestamp / time positional encoding)](#temporal-modeling-timestamp--time-positional-encoding)
    - [Downstream tasks](#downstream-tasks)
      - [Real-time Interaction](#real-time-interaction)
      - [Dense Video Captioning](#dense-video-captioning)
      - [Temporal Action Detection](#temporal-action-detection)
      - [Temporal Video Grounding](#temporal-video-grounding)
      - [Others](#others)
    - [Others](#others-1)

---


### Benchmark

*   **[LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos](http://arxiv.org/abs/2411.19772v3)** (CVPR2025 2024.11)
    *   Focus: Proposes a framework for fine-grained omni-modal video understanding using hierarchical alignment and contrastive learning.
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.19772) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVALE%3A%20Vision-Audio-Language-Event%20Benchmark%20Towards%20Time-Aware%20Omni-Modal%20Perception%20of%20Long%20Videos%22)

*   **[EgoLife: Towards Egocentric Life Assistant](http://arxiv.org/abs/2503.03803v1)** (CVPR2025 2025.03)
    *   Focus: EgoLife develops AI-powered wearable glasses for personal efficiency as an egocentric life assistant.
    *   code: [https://github.com/EvolvingLMMs-Lab/EgoLife](https://github.com/EvolvingLMMs-Lab/EgoLife)
    *   citation: 28 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.03803) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EgoLife%3A%20Towards%20Egocentric%20Life%20Assistant%22)

*   **[TeleEgo: Benchmarking Egocentric AI Assistants in the Wild](http://arxiv.org/abs/2510.23981v2)** (2025.10)
    *   Focus: Existing benchmarks for egocentric AI assistants lack real-time processing and long-term memory requirements.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.23981) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TeleEgo%3A%20Benchmarking%20Egocentric%20AI%20Assistants%20in%20the%20Wild%22)

*   **[PlanarTrack: A high-quality and challenging benchmark for large-scale planar object tracking](http://arxiv.org/abs/2510.23368v1)** (ICCV2023 2025.10)
    *   Focus: Planar tracking advances for robotics and AR, focusing on degenerate cases.
    *   code: [https://github.com/HengLan/PlanarTrack](https://github.com/HengLan/PlanarTrack)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.23368) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PlanarTrack%3A%20A%20high-quality%20and%20challenging%20benchmark%20for%20large-scale%20planar%20object%20tracking%22)

*   **[MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](http://arxiv.org/abs/2510.21406v1)** (NeurIPS2025 2025.10)
    *   Focus: Proposes MUVR benchmark for multi-modal untrimmed video retrieval on long videos.
    *   code: [https://github.com/debby-0527/MUVR](https://github.com/debby-0527/MUVR)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.21406) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MUVR%3A%20A%20Multi-Modal%20Untrimmed%20Video%20Retrieval%20Benchmark%20with%20Multi-Level%20Visual%20Correspondence%22)

*   **[LongInsightBench: A Comprehensive Benchmark for Evaluating Omni-Modal Models on Human-Centric Long-Video Understanding](http://arxiv.org/abs/2510.17305v2)** (2025.10)
    *   Focus: LongInsightBench is the first benchmark for evaluating long video understanding of language, actions, and context.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.17305) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongInsightBench%3A%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Omni-Modal%20Models%20on%20Human-Centric%20Long-Video%20Understanding%22)

*   **[ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](http://arxiv.org/abs/2510.11606v1)** (2025.10)
    *   Focus: MLLMs' potential for scientific discovery is unclear due to limited evaluation of their capabilities.
    *   code: [https://github.com/OpenGVLab/ExpVid](https://github.com/OpenGVLab/ExpVid)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.11606) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ExpVid%3A%20A%20Benchmark%20for%20Experiment%20Video%20Understanding%20%26%20Reasoning%22)

*   **[StreamingVLM: Real-Time Understanding for Infinite Video Streams](http://arxiv.org/abs/2510.09608v1)** (2025.10)
    *   Focus: VLMs struggle with real-time video understanding due to latency and memory constraints.
    *   code: [https://github.com/mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.09608) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamingVLM%3A%20Real-Time%20Understanding%20for%20Infinite%20Video%20Streams%22)

*   **[CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation](http://arxiv.org/abs/2510.09266v1)** (2025.10)
    *   Focus: Video-based MRAG benchmarks enable MLLMs to generate responses using external multimodal evidence.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.09266) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CFVBench%3A%20A%20Comprehensive%20Video%20Benchmark%20for%20Fine-grained%20Multimodal%20Retrieval-Augmented%20Generation%22)

*   **[StreamForest: Efficient Online Video Understanding with Persistent Event Memory](http://arxiv.org/abs/2509.24871v1)** (NeurIPS2025 2025.09)
    *   Focus: MLLMs struggle with real-time video streaming due to storage constraints.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24871) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamForest%3A%20Efficient%20Online%20Video%20Understanding%20with%20Persistent%20Event%20Memory%22)

*   **[NeMo: Needle in a Montage for Video-Language Understanding](http://arxiv.org/abs/2509.24563v2)** (2025.09)
    *   Focus: Proposes new benchmarks for evaluating temporal reasoning in video-language models.
    *   project: [https://lavi-lab.github.io/NeMoBench](https://lavi-lab.github.io/NeMoBench)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24563) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NeMo%3A%20Needle%20in%20a%20Montage%20for%20Video-Language%20Understanding%22)

*   **[VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding](http://arxiv.org/abs/2509.21451v1)** (2025.09)
    *   Focus: Current video understanding metrics inadequately capture human judgment quality.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.21451) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoJudge%3A%20Bootstrapping%20Enables%20Scalable%20Supervision%20of%20MLLM-as-a-Judge%20for%20Video%20Understanding%22)

*   **[VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](http://arxiv.org/abs/2509.19002v2)** (2025.09)
    *   Focus: MLLMs improve video understanding but face benchmark limitations for practical use.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.19002) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VIR-Bench%3A%20Evaluating%20Geospatial%20and%20Temporal%20Understanding%20of%20MLLMs%20via%20Travel%20Video%20Itinerary%20Reconstruction%22)

*   **[NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning](http://arxiv.org/abs/2509.18041v2)** (2025.09)
    *   Focus: Vision-language models struggle with long video question answering due to complex temporal reasoning demands.
    *   project: [https://utaustin-swarmlab.github.io/NeuS-QA/](https://utaustin-swarmlab.github.io/NeuS-QA/)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.18041) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NeuS-QA%3A%20Grounding%20Long-Form%20Video%20Understanding%20in%20Temporal%20Logic%20and%20Neuro-Symbolic%20Reasoning%22)

*   **[Cinéaste: A Fine-grained Contextual Movie Question Answering Benchmark](http://arxiv.org/abs/2509.14227v1)** (2025.09)
    *   Focus: Diagnosing deep narrative comprehension in video-language models is challenging with current benchmarks.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.14227) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Cin%C3%A9aste%3A%20A%20Fine-grained%20Contextual%20Movie%20Question%20Answering%20Benchmark%22)

*   **[ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Long Video Understanding](http://arxiv.org/abs/2508.21496v2)** (2025.08)
    *   Focus: Video-MLLMs show strong video understanding but are prone to hallucination.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.21496) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ELV-Halluc%3A%20Benchmarking%20Semantic%20Aggregation%20Hallucinations%20in%20Long%20Video%20Understanding%22)

*   **[EmbRACE-3K: Embodied Reasoning and Action in Complex Environments](http://arxiv.org/abs/2507.10548v1)** (2025.07)
    *   Focus: VLMs excel in passive video understanding but struggle in embodied settings requiring active perception.
    *   project: [https://mxllc.github.io/EmbRACE-3K/](https://mxllc.github.io/EmbRACE-3K/)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.10548) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EmbRACE-3K%3A%20Embodied%20Reasoning%20and%20Action%20in%20Complex%20Environments%22)

*   **[HumanVideo-MME: Benchmarking MLLMs for Human-Centric Video Understanding](http://arxiv.org/abs/2507.04909v2)** (2025.07)
    *   Focus: MLLMs advance in visual tasks but struggle with human-centric video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.04909) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HumanVideo-MME%3A%20Benchmarking%20MLLMs%20for%20Human-Centric%20Video%20Understanding%22)

*   **[MOMENTS: A Comprehensive Multimodal Benchmark for Theory of Mind](http://arxiv.org/abs/2507.04415v2)** (2025.07)
    *   Focus: MoMentS is a multimodal dataset for understanding mental states in social interactions.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.04415) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MOMENTS%3A%20A%20Comprehensive%20Multimodal%20Benchmark%20for%20Theory%20of%20Mind%22)

*   **[PhysLab: A Benchmark Dataset for Multi-Granularity Visual Parsing of Physics Experiments](http://arxiv.org/abs/2506.06631v2)** (2025.06)
    *   Focus: Visual parsing progress is limited by dataset constraints like insufficient annotations.
    *   code: [https://github.com/ZMH-SDUST/PhysLab](https://github.com/ZMH-SDUST/PhysLab)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.06631) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PhysLab%3A%20A%20Benchmark%20Dataset%20for%20Multi-Granularity%20Visual%20Parsing%20of%20Physics%20Experiments%22)

*   **[Movie Facts and Fibs (MF$^2$): A Benchmark for Long Movie Understanding](http://arxiv.org/abs/2506.06275v1)** (2025.06)
    *   Focus: Current benchmarks limit VLMs' ability to understand long-form video content.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.06275) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Movie%20Facts%20and%20Fibs%20%28MF%24%5E2%24%29%3A%20A%20Benchmark%20for%20Long%20Movie%20Understanding%22)

*   **[EASG-Bench: Video Q&A Benchmark with Egocentric Action Scene Graphs](http://arxiv.org/abs/2506.05787v2)** (2025.06)
    *   Focus: EASG-Bench is a QA benchmark for egocentric videos using spatio-temporally grounded scene graphs.
    *   code: [https://github.com/fpv-iplab/EASG-bench](https://github.com/fpv-iplab/EASG-bench)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.05787) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EASG-Bench%3A%20Video%20Q%26A%20Benchmark%20with%20Egocentric%20Action%20Scene%20Graphs%22)

*   **[TextVidBench: A Benchmark for Long Video Scene Text Understanding](http://arxiv.org/abs/2506.04983v1)** (2025.06)
    *   Focus: Current Text-VQA datasets have limited video duration despite recent progress.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.04983) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TextVidBench%3A%20A%20Benchmark%20for%20Long%20Video%20Scene%20Text%20Understanding%22)

*   **[ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding](http://arxiv.org/abs/2505.23922v1)** (NeurIPS2023 2025.05)
    *   Focus: Existing benchmarks lack hierarchical temporal modeling for long-video understanding.
    *   code: [https://github.com/multimodal-art-projection/ScaleLong](https://github.com/multimodal-art-projection/ScaleLong)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.23922) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ScaleLong%3A%20A%20Multi-Timescale%20Benchmark%20for%20Long%20Video%20Understanding%22)

*   **[VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](http://arxiv.org/abs/2505.23359v1)** (2025.05)
    *   Focus: Long chain-of-thought reasoning improves LLMs but lacks demonstration for long video understanding tasks.
    *   project: [https://llyx97.github.io/video_reason_bench/](https://llyx97.github.io/video_reason_bench/)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.23359) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoReasonBench%3A%20Can%20MLLMs%20Perform%20Vision-Centric%20Complex%20Video%20Reasoning%3F%22)

*   **[Two Causally Related Needles in a Video Haystack](http://arxiv.org/abs/2505.19853v3)** (NeurIPS2025 2025.05)
    *   Focus: Causal2Needles benchmark evaluates VLMs on long video understanding tasks.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.19853) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Two%20Causally%20Related%20Needles%20in%20a%20Video%20Haystack%22)

*   **[VideoEval-Pro: Robust and Realistic Long Video Understanding Evaluation](http://arxiv.org/abs/2505.14640v1)** (2025.05)
    *   Focus: LMMs are advancing long video understanding, driving the creation of standardized benchmarks for evaluation.
    *   project: [https://tiger-ai-lab.github.io/VideoEval-Pro](https://tiger-ai-lab.github.io/VideoEval-Pro)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.14640) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoEval-Pro%3A%20Robust%20and%20Realistic%20Long%20Video%20Understanding%20Evaluation%22)

*   **[LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts](http://arxiv.org/abs/2505.13928v3)** (2025.05)
    *   Focus: Existing video-text retrieval benchmarks have limited video duration, hindering long video understanding.
    *   code: [https://github.com/TechNomad-ds/LoVR-benchmark](https://github.com/TechNomad-ds/LoVR-benchmark)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.13928) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LoVR%3A%20A%20Benchmark%20for%20Long%20Video%20Retrieval%20in%20Multimodal%20Contexts%22)

*   **[Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation](http://arxiv.org/abs/2505.12702v2)** (2025.05)
    *   Focus: RVOS segments and tracks video objects using language, but current methods have limitations.
    *   project: [https://isee-laboratory.github.io/Long-RVOS](https://isee-laboratory.github.io/Long-RVOS)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.12702) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long-RVOS%3A%20A%20Comprehensive%20Benchmark%20for%20Long-term%20Referring%20Video%20Object%20Segmentation%22)

*   **[VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models](http://arxiv.org/abs/2505.08455v1)** (2025.05)
    *   Focus: LVLMs lack video causal reasoning benchmarks, limiting their evaluation and development.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.08455) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VCRBench%3A%20Exploring%20Long-form%20Causal%20Reasoning%20Capabilities%20of%20Large%20Video%20Language%20Models%22)

*   **[RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](http://arxiv.org/abs/2505.02064v3)** (NeurIPS2025 2025.05)
    *   Focus: Current benchmarks fail to assess MLLMs' continuous perception and reasoning abilities.
    *   code: [https://github.com/LJungang/RTV-Bench](https://github.com/LJungang/RTV-Bench)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.02064) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22RTV-Bench%3A%20Benchmarking%20MLLM%20Continuous%20Perception%2C%20Understanding%20and%20Reasoning%20through%20Real-Time%20Video%22)

*   **[SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding](http://arxiv.org/abs/2504.21435v3)** (CVPR2025 2025.04)
    *   Focus: MLLM benchmarks are growing to assess video understanding capabilities.
    *   code: [https://github.com/zackhxn/SeriesBench-CVPR2025](https://github.com/zackhxn/SeriesBench-CVPR2025)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.21435) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SeriesBench%3A%20A%20Benchmark%20for%20Narrative-Driven%20Drama%20Series%20Understanding%22)

*   **[LVC: A Lightweight Compression Framework for Enhancing VLMs in Long Video Understanding](http://arxiv.org/abs/2504.06835v1)** (2025.04)
    *   Focus: VLMs achieve frame-level understanding but struggle with long video comprehension.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.06835) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LVC%3A%20A%20Lightweight%20Compression%20Framework%20for%20Enhancing%20VLMs%20in%20Long%20Video%20Understanding%22)

*   **[Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma?](http://arxiv.org/abs/2503.12496v2)** (ICCV2025 2025.03)
    *   Focus: LVLMs struggle with long videos due to the sampling dilemma between low-detail sparse and high-cost dense sampling.
    *   code: [https://github.com/dvlab-research/LSDBench](https://github.com/dvlab-research/LSDBench)
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.12496) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Does%20Your%20Vision-Language%20Model%20Get%20Lost%20in%20the%20Long%20Video%20Sampling%20Dilemma%3F%22)

*   **[ALLVB: All-in-One Long Video Understanding Benchmark](http://arxiv.org/abs/2503.07298v2)** (2025.03)
    *   Focus: Existing video benchmarks are too short for evaluating MLLMs on long videos.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.07298) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ALLVB%3A%20All-in-One%20Long%20Video%20Understanding%20Benchmark%22)
    
*   **[MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](http://arxiv.org/abs/2506.04141v1)** (2025.06)
    *   Focus: Proposes a new MLLM architecture for improved long video understanding and temporal reasoning.
    *   project: [https://mmr-v.github.io](https://mmr-v.github.io)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.04141) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MMR-V%3A%20What%27s%20Left%20Unsaid%3F%20A%20Benchmark%20for%20Multimodal%20Deep%20Reasoning%20in%20Videos%22)

*   **[MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval](http://arxiv.org/abs/2502.12558v4)** (NeurIPS2025 2025.02)
    *   Focus: Proposes a new benchmark for long video understanding with longer videos and more diverse tasks.
    *   project: [https://yhy-2000.github.io/MomentSeeker/](https://yhy-2000.github.io/MomentSeeker/)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.12558) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MomentSeeker%3A%20A%20Task-Oriented%20Benchmark%20For%20Long-Video%20Moment%20Retrieval%22)

*   **[SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding](http://arxiv.org/abs/2502.10810v2)** (ICLR2025 2025.02)
    *   Focus: LVLMs lack suitable evaluation for emerging applications.
    *   code: [https://github.com/sotayang/SVBench](https://github.com/sotayang/SVBench)
    *   citation: 20 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.10810) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SVBench%3A%20A%20Benchmark%20with%20Temporal%20Multi-Turn%20Dialogues%20for%20Streaming%20Video%20Understanding%22)


*   **[$\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation](http://arxiv.org/abs/2501.19098v2)** (2025.01)
    *   Focus: This paper introduces a new method for long-video understanding using compressed representations and temporal modeling.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.19098) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22%24%5Cinfty%24-Video%3A%20A%20Training-Free%20Approach%20to%20Long%20Video%20Understanding%20via%20Continuous-Time%20Memory%20Consolidation%22)

*   **[X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding](http://arxiv.org/abs/2501.06835v2)** (2025.01)
    *   Focus: Long-form egocentric videos offer insights into human behavior for embodied intelligence applications.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.06835) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22X-LeBench%3A%20A%20Benchmark%20for%20Extremely%20Long%20Egocentric%20Video%20Understanding%22)

*   **[HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding](http://arxiv.org/abs/2501.01645v3)** (2025.01)
    *   Focus: Multimodal LLMs advance visual understanding but struggle with hour-long video comprehension.
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.01645) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HLV-1K%3A%20A%20Large-scale%20Hour-Long%20Video%20Benchmark%20for%20Time-Specific%20Long%20Video%20Understanding%22)

*   **[FriendsQA: A New Large-Scale Deep Video Understanding Dataset with Fine-grained Topic Categorization for Story Videos](http://arxiv.org/abs/2412.17022v1)** (2024.12)
    *   Focus: VideoQA models struggle with complex questions despite good performance on factoid tasks.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.17022) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FriendsQA%3A%20A%20New%20Large-Scale%20Deep%20Video%20Understanding%20Dataset%20with%20Fine-grained%20Topic%20Categorization%20for%20Story%20Videos%22)

*   **[Do Language Models Understand Time?](http://arxiv.org/abs/2412.13845v3)** (2024.12)
    *   Focus: LLMs enhance video tasks like action recognition and anomaly detection despite unique challenges.
    *   code: [https://github.com/Darcyddx/Video-LLM](https://github.com/Darcyddx/Video-LLM)
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.13845) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Do%20Language%20Models%20Understand%20Time%3F%22)

*   **[CG-Bench: Clue-grounded Question Answering Benchmark for Long Video Understanding](http://arxiv.org/abs/2412.12075v1)** (ICLR2025 2024.12)
    *   Focus: Existing long video benchmarks for MLLMs rely on single annotations, limiting evaluation of temporal reasoning.
    *   project: [https://cg-bench.github.io/leaderboard/](https://cg-bench.github.io/leaderboard/)
    *   citation: 35 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.12075) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CG-Bench%3A%20Clue-grounded%20Question%20Answering%20Benchmark%20for%20Long%20Video%20Understanding%22)

*   **[Apollo: An Exploration of Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2412.10360v1)** (CVPR2025 2024.12)
    *   Focus: This paper investigates the mechanisms behind video understanding in large multimodal models.
    *   project: [https://apollo-lmms.github.io](https://apollo-lmms.github.io)
    *   citation: 51 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.10360) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Apollo%3A%20An%20Exploration%20of%20Video%20Understanding%20in%20Large%20Multimodal%20Models%22)

*   **[Neptune: The Long Orbit to Benchmarking Long Video Understanding](http://arxiv.org/abs/2412.09582v2)** (2024.12)
    *   Focus: Neptune is a new benchmark for long video understanding requiring multimodal reasoning over extended time.
    *   code: [https://github.com/google-deepmind/neptune](https://github.com/google-deepmind/neptune)
    *   citation: 14 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.09582) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Neptune%3A%20The%20Long%20Orbit%20to%20Benchmarking%20Long%20Video%20Understanding%22)

*   **[Perception Test 2024: Challenge Summary and a Novel Hour-Long VideoQA Benchmark](http://arxiv.org/abs/2411.19941v1)** (2024.11)
    *   Focus: The Second Perception Test challenge at ECCV 2024 continues benchmarking visual perception tasks.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.19941) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Perception%20Test%202024%3A%20Challenge%20Summary%20and%20a%20Novel%20Hour-Long%20VideoQA%20Benchmark%22)

*   **[HourVideo: 1-Hour Video-Language Understanding](http://arxiv.org/abs/2411.04998v1)** (NeurIPS2024 2024.11)
    *   Focus: HourVideo is a benchmark for hour-long video understanding with summarization, perception, and reasoning tasks.
    *   citation: 79 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.04998) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HourVideo%3A%201-Hour%20Video-Language%20Understanding%22)

*   **[FIOVA: A Multi-Annotator Benchmark for Human-Aligned Video Captioning](http://arxiv.org/abs/2410.15270v2)** (2024.10)
    *   Focus: Existing video caption benchmarks inadequately assess LVLM alignment with human understanding due to single-annotation limitations.
    *   project: [https://huuuuusy.github.io/fiova/](https://huuuuusy.github.io/fiova/)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.15270) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FIOVA%3A%20A%20Multi-Annotator%20Benchmark%20for%20Human-Aligned%20Video%20Captioning%22)

*   **[TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models](http://arxiv.org/abs/2410.10818v2)** (2024.10)
    *   Focus: Existing video benchmarks lack fine-grained temporal annotations for detailed video understanding.
    *   project: [https://temporalbench.github.io/](https://temporalbench.github.io/)
    *   citation: 38 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.10818) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TemporalBench%3A%20Benchmarking%20Fine-grained%20Temporal%20Understanding%20for%20Multimodal%20Video%20Models%22)

*   **[MM-Ego: Towards Building Egocentric Multimodal LLMs for Video QA](http://arxiv.org/abs/2410.07177v2)** (ICLR2025 2024.10)
    *   Focus: This research builds a multimodal foundation model for egocentric video understanding.
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.07177) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MM-Ego%3A%20Towards%20Building%20Egocentric%20Multimodal%20LLMs%20for%20Video%20QA%22)

*   **[LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding](http://arxiv.org/abs/2407.15754v1)** (NeurIPS2024 2024.07)
    *   Focus: Introduces a new benchmark for evaluating large multimodal models on long, rich inputs.
    *   citation: 311 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.15754) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVideoBench%3A%20A%20Benchmark%20for%20Long-context%20Interleaved%20Video-Language%20Understanding%22)

*   **[InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies and TV Shows](http://arxiv.org/abs/2406.19875v5)** (2024.06)
    *   Focus: Long video understanding is challenging due to inadequate benchmarks for multi-modal models.
    *   project: [https://vision-cair.github.io/Infinibench](https://vision-cair.github.io/Infinibench)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.19875) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22InfiniBench%3A%20A%20Benchmark%20for%20Large%20Multi-Modal%20Models%20in%20Long-Form%20Movies%20and%20TV%20Shows%22)

*   **[MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding](http://arxiv.org/abs/2406.14515v3)** (NeurIPS2024 2024.06)
    *   Focus: LVLMs advance video understanding beyond traditional VideoQA benchmarks.
    *   code: [https://github.com/open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
    *   citation: 137 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.14515) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MMBench-Video%3A%20A%20Long-Form%20Multi-Shot%20Benchmark%20for%20Holistic%20Video%20Understanding%22)

*   **[Towards Event-oriented Long Video Understanding](http://arxiv.org/abs/2406.14129v1)** (2024.06)
    *   Focus: Video MLLMs lack benchmarks with rich evidence for comprehensive evaluation.
    *   code: [https://github.com/RUCAIBox/Event-Bench](https://github.com/RUCAIBox/Event-Bench)
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.14129) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Event-oriented%20Long%20Video%20Understanding%22)

*   **[VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment](http://arxiv.org/abs/2406.10889v2)** (CVPR2025 2024.06)
    *   Focus: Video models advance but struggle with associating people and actions over time for compositional reasoning.
    *   project: [https://katha-ai.github.io/projects/velociti](https://katha-ai.github.io/projects/velociti)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.10889) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VELOCITI%3A%20Benchmarking%20Video-Language%20Compositional%20Reasoning%20with%20Strict%20Entailment%22)

*   **[LVBench: An Extreme Long Video Understanding Benchmark](http://arxiv.org/abs/2406.08035v3)** (ICCV2025 2024.06)
    *   Focus: Multimodal LLMs improve short video understanding, but lack benchmarks for long videos.
    *   project: [https://lvbench.github.io](https://lvbench.github.io)
    *   citation: 184 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.08035) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LVBench%3A%20An%20Extreme%20Long%20Video%20Understanding%20Benchmark%22)

*   **[Vript: A Video Is Worth Thousands of Words](http://arxiv.org/abs/2406.06040v2)** (NeurIPS2024 2024.06)
    *   Focus: Vript introduces a method to create high-quality video-text datasets for multimodal learning.
    *   code: [https://github.com/mutonix/Vript](https://github.com/mutonix/Vript)
    *   citation: 54 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.06040) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Vript%3A%20A%20Video%20Is%20Worth%20Thousands%20of%20Words%22)

*   **[MLVU: Benchmarking Multi-task Long Video Understanding](http://arxiv.org/abs/2406.04264v3)** (CVPR2025 2024.06)
    *   Focus: Existing benchmarks are insufficient for evaluating long video understanding performance.
    *   citation: 77 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.04264) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MLVU%3A%20Benchmarking%20Multi-task%20Long%20Video%20Understanding%22)

*   **[Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](http://arxiv.org/abs/2405.21075v3)** (CVPR2025 2024.05)
    *   Focus: MLLMs advance towards AGI but lack focus on long video understanding.
    *   project: [https://video-mme.github.io](https://video-mme.github.io)
    *   citation: 755 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.21075) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-MME%3A%20The%20First-Ever%20Comprehensive%20Evaluation%20Benchmark%20of%20Multi-modal%20LLMs%20in%20Video%20Analysis%22)

*   **[CinePile: A Long Video Question Answering Dataset and Benchmark](http://arxiv.org/abs/2405.08813v3)** (2024.05)
    *   Focus: Current long-form video datasets lack genuine comprehension challenges, as tasks are solvable by models without deep understanding.
    *   project: [https://ruchitrawal.github.io/cinepile/](https://ruchitrawal.github.io/cinepile/)
    *   citation: 85 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.08813) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CinePile%3A%20A%20Long%20Video%20Question%20Answering%20Dataset%20and%20Benchmark%22)

*   **[WorldQA: Multimodal World Knowledge in Videos through Long-Chain Reasoning](http://arxiv.org/abs/2405.03272v1)** (2024.05)
    *   Focus: LLMs and LMMs struggle to emulate human understanding of complex, dynamic multimodal information.
    *   citation: 16 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.03272) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22WorldQA%3A%20Multimodal%20World%20Knowledge%20in%20Videos%20through%20Long-Chain%20Reasoning%22)

*   **[LvBench: A Benchmark for Long-form Video Understanding with Versatile Multi-modal Question Answering](http://arxiv.org/abs/2312.04817v2)** (ICCV2025 2023.12)
    *   Focus: Current VideoQA datasets use short videos, limiting genuine long-form video understanding.
    *   citation: 22 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.04817) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LvBench%3A%20A%20Benchmark%20for%20Long-form%20Video%20Understanding%20with%20Versatile%20Multi-modal%20Question%20Answering%22)

*   **[Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives](http://arxiv.org/abs/2311.18259v4)** (CVPR2024 2023.11)
    *   Focus: Ego-Exo4D is a large multimodal multiview video dataset with simultaneous egocentric and exocentric recordings.
    *   citation: 301 [[arxiv bibtex]](https://arxiv.org/bibtex/2311.18259) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Ego-Exo4D%3A%20Understanding%20Skilled%20Human%20Activity%20from%20First-%20and%20Third-Person%20Perspectives%22)

*   **[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](http://arxiv.org/abs/2309.13925v2)** (CVPR2024 2023.09)
    *   Focus: Surveillance video tasks need expansion beyond classification to include temporal localization and dense captioning.
    *   project: [https://xuange923.github.io/Surveillance-Video-Understanding](https://xuange923.github.io/Surveillance-Video-Understanding)
    *   citation: 37 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.13925) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Surveillance%20Video-and-Language%20Understanding%3A%20New%20Dataset%2C%20Baselines%2C%20and%20Challenges%22)

*   **[So you think you can track?](http://arxiv.org/abs/2309.07268v1)** (2023.09)
    *   Focus: A 234-hour multi-camera tracking dataset covering 4.2 miles of interstate highway.
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.07268) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22So%20you%20think%20you%20can%20track%3F%22)

*   **[EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding](http://arxiv.org/abs/2308.09126v1)** (NeurIPS2023 2023.08)
    *   Focus: EgoSchema is a long-form video QA dataset and benchmark for evaluating video understanding systems.
    *   project: [http://egoschema.github.io](http://egoschema.github.io)
    *   citation: 463 [[arxiv bibtex]](https://arxiv.org/bibtex/2308.09126) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EgoSchema%3A%20A%20Diagnostic%20Benchmark%20for%20Very%20Long-form%20Video%20Language%20Understanding%22)

*   **[MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](http://arxiv.org/abs/2307.16449v4)** (CVPR2024 2023.07)
    *   Focus: Video foundation models and LLMs are integrated to overcome task-specific limitations in video understanding.
    *   project: [https://rese1f.github.io/MovieChat/](https://rese1f.github.io/MovieChat/)
    *   citation: 429 [[arxiv bibtex]](https://arxiv.org/bibtex/2307.16449) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MovieChat%3A%20From%20Dense%20Token%20to%20Sparse%20Memory%20for%20Long%20Video%20Understanding%22)

*   **[Towards Long Form Audio-visual Video Understanding](http://arxiv.org/abs/2306.09431v1)** (2023.06)
    *   Focus: Long audio-visual videos bridge multimodal information for real-world scenario understanding.
    *   project: [http://gewu-lab.github.io/LFAV/](http://gewu-lab.github.io/LFAV/)
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2306.09431) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Long%20Form%20Audio-visual%20Video%20Understanding%22)

*   **[Building Scalable Video Understanding Benchmarks through Sports](http://arxiv.org/abs/2301.06866v3)** (2023.01)
    *   Focus: Existing long video benchmarks lack scale and annotation quality due to collection difficulties.
    *   project: [https://asap-benchmark.github.io/](https://asap-benchmark.github.io/)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2301.06866) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Building%20Scalable%20Video%20Understanding%20Benchmarks%20through%20Sports%22)

*   **[EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations](http://arxiv.org/abs/2209.13064v1)** (NeurIPS2022 2022.09)
    *   Focus: VISOR introduces a pixel-level annotation dataset and benchmark for hand and active object segmentation in egocentric video.
    *   project: [http://epic-kitchens.github.io/VISOR](http://epic-kitchens.github.io/VISOR)
    *   citation: 127 [[arxiv bibtex]](https://arxiv.org/bibtex/2209.13064) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EPIC-KITCHENS%20VISOR%20Benchmark%3A%20VIdeo%20Segmentations%20and%20Object%20Relations%22)

*   **[Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities](http://arxiv.org/abs/2203.14712v2)** (CVPR2022 2022.03)
    *   Focus: Assembly101 dataset contains 4321 videos of people assembling toy vehicles without fixed instructions.
    *   project: [https://assembly-101.github.io/](https://assembly-101.github.io/)
    *   citation: 280 [[arxiv bibtex]](https://arxiv.org/bibtex/2203.14712) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Assembly101%3A%20A%20Large-Scale%20Multi-View%20Video%20Dataset%20for%20Understanding%20Procedural%20Activities%22)

*   **[VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models](http://arxiv.org/abs/2505.08455v1)** (2025.05)
    *   Focus: LVLMs lack video causal reasoning benchmarks, limiting their evaluation and development.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.08455) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VCRBench%3A%20Exploring%20Long-form%20Causal%20Reasoning%20Capabilities%20of%20Large%20Video%20Language%20Models%22)


### Vision-Language Models
*   **[Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](http://arxiv.org/abs/2507.06261v5)** (2025.07)
    *   Focus: Introduces the Gemini 2.X model family including Pro and Flash variants.
    *   citation: 1054 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.06261) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Gemini%202.5%3A%20Pushing%20the%20Frontier%20with%20Advanced%20Reasoning%2C%20Multimodality%2C%20Long%20Context%2C%20and%20Next%20Generation%20Agentic%20Capabilities%22)

*   **[Qwen3-VL Technical Report](http://arxiv.org/abs/2511.21631v1)** (2025.11)
    *   Focus: Qwen3-VL is a top-performing vision-language model excelling in multimodal benchmarks.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.21631) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Qwen3-VL%20Technical%20Report%22)


*   **[NVIDIA Nemotron Nano V2 VL](http://arxiv.org/abs/2511.03929v2)** (2025.11)
    *   Focus: Nemotron Nano V2 VL advances document understanding, long video comprehension, and reasoning tasks.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.03929) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NVIDIA%20Nemotron%20Nano%20V2%20VL%22)

*   **[Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models](http://arxiv.org/abs/2504.15271v1)** (NeurIPS2025 2025.04)
    *   Focus: Eagle 2.5 introduces vision-language models for long video and high-resolution image understanding.
    *   citation: 20 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.15271) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Eagle%202.5%3A%20Boosting%20Long-Context%20Post-Training%20for%20Frontier%20Vision-Language%20Models%22)

*   **[Qwen2.5-VL Technical Report](http://arxiv.org/abs/2502.13923v1)** (2025.02)
    *   Focus: Qwen2.5-VL advances vision-language capabilities with new features and improved performance.
    *   citation: 2376 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.13923) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Qwen2.5-VL%20Technical%20Report%22)

*   **[Kimi-VL Technical Report](http://arxiv.org/abs/2504.07491v3)** (2025.04)
    *   Focus: Kimi-VL is an efficient open-source MoE vision-language model for multimodal reasoning and long-context understanding.
    *   code: [https://github.com/MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL)
    *   citation: 122 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.07491) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Kimi-VL%20Technical%20Report%22)

*   **[InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling](http://arxiv.org/abs/2501.12386v3)** (2025.01)
    *   Focus: InternVideo2.5 improves video MLLMs using long and rich context modeling.
    *   code: [https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5)
    *   citation: 100 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.12386) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22InternVideo2.5%3A%20Empowering%20Video%20MLLMs%20with%20Long%20and%20Rich%20Context%20Modeling%22)

*   **[GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](http://arxiv.org/abs/2507.01006v5)** (2025.07)
    *   Focus: GLM-4.1V-Thinking and GLM-4.5V are new vision-language models for multimodal understanding and reasoning.
    *   code: [https://github.com/zai-org/GLM-V](https://github.com/zai-org/GLM-V)
    *   citation: 64 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.01006) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22GLM-4.5V%20and%20GLM-4.1V-Thinking%3A%20Towards%20Versatile%20Multimodal%20Reasoning%20with%20Scalable%20Reinforcement%20Learning%22)

*   **[VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation](http://arxiv.org/abs/2412.00927v1)** (CVPR2025 2024.12)
    *   Focus: Lack of high-quality datasets limits large multimodal models' ability to process long or high-resolution videos.
    *   project: [https://tiger-ai-lab.github.io/VISTA/](https://tiger-ai-lab.github.io/VISTA/)
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.00927) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VISTA%3A%20Enhancing%20Long-Duration%20and%20High-Resolution%20Video%20Understanding%20by%20Video%20Spatiotemporal%20Augmentation%22)


*   **[Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy](http://arxiv.org/abs/2502.05177v3)** (2025.02)
    *   Focus: Long-VITA is a multi-modal model for long-context visual-language understanding tasks.
    *   code: [https://github.com/VITA-MLLM/Long-VITA](https://github.com/VITA-MLLM/Long-VITA)
    *   citation: 23 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.05177) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long-VITA%3A%20Scaling%20Large%20Multi-modal%20Models%20to%201%20Million%20Tokens%20with%20Leading%20Short-Context%20Accuracy%22)

*   **[VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling](http://arxiv.org/abs/2501.00574v4)** (2024.12)
    *   Focus: Long-context video modeling is essential for MLLMs but remains challenging.
    *   citation: 94 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.00574) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoChat-Flash%3A%20Hierarchical%20Compression%20for%20Long-Context%20Video%20Modeling%22)

*   **[Cambrian-S: Towards Spatial Supersensing in Video](http://arxiv.org/abs/2511.04670v1)** (2025.11)
    *   Focus: Argues for shifting from reactive systems to supersensing for multimodal intelligence.
    *   project: [https://cambrian-mllm.github.io/](https://cambrian-mllm.github.io/)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.04670) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Cambrian-S%3A%20Towards%20Spatial%20Supersensing%20in%20Video%22)

*   **[From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding](http://arxiv.org/abs/2409.18938v2)** (2024.09)
    *   Focus: LLMs combined with visual encoders improve visual understanding tasks.
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.18938) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Seconds%20to%20Hours%3A%20Reviewing%20MultiModal%20Large%20Language%20Models%20on%20Comprehensive%20Long%20Video%20Understanding%22)

*   **[mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models](http://arxiv.org/abs/2408.04840v2)** (ICLR2025 2024.08)
    *   Focus: MLLMs excel at single-image tasks but face challenges in other areas.
    *   citation: 206 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.04840) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22mPLUG-Owl3%3A%20Towards%20Long%20Image-Sequence%20Understanding%20in%20Multi-Modal%20Large%20Language%20Models%22)

*   **[ST-LLM: Large Language Models Are Effective Temporal Learners](http://arxiv.org/abs/2404.00308v1)** (ECCV2024 2024.03)
    *   Focus: Research explores video LLMs for human-AI interaction using text comprehension and generation.
    *   code: [https://github.com/TencentARC/ST-LLM](https://github.com/TencentARC/ST-LLM)
    *   citation: 119 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.00308) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ST-LLM%3A%20Large%20Language%20Models%20Are%20Effective%20Temporal%20Learners%22)

*   **[Valley: Video Assistant with Large Language model Enhanced abilitY](http://arxiv.org/abs/2306.07207v3)** (2023.06)
    *   Focus: LLMs show promise as multimodal AI assistants but struggle with joint video and language understanding.
    *   code: [https://github.com/valley-vl/Valley](https://github.com/valley-vl/Valley)
    *   citation: 246 [[arxiv bibtex]](https://arxiv.org/bibtex/2306.07207) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Valley%3A%20Video%20Assistant%20with%20Large%20Language%20model%20Enhanced%20abilitY%22)

*   **[OmChat: A Recipe to Train Multimodal Language Models with Strong Long Context and Video Understanding](http://arxiv.org/abs/2407.04923v1)** (2024.07)
    *   Focus: OmChat is a new model for long-context video understanding with standardized visual input processing.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.04923) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22OmChat%3A%20A%20Recipe%20to%20Train%20Multimodal%20Language%20Models%20with%20Strong%20Long%20Context%20and%20Video%20Understanding%22)

*   **[Summarization of Multimodal Presentations with Vision-Language Models: Study of the Effect of Modalities and Structure](http://arxiv.org/abs/2504.10049v1)** (2025.04)
    *   Focus: Fine-grained analysis of Vision-Language Models processing various visual-textual formats including long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.10049) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Summarization%20of%20Multimodal%20Presentations%20with%20Vision-Language%20Models%3A%20Study%20of%20the%20Effect%20of%20Modalities%20and%20Structure%22)

*   **[Efficient Motion-Aware Video MLLM](http://arxiv.org/abs/2503.13016v1)** (CVPR2025 2025.03)
    *   Focus: EMA addresses inefficient video processing and motion awareness in MLLMs.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13016) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Efficient%20Motion-Aware%20Video%20MLLM%22)

*   **[Koala: Key frame-conditioned long video-LLM](http://arxiv.org/abs/2404.04346v3)** (CVPR2024 2024.04)
    *   Focus: Video LLMs struggle with long videos due to short-term focus and lack of fine-grained relationship reasoning.
    *   citation: 58 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.04346) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Koala%3A%20Key%20frame-conditioned%20long%20video-LLM%22)

### Subsampling methods
#### RAG / Memory / Agentic / Language Repository / Frame Sampling Methods

*   **[Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding](http://arxiv.org/abs/2510.14032v1)** (NeurIPS2025 2025.10)
    *   Focus: Long video understanding is limited by token processing constraints in large video language models.
    *   project: [https://xiaoqian-shen.github.io/Vgent](https://xiaoqian-shen.github.io/Vgent)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.14032) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Vgent%3A%20Graph-based%20Retrieval-Reasoning-Augmented%20Generation%20For%20Long%20Video%20Understanding%22)

*   **[VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos](http://arxiv.org/abs/2405.19209v3)** (CVPR2025 2024.05)
    *   Focus: VideoTree is a training-free method for long video understanding that addresses redundancy and irrelevant information.
    *   project: [https://videotree2024.github.io/](https://videotree2024.github.io/)
    *   citation: 135 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.19209) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoTree%3A%20Adaptive%20Tree-based%20Video%20Representation%20for%20LLM%20Reasoning%20on%20Long%20Videos%22)

*   **[REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding](http://arxiv.org/abs/2511.13026v1)** (2025.11)
    *   Focus: Text-based self-reflection struggles with long video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.13026) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22REVISOR%3A%20Beyond%20Textual%20Reflection%2C%20Towards%20Multimodal%20Introspective%20Reasoning%20in%20Long-Form%20Video%20Understanding%22)

*   **[Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding](http://arxiv.org/abs/2511.14446v1)** (2025.11)
    *   Focus: VLMs process videos frame-by-frame, lacking efficient long-range temporal reasoning.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.14446) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Agentic%20Video%20Intelligence%3A%20A%20Flexible%20Framework%20for%20Advanced%20Video%20Exploration%20and%20Understanding%22)


*   **[AVATAAR: Agentic Video Answering via Temporal Adaptive Alignment and Reasoning](http://arxiv.org/abs/2511.15578v1)** (2025.11)
    *   Focus: Proposes a method for understanding and answering questions about long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.15578) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AVATAAR%3A%20Agentic%20Video%20Answering%20via%20Temporal%20Adaptive%20Alignment%20and%20Reasoning%22)

*   **[Adaptive Video Understanding Agent: Enhancing efficiency with dynamic frame sampling and feedback-driven reasoning](http://arxiv.org/abs/2410.20252v1)** (2024.10)
    *   Focus: An agent-based approach for understanding long videos, addressing temporal complexity and computational demands.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.20252) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Adaptive%20Video%20Understanding%20Agent%3A%20Enhancing%20efficiency%20with%20dynamic%20frame%20sampling%20and%20feedback-driven%20reasoning%22)
    
*   **[LAST: LeArning to Think in Space and Time for Generalist Vision-Language Models](http://arxiv.org/abs/2511.19261v1)** (2025.11)
    *   Focus: Vision-language models struggle to understand 3D space and long videos like humans.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.19261) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LAST%3A%20LeArning%20to%20Think%20in%20Space%20and%20Time%20for%20Generalist%20Vision-Language%20Models%22)

*   **[LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling](http://arxiv.org/abs/2511.20785v1)** (2025.11)
    *   Focus: Large multimodal models for video reasoning are prone to hallucinations in long-form content.
    *   code: [https://github.com/EvolvingLMMs-Lab/LongVT](https://github.com/EvolvingLMMs-Lab/LongVT)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.20785) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVT%3A%20Incentivizing%20Thinking%20with%20Long%20Videos%20via%20Native%20Tool%20Calling%22)
    *   code: https://github.com/EvolvingLMMs-Lab/LongVT

*   **[Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding](http://arxiv.org/abs/2511.14446v1)** (2025.11)
    *   Focus: VLMs process videos frame-by-frame, lacking efficient long-range temporal reasoning.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.14446) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Agentic%20Video%20Intelligence%3A%20A%20Flexible%20Framework%20for%20Advanced%20Video%20Exploration%20and%20Understanding%22)

*   **[Vision-Language Memory for Spatial Reasoning](http://arxiv.org/abs/2511.20644v1)** (2025.11)
    *   Focus: Vision-language models underperform humans in video spatial reasoning, highlighting a key research gap.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.20644) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Vision-Language%20Memory%20for%20Spatial%20Reasoning%22)

*   **[iRAG: Advancing RAG for Videos with an Incremental Approach](http://arxiv.org/abs/2404.12309v2)** (2024.04)
    *   Focus: RAG systems combine language generation and retrieval for video understanding tasks.
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.12309) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22iRAG%3A%20Advancing%20RAG%20for%20Videos%20with%20an%20Incremental%20Approach%22)

*   **[Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding](http://arxiv.org/abs/2505.23990v2)** (2025.05)
    *   Focus: Robots need adaptive decision-making and information filtering for effective human interaction.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.23990) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-RAG%3A%20A%20Multimodal%20Retrieval-Augmented%20Generation%20System%20for%20Adaptive%20Video%20Understanding%22)

*   **[F4D: Factorized 4D Convolutional Neural Network for Efficient Video-level Representation Learning](http://arxiv.org/abs/2401.08609v1)** (2023.11)
    *   Focus: Video-level representation learning captures long-range temporal structure for action recognition.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2401.08609) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22F4D%3A%20Factorized%204D%20Convolutional%20Neural%20Network%20for%20Efficient%20Video-level%20Representation%20Learning%22)

*   **[VideoAgent2: Enhancing the LLM-Based Agent System for Long-Form Video Understanding by Uncertainty-Aware CoT](http://arxiv.org/abs/2504.04471v1)** (2025.04)
    *   Focus: Agent-based approaches are gaining popularity for processing long videos.
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.04471) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoAgent2%3A%20Enhancing%20the%20LLM-Based%20Agent%20System%20for%20Long-Form%20Video%20Understanding%20by%20Uncertainty-Aware%20CoT%22)

*   **[VideoLucy: Deep Memory Backtracking for Long Video Understanding](http://arxiv.org/abs/2510.12422v1)** (NeurIPS2025 2025.10)
    *   Focus: Agent-based systems using LLMs for information retrieval show promise in long video understanding.
    *   project: [https://videolucy.github.io](https://videolucy.github.io)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.12422) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoLucy%3A%20Deep%20Memory%20Backtracking%20for%20Long%20Video%20Understanding%22)


*   **[GCAgent: Long-Video Understanding via Schematic and Narrative Episodic Memory](http://arxiv.org/abs/2511.12027v1)** (2025.11)
    *   Focus: MLLMs struggle with long videos due to token limits and temporal dependency complexity.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.12027) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22GCAgent%3A%20Long-Video%20Understanding%20via%20Schematic%20and%20Narrative%20Episodic%20Memory%22)

*   **[VideoSSR: Video Self-Supervised Reinforcement Learning](http://arxiv.org/abs/2511.06281v1)** (2025.11)
    *   Focus: RLVR improves MLLM video understanding, but rapid progress challenges evaluation.
    *   code: [https://github.com/lcqysl/VideoSSR](https://github.com/lcqysl/VideoSSR)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.06281) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoSSR%3A%20Video%20Self-Supervised%20Reinforcement%20Learning%22)

*   **[VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning with LLMs](http://arxiv.org/abs/2409.20365v2)** (2024.09)
    *   Focus: Zero-shot LLM reasoning challenges end-to-end video models but faces efficiency issues.
    *   code: [https://github.com/mayhugotong/VideoINSTA](https://github.com/mayhugotong/VideoINSTA)
    *   citation: 24 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.20365) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoINSTA%3A%20Zero-shot%20Long%20Video%20Understanding%20via%20Informative%20Spatial-Temporal%20Reasoning%20with%20LLMs%22)

*   **[FRAG: Frame Selection Augmented Generation for Long Video and Long Document Understanding](http://arxiv.org/abs/2504.17447v1)** (2025.04)
    *   Focus: Large Multimodal Models face challenges with long inputs due to size and performance constraints.
    *   code: [https://github.com/NVlabs/FRAG](https://github.com/NVlabs/FRAG)
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.17447) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FRAG%3A%20Frame%20Selection%20Augmented%20Generation%20for%20Long%20Video%20and%20Long%20Document%20Understanding%22)

*   **[TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning](http://arxiv.org/abs/2511.05489v1)** (2025.11)
    *   Focus: Temporal search finds minimal relevant frames from long videos for accurate video understanding.
    *   code: [https://github.com/Time-Search/TimeSearch-R](https://github.com/Time-Search/TimeSearch-R)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.05489) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeSearch-R%3A%20Adaptive%20Temporal%20Search%20for%20Long-Form%20Video%20Understanding%20via%20Self-Verification%20Reinforcement%20Learning%22)

*   **[Perceive, Reflect and Understand Long Video: Progressive Multi-Granular Clue Exploration with Interactive Agents](http://arxiv.org/abs/2509.24943v1)** (2025.09)
    *   Focus: LLM-based methods struggle with long video reasoning due to temporal complexity and sparse relevant information.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24943) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Perceive%2C%20Reflect%20and%20Understand%20Long%20Video%3A%20Progressive%20Multi-Granular%20Clue%20Exploration%20with%20Interactive%20Agents%22)

*   **[LOVE-R1: Advancing Long Video Understanding with an Adaptive Zoom-in Mechanism via Multi-Step Reasoning](http://arxiv.org/abs/2509.24786v1)** (2025.09)
    *   Focus: LVLMs struggle with long video understanding due to temporal-spatial perception conflicts.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24786) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LOVE-R1%3A%20Advancing%20Long%20Video%20Understanding%20with%20an%20Adaptive%20Zoom-in%20Mechanism%20via%20Multi-Step%20Reasoning%22)

*   **[FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting](http://arxiv.org/abs/2509.24304v2)** (2025.09)
    *   Focus: LVLMs struggle with long videos due to uniform frame sampling and static text.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24304) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FrameThinker%3A%20Learning%20to%20Think%20with%20Long%20Videos%20via%20Multi-Turn%20Frame%20Spotlighting%22)

*   **[Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](http://arxiv.org/abs/2508.20478v1)** (2025.08)
    *   Focus: Long video understanding faces challenges with temporal dependencies and multiple events.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.20478) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-MTR%3A%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding%22)

*   **[Episodic Memory Representation for Long-form Video Understanding](http://arxiv.org/abs/2508.09486v1)** (2025.08)
    *   Focus: Video-LLMs use keyframe retrieval to overcome context limits for long videos.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.09486) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Episodic%20Memory%20Representation%20for%20Long-form%20Video%20Understanding%22)

*   **[LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents](http://arxiv.org/abs/2503.10200v4)** (ICCV2025 2025.03)
    *   Focus: Agent-based methods use external tools to help MLLMs handle long video temporal context.
    *   code: [https://github.com/64327069/LVAgent](https://github.com/64327069/LVAgent)
    *   citation: 13 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.10200) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LVAgent%3A%20Long%20Video%20Understanding%20by%20Multi-Round%20Dynamical%20Collaboration%20of%20MLLM%20Agents%22)

*   **[Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](http://arxiv.org/abs/2508.09736v4)** (2025.08)
    *   Focus: M3-Agent is a multimodal framework with long-term memory for processing real-time visual and auditory inputs.
    *   code: [https://github.com/bytedance-seed/m3-agent](https://github.com/bytedance-seed/m3-agent)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.09736) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Seeing%2C%20Listening%2C%20Remembering%2C%20and%20Reasoning%3A%20A%20Multimodal%20Agent%20with%20Long-Term%20Memory%22)

*   **[Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning](http://arxiv.org/abs/2508.04416v2)** (2025.08)
    *   Focus: MLLMs need better video reasoning for tasks like QA and temporal grounding, but current methods rely too much on text.
    *   project: [https://zhang9302002.github.io/thinkingwithvideos-page/](https://zhang9302002.github.io/thinkingwithvideos-page/)
    *   citation: 18 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.04416) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Thinking%20With%20Videos%3A%20Multimodal%20Tool-Augmented%20Reinforcement%20Learning%20for%20Long%20Video%20Reasoning%22)

*   **[Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames](http://arxiv.org/abs/2507.02001v1)** (NeurIPS2025 2025.07)
    *   Focus: Long-video understanding remains challenging despite VLMs processing up to 1000 frames.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02001) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Chain%20of%20Thought%3A%20Long-Video%20Understanding%20by%20Thinking%20in%20Frames%22)

*   **[Iterative Zoom-In: Temporal Interval Exploration for Long Video Understanding](http://arxiv.org/abs/2507.02946v1)** (2025.06)
    *   Focus: MLLMs struggle with long videos due to inefficient temporal perception.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02946) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Iterative%20Zoom-In%3A%20Temporal%20Interval%20Exploration%20for%20Long%20Video%20Understanding%22)

*   **[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](http://arxiv.org/abs/2506.13589v3)** (NeurIPS2025 2025.06)
    *   Focus: MLLMs struggle with long videos due to fixed context and weak long-term modeling, suggesting retrieval-augmented generation as a solution.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.13589) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AdaVideoRAG%3A%20Omni-Contextual%20Adaptive%20Retrieval-Augmented%20Efficient%20Long%20Video%20Understanding%22)

*   **[VideoExplorer: Think With Videos For Agentic Long-Video Understanding](http://arxiv.org/abs/2506.10821v6)** (2025.06)
    *   Focus: Existing long-video methods sacrifice details or rely on text, lacking efficient visual modeling.
    *   code: [https://github.com/yhy-2000/VideoDeepResearch](https://github.com/yhy-2000/VideoDeepResearch)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.10821) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoExplorer%3A%20Think%20With%20Videos%20For%20Agentic%20Long-Video%20Understanding%22)

*   **[SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding](http://arxiv.org/abs/2506.07600v1)** (2025.06)
    *   Focus: Long-form video understanding is underexplored due to scale and complexity challenges.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.07600) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SceneRAG%3A%20Scene-level%20Retrieval-Augmented%20Generation%20for%20Video%20Understanding%22)

*   **[VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning](http://arxiv.org/abs/2506.06097v1)** (2025.06)
    *   Focus: MLLMs struggle with long video understanding despite advances in short video analysis.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.06097) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoChat-A1%3A%20Thinking%20with%20Long%20Videos%20by%20Chain-of-Shot%20Reasoning%22)

*   **[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](http://arxiv.org/abs/2505.18079v4)** (NeurIPS2025 2025.05)
    *   Focus: Long video understanding faces challenges from temporal-spatial complexity and extended context question answering.
    *   code: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.18079) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Deep%20Video%20Discovery%3A%20Agentic%20Search%20with%20Tool%20Use%20for%20Long-form%20Video%20Understanding%22)

*   **[MASR: Self-Reflective Reasoning through Multimodal Hierarchical Attention Focusing for Agent-based Video Understanding](http://arxiv.org/abs/2504.17213v2)** (2025.04)
    *   Focus: Video understanding is challenging due to high information redundancy compared to text or images.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.17213) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MASR%3A%20Self-Reflective%20Reasoning%20through%20Multimodal%20Hierarchical%20Attention%20Focusing%20for%20Agent-based%20Video%20Understanding%22)

*   **[MR. Video: "MapReduce" is the Principle for Long Video Understanding](http://arxiv.org/abs/2504.16082v1)** (NeurIPS2025 2025.04)
    *   Focus: MR. Video uses MapReduce for dense perception and reasoning in long video understanding.
    *   code: [https://github.com/ziqipang/MR-Video](https://github.com/ziqipang/MR-Video)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.16082) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MR.%20Video%3A%20MapReduce%20is%20the%20Principle%20for%20Long%20Video%20Understanding%22)

*   **[QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v2)** (2025.05)
    *   Focus: Long-video understanding is crucial for real-world applications but faces challenges.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.16175) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22QuickVideo%3A%20Real-Time%20Long%20Video%20Understanding%20with%20System%20Algorithm%20Co-Design%22)

*   **[AVA: Towards Agentic Video Analytics with Vision Language Models](http://arxiv.org/abs/2505.00254v5)** (2025.05)
    *   Focus: AI video analytics systems lack adaptability for open-ended tasks beyond predefined functions.
    *   code: [https://github.com/I-ESC/Project-Ava](https://github.com/I-ESC/Project-Ava)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.00254) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AVA%3A%20Towards%20Agentic%20Video%20Analytics%20with%20Vision%20Language%20Models%22)

*   **[TimeSearch: Hierarchical Video Search with Spotlight and Reflection for Human-like Long Video Understanding](http://arxiv.org/abs/2504.01407v1)** (2025.04)
    *   Focus: LVLMs face challenges with long videos due to high computational demands and memory constraints.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.01407) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeSearch%3A%20Hierarchical%20Video%20Search%20with%20Spotlight%20and%20Reflection%20for%20Human-like%20Long%20Video%20Understanding%22)

*   **[RAG-Adapter: A Plug-and-Play RAG-enhanced Framework for Long Video Understanding](http://arxiv.org/abs/2503.08576v1)** (2025.03)
    *   Focus: MLLMs are advancing rapidly, requiring long video benchmarks to assess comprehension.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.08576) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22RAG-Adapter%3A%20A%20Plug-and-Play%20RAG-enhanced%20Framework%20for%20Long%20Video%20Understanding%22)

*   **[VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos](http://arxiv.org/abs/2502.01549v1)** (2025.02)
    *   Focus: RAG enhances LLMs with external knowledge but is under-explored for long video understanding.
    *   code: [https://github.com/HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG)
    *   citation: 23 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.01549) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoRAG%3A%20Retrieval-Augmented%20Generation%20with%20Extreme%20Long-Context%20Videos%22)

*   **[Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding](http://arxiv.org/abs/2501.00358v2)** (ICCV2025 2024.12)
    *   Focus: This paper explores dynamic 3D scene understanding from egocentric views for robotics and embodied AI.
    *   project: [https://embodied-videoagent.github.io/](https://embodied-videoagent.github.io/)
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.00358) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Embodied%20VideoAgent%3A%20Persistent%20Memory%20from%20Egocentric%20Videos%20and%20Embodied%20Sensors%20Enables%20Dynamic%20Scene%20Understanding%22)

*   **[Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](http://arxiv.org/abs/2411.13093v4)** (NeurIPS2025 2024.11)
    *   Focus: Fine-tuning long-context LVLMs and using GPT agents to improve long video understanding.
    *   citation: 63 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.13093) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-RAG%3A%20Visually-aligned%20Retrieval-Augmented%20Long%20Video%20Comprehension%22)


*   **[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](http://arxiv.org/abs/2506.13589v3)** (NeurIPS2025 2025.06)
    *   Focus: MLLMs struggle with long videos due to fixed context and weak long-term modeling, suggesting retrieval-augmented generation as a solution.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.13589) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AdaVideoRAG%3A%20Omni-Contextual%20Adaptive%20Retrieval-Augmented%20Efficient%20Long%20Video%20Understanding%22)

*   **[Video-VoT-R1: An efficient video inference model integrating image packing and AoE architecture](http://arxiv.org/abs/2503.15807v1)** (2025.03)
    *   Focus: Proposes a new video-language model to improve inference efficiency and multimodal processing.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.15807) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-VoT-R1%3A%20An%20efficient%20video%20inference%20model%20integrating%20image%20packing%20and%20AoE%20architecture%22)

*   **[AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding](http://arxiv.org/abs/2503.12559v2)** (2025.03)
    *   Focus: MLLMs compress long videos using visual-language models to overcome context length limits.
    *   code: [https://github.com/SCZwangxiao/video-FlexReduc.git](https://github.com/SCZwangxiao/video-FlexReduc.git)
    *   citation: 16 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.12559) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AdaReTaKe%3A%20Adaptive%20Redundancy%20Reduction%20to%20Perceive%20Longer%20for%20Video-language%20Understanding%22)

*   **[Visual Context Window Extension: A New Perspective for Long Video Understanding](http://arxiv.org/abs/2409.20018v2)** (2024.09)
    *   Focus: LMMs struggle with long videos, while LLMs excel using language as a compressed representation.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.20018) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Visual%20Context%20Window%20Extension%3A%20A%20New%20Perspective%20for%20Long%20Video%20Understanding%22)

*   **[Understanding Long Videos with Multimodal Language Models](http://arxiv.org/abs/2403.16998v5)** (ICLR2025 2024.03)
    *   Focus: LLMs' world knowledge and reasoning improve long-video understanding benchmarks.
    *   code: [https://github.com/kahnchana/mvu](https://github.com/kahnchana/mvu)
    *   citation: 15 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.16998) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Understanding%20Long%20Videos%20with%20Multimodal%20Language%20Models%22)
    *   code: https://github.com/kahnchana/mvu

*   **[Language Repository for Long Video Understanding](http://arxiv.org/abs/2403.14622v2)** (2024.03)
    *   Focus: LLMs' long-context effectiveness declines over time despite supporting extended inputs.
    *   code: [https://github.com/kkahatapitiya/LangRepo](https://github.com/kkahatapitiya/LangRepo)
    *   citation: 48 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.14622) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Language%20Repository%20for%20Long%20Video%20Understanding%22)

*   **[VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](http://arxiv.org/abs/2403.11481v2)** (ECCV2024 2024.03)
    *   Focus: A unified memory mechanism combines foundation models for improved video understanding.
    *   project: [http://videoagent.github.io](http://videoagent.github.io)
    *   citation: 139 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.11481) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoAgent%3A%20A%20Memory-augmented%20Multimodal%20Agent%20for%20Video%20Understanding%22)
    *   code: https://github.com/YueFan1014/VideoAgent

*   **[A Simple LLM Framework for Long-Range Video Question-Answering](http://arxiv.org/abs/2312.17235v3)** (2023.12)
    *   Focus: LLoVi is a language-based framework for efficient long-range video question-answering.
    *   code: [https://github.com/CeeZh/LLoVi](https://github.com/CeeZh/LLoVi)
    *   citation: 141 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.17235) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Simple%20LLM%20Framework%20for%20Long-Range%20Video%20Question-Answering%22)

*   **[VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](http://arxiv.org/abs/2510.06040v1)** (ICCV2025 2025.10)
    *   Focus: Proposes a compression method for efficient hour-long video understanding with multi-modal LLMs.
    *   code: [https://github.com/caoxinye/VideoMiner](https://github.com/caoxinye/VideoMiner)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.06040) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoMiner%3A%20Iteratively%20Grounding%20Key%20Frames%20of%20Hour-Long%20Videos%20via%20Tree-based%20Group%20Relative%20Policy%20Optimization%22)
    
*   **[From Captions to Keyframes: KeyScore for Multimodal Frame Scoring and Video-Language Understanding](http://arxiv.org/abs/2510.06509v2)** (2025.10)
    *   Focus: KeyScore selects informative video keyframes using semantics to reduce redundancy.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.06509) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Captions%20to%20Keyframes%3A%20KeyScore%20for%20Multimodal%20Frame%20Scoring%20and%20Video-Language%20Understanding%22)

*   **[FOCUS: Efficient Keyframe Selection for Long Video Understanding](http://arxiv.org/abs/2510.27280v2)** (2025.10)
    *   Focus: MLLMs face impractical token inflation when scaling from images to hour-long videos.
    *   code: [https://github.com/NUS-HPC-AI-Lab/FOCUS](https://github.com/NUS-HPC-AI-Lab/FOCUS)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.27280) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FOCUS%3A%20Efficient%20Keyframe%20Selection%20for%20Long%20Video%20Understanding%22)

*   **[K-frames: Scene-Driven Any-k Keyframe Selection for long video understanding](http://arxiv.org/abs/2510.13891v1)** (2025.10)
    *   Focus: MLLMs struggle with long videos due to context limits and high computational costs.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.13891) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22K-frames%3A%20Scene-Driven%20Any-k%20Keyframe%20Selection%20for%20long%20video%20understanding%22)

*   **[AdaRD-key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video understanding](http://arxiv.org/abs/2510.02778v1)** (2025.10)
    *   Focus: VLMs struggle with long videos due to length and density, needing better compression and modeling.
    *   code: [https://github.com/Xian867/AdaRD-Key](https://github.com/Xian867/AdaRD-Key)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.02778) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AdaRD-key%3A%20Adaptive%20Relevance-Diversity%20Keyframe%20Sampling%20for%20Long-form%20Video%20understanding%22)

*   **[From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding](http://arxiv.org/abs/2510.02262v1)** (2025.10)
    *   Focus: Video LLMs struggle with finding relevant information in large video data.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.02262) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Frames%20to%20Clips%3A%20Efficient%20Key%20Clip%20Selection%20for%20Long-Form%20Video%20Understanding%22)

*   **[KFFocus: Highlighting Keyframes for Enhanced Video Understanding](http://arxiv.org/abs/2508.08989v1)** (2025.08)
    *   Focus: Multimodal LLMs show strong video understanding but face computational challenges from long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.08989) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22KFFocus%3A%20Highlighting%20Keyframes%20for%20Enhanced%20Video%20Understanding%22)


*   **[VSI: Visual Subtitle Integration for Keyframe Selection to enhance Long Video Understanding](http://arxiv.org/abs/2508.06869v3)** (2025.08)
    *   Focus: MLLMs struggle with long videos due to context limits and high computational costs.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.06869) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VSI%3A%20Visual%20Subtitle%20Integration%20for%20Keyframe%20Selection%20to%20enhance%20Long%20Video%20Understanding%22)

*   **[TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](http://arxiv.org/abs/2508.04369v4)** (2025.08)
    *   Focus: MLLMs struggle with long video inputs due to computational and memory constraints.
    *   code: [https://github.com/Hui-design/TSPO](https://github.com/Hui-design/TSPO)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.04369) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TSPO%3A%20Temporal%20Sampling%20Policy%20Optimization%20for%20Long-form%20Video%20Language%20Understanding%22)

*   **[Enhancing Long Video Question Answering with Scene-Localized Frame Grouping](http://arxiv.org/abs/2508.03009v1)** (2025.08)
    *   Focus: MLLMs struggle with long videos due to resource constraints, limiting frame processing and associated text.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.03009) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Enhancing%20Long%20Video%20Question%20Answering%20with%20Scene-Localized%20Frame%20Grouping%22)

*   **[E-VRAG: Enhancing Long Video Understanding with Resource-Efficient Retrieval Augmented Generation](http://arxiv.org/abs/2508.01546v1)** (2025.08)
    *   Focus: Vision-language models advance video understanding but face context limitations.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.01546) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22E-VRAG%3A%20Enhancing%20Long%20Video%20Understanding%20with%20Resource-Efficient%20Retrieval%20Augmented%20Generation%22)

*   **[VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding](http://arxiv.org/abs/2507.13353v1)** (2025.07)
    *   Focus: Selecting informative video frames boosts Video-LLM performance by reducing redundancy.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.13353) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoITG%3A%20Multimodal%20Video%20Understanding%20with%20Instructed%20Temporal%20Grounding%22)

*   **[From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding](http://arxiv.org/abs/2507.02790v2)** (2025.07)
    *   Focus: Efficient video editing techniques are needed to condense long videos into concise summaries.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02790) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Long%20Videos%20to%20Engaging%20Clips%3A%20A%20Human-Inspired%20Video%20Editing%20Framework%20with%20Multimodal%20Narrative%20Understanding%22)

*   **[Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames](http://arxiv.org/abs/2507.02001v1)** (NeurIPS2025 2025.07)
    *   Focus: Long-video understanding remains challenging despite VLMs processing up to 1000 frames.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02001) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Chain%20of%20Thought%3A%20Long-Video%20Understanding%20by%20Thinking%20in%20Frames%22)

*   **[Iterative Zoom-In: Temporal Interval Exploration for Long Video Understanding](http://arxiv.org/abs/2507.02946v1)** (2025.06)
    *   Focus: MLLMs struggle with long videos due to inefficient temporal perception.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02946) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Iterative%20Zoom-In%3A%20Temporal%20Interval%20Exploration%20for%20Long%20Video%20Understanding%22)

*   **[Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](http://arxiv.org/abs/2506.13654v1)** (2025.06)
    *   Focus: Ego-R1 uses Chain-of-Tool-Thought reasoning for ultra-long egocentric video understanding.
    *   project: [https://egolife-ai.github.io/Ego-R1/](https://egolife-ai.github.io/Ego-R1/)
    *   citation: 14 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.13654) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Ego-R1%3A%20Chain-of-Tool-Thought%20for%20Ultra-Long%20Egocentric%20Video%20Reasoning%22)

*   **[Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis](http://arxiv.org/abs/2506.00667v1)** (2025.05)
    *   Focus: Scene segmentation and keyframe extraction are vital for video understanding tasks.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.00667) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Scene%20Detection%20Policies%20and%20Keyframe%20Extraction%20Strategies%20for%20Large-Scale%20Video%20Analysis%22)

*   **[SiLVR: A Simple Language-based Video Reasoning Framework](http://arxiv.org/abs/2505.24869v1)** (2025.05)
    *   Focus: Test-time optimization improves LLM reasoning but faces challenges with long video understanding.
    *   code: [https://github.com/CeeZh/SILVR](https://github.com/CeeZh/SILVR)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.24869) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SiLVR%3A%20A%20Simple%20Language-based%20Video%20Reasoning%20Framework%22)

*   **[Threading Keyframe with Narratives: MLLMs as Strong Long Video Comprehenders](http://arxiv.org/abs/2505.24158v1)** (2025.05)
    *   Focus: MLLMs struggle with long videos due to high frame counts and limited context windows.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.24158) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Threading%20Keyframe%20with%20Narratives%3A%20MLLMs%20as%20Strong%20Long%20Video%20Comprehenders%22)

*   **[BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding](http://arxiv.org/abs/2503.21483v1)** (CVPR2025 2025.03)
    *   Focus: Large video-language models struggle with long-form video analysis due to limited context constraints.
    *   code: [https://github.com/sming256/BOLT](https://github.com/sming256/BOLT)
    *   citation: 18 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.21483) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22BOLT%3A%20Boost%20Large%20Vision-Language%20Model%20Without%20Training%20for%20Long-form%20Video%20Understanding%22)

*   **[From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment](http://arxiv.org/abs/2503.20472v1)** (ICCV2025 2025.03)
    *   Focus: MLLMs struggle with long videos due to limited input capacity.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.20472) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Trial%20to%20Triumph%3A%20Advancing%20Long%20Video%20Understanding%20via%20Visual%20Context%20Sample%20Scaling%20and%20Self-reward%20Alignment%22)

*   **[Self-ReS: Self-Reflection in Large Vision-Language Models for Long Video Understanding](http://arxiv.org/abs/2503.20362v2)** (2025.03)
    *   Focus: LVLMs excel in short videos but struggle with long videos due to linear frame sampling limitations.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.20362) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Self-ReS%3A%20Self-Reflection%20in%20Large%20Vision-Language%20Models%20for%20Long%20Video%20Understanding%22)

*   **[Generative Frame Sampler for Long Video Understanding](http://arxiv.org/abs/2503.09146v2)** (2025.03)
    *   Focus: VideoLLMs struggle to understand long videos with thousands of frames.
    *   code: [https://github.com/yaolinli/GenS](https://github.com/yaolinli/GenS)
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.09146) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Generative%20Frame%20Sampler%20for%20Long%20Video%20Understanding%22)

*   **[DrVideo: Document Retrieval Based Long Video Understanding](http://arxiv.org/abs/2406.12846v2)** (CVPR2025 2024.06)
    *   Focus: Existing video understanding methods are limited to short videos, lacking techniques for long videos.
    *   citation: 31 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.12846) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DrVideo%3A%20Document%20Retrieval%20Based%20Long%20Video%20Understanding%22)

*   **[Adaptive Keyframe Sampling for Long Video Understanding](http://arxiv.org/abs/2502.21271v1)** (CVPR2025 2025.02)
    *   Focus: MLLMs face computational challenges with long videos due to excessive visual tokens.
    *   code: [https://github.com/ncTimTang/AKS](https://github.com/ncTimTang/AKS)
    *   citation: 47 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.21271) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Adaptive%20Keyframe%20Sampling%20for%20Long%20Video%20Understanding%22)

*   **[CoS: Chain-of-Shot Prompting for Long Video Understanding](http://arxiv.org/abs/2502.06428v2)** (2025.02)
    *   Focus: MLLMs face context length limits from excessive visual tokens in long videos.
    *   project: [https://lwpyh.github.io/CoS](https://lwpyh.github.io/CoS)
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.06428) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CoS%3A%20Chain-of-Shot%20Prompting%20for%20Long%20Video%20Understanding%22)
    
*   **[MaxInfo: A Training-Free Key-Frame Selection Method Using Maximum Volume for Enhanced Video Understanding](http://arxiv.org/abs/2502.03183v2)** (2025.02)
    *   Focus: Uniform frame sampling in VLLMs misses critical video information due to redundancy and inefficiency.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.03183) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MaxInfo%3A%20A%20Training-Free%20Key-Frame%20Selection%20Method%20Using%20Maximum%20Volume%20for%20Enhanced%20Video%20Understanding%22)


*   **[$\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation](http://arxiv.org/abs/2501.19098v2)** (2025.01)
    *   Focus: This paper introduces a new method for long-video understanding using compressed representations and temporal modeling.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.19098) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22%24%5Cinfty%24-Video%3A%20A%20Training-Free%20Approach%20to%20Long%20Video%20Understanding%20via%20Continuous-Time%20Memory%20Consolidation%22)

*   **[VCA: Video Curious Agent for Long Video Understanding](http://arxiv.org/abs/2412.10471v2)** (ICCV2025 2024.12)
    *   Focus: Recent methods sample many frames or use auxiliary tools for long video understanding.
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.10471) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VCA%3A%20Video%20Curious%20Agent%20for%20Long%20Video%20Understanding%22)

*   **[Towards Neuro-Symbolic Video Understanding](http://arxiv.org/abs/2403.11021v3)** (ECCV2024 2024.03)
    *   Focus: Efficient frame extraction methods are needed for long-term temporal reasoning in videos.
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.11021) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Neuro-Symbolic%20Video%20Understanding%22)

*   **[VideoAgent: Long-form Video Understanding with Large Language Model as Agent](http://arxiv.org/abs/2403.10517v1)** (ECCV2024 2024.03)
    *   Focus: Long-form video understanding requires models that reason over long multi-modal sequences, inspired by human cognition.
    *   citation: 204 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.10517) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoAgent%3A%20Long-form%20Video%20Understanding%20with%20Large%20Language%20Model%20as%20Agent%22)

*   **[LLMs Meet Long Video: Advancing Long Video Question Answering with An Interactive Visual Adapter in LLMs](http://arxiv.org/abs/2402.13546v2)** (2024.02)
    *   Focus: LLMs face challenges in long video understanding due to computational constraints.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2402.13546) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LLMs%20Meet%20Long%20Video%3A%20Advancing%20Long%20Video%20Question%20Answering%20with%20An%20Interactive%20Visual%20Adapter%20in%20LLMs%22)


### Compression methods
#### New LLM Architectures 
e.g., Mamba, linear attention

*   **[TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding](http://arxiv.org/abs/2511.16595v2)** (2025.11)
    *   Focus: TimeViper is a hybrid vision-language model for efficient long video understanding.
    *   code: [https://github.com/xiaomi-research/timeviper](https://github.com/xiaomi-research/timeviper)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.16595) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeViper%3A%20A%20Hybrid%20Mamba-Transformer%20Vision-Language%20Model%20for%20Efficient%20Long%20Video%20Understanding%22)

*   **[StretchySnake: Flexible SSM Training Unlocks Action Recognition Across Spatio-Temporal Scales](http://arxiv.org/abs/2510.16209v1)** (2025.10)
    *   Focus: State space models offer linear complexity and recurrence for efficient long-range modeling.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.16209) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StretchySnake%3A%20Flexible%20SSM%20Training%20Unlocks%20Action%20Recognition%20Across%20Spatio-Temporal%20Scales%22)

*   **[AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding](http://arxiv.org/abs/2507.02591v3)** (ICCV2025 2025.07)
    *   Focus: Long video understanding faces high computational and memory costs from quadratic scaling in transformers.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02591) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AuroraLong%3A%20Bringing%20RNNs%20Back%20to%20Efficient%20Open-Ended%20Video%20Understanding%22)

*   **[Video RWKV:Video Action Recognition Based RWKV](http://arxiv.org/abs/2411.05636v1)** (2024.11)
    *   Focus: RWKV architecture introduced for efficient long-range video understanding.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.05636) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20RWKV%3AVideo%20Action%20Recognition%20Based%20RWKV%22)

*   **[LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture](http://arxiv.org/abs/2409.02889v3)** (2024.09)
    *   Focus: Systematic approaches are needed to expand MLLMs' long-context capabilities for video understanding and high-resolution image analysis.
    *   citation: 82 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.02889) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongLLaVA%3A%20Scaling%20Multi-modal%20LLMs%20to%201000%20Images%20Efficiently%20via%20a%20Hybrid%20Architecture%22)

*   **[VideoMamba: Spatio-Temporal Selective State Space Model](http://arxiv.org/abs/2407.08476v1)** (ECCV2024 2024.07)
    *   Focus: VideoMamba adapts the Mamba architecture for efficient video recognition without self-attention.
    *   code: [http://github.com/jinyjelly/VideoMamba](http://github.com/jinyjelly/VideoMamba)
    *   citation: 22 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.08476) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoMamba%3A%20Spatio-Temporal%20Selective%20State%20Space%20Model%22)

*   **[Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding](http://arxiv.org/abs/2403.09626v1)** (2024.03)
    *   Focus: Video understanding research explores architectures like RNN, 3D CNN, and Transformers.
    *   code: [https://github.com/OpenGVLab/video-mamba-suite](https://github.com/OpenGVLab/video-mamba-suite)
    *   citation: 116 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.09626) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20Mamba%20Suite%3A%20State%20Space%20Model%20as%20a%20Versatile%20Alternative%20for%20Video%20Understanding%22)

*   **[VideoMamba: State Space Model for Efficient Video Understanding](http://arxiv.org/abs/2403.06977v2)** (ECCV2024 2024.03)
    *   Focus: VideoMamba adapts Mamba to video to address redundancy and global dependencies.
    *   code: [https://github.com/OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba)
    *   citation: 355 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.06977) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoMamba%3A%20State%20Space%20Model%20for%20Efficient%20Video%20Understanding%22)

*   **[World Model on Million-Length Video And Language With Blockwise RingAttention](http://arxiv.org/abs/2402.08268v4)** (ICLR2025 2024.02)
    *   Focus: Long-context understanding is a key challenge for scaling sequence models in AI.
    *   citation: 128 [[arxiv bibtex]](https://arxiv.org/bibtex/2402.08268) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22World%20Model%20on%20Million-Length%20Video%20And%20Language%20With%20Blockwise%20RingAttention%22)

*   **[Selective Structured State-Spaces for Long-Form Video Understanding](http://arxiv.org/abs/2303.14526v1)** (CVPR2023 2023.03)
    *   Focus: S4 model's linear complexity addresses spatiotemporal dependencies in long videos.
    *   citation: 152 [[arxiv bibtex]](https://arxiv.org/bibtex/2303.14526) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Selective%20Structured%20State-Spaces%20for%20Long-Form%20Video%20Understanding%22)

*   **[Multimodal Instruction Tuning with Hybrid State Space Models](http://arxiv.org/abs/2411.08840v1)** (2024.11)
    *   Focus: MLLMs need long context handling for high-resolution images and long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.08840) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multimodal%20Instruction%20Tuning%20with%20Hybrid%20State%20Space%20Models%22)

*   **[MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](http://arxiv.org/abs/2504.16083v2)** (2025.04)
    *   Focus: Long-context VLMs face quadratic attention complexity in pre-filling, limiting efficiency.
    *   citation: 16 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.16083) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MMInference%3A%20Accelerating%20Pre-filling%20for%20Long-Context%20VLMs%20via%20Modality-Aware%20Permutation%20Sparse%20Attention%22)

*   **[MambaMia: A State-Space-Model-Based Compression for Efficient Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2506.13564v1)** (2025.06)
    *   Focus: A framework compresses video-frame features to reduce token explosion in long videos.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.13564) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MambaMia%3A%20A%20State-Space-Model-Based%20Compression%20for%20Efficient%20Video%20Understanding%20in%20Large%20Multimodal%20Models%22)

#### Token Compression

*   **[LongVLM: Efficient Long Video Understanding via Large Language Models](http://arxiv.org/abs/2404.03384v3)** (ECCV2024 2024.04)
    *   Focus: VideoLLMs use LLMs for video understanding by encoding video representations.
    *   code: [https://github.com/ziplab/LongVLM](https://github.com/ziplab/LongVLM)
    *   citation: 114 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.03384) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVLM%3A%20Efficient%20Long%20Video%20Understanding%20via%20Large%20Language%20Models%22)
    *   code: https://github.com/ziplab/LongVLM

*   **[MM-Ego: Towards Building Egocentric Multimodal LLMs for Video QA](http://arxiv.org/abs/2410.07177v2)** (ICLR2025 2024.10)
    *   Focus: This research builds a multimodal foundation model for egocentric video understanding.
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.07177) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MM-Ego%3A%20Towards%20Building%20Egocentric%20Multimodal%20LLMs%20for%20Video%20QA%22)

*   **[Apollo: An Exploration of Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2412.10360v1)** (CVPR2025 2024.12)
    *   Focus: This paper investigates the mechanisms behind video understanding in large multimodal models.
    *   project: [https://apollo-lmms.github.io](https://apollo-lmms.github.io)
    *   citation: 51 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.10360) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Apollo%3A%20An%20Exploration%20of%20Video%20Understanding%20in%20Large%20Multimodal%20Models%22)

*   **[D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition](http://arxiv.org/abs/2510.08818v1)** (ICLR2022 2025.10)
    *   Focus: Vid-LLMs can be built by adapting image-pretrained VLMs, but face challenges with video-specific tasks.
    *   code: [https://github.com/hukcc/D-CoDe](https://github.com/hukcc/D-CoDe)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.08818) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22D-CoDe%3A%20Scaling%20Image-Pretrained%20VLMs%20to%20Video%20via%20Dynamic%20Compression%20and%20Question%20Decomposition%22)

*   **[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](http://arxiv.org/abs/2511.00141v1)** (2025.10)
    *   Focus: Video-LMMs use advanced visual-language reasoning for long video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.00141) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FLoC%3A%20Facility%20Location-Based%20Efficient%20Visual%20Token%20Compression%20for%20Long%20Video%20Understanding%22)

*   **[Unleashing Hour-Scale Video Training for Long Video-Language Understanding](http://arxiv.org/abs/2506.05332v1)** (NeurIPS2025 2025.06)
    *   Focus: Long video understanding benchmarks advance Video-LMMs, but scarce annotated data limits training.
    *   project: [https://videomarathon.github.io/](https://videomarathon.github.io/)
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.05332) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Unleashing%20Hour-Scale%20Video%20Training%20for%20Long%20Video-Language%20Understanding%22)

*   **[Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation](http://arxiv.org/abs/2511.20714v1)** (2025.11)
    *   Focus: World models simulate realistic, interactive long videos for AI agents and gaming.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.20714) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Inferix%3A%20A%20Block-Diffusion%20based%20Next-Generation%20Inference%20Engine%20for%20World%20Simulation%22)

*   **[EventSTU: Event-Guided Efficient Spatio-Temporal Understanding for Video Large Language Models](http://arxiv.org/abs/2511.18920v1)** (2025.11)
    *   Focus: Proposes event-based token compression to reduce inference costs in long video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.18920) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EventSTU%3A%20Event-Guided%20Efficient%20Spatio-Temporal%20Understanding%20for%20Video%20Large%20Language%20Models%22)

*   **[VideoPerceiver: Enhancing Fine-Grained Temporal Perception in Video Multimodal Large Language Models](http://arxiv.org/abs/2511.18823v1)** (2025.11)
    *   Focus: VideoPerceiver improves fine-grained perception in video understanding by enhancing reasoning about brief events.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.18823) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoPerceiver%3A%20Enhancing%20Fine-Grained%20Temporal%20Perception%20in%20Video%20Multimodal%20Large%20Language%20Models%22)

*   **[Test-Time Temporal Sampling for Efficient MLLM Video Understanding](http://arxiv.org/abs/2511.17945v1)** (2025.11)
    *   Focus: MLLMs face computational challenges in long video processing due to quadratic self-attention scaling.
    *   code: [https://github.com/kaibinwang3/T3S](https://github.com/kaibinwang3/T3S)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.17945) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Test-Time%20Temporal%20Sampling%20for%20Efficient%20MLLM%20Video%20Understanding%22)

*   **[CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding](http://arxiv.org/abs/2511.13644v1)** (2025.11)
    *   Focus: Long-form video QA challenges VLMs due to growing attention and KV caches, requiring costly inference.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.13644) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CacheFlow%3A%20Compressive%20Streaming%20Memory%20for%20Efficient%20Long-Form%20Video%20Understanding%22)

*   **[Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models](http://arxiv.org/abs/2511.11910v2)** (2025.11)
    *   Focus: Long video understanding is challenging for MLLMs due to high video token counts.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.11910) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Seeing%20the%20Forest%20and%20the%20Trees%3A%20Query-Aware%20Tokenizer%20for%20Long-Video%20Multimodal%20Language%20Models%22)

*   **[MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](http://arxiv.org/abs/2307.16449v4)** (CVPR2024 2023.07)
    *   Focus: Video foundation models and LLMs are integrated to overcome task-specific limitations in video understanding.
    *   project: [https://rese1f.github.io/MovieChat/](https://rese1f.github.io/MovieChat/)
    *   citation: 429 [[arxiv bibtex]](https://arxiv.org/bibtex/2307.16449) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MovieChat%3A%20From%20Dense%20Token%20to%20Sparse%20Memory%20for%20Long%20Video%20Understanding%22)

*   **[SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models](http://arxiv.org/abs/2407.15841v2)** (2024.07)
    *   Focus: SF-LLaVA is a training-free video LLM capturing spatial details and long-range temporal context.
    *   code: [https://github.com/apple/ml-slowfast-llava](https://github.com/apple/ml-slowfast-llava)
    *   citation: 89 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.15841) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SlowFast-LLaVA%3A%20A%20Strong%20Training-Free%20Baseline%20for%20Video%20Large%20Language%20Models%22)

*   **[Long Video Understanding with Learnable Retrieval in Video-Language Models](http://arxiv.org/abs/2312.04931v3)** (2023.12)
    *   Focus: LLMs are applied to video understanding for their language and reasoning capabilities.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.04931) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long%20Video%20Understanding%20with%20Learnable%20Retrieval%20in%20Video-Language%20Models%22)

*   **[Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](http://arxiv.org/abs/2510.17364v1)** (NeurIPS2025 2025.10)
    *   Focus: Video-LLMs struggle with streaming video understanding due to limited access to full video content.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.17364) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Recurrent%20Attention-based%20Token%20Selection%20for%20Efficient%20Streaming%20Video-LLMs%22)

*   **[Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference](http://arxiv.org/abs/2510.14624v1)** (2025.10)
    *   Focus: Video VLMs face scalability issues due to quadratic costs of dense frame processing.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.14624) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Efficient%20Video%20Sampling%3A%20Pruning%20Temporally%20Redundant%20Tokens%20for%20Faster%20VLM%20Inference%22)

*   **[MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](http://arxiv.org/abs/2510.07915v1)** (2025.10)
    *   Focus: Visual language models face high computational costs when extended from images.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.07915) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MARC%3A%20Memory-Augmented%20RL%20Token%20Compression%20for%20Efficient%20Video%20Understanding%22)

*   **[Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](http://arxiv.org/abs/2510.05836v1)** (ICCV2025 2025.10)
    *   Focus: Long video understanding is challenged by redundancy and limited context.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.05836) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Flow4Agent%3A%20Long-form%20Video%20Understanding%20via%20Motion%20Prior%20from%20Optical%20Flow%22)

*   **[VideoNSA: Native Sparse Attention Scales Video Understanding](http://arxiv.org/abs/2510.02295v1)** (2025.10)
    *   Focus: Addresses video understanding limitations in multimodal models due to short context lengths.
    *   code: [https://github.com/Espere-1119-Song/VideoNSA](https://github.com/Espere-1119-Song/VideoNSA)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.02295) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoNSA%3A%20Native%20Sparse%20Attention%20Scales%20Video%20Understanding%22)

*   **[StreamForest: Efficient Online Video Understanding with Persistent Event Memory](http://arxiv.org/abs/2509.24871v1)** (NeurIPS2025 2025.09)
    *   Focus: MLLMs struggle with real-time video streaming due to storage constraints.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24871) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamForest%3A%20Efficient%20Online%20Video%20Understanding%20with%20Persistent%20Event%20Memory%22)

*   **[Video Panels for Long Video Understanding](http://arxiv.org/abs/2509.23724v1)** (2025.09)
    *   Focus: Video-language models lag behind on long-video tasks compared to images and short videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.23724) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20Panels%20for%20Long%20Video%20Understanding%22)

*   **[Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding](http://arxiv.org/abs/2509.23672v1)** (2025.09)
    *   Focus: Vision Transformers excel in surgical video tasks but face high computational costs.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.23672) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Token%20Merging%20via%20Spatiotemporal%20Information%20Mining%20for%20Surgical%20Video%20Understanding%22)

*   **[Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](http://arxiv.org/abs/2509.01552v1)** (2025.09)
    *   Focus: LVLMs show strong multimodal understanding but face challenges with high-resolution images and long videos.
    *   code: [https://github.com/xuyang-liu16/V2Drop](https://github.com/xuyang-liu16/V2Drop)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.01552) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Variation-aware%20Vision%20Token%20Dropping%20for%20Faster%20Large%20Vision-Language%20Models%22)

*   **[Language-Guided Temporal Token Pruning for Efficient VideoLLM Processing](http://arxiv.org/abs/2508.17686v1)** (2025.08)
    *   Focus: LGTTP uses language guidance to prune temporal tokens, reducing attention complexity for long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.17686) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Language-Guided%20Temporal%20Token%20Pruning%20for%20Efficient%20VideoLLM%20Processing%22)

*   **[StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding](http://arxiv.org/abs/2508.15717v1)** (2025.08)
    *   Focus: MLLMs struggle with long video processing despite recent efficiency improvements.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.15717) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamMem%3A%20Query-Agnostic%20KV%20Cache%20Memory%20for%20Streaming%20Video%20Understanding%22)


*   **[Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](http://arxiv.org/abs/2508.02134v1)** (ICCV2025 2025.08)
    *   Focus: Video-MLLMs face context length limits, hindering long video understanding.
    *   code: [https://github.com/wkfdb/Free-MoRef](https://github.com/wkfdb/Free-MoRef)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.02134) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Free-MoRef%3A%20Instantly%20Multiplexing%20Context%20Perception%20Capabilities%20of%20Video-MLLMs%20within%20Single%20Inference%22)

*   **[Infinite Video Understanding](http://arxiv.org/abs/2507.09068v2)** (2025.07)
    *   Focus: LLMs and MLLMs advance video understanding but face efficiency challenges with long videos.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.09068) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Infinite%20Video%20Understanding%22)


*   **[AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding](http://arxiv.org/abs/2507.02591v3)** (ICCV2025 2025.07)
    *   Focus: Long video understanding faces high computational and memory costs from quadratic scaling in transformers.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02591) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AuroraLong%3A%20Bringing%20RNNs%20Back%20to%20Efficient%20Open-Ended%20Video%20Understanding%22)

*   **[LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs](http://arxiv.org/abs/2506.21862v1)** (2025.06)
    *   Focus: LLaVA-Scissor is a training-free token compression method for video multimodal LLMs.
    *   code: [https://github.com/HumanMLLM/LLaVA-Scissor](https://github.com/HumanMLLM/LLaVA-Scissor)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.21862) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LLaVA-Scissor%3A%20Token%20Compression%20with%20Semantic%20Connected%20Components%20for%20Video%20LLMs%22)

*   **[Task-Aware KV Compression For Cost-Effective Long Video Understanding](http://arxiv.org/abs/2506.21184v1)** (2025.06)
    *   Focus: KV compression methods address computational costs in long-video understanding for MLLMs.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.21184) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Task-Aware%20KV%20Compression%20For%20Cost-Effective%20Long%20Video%20Understanding%22)

*   **[PEVLM: Parallel Encoding for Vision-Language Models](http://arxiv.org/abs/2506.19651v3)** (2025.06)
    *   Focus: Vision-language models struggle with long videos due to high computational demands.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.19651) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PEVLM%3A%20Parallel%20Encoding%20for%20Vision-Language%20Models%22)

*   **[Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification](http://arxiv.org/abs/2506.19225v1)** (2025.06)
    *   Focus: MLLMs struggle with long video processing due to high computational demands.
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.19225) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-XL-2%3A%20Towards%20Very%20Long-Video%20Understanding%20Through%20Task-Aware%20KV%20Sparsification%22)

*   **[InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](http://arxiv.org/abs/2506.15745v2)** (NeurIPS2025 2025.06)
    *   Focus: MLLMs' KV cache grows linearly with video length, exceeding device memory limits.
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.15745) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22InfiniPot-V%3A%20Memory-Constrained%20KV%20Cache%20Compression%20for%20Streaming%20Video%20Understanding%22)

*   **[Memory Consolidation Enables Long-Context Video Understanding](http://arxiv.org/abs/2402.05861v2)** (2024.02)
    *   Focus: Transformer video encoders struggle with long contexts due to quadratic complexity, despite extension attempts.
    *   citation: 44 [[arxiv bibtex]](https://arxiv.org/bibtex/2402.05861) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Memory%20Consolidation%20Enables%20Long-Context%20Video%20Understanding%22)

*   **[CyberV: Cybernetics for Test-time Scaling in Video Understanding](http://arxiv.org/abs/2506.07971v1)** (2025.06)
    *   Focus: MLLMs face challenges with long videos due to high computation, low robustness, and limited accuracy.
    *   code: [https://github.com/marinero4972/CyberV](https://github.com/marinero4972/CyberV)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.07971) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CyberV%3A%20Cybernetics%20for%20Test-time%20Scaling%20in%20Video%20Understanding%22)

*   **[APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](http://arxiv.org/abs/2506.04953v3)** (2025.06)
    *   Focus: MLLMs face challenges in modeling hour-level videos due to high information volume.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.04953) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22APVR%3A%20Hour-Level%20Long%20Video%20Understanding%20with%20Adaptive%20Pivot%20Visual%20Information%20Retrieval%22)

*   **[DynTok: Dynamic Compression of Visual Tokens for Efficient and Effective Video Understanding](http://arxiv.org/abs/2506.03990v1)** (2025.06)
    *   Focus: Video modeling methods use visual tokens for LLM processing, but face challenges with long videos.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.03990) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DynTok%3A%20Dynamic%20Compression%20of%20Visual%20Tokens%20for%20Efficient%20and%20Effective%20Video%20Understanding%22)

*   **[METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding](http://arxiv.org/abs/2506.02850v2)** (2025.06)
    *   Focus: VLLMs struggle with long videos due to high computational demands.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.02850) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22METok%3A%20Multi-Stage%20Event-based%20Token%20Compression%20for%20Efficient%20Long%20Video%20Understanding%22)

*   **[FlexSelect: Flexible Token Selection for Efficient Long Video Understanding](http://arxiv.org/abs/2506.00993v1)** (NeurIPS2025 2025.06)
    *   Focus: FlexSelect reduces computational demands for long video understanding in VideoLLMs.
    *   project: [https://yunzhuzhang0918.github.io/flex_select](https://yunzhuzhang0918.github.io/flex_select)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.00993) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FlexSelect%3A%20Flexible%20Token%20Selection%20for%20Efficient%20Long%20Video%20Understanding%22)

*   **[Clapper: Compact Learning and Video Representation in VLMs](http://arxiv.org/abs/2505.15529v1)** (2025.05)
    *   Focus: Vision-language models need effective temporal modeling for video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.15529) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Clapper%3A%20Compact%20Learning%20and%20Video%20Representation%20in%20VLMs%22)

*   **[RAVU: Retrieval Augmented Video Understanding with Compositional Reasoning over Graph](http://arxiv.org/abs/2505.03173v1)** (2025.05)
    *   Focus: LMMs struggle with long videos due to limited memory and processing constraints.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.03173) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22RAVU%3A%20Retrieval%20Augmented%20Video%20Understanding%20with%20Compositional%20Reasoning%20over%20Graph%22)

*   **[FiLA-Video: Spatio-Temporal Compression for Fine-Grained Long Video Understanding](http://arxiv.org/abs/2504.20384v1)** (2025.04)
    *   Focus: Video understanding in VLLMs has advanced but faces challenges with data complexity and context processing.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.20384) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FiLA-Video%3A%20Spatio-Temporal%20Compression%20for%20Fine-Grained%20Long%20Video%20Understanding%22)

*   **[MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](http://arxiv.org/abs/2504.16083v2)** (2025.04)
    *   Focus: Long-context VLMs face quadratic attention complexity in pre-filling, limiting efficiency.
    *   citation: 16 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.16083) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MMInference%3A%20Accelerating%20Pre-filling%20for%20Long-Context%20VLMs%20via%20Modality-Aware%20Permutation%20Sparse%20Attention%22)

*   **[Multimodal Long Video Modeling Based on Temporal Dynamic Context](http://arxiv.org/abs/2504.10443v1)** (2025.04)
    *   Focus: LLMs advance video understanding but struggle with long video context length.
    *   code: [https://github.com/Hoar012/TDC-Video](https://github.com/Hoar012/TDC-Video)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.10443) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multimodal%20Long%20Video%20Modeling%20Based%20on%20Temporal%20Dynamic%20Context%22)

*   **[Mavors: Multi-granularity Video Representation for Multimodal Large Language Model](http://arxiv.org/abs/2504.10068v1)** (2025.04)
    *   Focus: MLLMs struggle to balance computational efficiency with fine-grained spatio-temporal pattern retention in long videos.
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.10068) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Mavors%3A%20Multi-granularity%20Video%20Representation%20for%20Multimodal%20Large%20Language%20Model%22)

*   **[LVC: A Lightweight Compression Framework for Enhancing VLMs in Long Video Understanding](http://arxiv.org/abs/2504.06835v1)** (2025.04)
    *   Focus: VLMs achieve frame-level understanding but struggle with long video comprehension.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.06835) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LVC%3A%20A%20Lightweight%20Compression%20Framework%20for%20Enhancing%20VLMs%20in%20Long%20Video%20Understanding%22)

*   **[Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation](http://arxiv.org/abs/2504.02438v5)** (2025.04)
    *   Focus: Token pruning and feature merging address computational costs in long video processing.
    *   code: [https://github.com/steven-ccq/ViLAMP](https://github.com/steven-ccq/ViLAMP)
    *   citation: 16 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.02438) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Scaling%20Video-Language%20Models%20to%2010K%20Frames%20via%20Hierarchical%20Differential%20Distillation%22)

*   **[SlowFast-LLaVA-1.5: A Family of Token-Efficient Video Large Language Models for Long-Form Video Understanding](http://arxiv.org/abs/2503.18943v2)** (2025.03)
    *   Focus: SF-LLaVA-1.5 is a token-efficient video LLM family for long video understanding.
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.18943) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SlowFast-LLaVA-1.5%3A%20A%20Family%20of%20Token-Efficient%20Video%20Large%20Language%20Models%20for%20Long-Form%20Video%20Understanding%22)

*   **[Video-XL-Pro: Reconstructive Token Compression for Extremely Long Video Understanding](http://arxiv.org/abs/2503.18478v2)** (2025.03)
    *   Focus: Video-XL-Pro is an efficient MLLM for long video understanding.
    *   citation: 29 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.18478) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-XL-Pro%3A%20Reconstructive%20Token%20Compression%20for%20Extremely%20Long%20Video%20Understanding%22)

*   **[XAttention: Block Sparse Attention with Antidiagonal Scoring](http://arxiv.org/abs/2503.16428v1)** (2025.03)
    *   Focus: Block-sparse attention reduces computational costs in long-context transformers.
    *   code: [https://github.com/mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention)
    *   citation: 48 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.16428) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22XAttention%3A%20Block%20Sparse%20Attention%20with%20Antidiagonal%20Scoring%22)

*   **[Long-VMNet: Accelerating Long-Form Video Understanding via Fixed Memory](http://arxiv.org/abs/2503.13707v1)** (2025.03)
    *   Focus: Long-form video understanding is essential but computationally intensive for traditional methods.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13707) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long-VMNet%3A%20Accelerating%20Long-Form%20Video%20Understanding%20via%20Fixed%20Memory%22)

*   **[Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding](http://arxiv.org/abs/2503.13139v2)** (NeurIPS2025 2025.03)
    *   Focus: Current long video understanding methods neglect logical relations in dense captions and feature selection.
    *   citation: 15 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13139) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Logic-in-Frames%3A%20Dynamic%20Keyframe%20Search%20via%20Visual%20Semantic-Logical%20Verification%20for%20Long%20Video%20Understanding%22)

*   **[Efficient Motion-Aware Video MLLM](http://arxiv.org/abs/2503.13016v1)** (CVPR2025 2025.03)
    *   Focus: EMA addresses inefficient video processing and motion awareness in MLLMs.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13016) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Efficient%20Motion-Aware%20Video%20MLLM%22)

*   **[Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](http://arxiv.org/abs/2503.11579v2)** (ICCV2025 2025.03)
    *   Focus: Transformers struggle with long videos due to quadratic attention complexity and high computational costs.
    *   project: [https://tiger-ai-lab.github.io/Vamba/](https://tiger-ai-lab.github.io/Vamba/)
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.11579) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Vamba%3A%20Understanding%20Hour-Long%20Videos%20with%20Hybrid%20Mamba-Transformers%22)

*   **[FastVID: Dynamic Density Pruning for Fast Video Large Language Models](http://arxiv.org/abs/2503.11187v2)** (NeurIPS2025 2025.03)
    *   Focus: Video LLMs have strong understanding but high inference costs from redundant tokens.
    *   code: [https://github.com/LunarShen/FastVID](https://github.com/LunarShen/FastVID)
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.11187) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FastVID%3A%20Dynamic%20Density%20Pruning%20for%20Fast%20Video%20Large%20Language%20Models%22)

*   **[Keyframe-oriented Vision Token Pruning: Enhancing Efficiency of Large Vision Language Models on Long-Form Video Processing](http://arxiv.org/abs/2503.10742v2)** (ICCV2025 2025.03)
    *   Focus: Vision language models face high computational costs from redundant visual data.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.10742) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Keyframe-oriented%20Vision%20Token%20Pruning%3A%20Enhancing%20Efficiency%20of%20Large%20Vision%20Language%20Models%20on%20Long-Form%20Video%20Processing%22)

*   **[VideoScan: Enabling Efficient Streaming Video Understanding via Frame-level Semantic Carriers](http://arxiv.org/abs/2503.09387v2)** (2025.03)
    *   Focus: VideoScan enables real-time video interaction with efficient VLM inference for streamed video comprehension.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.09387) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoScan%3A%20Enabling%20Efficient%20Streaming%20Video%20Understanding%20via%20Frame-level%20Semantic%20Carriers%22)

*   **[Memory-enhanced Retrieval Augmentation for Long Video Understanding](http://arxiv.org/abs/2503.09149v2)** (2025.03)
    *   Focus: Long-video understanding faces challenges from compression and brute-force methods in current models.
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.09149) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Memory-enhanced%20Retrieval%20Augmentation%20for%20Long%20Video%20Understanding%22)

*   **[QuoTA: Query-oriented Token Assignment via CoT Query Decouple for Long Video Comprehension](http://arxiv.org/abs/2503.08689v1)** (2025.03)
    *   Focus: Critiques attention-based pruning for long videos and proposes a new compression method.
    *   code: [https://github.com/MAC-AutoML/QuoTA](https://github.com/MAC-AutoML/QuoTA)
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.08689) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22QuoTA%3A%20Query-oriented%20Token%20Assignment%20via%20CoT%20Query%20Decouple%20for%20Long%20Video%20Comprehension%22)

*   **[HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](http://arxiv.org/abs/2503.08585v2)** (CVPR2025 2025.03)
    *   Focus: MLLMs struggle with long videos due to frame and context limitations.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.08585) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HierarQ%3A%20Task-Aware%20Hierarchical%20Q-Former%20for%20Enhanced%20Video%20Understanding%22)

*   **[STORM: Token-Efficient Long Video Understanding for Multimodal LLMs](http://arxiv.org/abs/2503.04130v4)** (2025.03)
    *   Focus: Video-LLMs process videos as image sequences but face efficiency and context length challenges.
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.04130) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22STORM%3A%20Token-Efficient%20Long%20Video%20Understanding%20for%20Multimodal%20LLMs%22)

*   **[iMOVE: Instance-Motion-Aware Video Understanding](http://arxiv.org/abs/2502.11594v2)** (2025.02)
    *   Focus: Improving Video LLMs' fine-grained motion perception for better temporal understanding.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.11594) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22iMOVE%3A%20Instance-Motion-Aware%20Video%20Understanding%22)

*   **[LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding](http://arxiv.org/abs/2501.05067v2)** (2025.01)
    *   Focus: LLaVA-Octopus adaptively weights visual features for video understanding based on user instructions.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.05067) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LLaVA-Octopus%3A%20Unlocking%20Instruction-Driven%20Adaptive%20Projector%20Fusion%20for%20Video%20Understanding%22)

*   **[VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling](http://arxiv.org/abs/2501.00574v4)** (2024.12)
    *   Focus: Long-context video modeling is essential for MLLMs but remains challenging.
    *   citation: 94 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.00574) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoChat-Flash%3A%20Hierarchical%20Compression%20for%20Long-Context%20Video%20Modeling%22)

*   **[FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models](http://arxiv.org/abs/2501.01986v2)** (ICCV2025 2024.12)
    *   Focus: Existing token reduction methods for long videos are reviewed and a new efficient compression approach is proposed.
    *   code: [https://github.com/thu-nics/FrameFusion](https://github.com/thu-nics/FrameFusion)
    *   citation: 20 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.01986) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FrameFusion%3A%20Combining%20Similarity%20and%20Importance%20for%20Video%20Token%20Reduction%20on%20Large%20Vision%20Language%20Models%22)

*   **[ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding](http://arxiv.org/abs/2412.20504v5)** (2024.12)
    *   Focus: VideoLLMs struggle with long videos due to LLM limitations; new compression methods are proposed.
    *   code: [https://github.com/SCZwangxiao/video-ReTaKe](https://github.com/SCZwangxiao/video-ReTaKe)
    *   citation: 23 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.20504) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ReTaKe%3A%20Reducing%20Temporal%20and%20Knowledge%20Redundancy%20for%20Long%20Video%20Understanding%22)

*   **[B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](http://arxiv.org/abs/2412.09919v2)** (ICCV2025 2024.12)
    *   Focus: Vision LLMs encode visual content into sequences for understanding.
    *   code: [https://github.com/zhuqiangLu/B-VLLM](https://github.com/zhuqiangLu/B-VLLM)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.09919) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22B-VLLM%3A%20A%20Vision%20Large%20Language%20Model%20with%20Balanced%20Spatio-Temporal%20Tokens%22)

*   **[IQViC: In-context, Question Adaptive Vision Compressor for Long-term Video Understanding LMMs](http://arxiv.org/abs/2412.09907v2)** (2024.12)
    *   Focus: Existing methods struggle with accurate long-term temporal understanding in complex videos.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.09907) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22IQViC%3A%20In-context%2C%20Question%20Adaptive%20Vision%20Compressor%20for%20Long-term%20Video%20Understanding%20LMMs%22)

*   **[PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models](http://arxiv.org/abs/2412.09613v1)** (CVPR2025 2024.12)
    *   Focus: VLMs use visual token compression to handle long video inputs efficiently.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.09613) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PVC%3A%20Progressive%20Visual%20Token%20Compression%20for%20Unified%20Image%20and%20Video%20Processing%20in%20Large%20Vision-Language%20Models%22)

*   **[Espresso: High Compression For Rich Extraction From Videos for Your Vision-Language Model](http://arxiv.org/abs/2412.04729v3)** (2024.12)
    *   Focus: Vision-language models struggle with long videos due to token growth.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.04729) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Espresso%3A%20High%20Compression%20For%20Rich%20Extraction%20From%20Videos%20for%20Your%20Vision-Language%20Model%22)

*   **[AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](http://arxiv.org/abs/2412.03248v2)** (ICCV2025 2024.12)
    *   Focus: Multi-modal LLMs show strong video understanding but require extensive visual token compression.
    *   code: [https://github.com/LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM)
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.03248) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AIM%3A%20Adaptive%20Inference%20of%20Multi-Modal%20LLMs%20via%20Token%20Merging%20and%20Pruning%22)

*   **[SEAL: Semantic Attention Learning for Long Video Representation](http://arxiv.org/abs/2412.01798v3)** (CVPR2025 2024.12)
    *   Focus: Long video understanding requires efficient representations to reduce computational complexity and temporal redundancy.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.01798) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SEAL%3A%20Semantic%20Attention%20Learning%20for%20Long%20Video%20Representation%22)

*   **[Look Every Frame All at Once: Video-Ma$^2$mba for Efficient Long-form Video Understanding with Multi-Axis Gradient Checkpointing](http://arxiv.org/abs/2411.19460v1)** (2024.11)
    *   Focus: Long video processing faces high computational costs from quadratic memory and time demands.
    *   project: [https://ivy-lvlm.github.io/Video-MA2MBA/](https://ivy-lvlm.github.io/Video-MA2MBA/)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.19460) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Look%20Every%20Frame%20All%20at%20Once%3A%20Video-Ma%24%5E2%24mba%20for%20Efficient%20Long-form%20Video%20Understanding%20with%20Multi-Axis%20Gradient%20Checkpointing%22)

*   **[SAVEn-Vid: Synergistic Audio-Visual Integration for Enhanced Understanding in Long Video Context](http://arxiv.org/abs/2411.16213v2)** (2024.11)
    *   Focus: Video-LLMs struggle with long video understanding despite recent advances.
    *   project: [https://ljungang.github.io/SAVEn-Vid/](https://ljungang.github.io/SAVEn-Vid/)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.16213) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SAVEn-Vid%3A%20Synergistic%20Audio-Visual%20Integration%20for%20Enhanced%20Understanding%20in%20Long%20Video%20Context%22)

*   **[SALOVA: Segment-Augmented Long Video Assistant for Targeted Retrieval and Routing in Long-Form Video Analysis](http://arxiv.org/abs/2411.16173v2)** (CVPR2025 2024.11)
    *   Focus: LMMs struggle with long videos due to context length limits and high memory usage.
    *   project: [https://ivy-lvlm.github.io/SALOVA/](https://ivy-lvlm.github.io/SALOVA/)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.16173) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SALOVA%3A%20Segment-Augmented%20Long%20Video%20Assistant%20for%20Targeted%20Retrieval%20and%20Routing%20in%20Long-Form%20Video%20Analysis%22)

*   **[ReWind: Understanding Long Videos with Instructed Learnable Memory](http://arxiv.org/abs/2411.15556v2)** (CVPR2025 2024.11)
    *   Focus: Vision-language models face computational inefficiency challenges when processing long videos.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.15556) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ReWind%3A%20Understanding%20Long%20Videos%20with%20Instructed%20Learnable%20Memory%22)

*   **[AdaCM$^2$: On Understanding Extremely Long-Term Video with Adaptive Cross-Modality Memory Reduction](http://arxiv.org/abs/2411.12593v3)** (CVPR2025 2024.11)
    *   Focus: LLM-based video models struggle with long videos due to high computational costs and limited context length.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.12593) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AdaCM%24%5E2%24%3A%20On%20Understanding%20Extremely%20Long-Term%20Video%20with%20Adaptive%20Cross-Modality%20Memory%20Reduction%22)

*   **[DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](http://arxiv.org/abs/2411.12355v2)** (CVPR2025 2024.11)
    *   Focus: LLM-based video understanding struggles with preserving information in long videos while managing token count.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.12355) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DynFocus%3A%20Dynamic%20Cooperative%20Network%20Empowers%20LLMs%20with%20Video%20Understanding%22)

*   **[PPLLaVA: Varied Video Sequence Understanding With Prompt Guidance](http://arxiv.org/abs/2411.02327v2)** (2024.11)
    *   Focus: Video LLMs advance but struggle with unified short and long video understanding.
    *   code: [https://github.com/farewellthree/PPLLaVA](https://github.com/farewellthree/PPLLaVA)
    *   citation: 16 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.02327) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PPLLaVA%3A%20Varied%20Video%20Sequence%20Understanding%20With%20Prompt%20Guidance%22)

*   **[Video Token Merging for Long-form Video Understanding](http://arxiv.org/abs/2410.23782v1)** (2024.10)
    *   Focus: Transformer models face challenges with long video inputs, requiring alternatives to sampling.
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.23782) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20Token%20Merging%20for%20Long-form%20Video%20Understanding%22)

*   **[LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding](http://arxiv.org/abs/2410.17434v1)** (2024.10)
    *   Focus: MLLMs struggle with long video processing due to LLM context limits.
    *   project: [https://vision-cair.github.io/LongVU](https://vision-cair.github.io/LongVU)
    *   citation: 150 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.17434) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVU%3A%20Spatiotemporal%20Adaptive%20Compression%20for%20Long%20Video-Language%20Understanding%22)

*   **[VidCompress: Memory-Enhanced Temporal Compression for Video Understanding in Large Language Models](http://arxiv.org/abs/2410.11417v1)** (2024.10)
    *   Focus: Video-LLMs treat videos as frame sequences, missing temporal dynamics.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.11417) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VidCompress%3A%20Memory-Enhanced%20Temporal%20Compression%20for%20Video%20Understanding%20in%20Large%20Language%20Models%22)

*   **[Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos](http://arxiv.org/abs/2410.02763v1)** (2024.10)
    *   Focus: Research shifts focus to long video understanding as short video challenges are considered solved.
    *   project: [https://vinoground.github.io](https://vinoground.github.io)
    *   citation: 15 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.02763) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Vinoground%3A%20Scrutinizing%20LMMs%20over%20Dense%20Temporal%20Reasoning%20with%20Short%20Videos%22)

*   **[Learning to Localize Actions in Instructional Videos with LLM-Based Multi-Pathway Text-Video Alignment](http://arxiv.org/abs/2409.16145v1)** (ECCV2024 2024.09)
    *   Focus: Proposes a method to localize steps in instructional videos using limited annotations.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.16145) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Learning%20to%20Localize%20Actions%20in%20Instructional%20Videos%20with%20LLM-Based%20Multi-Pathway%20Text-Video%20Alignment%22)

*   **[Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](http://arxiv.org/abs/2409.14485v4)** (CVPR2025 2024.09)
    *   Focus: MLLMs struggle with long videos due to limited context length and high computational costs.
    *   citation: 123 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.14485) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-XL%3A%20Extra-Long%20Vision%20Language%20Model%20for%20Hour-Scale%20Video%20Understanding%22)

*   **[Enhancing Long Video Understanding via Hierarchical Event-Based Memory](http://arxiv.org/abs/2409.06299v1)** (2024.09)
    *   Focus: Video understanding systems integrate visual models with LLMs, often compressing diverse video data.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.06299) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Enhancing%20Long%20Video%20Understanding%20via%20Hierarchical%20Event-Based%20Memory%22)

*   **[LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture](http://arxiv.org/abs/2409.02889v3)** (2024.09)
    *   Focus: Systematic approaches are needed to expand MLLMs' long-context capabilities for video understanding and high-resolution image analysis.
    *   citation: 82 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.02889) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongLLaVA%3A%20Scaling%20Multi-modal%20LLMs%20to%201000%20Images%20Efficiently%20via%20a%20Hybrid%20Architecture%22)

*   **[VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](http://arxiv.org/abs/2409.01071v2)** (ICCV2025 2024.09)
    *   Focus: Large video-language models face computational and data scarcity challenges for real-time planning.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.01071) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoLLaMB%3A%20Long%20Streaming%20Video%20Understanding%20with%20Recurrent%20Memory%20Bridges%22)

*   **[HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](http://arxiv.org/abs/2408.17443v4)** (ICCV2025 2024.08)
    *   Focus: Long-form video understanding faces challenges in capturing long-range dependencies and processing redundant information.
    *   project: [https://joslefaure.github.io/assets/html/hermes.html](https://joslefaure.github.io/assets/html/hermes.html)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.17443) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HERMES%3A%20temporal-coHERent%20long-forM%20understanding%20with%20Episodes%20and%20Semantics%22)

*   **[VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation](http://arxiv.org/abs/2408.16730v1)** (NeurIPS2024 2024.08)
    *   Focus: Increasing vision tokens improves understanding but raises memory costs in large vision-language models.
    *   citation: 26 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.16730) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoLLM-MoD%3A%20Efficient%20Video-Language%20Streaming%20with%20Mixture-of-Depths%20Vision%20Computation%22)

*   **[Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input](http://arxiv.org/abs/2408.15542v1)** (NeurIPS2024 2024.08)
    *   Focus: Extending LLMs to handle video input remains a challenging research area.
    *   citation: 99 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.15542) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Kangaroo%3A%20A%20Powerful%20Video-Language%20Model%20Supporting%20Long-context%20Video%20Input%22)

*   **[Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](http://arxiv.org/abs/2407.12679v1)** (ECCV2024 2024.07)
    *   Focus: LLM-based video models struggle with long videos due to noise, redundancy, and memory constraints.
    *   project: [https://vision-cair.github.io/Goldfish_website/](https://vision-cair.github.io/Goldfish_website/)
    *   citation: 31 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.12679) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Goldfish%3A%20Vision-Language%20Understanding%20of%20Arbitrarily%20Long%20Videos%22)

*   **[MovieChat+: Question-aware Sparse Memory for Long Video Question Answering](http://arxiv.org/abs/2404.17176v1)** (2024.04)
    *   Focus: Video foundation models and LLMs overcome task limitations but face efficiency challenges.
    *   code: [https://github.com/rese1f/MovieChat](https://github.com/rese1f/MovieChat)
    *   citation: 47 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.17176) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MovieChat%2B%3A%20Question-aware%20Sparse%20Memory%20for%20Long%20Video%20Question%20Answering%22)

*   **[MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](http://arxiv.org/abs/2404.05726v2)** (CVPR2024 2024.04)
    *   Focus: Vision-language models need better long video understanding, which current LLM-based methods struggle with.
    *   project: [https://boheumd.github.io/MA-LMM/](https://boheumd.github.io/MA-LMM/)
    *   citation: 160 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.05726) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MA-LMM%3A%20Memory-Augmented%20Large%20Multimodal%20Model%20for%20Long-Term%20Video%20Understanding%22)

*   **[Text-Conditioned Resampler For Long Form Video Understanding](http://arxiv.org/abs/2312.11897v3)** (ECCV2024 2023.12)
    *   Focus: A text-conditioned video resampler uses frozen visual and language models to process long videos.
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.11897) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Text-Conditioned%20Resampler%20For%20Long%20Form%20Video%20Understanding%22)

*   **[TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding](http://arxiv.org/abs/2312.02051v2)** (CVPR2024 2023.12)
    *   Focus: TimeChat is a time-sensitive MLLM for long video understanding with timestamp-aware frame tokenization and temporal attention.
    *   code: [https://github.com/RenShuhuai-Andy/TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
    *   citation: 326 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.02051) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeChat%3A%20A%20Time-sensitive%20Multimodal%20Large%20Language%20Model%20for%20Long%20Video%20Understanding%22)

*   **[LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](http://arxiv.org/abs/2311.17043v1)** (ECCV2024 2023.11)
    *   Focus: LLaMA-VID reduces tokens for efficient video/image understanding in Vision Language Models.
    *   code: [https://github.com/dvlab-research/LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
    *   citation: 450 [[arxiv bibtex]](https://arxiv.org/bibtex/2311.17043) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LLaMA-VID%3A%20An%20Image%20is%20Worth%202%20Tokens%20in%20Large%20Language%20Models%22)

*   **[TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding](http://arxiv.org/abs/2310.19060v1)** (2023.10)
    *   Focus: Video-language pre-training advances understanding but faces high computational costs from video encoding.
    *   code: [https://github.com/RenShuhuai-Andy/TESTA](https://github.com/RenShuhuai-Andy/TESTA)
    *   citation: 39 [[arxiv bibtex]](https://arxiv.org/bibtex/2310.19060) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TESTA%3A%20Temporal-Spatial%20Token%20Aggregation%20for%20Long-form%20Video-Language%20Understanding%22)

*   **[Query-aware Long Video Localization and Relation Discrimination for Deep Video Understanding](http://arxiv.org/abs/2310.12724v1)** (2023.10)
    *   Focus: Existing video understanding techniques excel with short formats but face challenges with long videos.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2310.12724) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Query-aware%20Long%20Video%20Localization%20and%20Relation%20Discrimination%20for%20Deep%20Video%20Understanding%22)

*   **[From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding](http://arxiv.org/abs/2409.18938v2)** (2024.09)
    *   Focus: LLMs combined with visual encoders improve visual understanding tasks.
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.18938) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Seconds%20to%20Hours%3A%20Reviewing%20MultiModal%20Large%20Language%20Models%20on%20Comprehensive%20Long%20Video%20Understanding%22)


### Temporal Modeling (timestamp / time positional encoding)

*   **[HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](http://arxiv.org/abs/2505.20444v2)** (NeurIPS2025 2025.05)
    *   Focus: VLMs struggle with long videos due to limited context windows, requiring new architectures for long-range dependencies.
    *   code: [https://github.com/hrlics/HoPE](https://github.com/hrlics/HoPE)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.20444) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HoPE%3A%20Hybrid%20of%20Position%20Embedding%20for%20Long%20Context%20Vision-Language%20Models%22)

*   **[VideoRoPE: What Makes for Good Video Rotary Position Embedding?](http://arxiv.org/abs/2502.05173v3)** (2025.02)
    *   Focus: Extending 1D RoPE to video remains challenging due to complex spatio-temporal structure.
    *   code: [https://github.com/Wiselnn570/VideoRoPE](https://github.com/Wiselnn570/VideoRoPE)
    *   citation: 28 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.05173) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoRoPE%3A%20What%20Makes%20for%20Good%20Video%20Rotary%20Position%20Embedding%3F%22)


### Downstream tasks 
#### Real-time Interaction

*   **[AHA -- Predicting What Matters Next: Online Highlight Detection Without Looking Ahead](http://arxiv.org/abs/2509.16421v2)** (NeurIPS2025 2025.09)
    *   Focus: Real-time video stream understanding is critical for autonomous vehicles, drones, and disaster response agents.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.16421) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AHA%20--%20Predicting%20What%20Matters%20Next%3A%20Online%20Highlight%20Detection%20Without%20Looking%20Ahead%22)

*   **[StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding](http://arxiv.org/abs/2508.01875v3)** (2025.08)
    *   Focus: Real-time video streaming for autonomous driving and surveillance requires continuous perception beyond offline methods.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.01875) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamAgent%3A%20Towards%20Anticipatory%20Agents%20for%20Streaming%20Video%20Understanding%22)

*   **[TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos](http://arxiv.org/abs/2504.17343v1)** (2025.04)
    *   Focus: Real-time video understanding is needed for live streaming services.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.17343) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeChat-Online%3A%2080%25%20Visual%20Tokens%20are%20Naturally%20Redundant%20in%20Streaming%20Videos%22)

*   **[Streaming Long Video Understanding with Large Language Models](http://arxiv.org/abs/2405.16009v1)** (NeurIPS2024 2024.05)
    *   Focus: VideoStreaming is a VLLM that processes arbitrary-length videos using a constant number of tokens.
    *   citation: 102 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.16009) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Streaming%20Long%20Video%20Understanding%20with%20Large%20Language%20Models%22)

*   **[Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge](http://arxiv.org/abs/2501.13468v1)** (ICLR2025 2025.01)
    *   Focus: Video-LLMs advance multimodal learning but struggle with long video understanding.
    *   code: [https://github.com/hmxiong/StreamChat](https://github.com/hmxiong/StreamChat)
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.13468) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Streaming%20Video%20Understanding%20and%20Multi-round%20Interaction%20with%20Memory-enhanced%20Knowledge%22)

*   **[Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding](http://arxiv.org/abs/2504.13915v1)** (2025.04)
    *   Focus: ProVideLLM is an end-to-end framework for real-time procedural video understanding with multimodal caching.
    *   project: [https://dibschat.github.io/ProVideLLM](https://dibschat.github.io/ProVideLLM)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.13915) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Memory-efficient%20Streaming%20VideoLLMs%20for%20Real-time%20Procedural%20Video%20Understanding%22)

*   **[LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval](http://arxiv.org/abs/2505.15269v1)** (2025.05)
    *   Focus: Video LLMs excel at long videos but lack benchmarks for temporal understanding.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.15269) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LiveVLM%3A%20Efficient%20Online%20Video%20Understanding%20via%20Streaming-Oriented%20KV%20Cache%20and%20Retrieval%22)

*   **[StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant](http://arxiv.org/abs/2505.05467v2)** (NeurIPS2025 2025.05)
    *   Focus: StreamBridge transforms offline Video-LLMs into streaming-capable models.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.05467) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamBridge%3A%20Turning%20Your%20Offline%20Video%20Large%20Language%20Model%20into%20a%20Proactive%20Streaming%20Assistant%22)

*   **[video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory](http://arxiv.org/abs/2510.11129v1)** (2025.10)
    *   Focus: Proposes a scalable method for continuous high-frame-rate video processing to overcome LLM limitations.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.11129) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22video-SALMONN%20S%3A%20Streaming%20Audio-Visual%20LLMs%20Beyond%20Length%20Limits%20via%20Memory%22)

*   **[StreamingVLM: Real-Time Understanding for Infinite Video Streams](http://arxiv.org/abs/2510.09608v1)** (2025.10)
    *   Focus: VLMs struggle with real-time video understanding due to latency and memory constraints.
    *   code: [https://github.com/mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.09608) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamingVLM%3A%20Real-Time%20Understanding%20for%20Infinite%20Video%20Streams%22)

*   **[An Egocentric Vision-Language Model based Portable Real-time Smart Assistant](http://arxiv.org/abs/2503.04250v1)** (2025.03)
    *   Focus: Vinci is a portable AI system using EgoVideo-VL for real-time vision-language assistance.
    *   code: [https://github.com/OpenGVLab/vinci](https://github.com/OpenGVLab/vinci)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.04250) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22An%20Egocentric%20Vision-Language%20Model%20based%20Portable%20Real-time%20Smart%20Assistant%22)

*   **[Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction](http://arxiv.org/abs/2501.03218v1)** (CVPR2025 2025.01)
    *   Focus: Video LLMs enable real-time interaction by understanding user intent and responding during continuous video processing.
    *   code: [https://github.com/Mark12Ding/Dispider](https://github.com/Mark12Ding/Dispider)
    *   citation: 28 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.03218) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Dispider%3A%20Enabling%20Video%20LLMs%20with%20Active%20Real-Time%20Interaction%20via%20Disentangled%20Perception%2C%20Decision%2C%20and%20Reaction%22)

*   **[Memory-augmented Online Video Anomaly Detection](http://arxiv.org/abs/2302.10719v2)** (2023.02)
    *   Focus: An online system for autonomous vehicles to understand scenes and provide immediate responses.
    *   code: [https://github.com/IMPLabUniPr/movad/tree/movad_vad](https://github.com/IMPLabUniPr/movad/tree/movad_vad)
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2302.10719) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Memory-augmented%20Online%20Video%20Anomaly%20Detection%22)

*   **[StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling](http://arxiv.org/abs/2507.05240v1)** (2025.07)
    *   Focus: VLN agents process continuous video streams with low latency to follow language instructions.
    *   project: [https://streamvln.github.io/](https://streamvln.github.io/)
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.05240) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22StreamVLN%3A%20Streaming%20Vision-and-Language%20Navigation%20via%20SlowFast%20Context%20Modeling%22)

*   **[Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams](http://arxiv.org/abs/2406.08085v2)** (ICCV2025 2024.06)
    *   Focus: Existing video understanding methods excel offline but face challenges in real-time applications.
    *   project: [https://invinciblewyq.github.io/vstream-page/](https://invinciblewyq.github.io/vstream-page/)
    *   citation: 87 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.08085) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Flash-VStream%3A%20Memory-Based%20Real-Time%20Understanding%20for%20Long%20Video%20Streams%22)

#### Dense Video Captioning

*   **[SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](http://arxiv.org/abs/2510.17777v1)** (ICCV2025 2025.10)
    *   Focus: VLMs advance visual-textual reasoning for high-res images, long videos, and multi-turn conversations.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.17777) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SparseVILA%3A%20Decoupling%20Visual%20Sparsity%20for%20Efficient%20VLM%20Inference%22)

*   **[SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding](http://arxiv.org/abs/2510.13016v2)** (2025.10)
    *   Focus: AI systems need to understand fine-grained actions and localize actors in space and time.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.13016) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SVAG-Bench%3A%20A%20Large-Scale%20Benchmark%20for%20Multi-Instance%20Spatio-temporal%20Video%20Action%20Grounding%22)

*   **[Addressing the ID-Matching Challenge in Long Video Captioning](http://arxiv.org/abs/2510.06973v1)** (2025.10)
    *   Focus: Addresses challenges in generating captions for long, complex videos for text-to-video and multi-modal applications.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.06973) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Addressing%20the%20ID-Matching%20Challenge%20in%20Long%20Video%20Captioning%22)

*   **[Time-Scaling State-Space Models for Dense Video Captioning](http://arxiv.org/abs/2509.03426v1)** (2025.09)
    *   Focus: Dense video captioning segments videos into events and generates detailed descriptions for each.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.03426) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Time-Scaling%20State-Space%20Models%20for%20Dense%20Video%20Captioning%22)


*   **[From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding](http://arxiv.org/abs/2507.02790v2)** (2025.07)
    *   Focus: Efficient video editing techniques are needed to condense long videos into concise summaries.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.02790) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20Long%20Videos%20to%20Engaging%20Clips%3A%20A%20Human-Inspired%20Video%20Editing%20Framework%20with%20Multimodal%20Narrative%20Understanding%22)

*   **[LongAnimation: Long Animation Generation with Dynamic Global-Local Memory](http://arxiv.org/abs/2507.01945v2)** (ICCV2025 2025.07)
    *   Focus: Automated colorization for long animation videos to reduce labor costs.
    *   project: [https://cn-makers.github.io/long_animation_web/](https://cn-makers.github.io/long_animation_web/)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.01945) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongAnimation%3A%20Long%20Animation%20Generation%20with%20Dynamic%20Global-Local%20Memory%22)

*   **[A Culturally-diverse Multilingual Multimodal Video Benchmark & Model](http://arxiv.org/abs/2506.07032v3)** (2025.06)
    *   Focus: The paper proposes a new Chinese large multimodal model for video understanding with improved efficiency.
    *   project: [https://mbzuai-oryx.github.io/ViMUL/](https://mbzuai-oryx.github.io/ViMUL/)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.07032) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Culturally-diverse%20Multilingual%20Multimodal%20Video%20Benchmark%20%26%20Model%22)

*   **[QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v2)** (2025.05)
    *   Focus: Long-video understanding is crucial for real-world applications but faces challenges.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.16175) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22QuickVideo%3A%20Real-Time%20Long%20Video%20Understanding%20with%20System%20Algorithm%20Co-Design%22)

*   **[Action Anticipation from SoccerNet Football Video Broadcasts](http://arxiv.org/abs/2504.12021v1)** (2025.04)
    *   Focus: AI enables analysis of long sports videos for action understanding and motion prediction.
    *   code: [https://github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.12021) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Action%20Anticipation%20from%20SoccerNet%20Football%20Video%20Broadcasts%22)

*   **[DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description](http://arxiv.org/abs/2503.24096v1)** (2025.03)
    *   Focus: Audio Description aids vision-impaired audiences by narrating key visual elements in videos.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.24096) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DANTE-AD%3A%20Dual-Vision%20Attention%20Network%20for%20Long-Term%20Audio%20Description%22)

*   **[Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding](http://arxiv.org/abs/2503.13139v2)** (NeurIPS2025 2025.03)
    *   Focus: Current long video understanding methods neglect logical relations in dense captions and feature selection.
    *   citation: 15 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13139) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Logic-in-Frames%3A%20Dynamic%20Keyframe%20Search%20via%20Visual%20Semantic-Logical%20Verification%20for%20Long%20Video%20Understanding%22)

*   **[Prompt2LVideos: Exploring Prompts for Understanding Long-Form Multimodal Videos](http://arxiv.org/abs/2503.08335v1)** (2025.03)
    *   Focus: Long video understanding is challenging due to reliance on manually annotated video-caption datasets.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.08335) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Prompt2LVideos%3A%20Exploring%20Prompts%20for%20Understanding%20Long-Form%20Multimodal%20Videos%22)

*   **[MANTA: Diffusion Mamba for Efficient and Effective Stochastic Long-Term Dense Anticipation](http://arxiv.org/abs/2501.08837v2)** (2025.01)
    *   Focus: Challenges in predicting future actions and durations from video observations.
    *   code: [https://github.com/olga-zats/DIFF_MANTA](https://github.com/olga-zats/DIFF_MANTA)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.08837) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MANTA%3A%20Diffusion%20Mamba%20for%20Efficient%20and%20Effective%20Stochastic%20Long-Term%20Dense%20Anticipation%22)

*   **[Video LLMs for Temporal Reasoning in Long Videos](http://arxiv.org/abs/2412.02930v4)** (2024.12)
    *   Focus: TemporalVLM enables temporal reasoning and fine-grained understanding in long videos.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.02930) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20LLMs%20for%20Temporal%20Reasoning%20in%20Long%20Videos%22)

*   **[LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos](http://arxiv.org/abs/2411.19772v3)** (CVPR2025 2024.11)
    *   Focus: Proposes a framework for fine-grained omni-modal video understanding using hierarchical alignment and contrastive learning.
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.19772) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVALE%3A%20Vision-Audio-Language-Event%20Benchmark%20Towards%20Time-Aware%20Omni-Modal%20Perception%20of%20Long%20Videos%22)

*   **[Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](http://arxiv.org/abs/2411.16932v1)** (CVPR2025 2024.11)
    *   Focus: Video LLMs need temporal awareness for tasks like dense captioning and temporal grounding.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.16932) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Seq2Time%3A%20Sequential%20Knowledge%20Transfer%20for%20Video%20LLM%20Temporal%20Grounding%22)

*   **[FIOVA: A Multi-Annotator Benchmark for Human-Aligned Video Captioning](http://arxiv.org/abs/2410.15270v2)** (2024.10)
    *   Focus: Existing video caption benchmarks inadequately assess LVLM alignment with human understanding due to single-annotation limitations.
    *   project: [https://huuuuusy.github.io/fiova/](https://huuuuusy.github.io/fiova/)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.15270) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FIOVA%3A%20A%20Multi-Annotator%20Benchmark%20for%20Human-Aligned%20Video%20Captioning%22)

*   **[AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark](http://arxiv.org/abs/2410.03051v4)** (ICLR2025 2024.10)
    *   Focus: This paper proposes a method for generating detailed and coherent video captions.
    *   project: [https://rese1f.github.io/aurora-web/](https://rese1f.github.io/aurora-web/)
    *   citation: 87 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.03051) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AuroraCap%3A%20Efficient%2C%20Performant%20Video%20Detailed%20Captioning%20and%20a%20New%20Benchmark%22)

*   **[YouTube Video Analytics for Patient Engagement: Evidence from Colonoscopy Preparation Videos](http://arxiv.org/abs/2410.02830v1)** (2024.10)
    *   Focus: Video analysis methods for medical education content are explored.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.02830) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22YouTube%20Video%20Analytics%20for%20Patient%20Engagement%3A%20Evidence%20from%20Colonoscopy%20Preparation%20Videos%22)

*   **[InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output](http://arxiv.org/abs/2407.03320v1)** (2024.07)
    *   Focus: IXC-2.5 is a versatile vision-language model for long-context text-image tasks.
    *   code: [https://github.com/InternLM/InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)
    *   citation: 166 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.03320) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22InternLM-XComposer-2.5%3A%20A%20Versatile%20Large%20Vision%20Language%20Model%20Supporting%20Long-Contextual%20Input%20and%20Output%22)

*   **[VIA: Unified Spatiotemporal Video Adaptation Framework for Global and Local Video Editing](http://arxiv.org/abs/2406.12831v3)** (2024.06)
    *   Focus: Video editing is crucial in digital media but existing methods often neglect key requirements.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.12831) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VIA%3A%20Unified%20Spatiotemporal%20Video%20Adaptation%20Framework%20for%20Global%20and%20Local%20Video%20Editing%22)

*   **[ST-LLM: Large Language Models Are Effective Temporal Learners](http://arxiv.org/abs/2404.00308v1)** (ECCV2024 2024.03)
    *   Focus: Research explores video LLMs for human-AI interaction using text comprehension and generation.
    *   code: [https://github.com/TencentARC/ST-LLM](https://github.com/TencentARC/ST-LLM)
    *   citation: 119 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.00308) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ST-LLM%3A%20Large%20Language%20Models%20Are%20Effective%20Temporal%20Learners%22)

*   **[Towards Multimodal Video Paragraph Captioning Models Robust to Missing Modality](http://arxiv.org/abs/2403.19221v1)** (2024.03)
    *   Focus: Video paragraph captioning models are constrained by limited data and inefficient architectures.
    *   code: [https://github.com/lancopku/MR-VPC](https://github.com/lancopku/MR-VPC)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.19221) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Multimodal%20Video%20Paragraph%20Captioning%20Models%20Robust%20to%20Missing%20Modality%22)

*   **[Panonut360: A Head and Eye Tracking Dataset for Panoramic Video](http://arxiv.org/abs/2403.17708v1)** (2024.03)
    *   Focus: VR/AR technology advances require personalized immersive panoramic video services.
    *   project: [https://dianvrlab.github.io/Panonut360/](https://dianvrlab.github.io/Panonut360/)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.17708) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Panonut360%3A%20A%20Head%20and%20Eye%20Tracking%20Dataset%20for%20Panoramic%20Video%22)

*   **[Video ReCap: Recursive Captioning of Hour-Long Videos](http://arxiv.org/abs/2402.13250v6)** (CVPR2024 2024.02)
    *   Focus: Proposes a model for long video understanding and dense captioning of high-level concepts.
    *   citation: 78 [[arxiv bibtex]](https://arxiv.org/bibtex/2402.13250) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20ReCap%3A%20Recursive%20Captioning%20of%20Hour-Long%20Videos%22)

*   **[Shot2Story: A New Benchmark for Comprehensive Understanding of Multi-shot Videos](http://arxiv.org/abs/2312.10300v3)** (ICLR2025 2023.12)
    *   Focus: Video understanding requires capturing individual events and their associations to comprehend storylines.
    *   citation: 41 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.10300) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Shot2Story%3A%20A%20New%20Benchmark%20for%20Comprehensive%20Understanding%20of%20Multi-shot%20Videos%22)

*   **[MM-VID: Advancing Video Understanding with GPT-4V(ision)](http://arxiv.org/abs/2310.19773v1)** (2023.10)
    *   Focus: MM-VID integrates GPT-4V with specialized tools for advanced video understanding.
    *   project: [https://multimodal-vid.github.io/](https://multimodal-vid.github.io/)
    *   citation: 84 [[arxiv bibtex]](https://arxiv.org/bibtex/2310.19773) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MM-VID%3A%20Advancing%20Video%20Understanding%20with%20GPT-4V%28ision%29%22)

*   **[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](http://arxiv.org/abs/2309.13925v2)** (CVPR2024 2023.09)
    *   Focus: Surveillance video tasks need expansion beyond classification to include temporal localization and dense captioning.
    *   project: [https://xuange923.github.io/Surveillance-Video-Understanding](https://xuange923.github.io/Surveillance-Video-Understanding)
    *   citation: 37 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.13925) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Surveillance%20Video-and-Language%20Understanding%3A%20New%20Dataset%2C%20Baselines%2C%20and%20Challenges%22)

*   **[KuaiSAR: A Unified Search And Recommendation Dataset](http://arxiv.org/abs/2306.07705v4)** (2023.06)
    *   Focus: Search and recommendation integration is key for online platforms like e-commerce and video services.
    *   citation: 24 [[arxiv bibtex]](https://arxiv.org/bibtex/2306.07705) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22KuaiSAR%3A%20A%20Unified%20Search%20And%20Recommendation%20Dataset%22)

*   **[MUG: A General Meeting Understanding and Generation Benchmark](http://arxiv.org/abs/2303.13939v2)** (2023.03)
    *   Focus: Proposes a method to efficiently extract key information from long video/audio recordings using ASR transcripts.
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2303.13939) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MUG%3A%20A%20General%20Meeting%20Understanding%20and%20Generation%20Benchmark%22)

*   **[METEOR Guided Divergence for Video Captioning](http://arxiv.org/abs/2212.10690v1)** (2022.12)
    *   Focus: Video captioning requires temporal context modeling and action comprehension for holistic scene understanding.
    *   code: [https://github.com/d-rothen/bmhrl](https://github.com/d-rothen/bmhrl)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2212.10690) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22METEOR%20Guided%20Divergence%20for%20Video%20Captioning%22)

*   **[REVECA -- Rich Encoder-decoder framework for Video Event CAptioner](http://arxiv.org/abs/2206.09178v1)** (2022.06)
    *   Focus: A rich encoder-decoder framework for video boundary event captioning.
    *   code: [https://github.com/TooTouch/REVECA](https://github.com/TooTouch/REVECA)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2206.09178) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22REVECA%20--%20Rich%20Encoder-decoder%20framework%20for%20Video%20Event%20CAptioner%22)


*   **[Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding](http://arxiv.org/abs/2504.13915v1)** (2025.04)
    *   Focus: ProVideLLM is an end-to-end framework for real-time procedural video understanding with multimodal caching.
    *   project: [https://dibschat.github.io/ProVideLLM](https://dibschat.github.io/ProVideLLM)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.13915) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Memory-efficient%20Streaming%20VideoLLMs%20for%20Real-time%20Procedural%20Video%20Understanding%22)


#### Temporal Action Detection

*   **[ContextDet: Temporal Action Detection with Adaptive Context Aggregation](http://arxiv.org/abs/2410.15279v1)** (2024.10)
    *   Focus: TAD faces challenges from variable segment lengths and ambiguous boundaries in video understanding.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.15279) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ContextDet%3A%20Temporal%20Action%20Detection%20with%20Adaptive%20Context%20Aggregation%22)

*   **[Harnessing Temporal Causality for Advanced Temporal Action Detection](http://arxiv.org/abs/2407.17792v2)** (2024.07)
    *   Focus: Temporal action detection identifies actions with precise boundaries in untrimmed videos.
    *   code: [https://github.com/sming256/OpenTAD/](https://github.com/sming256/OpenTAD/)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.17792) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Harnessing%20Temporal%20Causality%20for%20Advanced%20Temporal%20Action%20Detection%22)

*   **[TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization](http://arxiv.org/abs/2303.09055v1)** (2023.03)
    *   Focus: TAL identifies and localizes actions in videos, with recent focus on appearance features.
    *   code: [https://github.com/TuanTNG/TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer)
    *   citation: 38 [[arxiv bibtex]](https://arxiv.org/bibtex/2303.09055) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TemporalMaxer%3A%20Maximize%20Temporal%20Context%20with%20only%20Max%20Pooling%20for%20Temporal%20Action%20Localization%22)

*   **[An Efficient Spatio-Temporal Pyramid Transformer for Action Detection](http://arxiv.org/abs/2207.10448v1)** (ECCV2022 2022.07)
    *   Focus: Action detection in long videos using vision Transformers to classify and localize actions.
    *   citation: 30 [[arxiv bibtex]](https://arxiv.org/bibtex/2207.10448) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22An%20Efficient%20Spatio-Temporal%20Pyramid%20Transformer%20for%20Action%20Detection%22)

*   **[Temporal Action Segmentation: An Analysis of Modern Techniques](http://arxiv.org/abs/2210.10352v5)** (2022.10)
    *   Focus: Temporal action segmentation identifies action classes in long videos, requiring long-range understanding.
    *   code: [https://github.com/nus-cvml/awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation)
    *   citation: 111 [[arxiv bibtex]](https://arxiv.org/bibtex/2210.10352) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Action%20Segmentation%3A%20An%20Analysis%20of%20Modern%20Techniques%22)

*   **[Streaming Video Temporal Action Segmentation In Real Time](http://arxiv.org/abs/2209.13808v3)** (2022.09)
    *   Focus: TAS models use features over raw video for long-term understanding.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2209.13808) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Streaming%20Video%20Temporal%20Action%20Segmentation%20In%20Real%20Time%22)

#### Temporal Video Grounding

*   **[TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning](http://arxiv.org/abs/2410.19702v2)** (ICLR2025 2024.10)
    *   Focus: MLLMs struggle with long video understanding despite success with short videos.
    *   citation: 49 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.19702) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeSuite%3A%20Improving%20MLLMs%20for%20Long%20Video%20Understanding%20via%20Grounded%20Tuning%22)

*   **[TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability](http://arxiv.org/abs/2411.18211v1)** (2024.11)
    *   Focus: Video-language models struggle with long videos due to computational limits and lack of long-range benchmarks.
    *   code: [https://github.com/TimeMarker-LLM/TimeMarker/](https://github.com/TimeMarker-LLM/TimeMarker/)
    *   citation: 35 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.18211) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeMarker%3A%20A%20Versatile%20Video-LLM%20for%20Long%20and%20Short%20Video%20Understanding%20with%20Superior%20Temporal%20Localization%20Ability%22)

*   **[ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning](http://arxiv.org/abs/2505.15447v1)** (2025.05)
    *   Focus: MLLMs enable flexible video understanding by focusing on goal-relevant frames.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.15447) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ViaRL%3A%20Adaptive%20Temporal%20Grounding%20via%20Visual%20Iterated%20Amplification%20Reinforcement%20Learning%22)

*   **[LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling](http://arxiv.org/abs/2511.20785v1)** (2025.11)
    *   Focus: Large multimodal models for video reasoning are prone to hallucinations in long-form content.
    *   code: [https://github.com/EvolvingLMMs-Lab/LongVT](https://github.com/EvolvingLMMs-Lab/LongVT)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.20785) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LongVT%3A%20Incentivizing%20Thinking%20with%20Long%20Videos%20via%20Native%20Tool%20Calling%22)
    *   code: https://github.com/EvolvingLMMs-Lab/LongVT

*   **[LAST: LeArning to Think in Space and Time for Generalist Vision-Language Models](http://arxiv.org/abs/2511.19261v1)** (2025.11)
    *   Focus: Vision-language models struggle to understand 3D space and long videos like humans.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.19261) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LAST%3A%20LeArning%20to%20Think%20in%20Space%20and%20Time%20for%20Generalist%20Vision-Language%20Models%22)

*   **[VideoPerceiver: Enhancing Fine-Grained Temporal Perception in Video Multimodal Large Language Models](http://arxiv.org/abs/2511.18823v1)** (2025.11)
    *   Focus: VideoPerceiver improves fine-grained perception in video understanding by enhancing reasoning about brief events.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.18823) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoPerceiver%3A%20Enhancing%20Fine-Grained%20Temporal%20Perception%20in%20Video%20Multimodal%20Large%20Language%20Models%22)

*   **[FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos](http://arxiv.org/abs/2511.16183v1)** (2025.11)
    *   Focus: Soccer video datasets support tasks like action localization, detection, and tracking.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.16183) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FOOTPASS%3A%20A%20Multi-Modal%20Multi-Agent%20Tactical%20Context%20Dataset%20for%20Play-by-Play%20Action%20Spotting%20in%20Soccer%20Broadcast%20Videos%22)

*   **[TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning](http://arxiv.org/abs/2511.05489v1)** (2025.11)
    *   Focus: Temporal search finds minimal relevant frames from long videos for accurate video understanding.
    *   code: [https://github.com/Time-Search/TimeSearch-R](https://github.com/Time-Search/TimeSearch-R)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.05489) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeSearch-R%3A%20Adaptive%20Temporal%20Search%20for%20Long-Form%20Video%20Understanding%20via%20Self-Verification%20Reinforcement%20Learning%22)

*   **[NVIDIA Nemotron Nano V2 VL](http://arxiv.org/abs/2511.03929v2)** (2025.11)
    *   Focus: Nemotron Nano V2 VL advances document understanding, long video comprehension, and reasoning tasks.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.03929) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NVIDIA%20Nemotron%20Nano%20V2%20VL%22)

*   **[Conan: Progressive Learning to Reason Like a Detective over Multi-Scale Visual Evidence](http://arxiv.org/abs/2510.20470v2)** (2025.10)
    *   Focus: RL-based methods improve video reasoning by enabling multi-step deduction across frames.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.20470) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Conan%3A%20Progressive%20Learning%20to%20Reason%20Like%20a%20Detective%20over%20Multi-Scale%20Visual%20Evidence%22)

*   **[SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](http://arxiv.org/abs/2510.17777v1)** (ICCV2025 2025.10)
    *   Focus: VLMs advance visual-textual reasoning for high-res images, long videos, and multi-turn conversations.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.17777) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SparseVILA%3A%20Decoupling%20Visual%20Sparsity%20for%20Efficient%20VLM%20Inference%22)

*   **[Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](http://arxiv.org/abs/2510.17364v1)** (NeurIPS2025 2025.10)
    *   Focus: Video-LLMs struggle with streaming video understanding due to limited access to full video content.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.17364) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Recurrent%20Attention-based%20Token%20Selection%20for%20Efficient%20Streaming%20Video-LLMs%22)

*   **[SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding](http://arxiv.org/abs/2510.13016v2)** (2025.10)
    *   Focus: AI systems need to understand fine-grained actions and localize actors in space and time.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.13016) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SVAG-Bench%3A%20A%20Large-Scale%20Benchmark%20for%20Multi-Instance%20Spatio-temporal%20Video%20Action%20Grounding%22)

*   **[Tracking the Spatiotemporal Evolution of Landslide Scars Using a Vision Foundation Model: A Novel and Universal Framework](http://arxiv.org/abs/2510.10084v1)** (2025.10)
    *   Focus: Proposes a method for tracking large-scale landslide scar evolution to improve early-warning systems.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.10084) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Tracking%20the%20Spatiotemporal%20Evolution%20of%20Landslide%20Scars%20Using%20a%20Vision%20Foundation%20Model%3A%20A%20Novel%20and%20Universal%20Framework%22)

*   **[Online Generic Event Boundary Detection](http://arxiv.org/abs/2510.06855v1)** (ICCV2025 2025.10)
    *   Focus: GEBD detects event boundaries in long videos but current methods need full frame processing.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.06855) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Online%20Generic%20Event%20Boundary%20Detection%22)

*   **[Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models](http://arxiv.org/abs/2510.05034v6)** (2025.10)
    *   Focus: Video understanding requires reasoning about complex spatiotemporal relationships and long-term dependencies.
    *   code: [https://github.com/yunlong10/Awesome-Video-LMM-Post-Training](https://github.com/yunlong10/Awesome-Video-LMM-Post-Training)
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.05034) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-LMM%20Post-Training%3A%20A%20Deep%20Dive%20into%20Video%20Reasoning%20with%20Large%20Multimodal%20Models%22)

*   **[Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs](http://arxiv.org/abs/2510.00705v1)** (2025.10)
    *   Focus: MLLMs struggle with fine-grained perception in high-resolution images and long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.00705) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Training-free%20Uncertainty%20Guidance%20for%20Complex%20Visual%20Tasks%20with%20MLLMs%22)

*   **[TimeScope: Towards Task-Oriented Temporal Grounding In Long Videos](http://arxiv.org/abs/2509.26360v2)** (2025.09)
    *   Focus: Introduces Task-oriented Temporal Grounding (ToTG) for locating key moments in long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.26360) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeScope%3A%20Towards%20Task-Oriented%20Temporal%20Grounding%20In%20Long%20Videos%22)

*   **[NeMo: Needle in a Montage for Video-Language Understanding](http://arxiv.org/abs/2509.24563v2)** (2025.09)
    *   Focus: Proposes new benchmarks for evaluating temporal reasoning in video-language models.
    *   project: [https://lavi-lab.github.io/NeMoBench](https://lavi-lab.github.io/NeMoBench)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.24563) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NeMo%3A%20Needle%20in%20a%20Montage%20for%20Video-Language%20Understanding%22)

*   **[NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning](http://arxiv.org/abs/2509.18041v2)** (2025.09)
    *   Focus: Vision-language models struggle with long video question answering due to complex temporal reasoning demands.
    *   project: [https://utaustin-swarmlab.github.io/NeuS-QA/](https://utaustin-swarmlab.github.io/NeuS-QA/)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.18041) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NeuS-QA%3A%20Grounding%20Long-Form%20Video%20Understanding%20in%20Temporal%20Logic%20and%20Neuro-Symbolic%20Reasoning%22)

*   **[Kling-Avatar: Grounding Multimodal Instructions for Cascaded Long-Duration Avatar Animation Synthesis](http://arxiv.org/abs/2509.09595v2)** (2025.09)
    *   Focus: Audio-driven avatar generation lacks high-level instruction conditioning for semantic control.
    *   project: [https://klingavatar.github.io/](https://klingavatar.github.io/)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.09595) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Kling-Avatar%3A%20Grounding%20Multimodal%20Instructions%20for%20Cascaded%20Long-Duration%20Avatar%20Animation%20Synthesis%22)

*   **[DATE: Dynamic Absolute Time Enhancement for Long Video Understanding](http://arxiv.org/abs/2509.09263v1)** (2025.09)
    *   Focus: Long video understanding challenges MLLMs in temporal reasoning and event localization.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.09263) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DATE%3A%20Dynamic%20Absolute%20Time%20Enhancement%20for%20Long%20Video%20Understanding%22)

*   **[OOTSM: A Decoupled Linguistic Framework for Effective Scene Graph Anticipation](http://arxiv.org/abs/2509.05661v1)** (2025.09)
    *   Focus: Scene Graph Anticipation predicts future object relationships from video clips for applications.
    *   code: [https://github.com/ZhuXMMM/OOTSM](https://github.com/ZhuXMMM/OOTSM)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.05661) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22OOTSM%3A%20A%20Decoupled%20Linguistic%20Framework%20for%20Effective%20Scene%20Graph%20Anticipation%22)

*   **[Long-Horizon Visual Imitation Learning via Plan and Code Reflection](http://arxiv.org/abs/2509.05368v2)** (2025.09)
    *   Focus: Visual imitation learning struggles with long-horizon demonstrations and complex action sequences.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.05368) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long-Horizon%20Visual%20Imitation%20Learning%20via%20Plan%20and%20Code%20Reflection%22)

*   **[ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Long Video Understanding](http://arxiv.org/abs/2508.21496v2)** (2025.08)
    *   Focus: Video-MLLMs show strong video understanding but are prone to hallucination.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.21496) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ELV-Halluc%3A%20Benchmarking%20Semantic%20Aggregation%20Hallucinations%20in%20Long%20Video%20Understanding%22)

*   **[Language-Guided Temporal Token Pruning for Efficient VideoLLM Processing](http://arxiv.org/abs/2508.17686v1)** (2025.08)
    *   Focus: LGTTP uses language guidance to prune temporal tokens, reducing attention complexity for long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.17686) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Language-Guided%20Temporal%20Token%20Pruning%20for%20Efficient%20VideoLLM%20Processing%22)

*   **[Multi-Level LVLM Guidance for Untrimmed Video Action Recognition](http://arxiv.org/abs/2508.17442v1)** (2025.08)
    *   Focus: Current methods struggle with fine-grained action recognition and localization in untrimmed videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.17442) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-Level%20LVLM%20Guidance%20for%20Untrimmed%20Video%20Action%20Recognition%22)

*   **[When and What: Diffusion-Grounded VideoLLM with Entity Aware Segmentation for Long Video Understanding](http://arxiv.org/abs/2508.15641v1)** (2025.08)
    *   Focus: Video LLMs need temporal grounding and entity interaction modeling for comprehensive video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.15641) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22When%20and%20What%3A%20Diffusion-Grounded%20VideoLLM%20with%20Entity%20Aware%20Segmentation%20for%20Long%20Video%20Understanding%22)

*   **[Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency](http://arxiv.org/abs/2506.01908v1)** (2025.06)
    *   Focus: MLLMs advance long video understanding with complex semantics and temporal dependencies.
    *   code: [https://github.com/appletea233/Temporal-R1](https://github.com/appletea233/Temporal-R1)
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.01908) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Reinforcement%20Learning%20Tuning%20for%20VideoLLMs%3A%20Reward%20Design%20and%20Data%20Efficiency%22)
    
*   **[TAR-TVG: Enhancing VLMs with Timestamp Anchor-Constrained Reasoning for Temporal Video Grounding](http://arxiv.org/abs/2508.07683v1)** (2025.08)
    *   Focus: TVG localizes video segments from language queries for long video understanding.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.07683) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TAR-TVG%3A%20Enhancing%20VLMs%20with%20Timestamp%20Anchor-Constrained%20Reasoning%20for%20Temporal%20Video%20Grounding%22)

*   **[LET-US: Long Event-Text Understanding of Scenes](http://arxiv.org/abs/2508.07401v1)** (2025.08)
    *   Focus: Event cameras enable low-latency vision, but multimodal models struggle with their sparse, asynchronous data streams.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.07401) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LET-US%3A%20Long%20Event-Text%20Understanding%20of%20Scenes%22)

*   **[Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning](http://arxiv.org/abs/2508.04416v2)** (2025.08)
    *   Focus: MLLMs need better video reasoning for tasks like QA and temporal grounding, but current methods rely too much on text.
    *   project: [https://zhang9302002.github.io/thinkingwithvideos-page/](https://zhang9302002.github.io/thinkingwithvideos-page/)
    *   citation: 18 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.04416) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Thinking%20With%20Videos%3A%20Multimodal%20Tool-Augmented%20Reinforcement%20Learning%20for%20Long%20Video%20Reasoning%22)

*   **[ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks](http://arxiv.org/abs/2508.01943v1)** (NeurIPS2025 2025.08)
    *   Focus: Vision-language models struggle with reasoning over long video sequences.
    *   project: [https://rover-vlm.github.io](https://rover-vlm.github.io)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.01943) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ROVER%3A%20Recursive%20Reasoning%20Over%20Videos%20with%20Vision-Language%20Models%20for%20Embodied%20Tasks%22)

*   **[Fine-grained Spatiotemporal Grounding on Egocentric Videos](http://arxiv.org/abs/2508.00518v1)** (ICCV2025 2025.08)
    *   Focus: Sparsely studied egocentric spatiotemporal video grounding needs new methods for entity localization.
    *   code: [https://github.com/LaVi-Lab/EgoMask](https://github.com/LaVi-Lab/EgoMask)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.00518) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Fine-grained%20Spatiotemporal%20Grounding%20on%20Egocentric%20Videos%22)

*   **[LeAdQA: LLM-Driven Context-Aware Temporal Grounding for Video Question Answering](http://arxiv.org/abs/2507.14784v2)** (2025.07)
    *   Focus: VideoQA needs to find key moments and reason about their causal links in long videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.14784) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LeAdQA%3A%20LLM-Driven%20Context-Aware%20Temporal%20Grounding%20for%20Video%20Question%20Answering%22)

*   **[THYME: Temporal Hierarchical-Cyclic Interactivity Modeling for Video Scene Graphs in Aerial Footage](http://arxiv.org/abs/2507.09200v1)** (2025.07)
    *   Focus: Dynamic scene understanding methods are needed for video applications like autonomous driving and surveillance.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.09200) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22THYME%3A%20Temporal%20Hierarchical-Cyclic%20Interactivity%20Modeling%20for%20Video%20Scene%20Graphs%20in%20Aerial%20Footage%22)

*   **[HumanVideo-MME: Benchmarking MLLMs for Human-Centric Video Understanding](http://arxiv.org/abs/2507.04909v2)** (2025.07)
    *   Focus: MLLMs advance in visual tasks but struggle with human-centric video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.04909) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HumanVideo-MME%3A%20Benchmarking%20MLLMs%20for%20Human-Centric%20Video%20Understanding%22)

*   **[Universal Video Temporal Grounding with Generative Multi-modal Large Language Models](http://arxiv.org/abs/2506.18883v2)** (NeurIPS2025 2025.06)
    *   Focus: A model for localizing video moments using natural language queries.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.18883) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Universal%20Video%20Temporal%20Grounding%20with%20Generative%20Multi-modal%20Large%20Language%20Models%22)

*   **[Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](http://arxiv.org/abs/2506.13654v1)** (2025.06)
    *   Focus: Ego-R1 uses Chain-of-Tool-Thought reasoning for ultra-long egocentric video understanding.
    *   project: [https://egolife-ai.github.io/Ego-R1/](https://egolife-ai.github.io/Ego-R1/)
    *   citation: 14 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.13654) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Ego-R1%3A%20Chain-of-Tool-Thought%20for%20Ultra-Long%20Egocentric%20Video%20Reasoning%22)

*   **[EASG-Bench: Video Q&A Benchmark with Egocentric Action Scene Graphs](http://arxiv.org/abs/2506.05787v2)** (2025.06)
    *   Focus: EASG-Bench is a QA benchmark for egocentric videos using spatio-temporally grounded scene graphs.
    *   code: [https://github.com/fpv-iplab/EASG-bench](https://github.com/fpv-iplab/EASG-bench)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.05787) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EASG-Bench%3A%20Video%20Q%26A%20Benchmark%20with%20Egocentric%20Action%20Scene%20Graphs%22)

*   **[MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](http://arxiv.org/abs/2506.04141v1)** (2025.06)
    *   Focus: Proposes a new MLLM architecture for improved long video understanding and temporal reasoning.
    *   project: [https://mmr-v.github.io](https://mmr-v.github.io)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.04141) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MMR-V%3A%20What%27s%20Left%20Unsaid%3F%20A%20Benchmark%20for%20Multimodal%20Deep%20Reasoning%20in%20Videos%22)

*   **[Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](http://arxiv.org/abs/2505.23908v2)** (2025.05)
    *   Focus: Previews help users discover and evaluate long-form talk content like videos and podcasts efficiently.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.23908) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Transforming%20Podcast%20Preview%20Generation%3A%20From%20Expert%20Models%20to%20LLM-Based%20Systems%22)

*   **[VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](http://arxiv.org/abs/2505.23359v1)** (2025.05)
    *   Focus: Long chain-of-thought reasoning improves LLMs but lacks demonstration for long video understanding tasks.
    *   project: [https://llyx97.github.io/video_reason_bench/](https://llyx97.github.io/video_reason_bench/)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.23359) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoReasonBench%3A%20Can%20MLLMs%20Perform%20Vision-Centric%20Complex%20Video%20Reasoning%3F%22)

*   **[Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](http://arxiv.org/abs/2505.18110v2)** (NeurIPS2025 2025.05)
    *   Focus: Video moment localization integrates visual and auditory cues to identify specific scenes.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.18110) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Watch%20and%20Listen%3A%20Understanding%20Audio-Visual-Speech%20Moments%20with%20Multimodal%20LLM%22)

*   **[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](http://arxiv.org/abs/2505.18079v4)** (NeurIPS2025 2025.05)
    *   Focus: Long video understanding faces challenges from temporal-spatial complexity and extended context question answering.
    *   code: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.18079) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Deep%20Video%20Discovery%3A%20Agentic%20Search%20with%20Tool%20Use%20for%20Long-form%20Video%20Understanding%22)

*   **[QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v2)** (2025.05)
    *   Focus: Long-video understanding is crucial for real-world applications but faces challenges.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.16175) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22QuickVideo%3A%20Real-Time%20Long%20Video%20Understanding%20with%20System%20Algorithm%20Co-Design%22)

*   **[Clapper: Compact Learning and Video Representation in VLMs](http://arxiv.org/abs/2505.15529v1)** (2025.05)
    *   Focus: Vision-language models need effective temporal modeling for video understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.15529) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Clapper%3A%20Compact%20Learning%20and%20Video%20Representation%20in%20VLMs%22)

*   **[CrayonRobo: Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation](http://arxiv.org/abs/2505.02166v1)** (CVPR2025 2025.05)
    *   Focus: The paper explores using goal videos to reduce ambiguity in robotic task specification.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.02166) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CrayonRobo%3A%20Object-Centric%20Prompt-Driven%20Vision-Language-Action%20Model%20for%20Robotic%20Manipulation%22)

*   **[An LLM-Empowered Low-Resolution Vision System for On-Device Human Behavior Understanding](http://arxiv.org/abs/2505.01743v1)** (2025.05)
    *   Focus: LVLMs can generate detailed descriptions for on-device human behavior understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.01743) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22An%20LLM-Empowered%20Low-Resolution%20Vision%20System%20for%20On-Device%20Human%20Behavior%20Understanding%22)

*   **[AVA: Towards Agentic Video Analytics with Vision Language Models](http://arxiv.org/abs/2505.00254v5)** (2025.05)
    *   Focus: AI video analytics systems lack adaptability for open-ended tasks beyond predefined functions.
    *   code: [https://github.com/I-ESC/Project-Ava](https://github.com/I-ESC/Project-Ava)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.00254) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AVA%3A%20Towards%20Agentic%20Video%20Analytics%20with%20Vision%20Language%20Models%22)

*   **[Multi-Stage Boundary-Aware Transformer Network for Action Segmentation in Untrimmed Surgical Videos](http://arxiv.org/abs/2504.18756v2)** (2025.04)
    *   Focus: Analyzing long surgical action sequences improves outcomes, training, and efficiency.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.18756) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-Stage%20Boundary-Aware%20Transformer%20Network%20for%20Action%20Segmentation%20in%20Untrimmed%20Surgical%20Videos%22)

*   **[TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation](http://arxiv.org/abs/2504.17365v3)** (2025.04)
    *   Focus: MLLMs show potential for analyzing long soccer videos and identifying key highlights.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.17365) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeSoccer%3A%20An%20End-to-End%20Multimodal%20Large%20Language%20Model%20for%20Soccer%20Commentary%20Generation%22)

*   **[Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](http://arxiv.org/abs/2504.12083v2)** (NeurIPS2025 2025.04)
    *   Focus: LVLMs struggle with temporal details, hallucinate, and make errors on simple video QA tasks.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.12083) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Self-alignment%20of%20Large%20Video%20Language%20Models%20with%20Refined%20Regularized%20Preference%20Optimization%22)

*   **[Action Anticipation from SoccerNet Football Video Broadcasts](http://arxiv.org/abs/2504.12021v1)** (2025.04)
    *   Focus: AI enables analysis of long sports videos for action understanding and motion prediction.
    *   code: [https://github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.12021) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Action%20Anticipation%20from%20SoccerNet%20Football%20Video%20Broadcasts%22)

*   **[Audio-visual Event Localization on Portrait Mode Short Videos](http://arxiv.org/abs/2504.06884v1)** (2025.04)
    *   Focus: AVEL datasets focus on landscape-oriented long videos with clean audio, limiting real-world applicability.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.06884) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Audio-visual%20Event%20Localization%20on%20Portrait%20Mode%20Short%20Videos%22)

*   **[Pose-Aware Weakly-Supervised Action Segmentation](http://arxiv.org/abs/2504.05700v1)** (2025.04)
    *   Focus: Action segment labeling is a costly challenge for human behavior understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.05700) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Pose-Aware%20Weakly-Supervised%20Action%20Segmentation%22)

*   **[T*: Re-thinking Temporal Search for Long-Form Video Understanding](http://arxiv.org/abs/2504.02259v3)** (CVPR2025 2025.04)
    *   Focus: Revisits temporal search paradigms to address fundamental challenges in long-form video understanding.
    *   citation: 32 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.02259) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22T%2A%3A%20Re-thinking%20Temporal%20Search%20for%20Long-Form%20Video%20Understanding%22)

*   **[MammAlps: A multi-view video behavior monitoring dataset of wild mammals in the Swiss Alps](http://arxiv.org/abs/2503.18223v2)** (CVPR2025 2025.03)
    *   Focus: Camera traps enable habitat-centric wildlife monitoring for ecology and ethology studies.
    *   code: [https://github.com/eceo-epfl/MammAlps](https://github.com/eceo-epfl/MammAlps)
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.18223) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MammAlps%3A%20A%20multi-view%20video%20behavior%20monitoring%20dataset%20of%20wild%20mammals%20in%20the%20Swiss%20Alps%22)

*   **[VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning](http://arxiv.org/abs/2503.13444v2)** (2025.03)
    *   Focus: Video understanding requires precise grounding of answers to visual evidence.
    *   project: [https://videomind.github.io/](https://videomind.github.io/)
    *   citation: 29 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13444) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoMind%3A%20A%20Chain-of-LoRA%20Agent%20for%20Long%20Video%20Reasoning%22)

*   **[Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding](http://arxiv.org/abs/2503.13377v3)** (NeurIPS2025 2025.03)
    *   Focus: TVG locates video segments from language queries, a key challenge in long video understanding.
    *   project: [https://xuboshen.github.io/Time-R1/](https://xuboshen.github.io/Time-R1/)
    *   citation: 33 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.13377) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Time-R1%3A%20Post-Training%20Large%20Vision%20Language%20Model%20for%20Temporal%20Video%20Grounding%22)

*   **[ST-Think: How Multimodal Large Language Models Reason About 4D Worlds from Ego-Centric Videos](http://arxiv.org/abs/2503.12542v2)** (2025.03)
    *   Focus: The abstract questions if multimodal LLMs can match human spatial-temporal reasoning in egocentric video understanding.
    *   citation: 14 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.12542) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ST-Think%3A%20How%20Multimodal%20Large%20Language%20Models%20Reason%20About%204D%20Worlds%20from%20Ego-Centric%20Videos%22)

*   **[VideoScan: Enabling Efficient Streaming Video Understanding via Frame-level Semantic Carriers](http://arxiv.org/abs/2503.09387v2)** (2025.03)
    *   Focus: VideoScan enables real-time video interaction with efficient VLM inference for streamed video comprehension.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.09387) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoScan%3A%20Enabling%20Efficient%20Streaming%20Video%20Understanding%20via%20Frame-level%20Semantic%20Carriers%22)

*   **[TimeLoc: A Unified End-to-End Framework for Precise Timestamp Localization in Long Videos](http://arxiv.org/abs/2503.06526v1)** (2025.03)
    *   Focus: Temporal localization identifies specific timestamps in untrimmed videos for video understanding.
    *   code: [https://github.com/sming256/TimeLoc](https://github.com/sming256/TimeLoc)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.06526) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeLoc%3A%20A%20Unified%20End-to-End%20Framework%20for%20Precise%20Timestamp%20Localization%20in%20Long%20Videos%22)

*   **[An Egocentric Vision-Language Model based Portable Real-time Smart Assistant](http://arxiv.org/abs/2503.04250v1)** (2025.03)
    *   Focus: Vinci is a portable AI system using EgoVideo-VL for real-time vision-language assistance.
    *   code: [https://github.com/OpenGVLab/vinci](https://github.com/OpenGVLab/vinci)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.04250) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22An%20Egocentric%20Vision-Language%20Model%20based%20Portable%20Real-time%20Smart%20Assistant%22)

*   **[iMOVE: Instance-Motion-Aware Video Understanding](http://arxiv.org/abs/2502.11594v2)** (2025.02)
    *   Focus: Improving Video LLMs' fine-grained motion perception for better temporal understanding.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.11594) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22iMOVE%3A%20Instance-Motion-Aware%20Video%20Understanding%22)

*   **[Understanding Long Videos via LLM-Powered Entity Relation Graphs](http://arxiv.org/abs/2501.15953v1)** (2025.01)
    *   Focus: Analyzing long videos is challenging for AI due to tracking and understanding visual elements over time.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.15953) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Understanding%20Long%20Videos%20via%20LLM-Powered%20Entity%20Relation%20Graphs%22)

*   **[TinyLLaVA-Video: Towards Smaller LMMs for Video Understanding with Group Resampler](http://arxiv.org/abs/2501.15513v2)** (2025.01)
    *   Focus: Video behavior recognition and scene understanding are fundamental multimodal intelligence tasks for real-world applications.
    *   code: [https://github.com/ZhangXJ199/TinyLLaVA-Video](https://github.com/ZhangXJ199/TinyLLaVA-Video)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.15513) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TinyLLaVA-Video%3A%20Towards%20Smaller%20LMMs%20for%20Video%20Understanding%20with%20Group%20Resampler%22)

*   **[Temporal Preference Optimization for Long-Form Video Understanding](http://arxiv.org/abs/2501.13919v3)** (2025.01)
    *   Focus: Video-LMMs struggle with temporal grounding in long videos, requiring new methods.
    *   project: [https://ruili33.github.io/tpo_website](https://ruili33.github.io/tpo_website)
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.13919) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Preference%20Optimization%20for%20Long-Form%20Video%20Understanding%22)

*   **[X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding](http://arxiv.org/abs/2501.06835v2)** (2025.01)
    *   Focus: Long-form egocentric videos offer insights into human behavior for embodied intelligence applications.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.06835) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22X-LeBench%3A%20A%20Benchmark%20for%20Extremely%20Long%20Egocentric%20Video%20Understanding%22)

*   **[LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding](http://arxiv.org/abs/2501.05067v2)** (2025.01)
    *   Focus: LLaVA-Octopus adaptively weights visual features for video understanding based on user instructions.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.05067) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LLaVA-Octopus%3A%20Unlocking%20Instruction-Driven%20Adaptive%20Projector%20Fusion%20for%20Video%20Understanding%22)

*   **[V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding](http://arxiv.org/abs/2412.09616v2)** (ICCV2025 2024.12)
    *   Focus: VLMs struggle with long-context video and high-resolution tasks despite multimodal capabilities.
    *   code: [https://github.com/OpenGVLab/V2PE](https://github.com/OpenGVLab/V2PE)
    *   citation: 15 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.09616) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22V2PE%3A%20Improving%20Multimodal%20Long-Context%20Capability%20of%20Vision-Language%20Models%20with%20Variable%20Visual%20Position%20Encoding%22)

*   **[Multi-Scale Contrastive Learning for Video Temporal Grounding](http://arxiv.org/abs/2412.07157v2)** (2024.12)
    *   Focus: Proposes a method to encode variable-length video moments for temporal grounding with natural language queries.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.07157) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-Scale%20Contrastive%20Learning%20for%20Video%20Temporal%20Grounding%22)

*   **[Towards Long Video Understanding via Fine-detailed Video Story Generation](http://arxiv.org/abs/2412.06182v2)** (2024.12)
    *   Focus: Long video understanding is a critical computer vision task with applications in surveillance and content retrieval.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.06182) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Long%20Video%20Understanding%20via%20Fine-detailed%20Video%20Story%20Generation%22)

*   **[Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](http://arxiv.org/abs/2412.06171v2)** (CVPR2025 2024.12)
    *   Focus: Proposes methods for video anomaly understanding across temporal scales and contexts.
    *   code: [https://github.com/pipixin321/HolmesVAU](https://github.com/pipixin321/HolmesVAU)
    *   citation: 29 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.06171) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Holmes-VAU%3A%20Towards%20Long-term%20Video%20Anomaly%20Understanding%20at%20Any%20Granularity%22)

*   **[Video LLMs for Temporal Reasoning in Long Videos](http://arxiv.org/abs/2412.02930v4)** (2024.12)
    *   Focus: TemporalVLM enables temporal reasoning and fine-grained understanding in long videos.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.02930) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20LLMs%20for%20Temporal%20Reasoning%20in%20Long%20Videos%22)

*   **[Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](http://arxiv.org/abs/2411.16932v1)** (CVPR2025 2024.11)
    *   Focus: Video LLMs need temporal awareness for tasks like dense captioning and temporal grounding.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.16932) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Seq2Time%3A%20Sequential%20Knowledge%20Transfer%20for%20Video%20LLM%20Temporal%20Grounding%22)

*   **[LLaVA-MR: Large Language-and-Vision Assistant for Video Moment Retrieval](http://arxiv.org/abs/2411.14505v1)** (2024.11)
    *   Focus: MLLMs struggle with long video processing and precise moment retrieval due to LLM limitations.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.14505) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LLaVA-MR%3A%20Large%20Language-and-Vision%20Assistant%20for%20Video%20Moment%20Retrieval%22)

*   **[BuckTales : A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes](http://arxiv.org/abs/2411.06896v1)** (NeurIPS2024 2024.11)
    *   Focus: Animal behavior understanding is crucial for ecological impact assessment but faces data acquisition and analysis challenges.
    *   citation: 13 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.06896) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22BuckTales%20%3A%20A%20multi-UAV%20dataset%20for%20multi-object%20tracking%20and%20re-identification%20of%20wild%20antelopes%22)

*   **[Zero-shot Action Localization via the Confidence of Large Vision-Language Models](http://arxiv.org/abs/2410.14340v2)** (2024.10)
    *   Focus: Action localization in untrimmed videos is crucial for sports and surgery applications.
    *   code: [https://github.com/josaklil-ai/zeal](https://github.com/josaklil-ai/zeal)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.14340) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Zero-shot%20Action%20Localization%20via%20the%20Confidence%20of%20Large%20Vision-Language%20Models%22)

*   **[Deep learning for action spotting in association football videos](http://arxiv.org/abs/2410.01304v1)** (2024.10)
    *   Focus: Action spotting identifies and precisely localizes actions with timestamps in long untrimmed videos.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.01304) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Deep%20learning%20for%20action%20spotting%20in%20association%20football%20videos%22)

*   **[UAL-Bench: The First Comprehensive Unusual Activity Localization Benchmark](http://arxiv.org/abs/2410.01180v1)** (2024.10)
    *   Focus: Models struggle to localize unusual activities like human errors in videos.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.01180) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22UAL-Bench%3A%20The%20First%20Comprehensive%20Unusual%20Activity%20Localization%20Benchmark%22)

*   **[YouTube Video Analytics for Patient Engagement: Evidence from Colonoscopy Preparation Videos](http://arxiv.org/abs/2410.02830v1)** (2024.10)
    *   Focus: Video analysis methods for medical education content are explored.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.02830) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22YouTube%20Video%20Analytics%20for%20Patient%20Engagement%3A%20Evidence%20from%20Colonoscopy%20Preparation%20Videos%22)

*   **[MECD: Unlocking Multi-Event Causal Discovery in Video Reasoning](http://arxiv.org/abs/2409.17647v4)** (NeurIPS2024 2024.09)
    *   Focus: Video causal reasoning tasks are currently limited in scope and executed as question-answering.
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.17647) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MECD%3A%20Unlocking%20Multi-Event%20Causal%20Discovery%20in%20Video%20Reasoning%22)

*   **[Learning to Localize Actions in Instructional Videos with LLM-Based Multi-Pathway Text-Video Alignment](http://arxiv.org/abs/2409.16145v1)** (ECCV2024 2024.09)
    *   Focus: Proposes a method to localize steps in instructional videos using limited annotations.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.16145) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Learning%20to%20Localize%20Actions%20in%20Instructional%20Videos%20with%20LLM-Based%20Multi-Pathway%20Text-Video%20Alignment%22)

*   **[ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation](http://arxiv.org/abs/2409.13682v1)** (2024.09)
    *   Focus: Robots face challenges in long-term environment understanding and answering human questions.
    *   project: [https://nvidia-ai-iot.github.io/remembr](https://nvidia-ai-iot.github.io/remembr)
    *   citation: 29 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.13682) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ReMEmbR%3A%20Building%20and%20Reasoning%20Over%20Long-Horizon%20Spatio-Temporal%20Memory%20for%20Robot%20Navigation%22)

*   **[AMEGO: Active Memory from long EGOcentric videos](http://arxiv.org/abs/2409.10917v1)** (ECCV2024 2024.09)
    *   Focus: AMEGO is a novel approach for understanding unstructured egocentric videos.
    *   project: [https://gabrielegoletto.github.io/AMEGO/](https://gabrielegoletto.github.io/AMEGO/)
    *   citation: 18 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.10917) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AMEGO%3A%20Active%20Memory%20from%20long%20EGOcentric%20videos%22)

*   **[Open-Vocabulary Action Localization with Iterative Visual Prompting](http://arxiv.org/abs/2408.17422v5)** (2024.08)
    *   Focus: Video action localization finds action timings but requires costly video annotations.
    *   project: [https://microsoft.github.io/VLM-Video-Action-Localization/](https://microsoft.github.io/VLM-Video-Action-Localization/)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.17422) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Open-Vocabulary%20Action%20Localization%20with%20Iterative%20Visual%20Prompting%22)

*   **[HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](http://arxiv.org/abs/2408.06437v1)** (ECCV2024 2024.08)
    *   Focus: Online video understanding uses frame-by-frame predictions, extended by OnTAL for temporal action localization.
    *   code: [https://github.com/sakibreza/ECCV24-HAT/](https://github.com/sakibreza/ECCV24-HAT/)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.06437) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HAT%3A%20History-Augmented%20Anchor%20Transformer%20for%20Online%20Temporal%20Action%20Localization%22)

*   **[mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models](http://arxiv.org/abs/2408.04840v2)** (ICLR2025 2024.08)
    *   Focus: MLLMs excel at single-image tasks but face challenges in other areas.
    *   citation: 206 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.04840) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22mPLUG-Owl3%3A%20Towards%20Long%20Image-Sequence%20Understanding%20in%20Multi-Modal%20Large%20Language%20Models%22)

*   **[SynopGround: A Large-Scale Dataset for Multi-Paragraph Video Grounding from TV Dramas and Synopses](http://arxiv.org/abs/2408.01669v4)** (2024.08)
    *   Focus: Video grounding localizes language queries in untrimmed videos, but current datasets are limited.
    *   project: [https://synopground.github.io/](https://synopground.github.io/)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.01669) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SynopGround%3A%20A%20Large-Scale%20Dataset%20for%20Multi-Paragraph%20Video%20Grounding%20from%20TV%20Dramas%20and%20Synopses%22)

*   **[Fine-grained Dynamic Network for Generic Event Boundary Detection](http://arxiv.org/abs/2407.04274v1)** (ECCV2024 2024.07)
    *   Focus: Generic event boundary detection identifies human-perceived boundaries in long videos for better understanding.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.04274) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Fine-grained%20Dynamic%20Network%20for%20Generic%20Event%20Boundary%20Detection%22)

*   **[MLLM as Video Narrator: Mitigating Modality Imbalance in Video Moment Retrieval](http://arxiv.org/abs/2406.17880v1)** (2024.06)
    *   Focus: Video Moment Retrieval localizes video segments from text queries but struggles with limited training annotations.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.17880) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MLLM%20as%20Video%20Narrator%3A%20Mitigating%20Modality%20Imbalance%20in%20Video%20Moment%20Retrieval%22)

*   **[Zero-Shot Long-Form Video Understanding through Screenplay](http://arxiv.org/abs/2406.17309v1)** (2024.06)
    *   Focus: Long-form video QA requires temporal and contextual analysis for accurate responses.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.17309) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Zero-Shot%20Long-Form%20Video%20Understanding%20through%20Screenplay%22)

*   **[VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment](http://arxiv.org/abs/2406.10889v2)** (CVPR2025 2024.06)
    *   Focus: Video models advance but struggle with associating people and actions over time for compositional reasoning.
    *   project: [https://katha-ai.github.io/projects/velociti](https://katha-ai.github.io/projects/velociti)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.10889) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VELOCITI%3A%20Benchmarking%20Video-Language%20Compositional%20Reasoning%20with%20Strict%20Entailment%22)

*   **[MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models](http://arxiv.org/abs/2405.14338v3)** (CVPR2025 2024.05)
    *   Focus: Point cloud videos capture spatial and temporal dynamics for intelligent agents.
    *   code: [https://github.com/IRMVLab/Mamba4D](https://github.com/IRMVLab/Mamba4D)
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.14338) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MAMBA4D%3A%20Efficient%20Long-Sequence%20Point%20Cloud%20Video%20Understanding%20with%20Disentangled%20Spatial-Temporal%20State%20Space%20Models%22)
    *   code: https://github.com/IRMVLab/Mamba4D

*   **[DTLLM-VLT: Diverse Text Generation for Visual Language Tracking Based on LLM](http://arxiv.org/abs/2405.12139v2)** (2024.05)
    *   Focus: VLT improves object tracking by using language descriptions for precise video object localization.
    *   citation: 30 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.12139) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DTLLM-VLT%3A%20Diverse%20Text%20Generation%20for%20Visual%20Language%20Tracking%20Based%20on%20LLM%22)

*   **[Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis](http://arxiv.org/abs/2405.08944v1)** (2024.05)
    *   Focus: Transformer models enable long-context AI applications like video understanding and coding agents.
    *   citation: 35 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.08944) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Challenges%20in%20Deploying%20Long-Context%20Transformers%3A%20A%20Theoretical%20Peak%20Performance%20Analysis%22)

*   **[SnAG: Scalable and Accurate Video Grounding](http://arxiv.org/abs/2404.02257v2)** (CVPR2024 2024.04)
    *   Focus: Existing video grounding methods prioritize accuracy over scalability.
    *   code: [https://github.com/fmu2/snag_release](https://github.com/fmu2/snag_release)
    *   citation: 23 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.02257) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SnAG%3A%20Scalable%20and%20Accurate%20Video%20Grounding%22)
    *   code: https://github.com/fmu2/snag_release

*   **[SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding](http://arxiv.org/abs/2404.01174v2)** (2024.04)
    *   Focus: Temporal video grounding aligns video content with language instructions for precise understanding.
    *   citation: 27 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.01174) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22SpikeMba%3A%20Multi-Modal%20Spiking%20Saliency%20Mamba%20for%20Temporal%20Video%20Grounding%22)

*   **[Towards Neuro-Symbolic Video Understanding](http://arxiv.org/abs/2403.11021v3)** (ECCV2024 2024.03)
    *   Focus: Efficient frame extraction methods are needed for long-term temporal reasoning in videos.
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.11021) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Neuro-Symbolic%20Video%20Understanding%22)

*   **[HawkEye: Training Video-Text LLMs for Grounding Text in Videos](http://arxiv.org/abs/2403.10228v1)** (2024.03)
    *   Focus: Video-text LLMs perform poorly on complex videos, similar to random guessing.
    *   citation: 54 [[arxiv bibtex]](https://arxiv.org/bibtex/2403.10228) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HawkEye%3A%20Training%20Video-Text%20LLMs%20for%20Grounding%20Text%20in%20Videos%22)

*   **[Multi-modal News Understanding with Professionally Labelled Videos (ReutersViLNews)](http://arxiv.org/abs/2401.12419v1)** (2024.01)
    *   Focus: Current video-language models struggle with high-level abstract understanding.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2401.12419) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-modal%20News%20Understanding%20with%20Professionally%20Labelled%20Videos%20%28ReutersViLNews%29%22)

*   **[A Simple LLM Framework for Long-Range Video Question-Answering](http://arxiv.org/abs/2312.17235v3)** (2023.12)
    *   Focus: LLoVi is a language-based framework for efficient long-range video question-answering.
    *   code: [https://github.com/CeeZh/LLoVi](https://github.com/CeeZh/LLoVi)
    *   citation: 141 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.17235) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Simple%20LLM%20Framework%20for%20Long-Range%20Video%20Question-Answering%22)

*   **[Grounding-Prompter: Prompting LLM with Multimodal Information for Temporal Sentence Grounding in Long Videos](http://arxiv.org/abs/2312.17117v1)** (2023.12)
    *   Focus: TSG localizes video moments using language queries, with current methods for short videos.
    *   citation: 17 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.17117) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Grounding-Prompter%3A%20Prompting%20LLM%20with%20Multimodal%20Information%20for%20Temporal%20Sentence%20Grounding%20in%20Long%20Videos%22)

*   **[Hierarchical Graph Pattern Understanding for Zero-Shot VOS](http://arxiv.org/abs/2312.09525v1)** (2023.12)
    *   Focus: Optical flow is ideal for video motion but current methods have limitations.
    *   code: [https://github.com/NUST-Machine-Intelligence-Laboratory/HGPU](https://github.com/NUST-Machine-Intelligence-Laboratory/HGPU)
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.09525) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Hierarchical%20Graph%20Pattern%20Understanding%20for%20Zero-Shot%20VOS%22)

*   **[Spatiotemporal Event Graphs for Dynamic Scene Understanding](http://arxiv.org/abs/2312.07621v1)** (2023.12)
    *   Focus: This thesis presents methods for dynamic scene understanding in videos.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.07621) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Spatiotemporal%20Event%20Graphs%20for%20Dynamic%20Scene%20Understanding%22)

*   **[Grounded Question-Answering in Long Egocentric Videos](http://arxiv.org/abs/2312.06505v4)** (CVPR2024 2023.12)
    *   Focus: Proposes a new approach for long, egocentric video understanding to address robotics applications.
    *   code: [https://github.com/Becomebright/GroundVQA](https://github.com/Becomebright/GroundVQA)
    *   citation: 42 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.06505) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Grounded%20Question-Answering%20in%20Long%20Egocentric%20Videos%22)

*   **[RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](http://arxiv.org/abs/2312.06729v3)** (ECCV2024 2023.12)
    *   Focus: Adapting short video grounding methods to locate moments in long videos.
    *   code: [https://github.com/Tanveer81/RGNet](https://github.com/Tanveer81/RGNet)
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.06729) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22RGNet%3A%20A%20Unified%20Clip%20Retrieval%20and%20Grounding%20Network%20for%20Long%20Videos%22)

*   **[TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding](http://arxiv.org/abs/2312.02051v2)** (CVPR2024 2023.12)
    *   Focus: TimeChat is a time-sensitive MLLM for long video understanding with timestamp-aware frame tokenization and temporal attention.
    *   code: [https://github.com/RenShuhuai-Andy/TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
    *   citation: 326 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.02051) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TimeChat%3A%20A%20Time-sensitive%20Multimodal%20Large%20Language%20Model%20for%20Long%20Video%20Understanding%22)

*   **[Multi-Modal Video Topic Segmentation with Dual-Contrastive Domain Adaptation](http://arxiv.org/abs/2312.00220v1)** (2023.11)
    *   Focus: Video topic segmentation reveals semantic structure and is vital for video understanding tasks.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.00220) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-Modal%20Video%20Topic%20Segmentation%20with%20Dual-Contrastive%20Domain%20Adaptation%22)

*   **[PALM: Predicting Actions through Language Models](http://arxiv.org/abs/2311.17944v2)** (ECCV2024 2023.11)
    *   Focus: Explores challenges in egocentric vision for human activity understanding.
    *   citation: 20 [[arxiv bibtex]](https://arxiv.org/bibtex/2311.17944) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PALM%3A%20Predicting%20Actions%20through%20Language%20Models%22)

*   **[A Hybrid Graph Network for Complex Activity Detection in Video](http://arxiv.org/abs/2310.17493v2)** (2023.10)
    *   Focus: Video understanding is challenging in fields like autonomous driving and sports analytics.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2310.17493) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Hybrid%20Graph%20Network%20for%20Complex%20Activity%20Detection%20in%20Video%22)

*   **[End-to-End Streaming Video Temporal Action Segmentation with Reinforce Learning](http://arxiv.org/abs/2309.15683v2)** (2023.09)
    *   Focus: Streaming temporal action segmentation is an understudied video understanding task.
    *   code: [https://github.com/Thinksky5124/SVTAS](https://github.com/Thinksky5124/SVTAS)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.15683) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22End-to-End%20Streaming%20Video%20Temporal%20Action%20Segmentation%20with%20Reinforce%20Learning%22)

*   **[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](http://arxiv.org/abs/2309.13925v2)** (CVPR2024 2023.09)
    *   Focus: Surveillance video tasks need expansion beyond classification to include temporal localization and dense captioning.
    *   project: [https://xuange923.github.io/Surveillance-Video-Understanding](https://xuange923.github.io/Surveillance-Video-Understanding)
    *   citation: 37 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.13925) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Towards%20Surveillance%20Video-and-Language%20Understanding%3A%20New%20Dataset%2C%20Baselines%2C%20and%20Challenges%22)

*   **[Language-Conditioned Change-point Detection to Identify Sub-Tasks in Robotics Domains](http://arxiv.org/abs/2309.00743v1)** (2023.09)
    *   Focus: An approach identifies robot trajectory sub-tasks using language instructions from demonstrations.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.00743) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Language-Conditioned%20Change-point%20Detection%20to%20Identify%20Sub-Tasks%20in%20Robotics%20Domains%22)

*   **[Helping Hands: An Object-Aware Ego-Centric Video Recognition Model](http://arxiv.org/abs/2308.07918v1)** (ICCV2023 2023.08)
    *   Focus: An object-aware decoder improves ego-centric video understanding by enhancing object-awareness during training.
    *   citation: 34 [[arxiv bibtex]](https://arxiv.org/bibtex/2308.07918) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Helping%20Hands%3A%20An%20Object-Aware%20Ego-Centric%20Video%20Recognition%20Model%22)

*   **[Single-Stage Visual Query Localization in Egocentric Videos](http://arxiv.org/abs/2306.09324v1)** (NeurIPS2023 2023.06)
    *   Focus: This paper addresses visual query localization in long egocentric videos for episodic memory systems.
    *   project: [https://hwjiang1510.github.io/VQLoC/](https://hwjiang1510.github.io/VQLoC/)
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2306.09324) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Single-Stage%20Visual%20Query%20Localization%20in%20Egocentric%20Videos%22)

*   **[Boundary-Denoising for Video Activity Localization](http://arxiv.org/abs/2304.02934v1)** (2023.04)
    *   Focus: Video activity localization retrieves actions of interest with timestamps from long untrimmed videos.
    *   citation: 13 [[arxiv bibtex]](https://arxiv.org/bibtex/2304.02934) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Boundary-Denoising%20for%20Video%20Activity%20Localization%22)

*   **[Diffusion Action Segmentation](http://arxiv.org/abs/2303.17959v2)** (ICCV2023 2023.03)
    *   Focus: A novel method for temporal action segmentation in long videos is proposed.
    *   citation: 95 [[arxiv bibtex]](https://arxiv.org/bibtex/2303.17959) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Diffusion%20Action%20Segmentation%22)

*   **[MINOTAUR: Multi-task Video Grounding From Multimodal Queries](http://arxiv.org/abs/2302.08063v2)** (2023.02)
    *   Focus: Video understanding tasks vary in inputs and goals, requiring flexible models.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2302.08063) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MINOTAUR%3A%20Multi-task%20Video%20Grounding%20From%20Multimodal%20Queries%22)

*   **[Efficient Movie Scene Detection using State-Space Transformers](http://arxiv.org/abs/2212.14427v2)** (CVPR2023 2022.12)
    *   Focus: Movie scene detection is challenging due to the need to understand complex storylines and temporal dynamics.
    *   code: [https://github.com/md-mohaiminul/TranS4mer](https://github.com/md-mohaiminul/TranS4mer)
    *   citation: 62 [[arxiv bibtex]](https://arxiv.org/bibtex/2212.14427) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Efficient%20Movie%20Scene%20Detection%20using%20State-Space%20Transformers%22)

*   **[Nonlinear and Machine Learning Analyses on High-Density EEG data of Math Experts and Novices](http://arxiv.org/abs/2212.00712v1)** (2022.12)
    *   Focus: Neuroscience uses naturalistic stimuli like cinema and video games to study brain function in ecologically valid conditions.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2212.00712) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Nonlinear%20and%20Machine%20Learning%20Analyses%20on%20High-Density%20EEG%20data%20of%20Math%20Experts%20and%20Novices%22)

*   **[Exploring Anchor-based Detection for Ego4D Natural Language Query](http://arxiv.org/abs/2208.05375v1)** (2022.08)
    *   Focus: Report on Ego4D natural language query challenge for comprehensive video understanding.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2208.05375) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Exploring%20Anchor-based%20Detection%20for%20Ego4D%20Natural%20Language%20Query%22)

*   **[Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding](http://arxiv.org/abs/2207.14698v2)** (ECCV2022 2022.07)
    *   Focus: Current temporal grounding methods struggle with severe performance issues in untrimmed videos.
    *   code: [https://github.com/haojc/ShufflingVideosForTSG](https://github.com/haojc/ShufflingVideosForTSG)
    *   citation: 33 [[arxiv bibtex]](https://arxiv.org/bibtex/2207.14698) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Can%20Shuffling%20Video%20Benefit%20Temporal%20Bias%20Problem%3A%20A%20Novel%20Training%20Framework%20for%20Temporal%20Grounding%22)

*   **[Video + CLIP Baseline for Ego4D Long-term Action Anticipation](http://arxiv.org/abs/2207.00579v1)** (2022.07)
    *   Focus: Video+CLIP adapts image-text models for long-term action anticipation using CLIP.
    *   code: [http://github.com/srijandas07/clip_baseline_LTA_Ego4d](http://github.com/srijandas07/clip_baseline_LTA_Ego4d)
    *   citation: 23 [[arxiv bibtex]](https://arxiv.org/bibtex/2207.00579) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20%2B%20CLIP%20Baseline%20for%20Ego4D%20Long-term%20Action%20Anticipation%22)

*   **[Technical Report for CVPR 2022 LOVEU AQTC Challenge](http://arxiv.org/abs/2206.14555v1)** (2022.06)
    *   Focus: Winning model for AQTC task in LOVEU challenge addresses multi-step video understanding difficulties.
    *   code: [https://github.com/jaykim9870/](https://github.com/jaykim9870/)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2206.14555) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Technical%20Report%20for%20CVPR%202022%20LOVEU%20AQTC%20Challenge%22)

*   **[Scene Consistency Representation Learning for Video Scene Segmentation](http://arxiv.org/abs/2205.05487v1)** (CVPR2022 2022.05)
    *   Focus: Scene boundary detection in long videos using semantic story consistency.
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2205.05487) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Scene%20Consistency%20Representation%20Learning%20for%20Video%20Scene%20Segmentation%22)

*   **[Contrastive Language-Action Pre-training for Temporal Localization](http://arxiv.org/abs/2204.12293v1)** (2022.04)
    *   Focus: Long video understanding faces memory limits for end-to-end training of temporal localization tasks.
    *   citation: 26 [[arxiv bibtex]](https://arxiv.org/bibtex/2204.12293) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Contrastive%20Language-Action%20Pre-training%20for%20Temporal%20Localization%22)

*   **[Temporal Alignment Networks for Long-term Video](http://arxiv.org/abs/2204.02968v1)** (CVPR2022 2022.04)
    *   Focus: A network aligns long videos with text sentences, determining if alignment is possible.
    *   citation: 101 [[arxiv bibtex]](https://arxiv.org/bibtex/2204.02968) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Alignment%20Networks%20for%20Long-term%20Video%22)

*   **[Long Movie Clip Classification with State-Space Video Models](http://arxiv.org/abs/2204.01692v3)** (ECCV2022 2022.04)
    *   Focus: Short video models struggle with long movie understanding tasks.
    *   code: [https://github.com/md-mohaiminul/ViS4mer](https://github.com/md-mohaiminul/ViS4mer)
    *   citation: 136 [[arxiv bibtex]](https://arxiv.org/bibtex/2204.01692) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long%20Movie%20Clip%20Classification%20with%20State-Space%20Video%20Models%22)

*   **[MSPred: Video Prediction at Multiple Spatio-Temporal Scales with Hierarchical Recurrent Networks](http://arxiv.org/abs/2203.09303v4)** (2022.03)
    *   Focus: Autonomous systems must understand past states to predict future actions from camera frames.
    *   code: [https://github.com/AIS-Bonn/MSPred](https://github.com/AIS-Bonn/MSPred)
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2203.09303) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MSPred%3A%20Video%20Prediction%20at%20Multiple%20Spatio-Temporal%20Scales%20with%20Hierarchical%20Recurrent%20Networks%22)

*   **[Behavior Recognition Based on the Integration of Multigranular Motion Features](http://arxiv.org/abs/2203.03097v1)** (2022.03)
    *   Focus: Video behavior recognition combines spatial object data with temporal action dynamics.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2203.03097) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Behavior%20Recognition%20Based%20on%20the%20Integration%20of%20Multigranular%20Motion%20Features%22)

#### Others

*   **[RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation](http://arxiv.org/abs/2506.22007v1)** (2025.06)
    *   Focus: Generating long-horizon videos for robotic tasks using text-to-video diffusion models.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.22007) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22RoboEnvision%3A%20A%20Long-Horizon%20Video%20Generation%20Model%20for%20Multi-Task%20Robot%20Manipulation%22)

*   **[Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation](http://arxiv.org/abs/2502.16707v1)** (2025.02)
    *   Focus: Robotic manipulation requires high-level planning, physical reasoning, and reactive motion selection.
    *   project: [https://reflect-vlm.github.io](https://reflect-vlm.github.io)
    *   citation: 21 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.16707) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Reflective%20Planning%3A%20Vision-Language%20Models%20for%20Multi-Stage%20Long-Horizon%20Robotic%20Manipulation%22)

*   **[Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](http://arxiv.org/abs/2412.09621v2)** (CVPR2025 2024.12)
    *   Focus: Dynamic 3D scene understanding from imagery lacks large-scale supervised training data.
    *   project: [https://stereo4d.github.io](https://stereo4d.github.io)
    *   citation: 39 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.09621) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Stereo4D%3A%20Learning%20How%20Things%20Move%20in%203D%20from%20Internet%20Stereo%20Videos%22)

*   **[Memory-augmented Online Video Anomaly Detection](http://arxiv.org/abs/2302.10719v2)** (2023.02)
    *   Focus: An online system for autonomous vehicles to understand scenes and provide immediate responses.
    *   code: [https://github.com/IMPLabUniPr/movad/tree/movad_vad](https://github.com/IMPLabUniPr/movad/tree/movad_vad)
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2302.10719) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Memory-augmented%20Online%20Video%20Anomaly%20Detection%22)


*   **[On the Pitfalls of Batch Normalization for End-to-End Video Learning: A Study on Surgical Workflow Analysis](http://arxiv.org/abs/2203.07976v5)** (2022.03)
    *   Focus: Batch Normalization's batch-dependent property causes problems in sequence modeling but is understudied.
    *   citation: 24 [[arxiv bibtex]](https://arxiv.org/bibtex/2203.07976) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22On%20the%20Pitfalls%20of%20Batch%20Normalization%20for%20End-to-End%20Video%20Learning%3A%20A%20Study%20on%20Surgical%20Workflow%20Analysis%22)

*   **[Vision-Language Memory for Spatial Reasoning](http://arxiv.org/abs/2511.20644v1)** (2025.11)
    *   Focus: Vision-language models underperform humans in video spatial reasoning, highlighting a key research gap.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.20644) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Vision-Language%20Memory%20for%20Spatial%20Reasoning%22)

*   **[DeepSport: A Multimodal Large Language Model for Comprehensive Sports Video Reasoning via Agentic Reinforcement Learning](http://arxiv.org/abs/2511.12908v1)** (2025.11)
    *   Focus: Sports video understanding requires models to handle high-speed dynamics, complex rules, and long temporal contexts.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.12908) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DeepSport%3A%20A%20Multimodal%20Large%20Language%20Model%20for%20Comprehensive%20Sports%20Video%20Reasoning%20via%20Agentic%20Reinforcement%20Learning%22)

*   **[Synopses of Movie Narratives: a Video-Language Dataset for Story Understanding](http://arxiv.org/abs/2203.05711v4)** (2022.03)
    *   Focus: A new video-language dataset for movie story understanding is collected and released.
    *   citation: 11 [[arxiv bibtex]](https://arxiv.org/bibtex/2203.05711) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Synopses%20of%20Movie%20Narratives%3A%20a%20Video-Language%20Dataset%20for%20Story%20Understanding%22)

*   **[HumanMM: Global Human Motion Recovery from Multi-shot Videos](http://arxiv.org/abs/2503.07597v1)** (CVPR2025 2025.03)
    *   Focus: A framework reconstructs long-sequence 3D human motion from in-the-wild videos with shot transitions.
    *   project: [https://zhangyuhong01.github.io/HumanMM/](https://zhangyuhong01.github.io/HumanMM/)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.07597) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22HumanMM%3A%20Global%20Human%20Motion%20Recovery%20from%20Multi-shot%20Videos%22)

*   **[Video-Mined Task Graphs for Keystep Recognition in Instructional Videos](http://arxiv.org/abs/2307.08763v2)** (NeurIPS2023 2023.07)
    *   Focus: Procedural activity understanding analyzes sequential human actions in long videos to achieve task goals.
    *   citation: 33 [[arxiv bibtex]](https://arxiv.org/bibtex/2307.08763) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video-Mined%20Task%20Graphs%20for%20Keystep%20Recognition%20in%20Instructional%20Videos%22)


*   **[Temporal Action Segmentation: An Analysis of Modern Techniques](http://arxiv.org/abs/2210.10352v5)** (2022.10)
    *   Focus: Temporal action segmentation identifies action classes in long videos, requiring long-range understanding.
    *   code: [https://github.com/nus-cvml/awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation)
    *   citation: 111 [[arxiv bibtex]](https://arxiv.org/bibtex/2210.10352) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Action%20Segmentation%3A%20An%20Analysis%20of%20Modern%20Techniques%22)

*   **[Controllable Hybrid Captioner for Improved Long-form Video Understanding](http://arxiv.org/abs/2507.17047v4)** (2025.07)
    *   Focus: Text summaries compress dense video data into compact, query-relevant representations.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.17047) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Controllable%20Hybrid%20Captioner%20for%20Improved%20Long-form%20Video%20Understanding%22)

*   **[MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding](http://arxiv.org/abs/2507.06072v1)** (ICCV2025 2025.07)
    *   Focus: Proposes a method for deep causal reasoning in autonomous driving video behavior recognition.
    *   code: [https://github.com/SixCorePeach/MCAM](https://github.com/SixCorePeach/MCAM)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.06072) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MCAM%3A%20Multimodal%20Causal%20Analysis%20Model%20for%20Ego-Vehicle-Level%20Driving%20Video%20Understanding%22)

*   **[Text-guided Weakly Supervised Framework for Dynamic Facial Expression Recognition](http://arxiv.org/abs/2511.10958v1)** (2025.11)
    *   Focus: Dynamic facial expression recognition models temporal facial changes in videos, addressing many-to-one mapping challenges.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.10958) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Text-guided%20Weakly%20Supervised%20Framework%20for%20Dynamic%20Facial%20Expression%20Recognition%22)

*   **[FloCoDe: Unbiased Dynamic Scene Graph Generation with Temporal Consistency and Correlation Debiasing](http://arxiv.org/abs/2310.16073v3)** (2023.10)
    *   Focus: Dynamic scene graph generation from videos requires understanding objects, motions, and interactions over time.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2310.16073) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FloCoDe%3A%20Unbiased%20Dynamic%20Scene%20Graph%20Generation%20with%20Temporal%20Consistency%20and%20Correlation%20Debiasing%22)

*   **[Local-Global Information Interaction Debiasing for Dynamic Scene Graph Generation](http://arxiv.org/abs/2308.05274v2)** (2023.08)
    *   Focus: Dynamic scene graph generation for videos faces challenges from long-tail distributions in spatial-temporal modeling.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2308.05274) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Local-Global%20Information%20Interaction%20Debiasing%20for%20Dynamic%20Scene%20Graph%20Generation%22)

*   **[TeleEgo: Benchmarking Egocentric AI Assistants in the Wild](http://arxiv.org/abs/2510.23981v2)** (2025.10)
    *   Focus: Existing benchmarks for egocentric AI assistants lack real-time processing and long-term memory requirements.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.23981) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22TeleEgo%3A%20Benchmarking%20Egocentric%20AI%20Assistants%20in%20the%20Wild%22)

*   **[EmbRACE-3K: Embodied Reasoning and Action in Complex Environments](http://arxiv.org/abs/2507.10548v1)** (2025.07)
    *   Focus: VLMs excel in passive video understanding but struggle in embodied settings requiring active perception.
    *   project: [https://mxllc.github.io/EmbRACE-3K/](https://mxllc.github.io/EmbRACE-3K/)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2507.10548) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EmbRACE-3K%3A%20Embodied%20Reasoning%20and%20Action%20in%20Complex%20Environments%22)

*   **[CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning](http://arxiv.org/abs/2506.17629v1)** (2025.06)
    *   Focus: EVR uses egocentric video for complex instruction following and spatiotemporal reasoning.
    *   code: [https://github.com/Teacher-Tom/CLiViS](https://github.com/Teacher-Tom/CLiViS)
    *   citation: 6 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.17629) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22CLiViS%3A%20Unleashing%20Cognitive%20Map%20through%20Linguistic-Visual%20Synergy%20for%20Embodied%20Visual%20Reasoning%22)

*   **[Mamba-Enhanced Text-Audio-Video Alignment Network for Emotion Recognition in Conversations](http://arxiv.org/abs/2409.05243v1)** (2024.09)
    *   Focus: ERC research identifies and classifies speaker emotions in multimodal conversations.
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2409.05243) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Mamba-Enhanced%20Text-Audio-Video%20Alignment%20Network%20for%20Emotion%20Recognition%20in%20Conversations%22)

*   **[Efficient Long-distance Latent Relation-aware Graph Neural Network for Multi-modal Emotion Recognition in Conversations](http://arxiv.org/abs/2407.00119v2)** (2024.06)
    *   Focus: MERC analyzes emotional states in conversations using multi-modal data.
    *   citation: 25 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.00119) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Efficient%20Long-distance%20Latent%20Relation-aware%20Graph%20Neural%20Network%20for%20Multi-modal%20Emotion%20Recognition%20in%20Conversations%22)

*   **[EALD-MLLM: Emotion Analysis in Long-sequential and De-identity videos with Multi-modal Large Language Model](http://arxiv.org/abs/2405.00574v2)** (2024.05)
    *   Focus: Emotion AI research needs better multimodal fusion and temporal modeling for improved emotion understanding.
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.00574) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EALD-MLLM%3A%20Emotion%20Analysis%20in%20Long-sequential%20and%20De-identity%20videos%20with%20Multi-modal%20Large%20Language%20Model%22)

*   **[EmpathicStories++: A Multimodal Dataset for Empathy towards Personal Experiences](http://arxiv.org/abs/2405.15708v1)** (2024.05)
    *   Focus: Existing empathy datasets lack interpersonal and experiential dimensions needed for AI modeling.
    *   project: [https://mitmedialab.github.io/empathic-stories-multimodal/](https://mitmedialab.github.io/empathic-stories-multimodal/)
    *   citation: 12 [[arxiv bibtex]](https://arxiv.org/bibtex/2405.15708) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EmpathicStories%2B%2B%3A%20A%20Multimodal%20Dataset%20for%20Empathy%20towards%20Personal%20Experiences%22)

*   **[DIV-FF: Dynamic Image-Video Feature Fields For Environment Understanding in Egocentric Videos](http://arxiv.org/abs/2503.08344v1)** (CVPR2025 2025.03)
    *   Focus: Egocentric videos enable dynamic environment understanding for robotics and AR applications.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2503.08344) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22DIV-FF%3A%20Dynamic%20Image-Video%20Feature%20Fields%20For%20Environment%20Understanding%20in%20Egocentric%20Videos%22)

*   **[MEGC2025: Micro-Expression Grand Challenge on Spot Then Recognize and Visual Question Answering](http://arxiv.org/abs/2506.15298v2)** (2025.06)
    *   Focus: Facial micro-expressions are involuntary facial movements during suppressed emotions.
    *   project: [https://megc2025.github.io](https://megc2025.github.io)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.15298) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MEGC2025%3A%20Micro-Expression%20Grand%20Challenge%20on%20Spot%20Then%20Recognize%20and%20Visual%20Question%20Answering%22)

*   **[A Space-Time Transformer for Precipitation Forecasting](http://arxiv.org/abs/2511.11090v1)** (2025.11)
    *   Focus: Traditional NWP models are static, but new AI methods improve real-time flood forecasting.
    *   code: [https://github.com/leharris3/satformer](https://github.com/leharris3/satformer)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.11090) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Space-Time%20Transformer%20for%20Precipitation%20Forecasting%22)

*   **[Gait Disorder Assessment Based on a Large-Scale Clinical Trial: WiFi vs. Video vs. Doctor's Visual Inspection](http://arxiv.org/abs/2502.05328v2)** (2025.02)
    *   Focus: This paper explores WiFi sensing for diagnosing neurological gait disorders.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2502.05328) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Gait%20Disorder%20Assessment%20Based%20on%20a%20Large-Scale%20Clinical%20Trial%3A%20WiFi%20vs.%20Video%20vs.%20Doctor%27s%20Visual%20Inspection%22)

*   **[Temporal Perceiver: A General Architecture for Arbitrary Boundary Detection](http://arxiv.org/abs/2203.00307v2)** (2022.03)
    *   Focus: GBD detects general boundaries to segment videos into coherent units for preprocessing.
    *   citation: 19 [[arxiv bibtex]](https://arxiv.org/bibtex/2203.00307) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Temporal%20Perceiver%3A%20A%20General%20Architecture%20for%20Arbitrary%20Boundary%20Detection%22)

*   **[AirLetters: An Open Video Dataset of Characters Drawn in the Air](http://arxiv.org/abs/2410.02921v1)** (2024.10)
    *   Focus: AirLetters is a dataset of human motion videos for predicting articulated letter gestures.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.02921) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22AirLetters%3A%20An%20Open%20Video%20Dataset%20of%20Characters%20Drawn%20in%20the%20Air%22)

*   **[Dynamic Gesture Recognition in Ultra-Range Distance for Effective Human-Robot Interaction](http://arxiv.org/abs/2407.21374v1)** (2024.07)
    *   Focus: Novel approach for ultra-range gesture recognition in human-robot interaction using video data.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.21374) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Dynamic%20Gesture%20Recognition%20in%20Ultra-Range%20Distance%20for%20Effective%20Human-Robot%20Interaction%22)

*   **[Multimodal Language Models for Domain-Specific Procedural Video Summarization](http://arxiv.org/abs/2407.05419v1)** (2024.07)
    *   Focus: Long-format video tutorials are effective for learning new skills.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.05419) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multimodal%20Language%20Models%20for%20Domain-Specific%20Procedural%20Video%20Summarization%22)

*   **[NOVIS: A Case for End-to-End Near-Online Video Instance Segmentation](http://arxiv.org/abs/2308.15266v2)** (2023.08)
    *   Focus: Recent findings challenge the belief that offline methods outperform online frame-by-frame processing in Video Instance Segmentation.
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2308.15266) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22NOVIS%3A%20A%20Case%20for%20End-to-End%20Near-Online%20Video%20Instance%20Segmentation%22)

*   **[VITA: Video Instance Segmentation via Object Token Association](http://arxiv.org/abs/2206.04403v2)** (NeurIPS2022 2022.06)
    *   Focus: A new offline VIS method uses object-oriented information to improve video context understanding.
    *   code: [https://github.com/sukjunhwang/VITA](https://github.com/sukjunhwang/VITA)
    *   citation: 121 [[arxiv bibtex]](https://arxiv.org/bibtex/2206.04403) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VITA%3A%20Video%20Instance%20Segmentation%20via%20Object%20Token%20Association%22)

*   **[Discriminative Spatial-Semantic VOS Solution: 1st Place Solution for 6th LSVOS](http://arxiv.org/abs/2408.16431v1)** (2024.08)
    *   Focus: The MOSE dataset addresses video object segmentation challenges in complex scenes and long motions.
    *   code: [https://github.com/yahooo-m/VOS-Solution](https://github.com/yahooo-m/VOS-Solution)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2408.16431) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Discriminative%20Spatial-Semantic%20VOS%20Solution%3A%201st%20Place%20Solution%20for%206th%20LSVOS%22)

*   **[Joint Modeling of Feature, Correspondence, and a Compressed Memory for Video Object Segmentation](http://arxiv.org/abs/2308.13505v2)** (2023.08)
    *   Focus: Video Object Segmentation methods use extraction-then-matching pipelines with independent feature extraction and dense matching.
    *   code: [https://github.com/MCG-NJU/JointFormer](https://github.com/MCG-NJU/JointFormer)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2308.13505) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Joint%20Modeling%20of%20Feature%2C%20Correspondence%2C%20and%20a%20Compressed%20Memory%20for%20Video%20Object%20Segmentation%22)

*   **[Look Before You Match: Instance Understanding Matters in Video Object Segmentation](http://arxiv.org/abs/2212.06826v1)** (CVPR2023 2022.12)
    *   Focus: Memory-based methods use dense frame matching for long-range context in video object segmentation.
    *   citation: 52 [[arxiv bibtex]](https://arxiv.org/bibtex/2212.06826) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Look%20Before%20You%20Match%3A%20Instance%20Understanding%20Matters%20in%20Video%20Object%20Segmentation%22)

*   **[FriendsQA: A New Large-Scale Deep Video Understanding Dataset with Fine-grained Topic Categorization for Story Videos](http://arxiv.org/abs/2412.17022v1)** (2024.12)
    *   Focus: VideoQA models struggle with complex questions despite good performance on factoid tasks.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.17022) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FriendsQA%3A%20A%20New%20Large-Scale%20Deep%20Video%20Understanding%20Dataset%20with%20Fine-grained%20Topic%20Categorization%20for%20Story%20Videos%22)

*   **[EVQAScore: A Fine-grained Metric for Video Question Answering Data Quality Evaluation](http://arxiv.org/abs/2411.06908v3)** (2024.11)
    *   Focus: Evaluating video QA and caption data quality is essential for training effective VideoLLMs.
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2411.06908) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EVQAScore%3A%20A%20Fine-grained%20Metric%20for%20Video%20Question%20Answering%20Data%20Quality%20Evaluation%22)

*   **[Zero-Shot Video Question Answering with Procedural Programs](http://arxiv.org/abs/2312.00937v1)** (2023.12)
    *   Focus: Proposes zero-shot video QA via procedural programs solving visual subtasks.
    *   project: [https://rccchoudhury.github.io/proviq2023](https://rccchoudhury.github.io/proviq2023)
    *   citation: 37 [[arxiv bibtex]](https://arxiv.org/bibtex/2312.00937) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Zero-Shot%20Video%20Question%20Answering%20with%20Procedural%20Programs%22)

*   **[EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding](http://arxiv.org/abs/2308.09126v1)** (NeurIPS2023 2023.08)
    *   Focus: EgoSchema is a long-form video QA dataset and benchmark for evaluating video understanding systems.
    *   project: [http://egoschema.github.io](http://egoschema.github.io)
    *   citation: 463 [[arxiv bibtex]](https://arxiv.org/bibtex/2308.09126) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EgoSchema%3A%20A%20Diagnostic%20Benchmark%20for%20Very%20Long-form%20Video%20Language%20Understanding%22)

*   **[Dense but Efficient VideoQA for Intricate Compositional Reasoning](http://arxiv.org/abs/2210.10300v1)** (2022.10)
    *   Focus: Proposes a new VideoQA benchmark with complex reasoning on long videos.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2210.10300) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Dense%20but%20Efficient%20VideoQA%20for%20Intricate%20Compositional%20Reasoning%22)

*   **[Locate before Answering: Answer Guided Question Localization for Video Question Answering](http://arxiv.org/abs/2210.02081v2)** (2022.10)
    *   Focus: VideoQA is a key vision-language task with growing research interest.
    *   citation: 24 [[arxiv bibtex]](https://arxiv.org/bibtex/2210.02081) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Locate%20before%20Answering%3A%20Answer%20Guided%20Question%20Localization%20for%20Video%20Question%20Answering%22)

*   **[Structured Two-stream Attention Network for Video Question Answering](http://arxiv.org/abs/2206.01017v1)** (2022.06)
    *   Focus: Video QA remains a major challenge in vision-language understanding compared to image QA.
    *   citation: 70 [[arxiv bibtex]](https://arxiv.org/bibtex/2206.01017) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Structured%20Two-stream%20Attention%20Network%20for%20Video%20Question%20Answering%22)

*   **[MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](http://arxiv.org/abs/2510.21406v1)** (NeurIPS2025 2025.10)
    *   Focus: Proposes MUVR benchmark for multi-modal untrimmed video retrieval on long videos.
    *   code: [https://github.com/debby-0527/MUVR](https://github.com/debby-0527/MUVR)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.21406) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22MUVR%3A%20A%20Multi-Modal%20Untrimmed%20Video%20Retrieval%20Benchmark%20with%20Multi-Level%20Visual%20Correspondence%22)

*   **[ViSMaP: Unsupervised Hour-long Video Summarisation by Meta-Prompting](http://arxiv.org/abs/2504.15921v1)** (2025.04)
    *   Focus: ViSMap is an unsupervised system for summarizing hour-long videos without supervision.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.15921) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22ViSMaP%3A%20Unsupervised%20Hour-long%20Video%20Summarisation%20by%20Meta-Prompting%22)

*   **[FullTransNet: Full Transformer with Local-Global Attention for Video Summarization](http://arxiv.org/abs/2501.00882v2)** (2025.01)
    *   Focus: Video summarization creates compact synopses for efficient video browsing and analysis.
    *   code: [https://github.com/ChiangLu/FullTransNet](https://github.com/ChiangLu/FullTransNet)
    *   citation: 3 [[arxiv bibtex]](https://arxiv.org/bibtex/2501.00882) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22FullTransNet%3A%20Full%20Transformer%20with%20Local-Global%20Attention%20for%20Video%20Summarization%22)

*   **[Do Language Models Understand Time?](http://arxiv.org/abs/2412.13845v3)** (2024.12)
    *   Focus: LLMs enhance video tasks like action recognition and anomaly detection despite unique challenges.
    *   code: [https://github.com/Darcyddx/Video-LLM](https://github.com/Darcyddx/Video-LLM)
    *   citation: 9 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.13845) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Do%20Language%20Models%20Understand%20Time%3F%22)

*   **[Video Repurposing from User Generated Content: A Large-scale Dataset and Benchmark](http://arxiv.org/abs/2412.08879v2)** (ICCV2025 2024.12)
    *   Focus: Short-form video demand grows, but current summarization methods remain inadequate.
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2412.08879) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20Repurposing%20from%20User%20Generated%20Content%3A%20A%20Large-scale%20Dataset%20and%20Benchmark%22)

*   **[LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts](http://arxiv.org/abs/2505.13928v3)** (2025.05)
    *   Focus: Existing video-text retrieval benchmarks have limited video duration, hindering long video understanding.
    *   code: [https://github.com/TechNomad-ds/LoVR-benchmark](https://github.com/TechNomad-ds/LoVR-benchmark)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.13928) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22LoVR%3A%20A%20Benchmark%20for%20Long%20Video%20Retrieval%20in%20Multimodal%20Contexts%22)


### Others

*   **[Video Finetuning Improves Reasoning Between Frames](http://arxiv.org/abs/2511.12868v1)** (2025.11)
    *   Focus: Critiques naive frame token concatenation in video LLMs, proposing improved multimodal architectures.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2511.12868) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Video%20Finetuning%20Improves%20Reasoning%20Between%20Frames%22)

*   **[Dual Band Video Thermography Near Ambient Conditions](http://arxiv.org/abs/2509.11334v1)** (2025.09)
    *   Focus: Thermal camera images combine reflected environmental light and emitted surface radiation.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.11334) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Dual%20Band%20Video%20Thermography%20Near%20Ambient%20Conditions%22)

*   **[RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement](http://arxiv.org/abs/2404.06483v2)** (2024.04)
    *   Focus: rPPG enables non-contact physiological signal measurement from facial videos for healthcare and affective computing.
    *   code: [https://github.com/zizheng-guo/RhythmMamba](https://github.com/zizheng-guo/RhythmMamba)
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2404.06483) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22RhythmMamba%3A%20Fast%2C%20Lightweight%2C%20and%20Accurate%20Remote%20Physiological%20Measurement%22)

*   **[The Dawn of Video Generation: Preliminary Explorations with SORA-like Models](http://arxiv.org/abs/2410.05227v2)** (2024.10)
    *   Focus: High-quality video generation methods like T2V, I2V, and V2V are important for content creation.
    *   project: [https://ailab-cvc.github.io/VideoGen-Eval/](https://ailab-cvc.github.io/VideoGen-Eval/)
    *   citation: 24 [[arxiv bibtex]](https://arxiv.org/bibtex/2410.05227) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22The%20Dawn%20of%20Video%20Generation%3A%20Preliminary%20Explorations%20with%20SORA-like%20Models%22)

*   **[A Video Is Worth 4096 Tokens: Verbalize Videos To Understand Them In Zero Shot](http://arxiv.org/abs/2305.09758v3)** (2023.05)
    *   Focus: Multimedia content combines text, visuals, audio, and storytelling for creative expression.
    *   citation: 14 [[arxiv bibtex]](https://arxiv.org/bibtex/2305.09758) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Video%20Is%20Worth%204096%20Tokens%3A%20Verbalize%20Videos%20To%20Understand%20Them%20In%20Zero%20Shot%22)

*   **[VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform](http://arxiv.org/abs/2504.14904v1)** (2025.04)
    *   Focus: SVPs struggle to moderate harmful content affecting minors' mental health.
    *   project: [https://kuaimod.github.io](https://kuaimod.github.io)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.14904) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VLM%20as%20Policy%3A%20Common-Law%20Content%20Moderation%20Framework%20for%20Short%20Video%20Platform%22)

*   **[Designing Loving-Kindness Meditation in Virtual Reality for Long-Distance Romantic Relationships](http://arxiv.org/abs/2309.11816v1)** (2023.09)
    *   Focus: Virtual reality may enable loving-kindness meditation for isolated couples in therapy.
    *   citation: 10 [[arxiv bibtex]](https://arxiv.org/bibtex/2309.11816) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Designing%20Loving-Kindness%20Meditation%20in%20Virtual%20Reality%20for%20Long-Distance%20Romantic%20Relationships%22)

*   **[EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations](http://arxiv.org/abs/2209.13064v1)** (NeurIPS2022 2022.09)
    *   Focus: VISOR introduces a pixel-level annotation dataset and benchmark for hand and active object segmentation in egocentric video.
    *   project: [http://epic-kitchens.github.io/VISOR](http://epic-kitchens.github.io/VISOR)
    *   citation: 127 [[arxiv bibtex]](https://arxiv.org/bibtex/2209.13064) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EPIC-KITCHENS%20VISOR%20Benchmark%3A%20VIdeo%20Segmentations%20and%20Object%20Relations%22)

*   **[Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding](http://arxiv.org/abs/2505.23990v2)** (2025.05)
    *   Focus: Robots need adaptive decision-making and information filtering for effective human interaction.
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2505.23990) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Multi-RAG%3A%20A%20Multimodal%20Retrieval-Augmented%20Generation%20System%20for%20Adaptive%20Video%20Understanding%22)

*   **[Long-Term 3D Point Tracking By Cost Volume Fusion](http://arxiv.org/abs/2407.13337v1)** (2024.07)
    *   Focus: Deep learning improves long-term point tracking for non-rigid motion understanding.
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.13337) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long-Term%203D%20Point%20Tracking%20By%20Cost%20Volume%20Fusion%22)

*   **[PAD3R: Pose-Aware Dynamic 3D Reconstruction from Casual Videos](http://arxiv.org/abs/2509.25183v1)** (2025.09)
    *   Focus: PAD3R reconstructs deformable 3D objects from long, unposed monocular videos.
    *   project: [https://pad3r.github.io/](https://pad3r.github.io/)
    *   citation: 0 [[arxiv bibtex]](https://arxiv.org/bibtex/2509.25183) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22PAD3R%3A%20Pose-Aware%20Dynamic%203D%20Reconstruction%20from%20Casual%20Videos%22)

*   **[EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations](http://arxiv.org/abs/2209.13064v1)** (NeurIPS2022 2022.09)
    *   Focus: VISOR introduces a pixel-level annotation dataset and benchmark for hand and active object segmentation in egocentric video.
    *   project: [http://epic-kitchens.github.io/VISOR](http://epic-kitchens.github.io/VISOR)
    *   citation: 127 [[arxiv bibtex]](https://arxiv.org/bibtex/2209.13064) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EPIC-KITCHENS%20VISOR%20Benchmark%3A%20VIdeo%20Segmentations%20and%20Object%20Relations%22)

*   **[A Comprehensive Survey on World Models for Embodied AI](http://arxiv.org/abs/2510.16732v1)** (2025.10)
    *   Focus: World models simulate environment dynamics for embodied AI agents to perceive, act, and anticipate future states.
    *   code: [https://github.com/Li-Zn-H/AwesomeWorldModels](https://github.com/Li-Zn-H/AwesomeWorldModels)
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2510.16732) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22A%20Comprehensive%20Survey%20on%20World%20Models%20for%20Embodied%20AI%22)

*   **[EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control](http://arxiv.org/abs/2508.21112v4)** (2025.08)
    *   Focus: Vision-language-action models enable embodied agents to perform multimodal reasoning and physical interaction.
    *   citation: 7 [[arxiv bibtex]](https://arxiv.org/bibtex/2508.21112) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EO-1%3A%20Interleaved%20Vision-Text-Action%20Pretraining%20for%20General%20Robot%20Control%22)

*   **[Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs](http://arxiv.org/abs/2407.07775v2)** (2024.07)
    *   Focus: Research aims to build agents that understand multimodal instructions for navigation.
    *   citation: 53 [[arxiv bibtex]](https://arxiv.org/bibtex/2407.07775) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Mobility%20VLA%3A%20Multimodal%20Instruction%20Navigation%20with%20Long-Context%20VLMs%20and%20Topological%20Graphs%22)

*   **[Spacewalk-18: A Benchmark for Multimodal and Long-form Procedural Video Understanding in Novel Domains](http://arxiv.org/abs/2311.18773v3)** (2023.11)
    *   Focus: Procedural videos help embodied agents learn skills from human demonstrations.
    *   project: [https://brown-palm.github.io/Spacewalk-18/](https://brown-palm.github.io/Spacewalk-18/)
    *   citation: 2 [[arxiv bibtex]](https://arxiv.org/bibtex/2311.18773) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Spacewalk-18%3A%20A%20Benchmark%20for%20Multimodal%20and%20Long-form%20Procedural%20Video%20Understanding%20in%20Novel%20Domains%22)

*   **[EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](http://arxiv.org/abs/2305.15021v2)** (NeurIPS2023 2023.05)
    *   Focus: Introduces Embodied AI for long-horizon robot task planning and execution in physical environments.
    *   citation: 334 [[arxiv bibtex]](https://arxiv.org/bibtex/2305.15021) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22EmbodiedGPT%3A%20Vision-Language%20Pre-Training%20via%20Embodied%20Chain%20of%20Thought%22)

*   **[Enhancing Object Search in Indoor Spaces via Personalized Object-factored Ontologies](http://arxiv.org/abs/2506.14422v1)** (2025.06)
    *   Focus: Service robots require personalized environmental understanding and change awareness for effective operation.
    *   citation: 1 [[arxiv bibtex]](https://arxiv.org/bibtex/2506.14422) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Enhancing%20Object%20Search%20in%20Indoor%20Spaces%20via%20Personalized%20Object-factored%20Ontologies%22)

*   **[From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models](http://arxiv.org/abs/2504.06214v1)** (2025.04)
    *   Focus: Long-context models are vital for document/video understanding, in-context learning, and scaling.
    *   project: [https://ultralong.github.io/](https://ultralong.github.io/)
    *   citation: 8 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.06214) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22From%20128K%20to%204M%3A%20Efficient%20Training%20of%20Ultra-Long%20Context%20Large%20Language%20Models%22)

*   **[Long Context Transfer from Language to Vision](http://arxiv.org/abs/2406.16852v2)** (2024.06)
    *   Focus: Proposes methods to reduce video tokens for long video understanding in large multimodal models.
    *   code: [https://github.com/EvolvingLMMs-Lab/LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)
    *   citation: 320 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.16852) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Long%20Context%20Transfer%20from%20Language%20to%20Vision%22)

*   **[Hallucination Mitigation Prompts Long-term Video Understanding](http://arxiv.org/abs/2406.11333v1)** (2024.06)
    *   Focus: Multimodal LLMs struggle with unprocessed long videos due to computational constraints.
    *   code: [https://github.com/lntzm/CVPR24Track-LongVideo](https://github.com/lntzm/CVPR24Track-LongVideo)
    *   citation: 5 [[arxiv bibtex]](https://arxiv.org/bibtex/2406.11333) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22Hallucination%20Mitigation%20Prompts%20Long-term%20Video%20Understanding%22)

*   **[VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment](http://arxiv.org/abs/2504.14096v3)** (2025.04)
    *   Focus: Introduces a new method to improve Video-LLMs' spatial, temporal, and cross-frame understanding.    
    *   citation: 4 [[arxiv bibtex]](https://arxiv.org/bibtex/2504.14096) [[google scholar bibtex]](https://scholar.google.com/scholar?hl=en&q=%22VideoPASTA%3A%207K%20Preference%20Pairs%20That%20Matter%20for%20Video-LLM%20Alignment%22)
