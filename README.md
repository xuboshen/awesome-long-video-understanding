# 🎥 Awesome-Long-Video-Understanding

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Update date: 6th Dec 2025

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
    *   citation: 21

*   **[EgoLife: Towards Egocentric Life Assistant](http://arxiv.org/abs/2503.03803v1)** (CVPR2025 2025.03)
    *   Focus: EgoLife develops AI-powered wearable glasses for personal efficiency as an egocentric life assistant.
    *   code: [https://github.com/EvolvingLMMs-Lab/EgoLife](https://github.com/EvolvingLMMs-Lab/EgoLife)
    *   citation: 28

*   **[TeleEgo: Benchmarking Egocentric AI Assistants in the Wild](http://arxiv.org/abs/2510.23981v2)** (2025.10)
    *   Focus: Existing benchmarks for egocentric AI assistants lack real-time processing and long-term memory requirements.
    *   citation: 0

*   **[PlanarTrack: A high-quality and challenging benchmark for large-scale planar object tracking](http://arxiv.org/abs/2510.23368v1)** (2025.10)
    *   Focus: Planar tracking advances for robotics and AR, focusing on degenerate cases.
    *   code: [https://github.com/HengLan/PlanarTrack](https://github.com/HengLan/PlanarTrack)
    *   citation: 0

*   **[MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](http://arxiv.org/abs/2510.21406v1)** (NeurIPS2025 2025.10)
    *   Focus: Proposes MUVR benchmark for multi-modal untrimmed video retrieval on long videos.
    *   code: [https://github.com/debby-0527/MUVR](https://github.com/debby-0527/MUVR)
    *   citation: 0

*   **[LongInsightBench: A Comprehensive Benchmark for Evaluating Omni-Modal Models on Human-Centric Long-Video Understanding](http://arxiv.org/abs/2510.17305v2)** (2025.10)
    *   Focus: LongInsightBench is the first benchmark for evaluating long video understanding of language, actions, and context.
    *   citation: 0

*   **[ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](http://arxiv.org/abs/2510.11606v1)** (2025.10)
    *   Focus: MLLMs' potential for scientific discovery is unclear due to limited evaluation of their capabilities.
    *   code: [https://github.com/OpenGVLab/ExpVid](https://github.com/OpenGVLab/ExpVid)
    *   citation: 1

*   **[StreamingVLM: Real-Time Understanding for Infinite Video Streams](http://arxiv.org/abs/2510.09608v1)** (2025.10)
    *   Focus: VLMs struggle with real-time video understanding due to latency and memory constraints.
    *   code: [https://github.com/mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm)
    *   citation: 4

*   **[CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation](http://arxiv.org/abs/2510.09266v1)** (2025.10)
    *   Focus: Video-based MRAG benchmarks enable MLLMs to generate responses using external multimodal evidence.
    *   citation: 0

*   **[StreamForest: Efficient Online Video Understanding with Persistent Event Memory](http://arxiv.org/abs/2509.24871v1)** (NeurIPS2025 2025.09)
    *   Focus: MLLMs struggle with real-time video streaming due to storage constraints.
    *   citation: 2

*   **[NeMo: Needle in a Montage for Video-Language Understanding](http://arxiv.org/abs/2509.24563v2)** (2025.09)
    *   Focus: Proposes new benchmarks for evaluating temporal reasoning in video-language models.
    *   project: [https://lavi-lab.github.io/NeMoBench](https://lavi-lab.github.io/NeMoBench)
    *   citation: 1

*   **[VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding](http://arxiv.org/abs/2509.21451v1)** (2025.09)
    *   Focus: Current video understanding metrics inadequately capture human judgment quality.
    *   citation: 0

*   **[VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](http://arxiv.org/abs/2509.19002v2)** (2025.09)
    *   Focus: MLLMs improve video understanding but face benchmark limitations for practical use.
    *   citation: 0

*   **[NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning](http://arxiv.org/abs/2509.18041v2)** (2025.09)
    *   Focus: Vision-language models struggle with long video question answering due to complex temporal reasoning demands.
    *   project: [https://utaustin-swarmlab.github.io/NeuS-QA/](https://utaustin-swarmlab.github.io/NeuS-QA/)
    *   citation: 0

*   **[Cinéaste: A Fine-grained Contextual Movie Question Answering Benchmark](http://arxiv.org/abs/2509.14227v1)** (2025.09)
    *   Focus: Diagnosing deep narrative comprehension in video-language models is challenging with current benchmarks.
    *   citation: 0

*   **[ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Long Video Understanding](http://arxiv.org/abs/2508.21496v2)** (2025.08)
    *   Focus: Video-MLLMs show strong video understanding but are prone to hallucination.
    *   citation: 0

*   **[EmbRACE-3K: Embodied Reasoning and Action in Complex Environments](http://arxiv.org/abs/2507.10548v1)** (2025.07)
    *   Focus: VLMs excel in passive video understanding but struggle in embodied settings requiring active perception.
    *   project: [https://mxllc.github.io/EmbRACE-3K/](https://mxllc.github.io/EmbRACE-3K/)
    *   citation: 0

*   **[HumanVideo-MME: Benchmarking MLLMs for Human-Centric Video Understanding](http://arxiv.org/abs/2507.04909v2)** (2025.07)
    *   Focus: MLLMs advance in visual tasks but struggle with human-centric video understanding.
    *   citation: 0

*   **[MOMENTS: A Comprehensive Multimodal Benchmark for Theory of Mind](http://arxiv.org/abs/2507.04415v2)** (2025.07)
    *   Focus: MoMentS is a multimodal dataset for understanding mental states in social interactions.
    *   citation: 1

*   **[PhysLab: A Benchmark Dataset for Multi-Granularity Visual Parsing of Physics Experiments](http://arxiv.org/abs/2506.06631v2)** (2025.06)
    *   Focus: Visual parsing progress is limited by dataset constraints like insufficient annotations.
    *   code: [https://github.com/ZMH-SDUST/PhysLab](https://github.com/ZMH-SDUST/PhysLab)
    *   citation: 1

*   **[Movie Facts and Fibs (MF$^2$): A Benchmark for Long Movie Understanding](http://arxiv.org/abs/2506.06275v1)** (2025.06)
    *   Focus: Current benchmarks limit VLMs' ability to understand long-form video content.
    *   citation: 3

*   **[EASG-Bench: Video Q&A Benchmark with Egocentric Action Scene Graphs](http://arxiv.org/abs/2506.05787v2)** (2025.06)
    *   Focus: EASG-Bench is a QA benchmark for egocentric videos using spatio-temporally grounded scene graphs.
    *   code: [https://github.com/fpv-iplab/EASG-bench](https://github.com/fpv-iplab/EASG-bench)
    *   citation: 0

*   **[TextVidBench: A Benchmark for Long Video Scene Text Understanding](http://arxiv.org/abs/2506.04983v1)** (2025.06)
    *   Focus: Current Text-VQA datasets have limited video duration despite recent progress.
    *   citation: 0

*   **[ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding](http://arxiv.org/abs/2505.23922v1)** (2025.05)
    *   Focus: Existing benchmarks lack hierarchical temporal modeling for long-video understanding.
    *   code: [https://github.com/multimodal-art-projection/ScaleLong](https://github.com/multimodal-art-projection/ScaleLong)
    *   citation: 3

*   **[VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](http://arxiv.org/abs/2505.23359v1)** (2025.05)
    *   Focus: Long chain-of-thought reasoning improves LLMs but lacks demonstration for long video understanding tasks.
    *   project: [https://llyx97.github.io/video_reason_bench/](https://llyx97.github.io/video_reason_bench/)
    *   citation: 4

*   **[Two Causally Related Needles in a Video Haystack](http://arxiv.org/abs/2505.19853v3)** (NeurIPS2025 2025.05)
    *   Focus: Causal2Needles benchmark evaluates VLMs on long video understanding tasks.
    *   citation: 0

*   **[VideoEval-Pro: Robust and Realistic Long Video Understanding Evaluation](http://arxiv.org/abs/2505.14640v1)** (2025.05)
    *   Focus: LMMs are advancing long video understanding, driving the creation of standardized benchmarks for evaluation.
    *   project: [https://tiger-ai-lab.github.io/VideoEval-Pro](https://tiger-ai-lab.github.io/VideoEval-Pro)
    *   citation: 5

*   **[LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts](http://arxiv.org/abs/2505.13928v3)** (2025.05)
    *   Focus: Existing video-text retrieval benchmarks have limited video duration, hindering long video understanding.
    *   code: [https://github.com/TechNomad-ds/LoVR-benchmark](https://github.com/TechNomad-ds/LoVR-benchmark)
    *   citation: 2

*   **[Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation](http://arxiv.org/abs/2505.12702v2)** (2025.05)
    *   Focus: RVOS segments and tracks video objects using language, but current methods have limitations.
    *   project: [https://isee-laboratory.github.io/Long-RVOS](https://isee-laboratory.github.io/Long-RVOS)
    *   citation: 2

*   **[VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models](http://arxiv.org/abs/2505.08455v1)** (2025.05)
    *   Focus: LVLMs lack video causal reasoning benchmarks, limiting their evaluation and development.
    *   citation: 1

*   **[RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](http://arxiv.org/abs/2505.02064v3)** (NeurIPS2025 2025.05)
    *   Focus: Current benchmarks fail to assess MLLMs' continuous perception and reasoning abilities.
    *   code: [https://github.com/LJungang/RTV-Bench](https://github.com/LJungang/RTV-Bench)
    *   citation: 4

*   **[SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding](http://arxiv.org/abs/2504.21435v3)** (CVPR2025 2025.04)
    *   Focus: MLLM benchmarks are growing to assess video understanding capabilities.
    *   code: [https://github.com/zackhxn/SeriesBench-CVPR2025](https://github.com/zackhxn/SeriesBench-CVPR2025)
    *   citation: 0

*   **[LVC: A Lightweight Compression Framework for Enhancing VLMs in Long Video Understanding](http://arxiv.org/abs/2504.06835v1)** (2025.04)
    *   Focus: VLMs achieve frame-level understanding but struggle with long video comprehension.
    *   citation: 3

*   **[Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma?](http://arxiv.org/abs/2503.12496v2)** (ICCV2025 2025.03)
    *   Focus: LVLMs struggle with long videos due to the sampling dilemma between low-detail sparse and high-cost dense sampling.
    *   code: [https://github.com/dvlab-research/LSDBench](https://github.com/dvlab-research/LSDBench)
    *   citation: 10

*   **[ALLVB: All-in-One Long Video Understanding Benchmark](http://arxiv.org/abs/2503.07298v2)** (2025.03)
    *   Focus: Existing video benchmarks are too short for evaluating MLLMs on long videos.
    *   citation: 3
    
*   **[MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](http://arxiv.org/abs/2506.04141v1)** (2025.06)
    *   Focus: Proposes a new MLLM architecture for improved long video understanding and temporal reasoning.
    *   project: [https://mmr-v.github.io](https://mmr-v.github.io)
    *   citation: 6

*   **[MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval](http://arxiv.org/abs/2502.12558v4)** (NeurIPS2025 2025.02)
    *   Focus: Proposes a new benchmark for long video understanding with longer videos and more diverse tasks.
    *   project: [https://yhy-2000.github.io/MomentSeeker/](https://yhy-2000.github.io/MomentSeeker/)
    *   citation: 1

*   **[SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding](http://arxiv.org/abs/2502.10810v2)** (ICLR2025 2025.02)
    *   Focus: LVLMs lack suitable evaluation for emerging applications.
    *   code: [https://github.com/sotayang/SVBench](https://github.com/sotayang/SVBench)
    *   citation: 20


*   **[$\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation](http://arxiv.org/abs/2501.19098v2)** (2025.01)
    *   Focus: This paper introduces a new method for long-video understanding using compressed representations and temporal modeling.
    *   citation: 8

*   **[X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding](http://arxiv.org/abs/2501.06835v2)** (2025.01)
    *   Focus: Long-form egocentric videos offer insights into human behavior for embodied intelligence applications.
    *   citation: 4

*   **[HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding](http://arxiv.org/abs/2501.01645v3)** (2025.01)
    *   Focus: Multimodal LLMs advance visual understanding but struggle with hour-long video comprehension.
    *   citation: 6

*   **[FriendsQA: A New Large-Scale Deep Video Understanding Dataset with Fine-grained Topic Categorization for Story Videos](http://arxiv.org/abs/2412.17022v1)** (2024.12)
    *   Focus: VideoQA models struggle with complex questions despite good performance on factoid tasks.
    *   citation: 2

*   **[Do Language Models Understand Time?](http://arxiv.org/abs/2412.13845v3)** (2024.12)
    *   Focus: LLMs enhance video tasks like action recognition and anomaly detection despite unique challenges.
    *   code: [https://github.com/Darcyddx/Video-LLM](https://github.com/Darcyddx/Video-LLM)
    *   citation: 9

*   **[CG-Bench: Clue-grounded Question Answering Benchmark for Long Video Understanding](http://arxiv.org/abs/2412.12075v1)** (ICLR2025 2024.12)
    *   Focus: Existing long video benchmarks for MLLMs rely on single annotations, limiting evaluation of temporal reasoning.
    *   project: [https://cg-bench.github.io/leaderboard/](https://cg-bench.github.io/leaderboard/)
    *   citation: 35

*   **[Apollo: An Exploration of Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2412.10360v1)** (CVPR2025 2024.12)
    *   Focus: This paper investigates the mechanisms behind video understanding in large multimodal models.
    *   project: [https://apollo-lmms.github.io](https://apollo-lmms.github.io)
    *   citation: 51

*   **[Neptune: The Long Orbit to Benchmarking Long Video Understanding](http://arxiv.org/abs/2412.09582v2)** (2024.12)
    *   Focus: Neptune is a new benchmark for long video understanding requiring multimodal reasoning over extended time.
    *   code: [https://github.com/google-deepmind/neptune](https://github.com/google-deepmind/neptune)
    *   citation: 14

*   **[Perception Test 2024: Challenge Summary and a Novel Hour-Long VideoQA Benchmark](http://arxiv.org/abs/2411.19941v1)** (2024.11)
    *   Focus: The Second Perception Test challenge at ECCV 2024 continues benchmarking visual perception tasks.
    *   citation: 3

*   **[HourVideo: 1-Hour Video-Language Understanding](http://arxiv.org/abs/2411.04998v1)** (NeurIPS2024 2024.11)
    *   Focus: HourVideo is a benchmark for hour-long video understanding with summarization, perception, and reasoning tasks.
    *   citation: 79

*   **[FIOVA: A Multi-Annotator Benchmark for Human-Aligned Video Captioning](http://arxiv.org/abs/2410.15270v2)** (2024.10)
    *   Focus: Existing video caption benchmarks inadequately assess LVLM alignment with human understanding due to single-annotation limitations.
    *   project: [https://huuuuusy.github.io/fiova/](https://huuuuusy.github.io/fiova/)
    *   citation: 3

*   **[TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models](http://arxiv.org/abs/2410.10818v2)** (2024.10)
    *   Focus: Existing video benchmarks lack fine-grained temporal annotations for detailed video understanding.
    *   project: [https://temporalbench.github.io/](https://temporalbench.github.io/)
    *   citation: 38

*   **[MM-Ego: Towards Building Egocentric Multimodal LLMs for Video QA](http://arxiv.org/abs/2410.07177v2)** (ICLR2025 2024.10)
    *   Focus: This research builds a multimodal foundation model for egocentric video understanding.
    *   citation: 17

*   **[LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding](http://arxiv.org/abs/2407.15754v1)** (NeurIPS2024 2024.07)
    *   Focus: Introduces a new benchmark for evaluating large multimodal models on long, rich inputs.
    *   citation: 311

*   **[InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies and TV Shows](http://arxiv.org/abs/2406.19875v5)** (2024.06)
    *   Focus: Long video understanding is challenging due to inadequate benchmarks for multi-modal models.
    *   project: [https://vision-cair.github.io/Infinibench](https://vision-cair.github.io/Infinibench)
    *   citation: 1

*   **[MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding](http://arxiv.org/abs/2406.14515v3)** (NeurIPS2024 2024.06)
    *   Focus: LVLMs advance video understanding beyond traditional VideoQA benchmarks.
    *   code: [https://github.com/open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
    *   citation: 137

*   **[Towards Event-oriented Long Video Understanding](http://arxiv.org/abs/2406.14129v1)** (2024.06)
    *   Focus: Video MLLMs lack benchmarks with rich evidence for comprehensive evaluation.
    *   code: [https://github.com/RUCAIBox/Event-Bench](https://github.com/RUCAIBox/Event-Bench)
    *   citation: 19

*   **[VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment](http://arxiv.org/abs/2406.10889v2)** (CVPR2025 2024.06)
    *   Focus: Video models advance but struggle with associating people and actions over time for compositional reasoning.
    *   project: [https://katha-ai.github.io/projects/velociti](https://katha-ai.github.io/projects/velociti)
    *   citation: 3

*   **[LVBench: An Extreme Long Video Understanding Benchmark](http://arxiv.org/abs/2406.08035v3)** (ICCV2025 2024.06)
    *   Focus: Multimodal LLMs improve short video understanding, but lack benchmarks for long videos.
    *   project: [https://lvbench.github.io](https://lvbench.github.io)
    *   citation: 184

*   **[Vript: A Video Is Worth Thousands of Words](http://arxiv.org/abs/2406.06040v2)** (NeurIPS2024 2024.06)
    *   Focus: Vript introduces a method to create high-quality video-text datasets for multimodal learning.
    *   code: [https://github.com/mutonix/Vript](https://github.com/mutonix/Vript)
    *   citation: 54

*   **[MLVU: Benchmarking Multi-task Long Video Understanding](http://arxiv.org/abs/2406.04264v3)** (CVPR2025 2024.06)
    *   Focus: Existing benchmarks are insufficient for evaluating long video understanding performance.
    *   citation: 77

*   **[Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](http://arxiv.org/abs/2405.21075v3)** (CVPR2025 2024.05)
    *   Focus: MLLMs advance towards AGI but lack focus on long video understanding.
    *   project: [https://video-mme.github.io](https://video-mme.github.io)
    *   citation: 755

*   **[CinePile: A Long Video Question Answering Dataset and Benchmark](http://arxiv.org/abs/2405.08813v3)** (2024.05)
    *   Focus: Current long-form video datasets lack genuine comprehension challenges, as tasks are solvable by models without deep understanding.
    *   project: [https://ruchitrawal.github.io/cinepile/](https://ruchitrawal.github.io/cinepile/)
    *   citation: 85

*   **[WorldQA: Multimodal World Knowledge in Videos through Long-Chain Reasoning](http://arxiv.org/abs/2405.03272v1)** (2024.05)
    *   Focus: LLMs and LMMs struggle to emulate human understanding of complex, dynamic multimodal information.
    *   citation: 16

*   **[LvBench: A Benchmark for Long-form Video Understanding with Versatile Multi-modal Question Answering](http://arxiv.org/abs/2312.04817v2)** (2023.12)
    *   Focus: Current VideoQA datasets use short videos, limiting genuine long-form video understanding.
    *   citation: 22

*   **[Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives](http://arxiv.org/abs/2311.18259v4)** (CVPR2024 2023.11)
    *   Focus: Ego-Exo4D is a large multimodal multiview video dataset with simultaneous egocentric and exocentric recordings.
    *   citation: 301

*   **[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](http://arxiv.org/abs/2309.13925v2)** (2023.09)
    *   Focus: Surveillance video tasks need expansion beyond classification to include temporal localization and dense captioning.
    *   project: [https://xuange923.github.io/Surveillance-Video-Understanding](https://xuange923.github.io/Surveillance-Video-Understanding)
    *   citation: 37

*   **[So you think you can track?](http://arxiv.org/abs/2309.07268v1)** (2023.09)
    *   Focus: A 234-hour multi-camera tracking dataset covering 4.2 miles of interstate highway.
    *   citation: 21

*   **[EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding](http://arxiv.org/abs/2308.09126v1)** (NeurIPS2023 2023.08)
    *   Focus: EgoSchema is a long-form video QA dataset and benchmark for evaluating video understanding systems.
    *   project: [http://egoschema.github.io](http://egoschema.github.io)
    *   citation: 463

*   **[MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](http://arxiv.org/abs/2307.16449v4)** (CVPR2024 2023.07)
    *   Focus: Video foundation models and LLMs are integrated to overcome task-specific limitations in video understanding.
    *   project: [https://rese1f.github.io/MovieChat/](https://rese1f.github.io/MovieChat/)
    *   citation: 429

*   **[Towards Long Form Audio-visual Video Understanding](http://arxiv.org/abs/2306.09431v1)** (2023.06)
    *   Focus: Long audio-visual videos bridge multimodal information for real-world scenario understanding.
    *   project: [http://gewu-lab.github.io/LFAV/](http://gewu-lab.github.io/LFAV/)
    *   citation: 12

*   **[Building Scalable Video Understanding Benchmarks through Sports](http://arxiv.org/abs/2301.06866v3)** (2023.01)
    *   Focus: Existing long video benchmarks lack scale and annotation quality due to collection difficulties.
    *   project: [https://asap-benchmark.github.io/](https://asap-benchmark.github.io/)
    *   citation: 2

*   **[EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations](http://arxiv.org/abs/2209.13064v1)** (NeurIPS2022 2022.09)
    *   Focus: VISOR introduces a pixel-level annotation dataset and benchmark for hand and active object segmentation in egocentric video.
    *   project: [http://epic-kitchens.github.io/VISOR](http://epic-kitchens.github.io/VISOR)
    *   citation: 127

*   **[Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities](http://arxiv.org/abs/2203.14712v2)** (CVPR2022 2022.03)
    *   Focus: Assembly101 dataset contains 4321 videos of people assembling toy vehicles without fixed instructions.
    *   project: [https://assembly-101.github.io/](https://assembly-101.github.io/)
    *   citation: 280

*   **[VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models](http://arxiv.org/abs/2505.08455v1)** (2025.05)
    *   Focus: LVLMs lack video causal reasoning benchmarks, limiting their evaluation and development.
    *   citation: 1


### Vision-Language Models
*   **[Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](http://arxiv.org/abs/2507.06261v5)** (2025.07)
    *   Focus: Introduces the Gemini 2.X model family including Pro and Flash variants.
    *   citation: 1054

*   **[Qwen3-VL Technical Report](http://arxiv.org/abs/2511.21631v1)** (2025.11)
    *   Focus: Qwen3-VL is a top-performing vision-language model excelling in multimodal benchmarks.
    *   citation: 11


*   **[NVIDIA Nemotron Nano V2 VL](http://arxiv.org/abs/2511.03929v2)** (2025.11)
    *   Focus: Nemotron Nano V2 VL advances document understanding, long video comprehension, and reasoning tasks.
    *   citation: 1

*   **[Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models](http://arxiv.org/abs/2504.15271v1)** (NeurIPS2025 2025.04)
    *   Focus: Eagle 2.5 introduces vision-language models for long video and high-resolution image understanding.
    *   citation: 20

*   **[Qwen2.5-VL Technical Report](http://arxiv.org/abs/2502.13923v1)** (2025.02)
    *   Focus: Qwen2.5-VL advances vision-language capabilities with new features and improved performance.
    *   citation: 2376

*   **[Kimi-VL Technical Report](http://arxiv.org/abs/2504.07491v3)** (2025.04)
    *   Focus: Kimi-VL is an efficient open-source MoE vision-language model for multimodal reasoning and long-context understanding.
    *   code: [https://github.com/MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL)
    *   citation: 122

*   **[InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling](http://arxiv.org/abs/2501.12386v3)** (2025.01)
    *   Focus: InternVideo2.5 improves video MLLMs using long and rich context modeling.
    *   code: [https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5)
    *   citation: 100

*   **[GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](http://arxiv.org/abs/2507.01006v5)** (2025.07)
    *   Focus: GLM-4.1V-Thinking and GLM-4.5V are new vision-language models for multimodal understanding and reasoning.
    *   code: [https://github.com/zai-org/GLM-V](https://github.com/zai-org/GLM-V)
    *   citation: 64

*   **[VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation](http://arxiv.org/abs/2412.00927v1)** (CVPR2025 2024.12)
    *   Focus: Lack of high-quality datasets limits large multimodal models' ability to process long or high-resolution videos.
    *   project: [https://tiger-ai-lab.github.io/VISTA/](https://tiger-ai-lab.github.io/VISTA/)
    *   citation: 9


*   **[Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy](http://arxiv.org/abs/2502.05177v3)** (2025.02)
    *   Focus: Long-VITA is a multi-modal model for long-context visual-language understanding tasks.
    *   code: [https://github.com/VITA-MLLM/Long-VITA](https://github.com/VITA-MLLM/Long-VITA)
    *   citation: 23

*   **[VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling](http://arxiv.org/abs/2501.00574v4)** (2024.12)
    *   Focus: Long-context video modeling is essential for MLLMs but remains challenging.
    *   citation: 94

*   **[Cambrian-S: Towards Spatial Supersensing in Video](http://arxiv.org/abs/2511.04670v1)** (2025.11)
    *   Focus: Argues for shifting from reactive systems to supersensing for multimodal intelligence.
    *   project: [https://cambrian-mllm.github.io/](https://cambrian-mllm.github.io/)
    *   citation: 6

*   **[From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding](http://arxiv.org/abs/2409.18938v2)** (2024.09)
    *   Focus: LLMs combined with visual encoders improve visual understanding tasks.
    *   citation: 17

*   **[mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models](http://arxiv.org/abs/2408.04840v2)** (ICLR2025 2024.08)
    *   Focus: MLLMs excel at single-image tasks but face challenges in other areas.
    *   citation: 206

*   **[ST-LLM: Large Language Models Are Effective Temporal Learners](http://arxiv.org/abs/2404.00308v1)** (ECCV2024 2024.03)
    *   Focus: Research explores video LLMs for human-AI interaction using text comprehension and generation.
    *   code: [https://github.com/TencentARC/ST-LLM](https://github.com/TencentARC/ST-LLM)
    *   citation: 119

*   **[Valley: Video Assistant with Large Language model Enhanced abilitY](http://arxiv.org/abs/2306.07207v3)** (2023.06)
    *   Focus: LLMs show promise as multimodal AI assistants but struggle with joint video and language understanding.
    *   code: [https://github.com/valley-vl/Valley](https://github.com/valley-vl/Valley)
    *   citation: 246

*   **[OmChat: A Recipe to Train Multimodal Language Models with Strong Long Context and Video Understanding](http://arxiv.org/abs/2407.04923v1)** (2024.07)
    *   Focus: OmChat is a new model for long-context video understanding with standardized visual input processing.
    *   citation: 8

*   **[Summarization of Multimodal Presentations with Vision-Language Models: Study of the Effect of Modalities and Structure](http://arxiv.org/abs/2504.10049v1)** (2025.04)
    *   Focus: Fine-grained analysis of Vision-Language Models processing various visual-textual formats including long videos.
    *   citation: 0

*   **[Efficient Motion-Aware Video MLLM](http://arxiv.org/abs/2503.13016v1)** (CVPR2025 2025.03)
    *   Focus: EMA addresses inefficient video processing and motion awareness in MLLMs.
    *   citation: 3

*   **[Koala: Key frame-conditioned long video-LLM](http://arxiv.org/abs/2404.04346v3)** (CVPR2024 2024.04)
    *   Focus: Video LLMs struggle with long videos due to short-term focus and lack of fine-grained relationship reasoning.
    *   citation: 58

### Subsampling methods
#### RAG / Memory / Agentic / Language Repository / Frame Sampling Methods

*   **[Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding](http://arxiv.org/abs/2510.14032v1)** (NeurIPS2025 2025.10)
    *   Focus: Long video understanding is limited by token processing constraints in large video language models.
    *   project: [https://xiaoqian-shen.github.io/Vgent](https://xiaoqian-shen.github.io/Vgent)
    *   citation: 2

*   **[VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos](http://arxiv.org/abs/2405.19209v3)** (CVPR2025 2024.05)
    *   Focus: VideoTree is a training-free method for long video understanding that addresses redundancy and irrelevant information.
    *   project: [https://videotree2024.github.io/](https://videotree2024.github.io/)
    *   citation: 135

*   **[REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding](http://arxiv.org/abs/2511.13026v1)** (2025.11)
    *   Focus: Text-based self-reflection struggles with long video understanding.
    *   citation: 0

*   **[Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding](http://arxiv.org/abs/2511.14446v1)** (2025.11)
    *   Focus: VLMs process videos frame-by-frame, lacking efficient long-range temporal reasoning.
    *   citation: 0


*   **[AVATAAR: Agentic Video Answering via Temporal Adaptive Alignment and Reasoning](http://arxiv.org/abs/2511.15578v1)** (2025.11)
    *   Focus: Proposes a method for understanding and answering questions about long videos.
    *   citation: 0

*   **[Adaptive Video Understanding Agent: Enhancing efficiency with dynamic frame sampling and feedback-driven reasoning](http://arxiv.org/abs/2410.20252v1)** (2024.10)
    *   Focus: An agent-based approach for understanding long videos, addressing temporal complexity and computational demands.
    *   citation: 4
    
*   **[LAST: LeArning to Think in Space and Time for Generalist Vision-Language Models](http://arxiv.org/abs/2511.19261v1)** (2025.11)
    *   Focus: Vision-language models struggle to understand 3D space and long videos like humans.
    *   citation: 0

*   **[LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling](http://arxiv.org/abs/2511.20785v1)** (2025.11)
    *   Focus: Large multimodal models for video reasoning are prone to hallucinations in long-form content.
    *   code: [https://github.com/EvolvingLMMs-Lab/LongVT](https://github.com/EvolvingLMMs-Lab/LongVT)
    *   citation: 3
    *   code: https://github.com/EvolvingLMMs-Lab/LongVT

*   **[Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding](http://arxiv.org/abs/2511.14446v1)** (2025.11)
    *   Focus: VLMs process videos frame-by-frame, lacking efficient long-range temporal reasoning.
    *   citation: 0

*   **[Vision-Language Memory for Spatial Reasoning](http://arxiv.org/abs/2511.20644v1)** (2025.11)
    *   Focus: Vision-language models underperform humans in video spatial reasoning, highlighting a key research gap.
    *   citation: 0

*   **[iRAG: Advancing RAG for Videos with an Incremental Approach](http://arxiv.org/abs/2404.12309v2)** (2024.04)
    *   Focus: RAG systems combine language generation and retrieval for video understanding tasks.
    *   citation: 12

*   **[Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding](http://arxiv.org/abs/2505.23990v2)** (2025.05)
    *   Focus: Robots need adaptive decision-making and information filtering for effective human interaction.
    *   citation: 5

*   **[F4D: Factorized 4D Convolutional Neural Network for Efficient Video-level Representation Learning](http://arxiv.org/abs/2401.08609v1)** (2023.11)
    *   Focus: Video-level representation learning captures long-range temporal structure for action recognition.
    *   citation: 3

*   **[VideoAgent2: Enhancing the LLM-Based Agent System for Long-Form Video Understanding by Uncertainty-Aware CoT](http://arxiv.org/abs/2504.04471v1)** (2025.04)
    *   Focus: Agent-based approaches are gaining popularity for processing long videos.
    *   citation: 12

*   **[VideoLucy: Deep Memory Backtracking for Long Video Understanding](http://arxiv.org/abs/2510.12422v1)** (NeurIPS2025 2025.10)
    *   Focus: Agent-based systems using LLMs for information retrieval show promise in long video understanding.
    *   project: [https://videolucy.github.io](https://videolucy.github.io)
    *   citation: 0


*   **[GCAgent: Long-Video Understanding via Schematic and Narrative Episodic Memory](http://arxiv.org/abs/2511.12027v1)** (2025.11)
    *   Focus: MLLMs struggle with long videos due to token limits and temporal dependency complexity.
    *   citation: 0

*   **[VideoSSR: Video Self-Supervised Reinforcement Learning](http://arxiv.org/abs/2511.06281v1)** (2025.11)
    *   Focus: RLVR improves MLLM video understanding, but rapid progress challenges evaluation.
    *   code: [https://github.com/lcqysl/VideoSSR](https://github.com/lcqysl/VideoSSR)
    *   citation: 0

*   **[VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning with LLMs](http://arxiv.org/abs/2409.20365v2)** (2024.09)
    *   Focus: Zero-shot LLM reasoning challenges end-to-end video models but faces efficiency issues.
    *   code: [https://github.com/mayhugotong/VideoINSTA](https://github.com/mayhugotong/VideoINSTA)
    *   citation: 24

*   **[FRAG: Frame Selection Augmented Generation for Long Video and Long Document Understanding](http://arxiv.org/abs/2504.17447v1)** (2025.04)
    *   Focus: Large Multimodal Models face challenges with long inputs due to size and performance constraints.
    *   code: [https://github.com/NVlabs/FRAG](https://github.com/NVlabs/FRAG)
    *   citation: 10

*   **[TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning](http://arxiv.org/abs/2511.05489v1)** (2025.11)
    *   Focus: Temporal search finds minimal relevant frames from long videos for accurate video understanding.
    *   code: [https://github.com/Time-Search/TimeSearch-R](https://github.com/Time-Search/TimeSearch-R)
    *   citation: 0

*   **[Perceive, Reflect and Understand Long Video: Progressive Multi-Granular Clue Exploration with Interactive Agents](http://arxiv.org/abs/2509.24943v1)** (2025.09)
    *   Focus: LLM-based methods struggle with long video reasoning due to temporal complexity and sparse relevant information.
    *   citation: 0

*   **[LOVE-R1: Advancing Long Video Understanding with an Adaptive Zoom-in Mechanism via Multi-Step Reasoning](http://arxiv.org/abs/2509.24786v1)** (2025.09)
    *   Focus: LVLMs struggle with long video understanding due to temporal-spatial perception conflicts.
    *   citation: 2

*   **[FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting](http://arxiv.org/abs/2509.24304v2)** (2025.09)
    *   Focus: LVLMs struggle with long videos due to uniform frame sampling and static text.
    *   citation: 5

*   **[Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](http://arxiv.org/abs/2508.20478v1)** (2025.08)
    *   Focus: Long video understanding faces challenges with temporal dependencies and multiple events.
    *   citation: 7

*   **[Episodic Memory Representation for Long-form Video Understanding](http://arxiv.org/abs/2508.09486v1)** (2025.08)
    *   Focus: Video-LLMs use keyframe retrieval to overcome context limits for long videos.
    *   citation: 3

*   **[LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents](http://arxiv.org/abs/2503.10200v4)** (ICCV2025 2025.03)
    *   Focus: Agent-based methods use external tools to help MLLMs handle long video temporal context.
    *   code: [https://github.com/64327069/LVAgent](https://github.com/64327069/LVAgent)
    *   citation: 13

*   **[Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](http://arxiv.org/abs/2508.09736v4)** (2025.08)
    *   Focus: M3-Agent is a multimodal framework with long-term memory for processing real-time visual and auditory inputs.
    *   code: [https://github.com/bytedance-seed/m3-agent](https://github.com/bytedance-seed/m3-agent)
    *   citation: 6

*   **[Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning](http://arxiv.org/abs/2508.04416v2)** (2025.08)
    *   Focus: MLLMs need better video reasoning for tasks like QA and temporal grounding, but current methods rely too much on text.
    *   project: [https://zhang9302002.github.io/thinkingwithvideos-page/](https://zhang9302002.github.io/thinkingwithvideos-page/)
    *   citation: 18

*   **[Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames](http://arxiv.org/abs/2507.02001v1)** (NeurIPS2025 2025.07)
    *   Focus: Long-video understanding remains challenging despite VLMs processing up to 1000 frames.
    *   citation: 7

*   **[Iterative Zoom-In: Temporal Interval Exploration for Long Video Understanding](http://arxiv.org/abs/2507.02946v1)** (2025.06)
    *   Focus: MLLMs struggle with long videos due to inefficient temporal perception.
    *   citation: 1

*   **[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](http://arxiv.org/abs/2506.13589v3)** (NeurIPS2025 2025.06)
    *   Focus: MLLMs struggle with long videos due to fixed context and weak long-term modeling, suggesting retrieval-augmented generation as a solution.
    *   citation: 4

*   **[VideoExplorer: Think With Videos For Agentic Long-Video Understanding](http://arxiv.org/abs/2506.10821v6)** (2025.06)
    *   Focus: Existing long-video methods sacrifice details or rely on text, lacking efficient visual modeling.
    *   code: [https://github.com/yhy-2000/VideoDeepResearch](https://github.com/yhy-2000/VideoDeepResearch)
    *   citation: 3

*   **[SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding](http://arxiv.org/abs/2506.07600v1)** (2025.06)
    *   Focus: Long-form video understanding is underexplored due to scale and complexity challenges.
    *   citation: 0

*   **[VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning](http://arxiv.org/abs/2506.06097v1)** (2025.06)
    *   Focus: MLLMs struggle with long video understanding despite advances in short video analysis.
    *   citation: 11

*   **[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](http://arxiv.org/abs/2505.18079v4)** (NeurIPS2025 2025.05)
    *   Focus: Long video understanding faces challenges from temporal-spatial complexity and extended context question answering.
    *   code: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)
    *   citation: 7

*   **[MASR: Self-Reflective Reasoning through Multimodal Hierarchical Attention Focusing for Agent-based Video Understanding](http://arxiv.org/abs/2504.17213v2)** (2025.04)
    *   Focus: Video understanding is challenging due to high information redundancy compared to text or images.
    *   citation: 3

*   **[MR. Video: "MapReduce" is the Principle for Long Video Understanding](http://arxiv.org/abs/2504.16082v1)** (2025.04)
    *   Focus: MR. Video uses MapReduce for dense perception and reasoning in long video understanding.
    *   code: [https://github.com/ziqipang/MR-Video](https://github.com/ziqipang/MR-Video)
    *   citation: 4

*   **[QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v2)** (2025.05)
    *   Focus: Long-video understanding is crucial for real-world applications but faces challenges.
    *   citation: 2

*   **[AVA: Towards Agentic Video Analytics with Vision Language Models](http://arxiv.org/abs/2505.00254v5)** (2025.05)
    *   Focus: AI video analytics systems lack adaptability for open-ended tasks beyond predefined functions.
    *   code: [https://github.com/I-ESC/Project-Ava](https://github.com/I-ESC/Project-Ava)
    *   citation: 4

*   **[TimeSearch: Hierarchical Video Search with Spotlight and Reflection for Human-like Long Video Understanding](http://arxiv.org/abs/2504.01407v1)** (2025.04)
    *   Focus: LVLMs face challenges with long videos due to high computational demands and memory constraints.
    *   citation: 4

*   **[RAG-Adapter: A Plug-and-Play RAG-enhanced Framework for Long Video Understanding](http://arxiv.org/abs/2503.08576v1)** (2025.03)
    *   Focus: MLLMs are advancing rapidly, requiring long video benchmarks to assess comprehension.
    *   citation: 3

*   **[VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos](http://arxiv.org/abs/2502.01549v1)** (2025.02)
    *   Focus: RAG enhances LLMs with external knowledge but is under-explored for long video understanding.
    *   code: [https://github.com/HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG)
    *   citation: 23

*   **[Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding](http://arxiv.org/abs/2501.00358v2)** (ICCV2025 2024.12)
    *   Focus: This paper explores dynamic 3D scene understanding from egocentric views for robotics and embodied AI.
    *   project: [https://embodied-videoagent.github.io/](https://embodied-videoagent.github.io/)
    *   citation: 10

*   **[Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](http://arxiv.org/abs/2411.13093v4)** (NeurIPS2025 2024.11)
    *   Focus: Fine-tuning long-context LVLMs and using GPT agents to improve long video understanding.
    *   citation: 63


*   **[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](http://arxiv.org/abs/2506.13589v3)** (NeurIPS2025 2025.06)
    *   Focus: MLLMs struggle with long videos due to fixed context and weak long-term modeling, suggesting retrieval-augmented generation as a solution.
    *   citation: 4

*   **[Video-VoT-R1: An efficient video inference model integrating image packing and AoE architecture](http://arxiv.org/abs/2503.15807v1)** (2025.03)
    *   Focus: Proposes a new video-language model to improve inference efficiency and multimodal processing.
    *   citation: 2

*   **[AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding](http://arxiv.org/abs/2503.12559v2)** (2025.03)
    *   Focus: MLLMs compress long videos using visual-language models to overcome context length limits.
    *   code: [https://github.com/SCZwangxiao/video-FlexReduc.git](https://github.com/SCZwangxiao/video-FlexReduc.git)
    *   citation: 16

*   **[Visual Context Window Extension: A New Perspective for Long Video Understanding](http://arxiv.org/abs/2409.20018v2)** (2024.09)
    *   Focus: LMMs struggle with long videos, while LLMs excel using language as a compressed representation.
    *   citation: 1

*   **[Understanding Long Videos with Multimodal Language Models](http://arxiv.org/abs/2403.16998v5)** (ICLR2025 2024.03)
    *   Focus: LLMs' world knowledge and reasoning improve long-video understanding benchmarks.
    *   code: [https://github.com/kahnchana/mvu](https://github.com/kahnchana/mvu)
    *   citation: 15
    *   code: https://github.com/kahnchana/mvu

*   **[Language Repository for Long Video Understanding](http://arxiv.org/abs/2403.14622v2)** (2024.03)
    *   Focus: LLMs' long-context effectiveness declines over time despite supporting extended inputs.
    *   code: [https://github.com/kkahatapitiya/LangRepo](https://github.com/kkahatapitiya/LangRepo)
    *   citation: 48

*   **[VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](http://arxiv.org/abs/2403.11481v2)** (ECCV2024 2024.03)
    *   Focus: A unified memory mechanism combines foundation models for improved video understanding.
    *   project: [http://videoagent.github.io](http://videoagent.github.io)
    *   citation: 139
    *   code: https://github.com/YueFan1014/VideoAgent

*   **[A Simple LLM Framework for Long-Range Video Question-Answering](http://arxiv.org/abs/2312.17235v3)** (2023.12)
    *   Focus: LLoVi is a language-based framework for efficient long-range video question-answering.
    *   code: [https://github.com/CeeZh/LLoVi](https://github.com/CeeZh/LLoVi)
    *   citation: 141

*   **[VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](http://arxiv.org/abs/2510.06040v1)** (ICCV2025 2025.10)
    *   Focus: Proposes a compression method for efficient hour-long video understanding with multi-modal LLMs.
    *   code: [https://github.com/caoxinye/VideoMiner](https://github.com/caoxinye/VideoMiner)
    *   citation: 0
    
*   **[From Captions to Keyframes: KeyScore for Multimodal Frame Scoring and Video-Language Understanding](http://arxiv.org/abs/2510.06509v2)** (2025.10)
    *   Focus: KeyScore selects informative video keyframes using semantics to reduce redundancy.
    *   citation: 1

*   **[FOCUS: Efficient Keyframe Selection for Long Video Understanding](http://arxiv.org/abs/2510.27280v2)** (2025.10)
    *   Focus: MLLMs face impractical token inflation when scaling from images to hour-long videos.
    *   code: [https://github.com/NUS-HPC-AI-Lab/FOCUS](https://github.com/NUS-HPC-AI-Lab/FOCUS)
    *   citation: 0

*   **[K-frames: Scene-Driven Any-k Keyframe Selection for long video understanding](http://arxiv.org/abs/2510.13891v1)** (2025.10)
    *   Focus: MLLMs struggle with long videos due to context limits and high computational costs.
    *   citation: 1

*   **[AdaRD-key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video understanding](http://arxiv.org/abs/2510.02778v1)** (2025.10)
    *   Focus: VLMs struggle with long videos due to length and density, needing better compression and modeling.
    *   code: [https://github.com/Xian867/AdaRD-Key](https://github.com/Xian867/AdaRD-Key)
    *   citation: 0

*   **[From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding](http://arxiv.org/abs/2510.02262v1)** (2025.10)
    *   Focus: Video LLMs struggle with finding relevant information in large video data.
    *   citation: 0

*   **[KFFocus: Highlighting Keyframes for Enhanced Video Understanding](http://arxiv.org/abs/2508.08989v1)** (2025.08)
    *   Focus: Multimodal LLMs show strong video understanding but face computational challenges from long videos.
    *   citation: 0


*   **[VSI: Visual Subtitle Integration for Keyframe Selection to enhance Long Video Understanding](http://arxiv.org/abs/2508.06869v3)** (2025.08)
    *   Focus: MLLMs struggle with long videos due to context limits and high computational costs.
    *   citation: 0

*   **[TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](http://arxiv.org/abs/2508.04369v4)** (2025.08)
    *   Focus: MLLMs struggle with long video inputs due to computational and memory constraints.
    *   code: [https://github.com/Hui-design/TSPO](https://github.com/Hui-design/TSPO)
    *   citation: 3

*   **[Enhancing Long Video Question Answering with Scene-Localized Frame Grouping](http://arxiv.org/abs/2508.03009v1)** (2025.08)
    *   Focus: MLLMs struggle with long videos due to resource constraints, limiting frame processing and associated text.
    *   citation: 1

*   **[E-VRAG: Enhancing Long Video Understanding with Resource-Efficient Retrieval Augmented Generation](http://arxiv.org/abs/2508.01546v1)** (2025.08)
    *   Focus: Vision-language models advance video understanding but face context limitations.
    *   citation: 1

*   **[VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding](http://arxiv.org/abs/2507.13353v1)** (2025.07)
    *   Focus: Selecting informative video frames boosts Video-LLM performance by reducing redundancy.
    *   citation: 8

*   **[From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding](http://arxiv.org/abs/2507.02790v2)** (2025.07)
    *   Focus: Efficient video editing techniques are needed to condense long videos into concise summaries.
    *   citation: 0

*   **[Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames](http://arxiv.org/abs/2507.02001v1)** (NeurIPS2025 2025.07)
    *   Focus: Long-video understanding remains challenging despite VLMs processing up to 1000 frames.
    *   citation: 7

*   **[Iterative Zoom-In: Temporal Interval Exploration for Long Video Understanding](http://arxiv.org/abs/2507.02946v1)** (2025.06)
    *   Focus: MLLMs struggle with long videos due to inefficient temporal perception.
    *   citation: 1

*   **[Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](http://arxiv.org/abs/2506.13654v1)** (2025.06)
    *   Focus: Ego-R1 uses Chain-of-Tool-Thought reasoning for ultra-long egocentric video understanding.
    *   project: [https://egolife-ai.github.io/Ego-R1/](https://egolife-ai.github.io/Ego-R1/)
    *   citation: 14

*   **[Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis](http://arxiv.org/abs/2506.00667v1)** (2025.05)
    *   Focus: Scene segmentation and keyframe extraction are vital for video understanding tasks.
    *   citation: 1

*   **[SiLVR: A Simple Language-based Video Reasoning Framework](http://arxiv.org/abs/2505.24869v1)** (2025.05)
    *   Focus: Test-time optimization improves LLM reasoning but faces challenges with long video understanding.
    *   code: [https://github.com/CeeZh/SILVR](https://github.com/CeeZh/SILVR)
    *   citation: 5

*   **[Threading Keyframe with Narratives: MLLMs as Strong Long Video Comprehenders](http://arxiv.org/abs/2505.24158v1)** (2025.05)
    *   Focus: MLLMs struggle with long videos due to high frame counts and limited context windows.
    *   citation: 3

*   **[BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding](http://arxiv.org/abs/2503.21483v1)** (CVPR2025 2025.03)
    *   Focus: Large video-language models struggle with long-form video analysis due to limited context constraints.
    *   code: [https://github.com/sming256/BOLT](https://github.com/sming256/BOLT)
    *   citation: 18

*   **[From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment](http://arxiv.org/abs/2503.20472v1)** (ICCV2025 2025.03)
    *   Focus: MLLMs struggle with long videos due to limited input capacity.
    *   citation: 5

*   **[Self-ReS: Self-Reflection in Large Vision-Language Models for Long Video Understanding](http://arxiv.org/abs/2503.20362v2)** (2025.03)
    *   Focus: LVLMs excel in short videos but struggle with long videos due to linear frame sampling limitations.
    *   citation: 1

*   **[Generative Frame Sampler for Long Video Understanding](http://arxiv.org/abs/2503.09146v2)** (2025.03)
    *   Focus: VideoLLMs struggle to understand long videos with thousands of frames.
    *   code: [https://github.com/yaolinli/GenS](https://github.com/yaolinli/GenS)
    *   citation: 12

*   **[DrVideo: Document Retrieval Based Long Video Understanding](http://arxiv.org/abs/2406.12846v2)** (CVPR2025 2024.06)
    *   Focus: Existing video understanding methods are limited to short videos, lacking techniques for long videos.
    *   citation: 31

*   **[Adaptive Keyframe Sampling for Long Video Understanding](http://arxiv.org/abs/2502.21271v1)** (CVPR2025 2025.02)
    *   Focus: MLLMs face computational challenges with long videos due to excessive visual tokens.
    *   code: [https://github.com/ncTimTang/AKS](https://github.com/ncTimTang/AKS)
    *   citation: 47

*   **[CoS: Chain-of-Shot Prompting for Long Video Understanding](http://arxiv.org/abs/2502.06428v2)** (2025.02)
    *   Focus: MLLMs face context length limits from excessive visual tokens in long videos.
    *   project: [https://lwpyh.github.io/CoS](https://lwpyh.github.io/CoS)
    *   citation: 17
    
*   **[MaxInfo: A Training-Free Key-Frame Selection Method Using Maximum Volume for Enhanced Video Understanding](http://arxiv.org/abs/2502.03183v2)** (2025.02)
    *   Focus: Uniform frame sampling in VLLMs misses critical video information due to redundancy and inefficiency.
    *   citation: 3


*   **[$\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation](http://arxiv.org/abs/2501.19098v2)** (2025.01)
    *   Focus: This paper introduces a new method for long-video understanding using compressed representations and temporal modeling.
    *   citation: 8

*   **[VCA: Video Curious Agent for Long Video Understanding](http://arxiv.org/abs/2412.10471v2)** (ICCV2025 2024.12)
    *   Focus: Recent methods sample many frames or use auxiliary tools for long video understanding.
    *   citation: 21

*   **[Towards Neuro-Symbolic Video Understanding](http://arxiv.org/abs/2403.11021v3)** (ECCV2024 2024.03)
    *   Focus: Efficient frame extraction methods are needed for long-term temporal reasoning in videos.
    *   citation: 19

*   **[VideoAgent: Long-form Video Understanding with Large Language Model as Agent](http://arxiv.org/abs/2403.10517v1)** (ECCV2024 2024.03)
    *   Focus: Long-form video understanding requires models that reason over long multi-modal sequences, inspired by human cognition.
    *   citation: 204

*   **[LLMs Meet Long Video: Advancing Long Video Question Answering with An Interactive Visual Adapter in LLMs](http://arxiv.org/abs/2402.13546v2)** (2024.02)
    *   Focus: LLMs face challenges in long video understanding due to computational constraints.
    *   citation: 4


### Compression methods
#### New LLM Architectures 
e.g., Mamba, linear attention

*   **[TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding](http://arxiv.org/abs/2511.16595v2)** (2025.11)
    *   Focus: TimeViper is a hybrid vision-language model for efficient long video understanding.
    *   code: [https://github.com/xiaomi-research/timeviper](https://github.com/xiaomi-research/timeviper)
    *   citation: 0

*   **[StretchySnake: Flexible SSM Training Unlocks Action Recognition Across Spatio-Temporal Scales](http://arxiv.org/abs/2510.16209v1)** (2025.10)
    *   Focus: State space models offer linear complexity and recurrence for efficient long-range modeling.
    *   citation: 0

*   **[AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding](http://arxiv.org/abs/2507.02591v3)** (ICCV2025 2025.07)
    *   Focus: Long video understanding faces high computational and memory costs from quadratic scaling in transformers.
    *   citation: 3

*   **[Video RWKV:Video Action Recognition Based RWKV](http://arxiv.org/abs/2411.05636v1)** (2024.11)
    *   Focus: RWKV architecture introduced for efficient long-range video understanding.
    *   citation: 3

*   **[LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture](http://arxiv.org/abs/2409.02889v3)** (2024.09)
    *   Focus: Systematic approaches are needed to expand MLLMs' long-context capabilities for video understanding and high-resolution image analysis.
    *   citation: 82

*   **[VideoMamba: Spatio-Temporal Selective State Space Model](http://arxiv.org/abs/2407.08476v1)** (ECCV2024 2024.07)
    *   Focus: VideoMamba adapts the Mamba architecture for efficient video recognition without self-attention.
    *   code: [http://github.com/jinyjelly/VideoMamba](http://github.com/jinyjelly/VideoMamba)
    *   citation: 22

*   **[Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding](http://arxiv.org/abs/2403.09626v1)** (2024.03)
    *   Focus: Video understanding research explores architectures like RNN, 3D CNN, and Transformers.
    *   code: [https://github.com/OpenGVLab/video-mamba-suite](https://github.com/OpenGVLab/video-mamba-suite)
    *   citation: 116

*   **[VideoMamba: State Space Model for Efficient Video Understanding](http://arxiv.org/abs/2403.06977v2)** (ECCV2024 2024.03)
    *   Focus: VideoMamba adapts Mamba to video to address redundancy and global dependencies.
    *   code: [https://github.com/OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba)
    *   citation: 355

*   **[World Model on Million-Length Video And Language With Blockwise RingAttention](http://arxiv.org/abs/2402.08268v4)** (ICLR2025 2024.02)
    *   Focus: Long-context understanding is a key challenge for scaling sequence models in AI.
    *   citation: 128

*   **[Selective Structured State-Spaces for Long-Form Video Understanding](http://arxiv.org/abs/2303.14526v1)** (CVPR2023 2023.03)
    *   Focus: S4 model's linear complexity addresses spatiotemporal dependencies in long videos.
    *   citation: 152

*   **[Multimodal Instruction Tuning with Hybrid State Space Models](http://arxiv.org/abs/2411.08840v1)** (2024.11)
    *   Focus: MLLMs need long context handling for high-resolution images and long videos.
    *   citation: 0

*   **[MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](http://arxiv.org/abs/2504.16083v2)** (2025.04)
    *   Focus: Long-context VLMs face quadratic attention complexity in pre-filling, limiting efficiency.
    *   citation: 16

*   **[MambaMia: A State-Space-Model-Based Compression for Efficient Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2506.13564v1)** (2025.06)
    *   Focus: A framework compresses video-frame features to reduce token explosion in long videos.
    *   citation: 1

#### Token Compression

*   **[LongVLM: Efficient Long Video Understanding via Large Language Models](http://arxiv.org/abs/2404.03384v3)** (ECCV2024 2024.04)
    *   Focus: VideoLLMs use LLMs for video understanding by encoding video representations.
    *   code: [https://github.com/ziplab/LongVLM](https://github.com/ziplab/LongVLM)
    *   citation: 114
    *   code: https://github.com/ziplab/LongVLM

*   **[MM-Ego: Towards Building Egocentric Multimodal LLMs for Video QA](http://arxiv.org/abs/2410.07177v2)** (ICLR2025 2024.10)
    *   Focus: This research builds a multimodal foundation model for egocentric video understanding.
    *   citation: 17

*   **[Apollo: An Exploration of Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2412.10360v1)** (CVPR2025 2024.12)
    *   Focus: This paper investigates the mechanisms behind video understanding in large multimodal models.
    *   project: [https://apollo-lmms.github.io](https://apollo-lmms.github.io)
    *   citation: 51

*   **[D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition](http://arxiv.org/abs/2510.08818v1)** (2025.10)
    *   Focus: Vid-LLMs can be built by adapting image-pretrained VLMs, but face challenges with video-specific tasks.
    *   code: [https://github.com/hukcc/D-CoDe](https://github.com/hukcc/D-CoDe)
    *   citation: 0

*   **[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](http://arxiv.org/abs/2511.00141v1)** (2025.10)
    *   Focus: Video-LMMs use advanced visual-language reasoning for long video understanding.
    *   citation: 0

*   **[Unleashing Hour-Scale Video Training for Long Video-Language Understanding](http://arxiv.org/abs/2506.05332v1)** (NeurIPS2025 2025.06)
    *   Focus: Long video understanding benchmarks advance Video-LMMs, but scarce annotated data limits training.
    *   project: [https://videomarathon.github.io/](https://videomarathon.github.io/)
    *   citation: 9

*   **[Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation](http://arxiv.org/abs/2511.20714v1)** (2025.11)
    *   Focus: World models simulate realistic, interactive long videos for AI agents and gaming.
    *   citation: 1

*   **[EventSTU: Event-Guided Efficient Spatio-Temporal Understanding for Video Large Language Models](http://arxiv.org/abs/2511.18920v1)** (2025.11)
    *   Focus: Proposes event-based token compression to reduce inference costs in long video understanding.
    *   citation: 0

*   **[VideoPerceiver: Enhancing Fine-Grained Temporal Perception in Video Multimodal Large Language Models](http://arxiv.org/abs/2511.18823v1)** (2025.11)
    *   Focus: VideoPerceiver improves fine-grained perception in video understanding by enhancing reasoning about brief events.
    *   citation: 0

*   **[Test-Time Temporal Sampling for Efficient MLLM Video Understanding](http://arxiv.org/abs/2511.17945v1)** (2025.11)
    *   Focus: MLLMs face computational challenges in long video processing due to quadratic self-attention scaling.
    *   code: [https://github.com/kaibinwang3/T3S](https://github.com/kaibinwang3/T3S)
    *   citation: 0

*   **[CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding](http://arxiv.org/abs/2511.13644v1)** (2025.11)
    *   Focus: Long-form video QA challenges VLMs due to growing attention and KV caches, requiring costly inference.
    *   citation: 0

*   **[Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models](http://arxiv.org/abs/2511.11910v2)** (2025.11)
    *   Focus: Long video understanding is challenging for MLLMs due to high video token counts.
    *   citation: 0

*   **[MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](http://arxiv.org/abs/2307.16449v4)** (CVPR2024 2023.07)
    *   Focus: Video foundation models and LLMs are integrated to overcome task-specific limitations in video understanding.
    *   project: [https://rese1f.github.io/MovieChat/](https://rese1f.github.io/MovieChat/)
    *   citation: 429

*   **[SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models](http://arxiv.org/abs/2407.15841v2)** (2024.07)
    *   Focus: SF-LLaVA is a training-free video LLM capturing spatial details and long-range temporal context.
    *   code: [https://github.com/apple/ml-slowfast-llava](https://github.com/apple/ml-slowfast-llava)
    *   citation: 89

*   **[Long Video Understanding with Learnable Retrieval in Video-Language Models](http://arxiv.org/abs/2312.04931v3)** (2023.12)
    *   Focus: LLMs are applied to video understanding for their language and reasoning capabilities.
    *   citation: 3

*   **[Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](http://arxiv.org/abs/2510.17364v1)** (NeurIPS2025 2025.10)
    *   Focus: Video-LLMs struggle with streaming video understanding due to limited access to full video content.
    *   citation: 0

*   **[Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference](http://arxiv.org/abs/2510.14624v1)** (2025.10)
    *   Focus: Video VLMs face scalability issues due to quadratic costs of dense frame processing.
    *   citation: 1

*   **[MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](http://arxiv.org/abs/2510.07915v1)** (2025.10)
    *   Focus: Visual language models face high computational costs when extended from images.
    *   citation: 1

*   **[Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](http://arxiv.org/abs/2510.05836v1)** (ICCV2025 2025.10)
    *   Focus: Long video understanding is challenged by redundancy and limited context.
    *   citation: 3

*   **[VideoNSA: Native Sparse Attention Scales Video Understanding](http://arxiv.org/abs/2510.02295v1)** (2025.10)
    *   Focus: Addresses video understanding limitations in multimodal models due to short context lengths.
    *   code: [https://github.com/Espere-1119-Song/VideoNSA](https://github.com/Espere-1119-Song/VideoNSA)
    *   citation: 1

*   **[StreamForest: Efficient Online Video Understanding with Persistent Event Memory](http://arxiv.org/abs/2509.24871v1)** (NeurIPS2025 2025.09)
    *   Focus: MLLMs struggle with real-time video streaming due to storage constraints.
    *   citation: 2

*   **[Video Panels for Long Video Understanding](http://arxiv.org/abs/2509.23724v1)** (2025.09)
    *   Focus: Video-language models lag behind on long-video tasks compared to images and short videos.
    *   citation: 0

*   **[Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding](http://arxiv.org/abs/2509.23672v1)** (2025.09)
    *   Focus: Vision Transformers excel in surgical video tasks but face high computational costs.
    *   citation: 0

*   **[Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](http://arxiv.org/abs/2509.01552v1)** (2025.09)
    *   Focus: LVLMs show strong multimodal understanding but face challenges with high-resolution images and long videos.
    *   code: [https://github.com/xuyang-liu16/V2Drop](https://github.com/xuyang-liu16/V2Drop)
    *   citation: 3

*   **[Language-Guided Temporal Token Pruning for Efficient VideoLLM Processing](http://arxiv.org/abs/2508.17686v1)** (2025.08)
    *   Focus: LGTTP uses language guidance to prune temporal tokens, reducing attention complexity for long videos.
    *   citation: 0

*   **[StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding](http://arxiv.org/abs/2508.15717v1)** (2025.08)
    *   Focus: MLLMs struggle with long video processing despite recent efficiency improvements.
    *   citation: 5


*   **[Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](http://arxiv.org/abs/2508.02134v1)** (ICCV2025 2025.08)
    *   Focus: Video-MLLMs face context length limits, hindering long video understanding.
    *   code: [https://github.com/wkfdb/Free-MoRef](https://github.com/wkfdb/Free-MoRef)
    *   citation: 1

*   **[Infinite Video Understanding](http://arxiv.org/abs/2507.09068v2)** (2025.07)
    *   Focus: LLMs and MLLMs advance video understanding but face efficiency challenges with long videos.
    *   citation: 1


*   **[AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding](http://arxiv.org/abs/2507.02591v3)** (ICCV2025 2025.07)
    *   Focus: Long video understanding faces high computational and memory costs from quadratic scaling in transformers.
    *   citation: 3

*   **[LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs](http://arxiv.org/abs/2506.21862v1)** (2025.06)
    *   Focus: LLaVA-Scissor is a training-free token compression method for video multimodal LLMs.
    *   code: [https://github.com/HumanMLLM/LLaVA-Scissor](https://github.com/HumanMLLM/LLaVA-Scissor)
    *   citation: 5

*   **[Task-Aware KV Compression For Cost-Effective Long Video Understanding](http://arxiv.org/abs/2506.21184v1)** (2025.06)
    *   Focus: KV compression methods address computational costs in long-video understanding for MLLMs.
    *   citation: 1

*   **[PEVLM: Parallel Encoding for Vision-Language Models](http://arxiv.org/abs/2506.19651v3)** (2025.06)
    *   Focus: Vision-language models struggle with long videos due to high computational demands.
    *   citation: 0

*   **[Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification](http://arxiv.org/abs/2506.19225v1)** (2025.06)
    *   Focus: MLLMs struggle with long video processing due to high computational demands.
    *   citation: 12

*   **[InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](http://arxiv.org/abs/2506.15745v2)** (NeurIPS2025 2025.06)
    *   Focus: MLLMs' KV cache grows linearly with video length, exceeding device memory limits.
    *   citation: 6

*   **[Memory Consolidation Enables Long-Context Video Understanding](http://arxiv.org/abs/2402.05861v2)** (2024.02)
    *   Focus: Transformer video encoders struggle with long contexts due to quadratic complexity, despite extension attempts.
    *   citation: 44

*   **[CyberV: Cybernetics for Test-time Scaling in Video Understanding](http://arxiv.org/abs/2506.07971v1)** (2025.06)
    *   Focus: MLLMs face challenges with long videos due to high computation, low robustness, and limited accuracy.
    *   code: [https://github.com/marinero4972/CyberV](https://github.com/marinero4972/CyberV)
    *   citation: 1

*   **[APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](http://arxiv.org/abs/2506.04953v3)** (2025.06)
    *   Focus: MLLMs face challenges in modeling hour-level videos due to high information volume.
    *   citation: 1

*   **[DynTok: Dynamic Compression of Visual Tokens for Efficient and Effective Video Understanding](http://arxiv.org/abs/2506.03990v1)** (2025.06)
    *   Focus: Video modeling methods use visual tokens for LLM processing, but face challenges with long videos.
    *   citation: 2

*   **[METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding](http://arxiv.org/abs/2506.02850v2)** (2025.06)
    *   Focus: VLLMs struggle with long videos due to high computational demands.
    *   citation: 1

*   **[FlexSelect: Flexible Token Selection for Efficient Long Video Understanding](http://arxiv.org/abs/2506.00993v1)** (2025.06)
    *   Focus: FlexSelect reduces computational demands for long video understanding in VideoLLMs.
    *   project: [https://yunzhuzhang0918.github.io/flex_select](https://yunzhuzhang0918.github.io/flex_select)
    *   citation: 4

*   **[Clapper: Compact Learning and Video Representation in VLMs](http://arxiv.org/abs/2505.15529v1)** (2025.05)
    *   Focus: Vision-language models need effective temporal modeling for video understanding.
    *   citation: 0

*   **[RAVU: Retrieval Augmented Video Understanding with Compositional Reasoning over Graph](http://arxiv.org/abs/2505.03173v1)** (2025.05)
    *   Focus: LMMs struggle with long videos due to limited memory and processing constraints.
    *   citation: 1

*   **[FiLA-Video: Spatio-Temporal Compression for Fine-Grained Long Video Understanding](http://arxiv.org/abs/2504.20384v1)** (2025.04)
    *   Focus: Video understanding in VLLMs has advanced but faces challenges with data complexity and context processing.
    *   citation: 5

*   **[MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](http://arxiv.org/abs/2504.16083v2)** (2025.04)
    *   Focus: Long-context VLMs face quadratic attention complexity in pre-filling, limiting efficiency.
    *   citation: 16

*   **[Multimodal Long Video Modeling Based on Temporal Dynamic Context](http://arxiv.org/abs/2504.10443v1)** (2025.04)
    *   Focus: LLMs advance video understanding but struggle with long video context length.
    *   code: [https://github.com/Hoar012/TDC-Video](https://github.com/Hoar012/TDC-Video)
    *   citation: 0

*   **[Mavors: Multi-granularity Video Representation for Multimodal Large Language Model](http://arxiv.org/abs/2504.10068v1)** (2025.04)
    *   Focus: MLLMs struggle to balance computational efficiency with fine-grained spatio-temporal pattern retention in long videos.
    *   citation: 9

*   **[LVC: A Lightweight Compression Framework for Enhancing VLMs in Long Video Understanding](http://arxiv.org/abs/2504.06835v1)** (2025.04)
    *   Focus: VLMs achieve frame-level understanding but struggle with long video comprehension.
    *   citation: 3

*   **[Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation](http://arxiv.org/abs/2504.02438v5)** (2025.04)
    *   Focus: Token pruning and feature merging address computational costs in long video processing.
    *   code: [https://github.com/steven-ccq/ViLAMP](https://github.com/steven-ccq/ViLAMP)
    *   citation: 16

*   **[SlowFast-LLaVA-1.5: A Family of Token-Efficient Video Large Language Models for Long-Form Video Understanding](http://arxiv.org/abs/2503.18943v2)** (2025.03)
    *   Focus: SF-LLaVA-1.5 is a token-efficient video LLM family for long video understanding.
    *   citation: 12

*   **[Video-XL-Pro: Reconstructive Token Compression for Extremely Long Video Understanding](http://arxiv.org/abs/2503.18478v2)** (2025.03)
    *   Focus: Video-XL-Pro is an efficient MLLM for long video understanding.
    *   citation: 29

*   **[XAttention: Block Sparse Attention with Antidiagonal Scoring](http://arxiv.org/abs/2503.16428v1)** (2025.03)
    *   Focus: Block-sparse attention reduces computational costs in long-context transformers.
    *   code: [https://github.com/mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention)
    *   citation: 48

*   **[Long-VMNet: Accelerating Long-Form Video Understanding via Fixed Memory](http://arxiv.org/abs/2503.13707v1)** (2025.03)
    *   Focus: Long-form video understanding is essential but computationally intensive for traditional methods.
    *   citation: 1

*   **[Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding](http://arxiv.org/abs/2503.13139v2)** (NeurIPS2025 2025.03)
    *   Focus: Current long video understanding methods neglect logical relations in dense captions and feature selection.
    *   citation: 15

*   **[Efficient Motion-Aware Video MLLM](http://arxiv.org/abs/2503.13016v1)** (CVPR2025 2025.03)
    *   Focus: EMA addresses inefficient video processing and motion awareness in MLLMs.
    *   citation: 3

*   **[Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](http://arxiv.org/abs/2503.11579v2)** (ICCV2025 2025.03)
    *   Focus: Transformers struggle with long videos due to quadratic attention complexity and high computational costs.
    *   project: [https://tiger-ai-lab.github.io/Vamba/](https://tiger-ai-lab.github.io/Vamba/)
    *   citation: 17

*   **[FastVID: Dynamic Density Pruning for Fast Video Large Language Models](http://arxiv.org/abs/2503.11187v2)** (NeurIPS2025 2025.03)
    *   Focus: Video LLMs have strong understanding but high inference costs from redundant tokens.
    *   code: [https://github.com/LunarShen/FastVID](https://github.com/LunarShen/FastVID)
    *   citation: 10

*   **[Keyframe-oriented Vision Token Pruning: Enhancing Efficiency of Large Vision Language Models on Long-Form Video Processing](http://arxiv.org/abs/2503.10742v2)** (ICCV2025 2025.03)
    *   Focus: Vision language models face high computational costs from redundant visual data.
    *   citation: 4

*   **[VideoScan: Enabling Efficient Streaming Video Understanding via Frame-level Semantic Carriers](http://arxiv.org/abs/2503.09387v2)** (2025.03)
    *   Focus: VideoScan enables real-time video interaction with efficient VLM inference for streamed video comprehension.
    *   citation: 3

*   **[Memory-enhanced Retrieval Augmentation for Long Video Understanding](http://arxiv.org/abs/2503.09149v2)** (2025.03)
    *   Focus: Long-video understanding faces challenges from compression and brute-force methods in current models.
    *   citation: 9

*   **[QuoTA: Query-oriented Token Assignment via CoT Query Decouple for Long Video Comprehension](http://arxiv.org/abs/2503.08689v1)** (2025.03)
    *   Focus: Critiques attention-based pruning for long videos and proposes a new compression method.
    *   code: [https://github.com/MAC-AutoML/QuoTA](https://github.com/MAC-AutoML/QuoTA)
    *   citation: 7

*   **[HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](http://arxiv.org/abs/2503.08585v2)** (CVPR2025 2025.03)
    *   Focus: MLLMs struggle with long videos due to frame and context limitations.
    *   citation: 11

*   **[STORM: Token-Efficient Long Video Understanding for Multimodal LLMs](http://arxiv.org/abs/2503.04130v4)** (2025.03)
    *   Focus: Video-LLMs process videos as image sequences but face efficiency and context length challenges.
    *   citation: 12

*   **[iMOVE: Instance-Motion-Aware Video Understanding](http://arxiv.org/abs/2502.11594v2)** (2025.02)
    *   Focus: Improving Video LLMs' fine-grained motion perception for better temporal understanding.
    *   citation: 8

*   **[LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding](http://arxiv.org/abs/2501.05067v2)** (2025.01)
    *   Focus: LLaVA-Octopus adaptively weights visual features for video understanding based on user instructions.
    *   citation: 8

*   **[VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling](http://arxiv.org/abs/2501.00574v4)** (2024.12)
    *   Focus: Long-context video modeling is essential for MLLMs but remains challenging.
    *   citation: 94

*   **[FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models](http://arxiv.org/abs/2501.01986v2)** (ICCV2025 2024.12)
    *   Focus: Existing token reduction methods for long videos are reviewed and a new efficient compression approach is proposed.
    *   code: [https://github.com/thu-nics/FrameFusion](https://github.com/thu-nics/FrameFusion)
    *   citation: 20

*   **[ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding](http://arxiv.org/abs/2412.20504v5)** (2024.12)
    *   Focus: VideoLLMs struggle with long videos due to LLM limitations; new compression methods are proposed.
    *   code: [https://github.com/SCZwangxiao/video-ReTaKe](https://github.com/SCZwangxiao/video-ReTaKe)
    *   citation: 23

*   **[B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](http://arxiv.org/abs/2412.09919v2)** (ICCV2025 2024.12)
    *   Focus: Vision LLMs encode visual content into sequences for understanding.
    *   code: [https://github.com/zhuqiangLu/B-VLLM](https://github.com/zhuqiangLu/B-VLLM)
    *   citation: 5

*   **[IQViC: In-context, Question Adaptive Vision Compressor for Long-term Video Understanding LMMs](http://arxiv.org/abs/2412.09907v2)** (2024.12)
    *   Focus: Existing methods struggle with accurate long-term temporal understanding in complex videos.
    *   citation: 1

*   **[PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models](http://arxiv.org/abs/2412.09613v1)** (CVPR2025 2024.12)
    *   Focus: VLMs use visual token compression to handle long video inputs efficiently.
    *   citation: 7

*   **[Espresso: High Compression For Rich Extraction From Videos for Your Vision-Language Model](http://arxiv.org/abs/2412.04729v3)** (2024.12)
    *   Focus: Vision-language models struggle with long videos due to token growth.
    *   citation: 2

*   **[AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](http://arxiv.org/abs/2412.03248v2)** (ICCV2025 2024.12)
    *   Focus: Multi-modal LLMs show strong video understanding but require extensive visual token compression.
    *   code: [https://github.com/LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM)
    *   citation: 17

*   **[SEAL: Semantic Attention Learning for Long Video Representation](http://arxiv.org/abs/2412.01798v3)** (CVPR2025 2024.12)
    *   Focus: Long video understanding requires efficient representations to reduce computational complexity and temporal redundancy.
    *   citation: 7

*   **[Look Every Frame All at Once: Video-Ma$^2$mba for Efficient Long-form Video Understanding with Multi-Axis Gradient Checkpointing](http://arxiv.org/abs/2411.19460v1)** (2024.11)
    *   Focus: Long video processing faces high computational costs from quadratic memory and time demands.
    *   project: [https://ivy-lvlm.github.io/Video-MA2MBA/](https://ivy-lvlm.github.io/Video-MA2MBA/)
    *   citation: 2

*   **[SAVEn-Vid: Synergistic Audio-Visual Integration for Enhanced Understanding in Long Video Context](http://arxiv.org/abs/2411.16213v2)** (2024.11)
    *   Focus: Video-LLMs struggle with long video understanding despite recent advances.
    *   project: [https://ljungang.github.io/SAVEn-Vid/](https://ljungang.github.io/SAVEn-Vid/)
    *   citation: 4

*   **[SALOVA: Segment-Augmented Long Video Assistant for Targeted Retrieval and Routing in Long-Form Video Analysis](http://arxiv.org/abs/2411.16173v2)** (CVPR2025 2024.11)
    *   Focus: LMMs struggle with long videos due to context length limits and high memory usage.
    *   project: [https://ivy-lvlm.github.io/SALOVA/](https://ivy-lvlm.github.io/SALOVA/)
    *   citation: 5

*   **[ReWind: Understanding Long Videos with Instructed Learnable Memory](http://arxiv.org/abs/2411.15556v2)** (CVPR2025 2024.11)
    *   Focus: Vision-language models face computational inefficiency challenges when processing long videos.
    *   citation: 4

*   **[AdaCM$^2$: On Understanding Extremely Long-Term Video with Adaptive Cross-Modality Memory Reduction](http://arxiv.org/abs/2411.12593v3)** (CVPR2025 2024.11)
    *   Focus: LLM-based video models struggle with long videos due to high computational costs and limited context length.
    *   citation: 3

*   **[DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](http://arxiv.org/abs/2411.12355v2)** (CVPR2025 2024.11)
    *   Focus: LLM-based video understanding struggles with preserving information in long videos while managing token count.
    *   citation: 5

*   **[PPLLaVA: Varied Video Sequence Understanding With Prompt Guidance](http://arxiv.org/abs/2411.02327v2)** (2024.11)
    *   Focus: Video LLMs advance but struggle with unified short and long video understanding.
    *   code: [https://github.com/farewellthree/PPLLaVA](https://github.com/farewellthree/PPLLaVA)
    *   citation: 16

*   **[Video Token Merging for Long-form Video Understanding](http://arxiv.org/abs/2410.23782v1)** (2024.10)
    *   Focus: Transformer models face challenges with long video inputs, requiring alternatives to sampling.
    *   citation: 12

*   **[LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding](http://arxiv.org/abs/2410.17434v1)** (2024.10)
    *   Focus: MLLMs struggle with long video processing due to LLM context limits.
    *   project: [https://vision-cair.github.io/LongVU](https://vision-cair.github.io/LongVU)
    *   citation: 150

*   **[VidCompress: Memory-Enhanced Temporal Compression for Video Understanding in Large Language Models](http://arxiv.org/abs/2410.11417v1)** (2024.10)
    *   Focus: Video-LLMs treat videos as frame sequences, missing temporal dynamics.
    *   citation: 4

*   **[Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos](http://arxiv.org/abs/2410.02763v1)** (2024.10)
    *   Focus: Research shifts focus to long video understanding as short video challenges are considered solved.
    *   project: [https://vinoground.github.io](https://vinoground.github.io)
    *   citation: 15

*   **[Learning to Localize Actions in Instructional Videos with LLM-Based Multi-Pathway Text-Video Alignment](http://arxiv.org/abs/2409.16145v1)** (ECCV2024 2024.09)
    *   Focus: Proposes a method to localize steps in instructional videos using limited annotations.
    *   citation: 5

*   **[Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](http://arxiv.org/abs/2409.14485v4)** (CVPR2025 2024.09)
    *   Focus: MLLMs struggle with long videos due to limited context length and high computational costs.
    *   citation: 123

*   **[Enhancing Long Video Understanding via Hierarchical Event-Based Memory](http://arxiv.org/abs/2409.06299v1)** (2024.09)
    *   Focus: Video understanding systems integrate visual models with LLMs, often compressing diverse video data.
    *   citation: 11

*   **[LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture](http://arxiv.org/abs/2409.02889v3)** (2024.09)
    *   Focus: Systematic approaches are needed to expand MLLMs' long-context capabilities for video understanding and high-resolution image analysis.
    *   citation: 82

*   **[VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](http://arxiv.org/abs/2409.01071v2)** (ICCV2025 2024.09)
    *   Focus: Large video-language models face computational and data scarcity challenges for real-time planning.
    *   citation: 2

*   **[HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](http://arxiv.org/abs/2408.17443v4)** (ICCV2025 2024.08)
    *   Focus: Long-form video understanding faces challenges in capturing long-range dependencies and processing redundant information.
    *   project: [https://joslefaure.github.io/assets/html/hermes.html](https://joslefaure.github.io/assets/html/hermes.html)
    *   citation: 6

*   **[VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation](http://arxiv.org/abs/2408.16730v1)** (NeurIPS2024 2024.08)
    *   Focus: Increasing vision tokens improves understanding but raises memory costs in large vision-language models.
    *   citation: 26

*   **[Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input](http://arxiv.org/abs/2408.15542v1)** (2024.08)
    *   Focus: Extending LLMs to handle video input remains a challenging research area.
    *   citation: 99

*   **[Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](http://arxiv.org/abs/2407.12679v1)** (ECCV2024 2024.07)
    *   Focus: LLM-based video models struggle with long videos due to noise, redundancy, and memory constraints.
    *   project: [https://vision-cair.github.io/Goldfish_website/](https://vision-cair.github.io/Goldfish_website/)
    *   citation: 31

*   **[MovieChat+: Question-aware Sparse Memory for Long Video Question Answering](http://arxiv.org/abs/2404.17176v1)** (2024.04)
    *   Focus: Video foundation models and LLMs overcome task limitations but face efficiency challenges.
    *   code: [https://github.com/rese1f/MovieChat](https://github.com/rese1f/MovieChat)
    *   citation: 47

*   **[MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](http://arxiv.org/abs/2404.05726v2)** (CVPR2024 2024.04)
    *   Focus: Vision-language models need better long video understanding, which current LLM-based methods struggle with.
    *   project: [https://boheumd.github.io/MA-LMM/](https://boheumd.github.io/MA-LMM/)
    *   citation: 160

*   **[Text-Conditioned Resampler For Long Form Video Understanding](http://arxiv.org/abs/2312.11897v3)** (ECCV2024 2023.12)
    *   Focus: A text-conditioned video resampler uses frozen visual and language models to process long videos.
    *   citation: 21

*   **[TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding](http://arxiv.org/abs/2312.02051v2)** (CVPR2024 2023.12)
    *   Focus: TimeChat is a time-sensitive MLLM for long video understanding with timestamp-aware frame tokenization and temporal attention.
    *   code: [https://github.com/RenShuhuai-Andy/TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
    *   citation: 326

*   **[LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](http://arxiv.org/abs/2311.17043v1)** (ECCV2024 2023.11)
    *   Focus: LLaMA-VID reduces tokens for efficient video/image understanding in Vision Language Models.
    *   code: [https://github.com/dvlab-research/LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
    *   citation: 450

*   **[TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding](http://arxiv.org/abs/2310.19060v1)** (2023.10)
    *   Focus: Video-language pre-training advances understanding but faces high computational costs from video encoding.
    *   code: [https://github.com/RenShuhuai-Andy/TESTA](https://github.com/RenShuhuai-Andy/TESTA)
    *   citation: 39

*   **[Query-aware Long Video Localization and Relation Discrimination for Deep Video Understanding](http://arxiv.org/abs/2310.12724v1)** (2023.10)
    *   Focus: Existing video understanding techniques excel with short formats but face challenges with long videos.
    *   citation: 2

*   **[From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding](http://arxiv.org/abs/2409.18938v2)** (2024.09)
    *   Focus: LLMs combined with visual encoders improve visual understanding tasks.
    *   citation: 17


### Temporal Modeling (timestamp / time positional encoding)

*   **[HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](http://arxiv.org/abs/2505.20444v2)** (NeurIPS2025 2025.05)
    *   Focus: VLMs struggle with long videos due to limited context windows, requiring new architectures for long-range dependencies.
    *   code: [https://github.com/hrlics/HoPE](https://github.com/hrlics/HoPE)
    *   citation: 2

*   **[VideoRoPE: What Makes for Good Video Rotary Position Embedding?](http://arxiv.org/abs/2502.05173v3)** (2025.02)
    *   Focus: Extending 1D RoPE to video remains challenging due to complex spatio-temporal structure.
    *   code: [https://github.com/Wiselnn570/VideoRoPE](https://github.com/Wiselnn570/VideoRoPE)
    *   citation: 28


### Downstream tasks 
#### Real-time Interaction

*   **[AHA -- Predicting What Matters Next: Online Highlight Detection Without Looking Ahead](http://arxiv.org/abs/2509.16421v2)** (NeurIPS2025 2025.09)
    *   Focus: Real-time video stream understanding is critical for autonomous vehicles, drones, and disaster response agents.
    *   citation: 0

*   **[StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding](http://arxiv.org/abs/2508.01875v3)** (2025.08)
    *   Focus: Real-time video streaming for autonomous driving and surveillance requires continuous perception beyond offline methods.
    *   citation: 1

*   **[TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos](http://arxiv.org/abs/2504.17343v1)** (2025.04)
    *   Focus: Real-time video understanding is needed for live streaming services.
    *   citation: 11

*   **[Streaming Long Video Understanding with Large Language Models](http://arxiv.org/abs/2405.16009v1)** (NeurIPS2024 2024.05)
    *   Focus: VideoStreaming is a VLLM that processes arbitrary-length videos using a constant number of tokens.
    *   citation: 102

*   **[Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge](http://arxiv.org/abs/2501.13468v1)** (ICLR2025 2025.01)
    *   Focus: Video-LLMs advance multimodal learning but struggle with long video understanding.
    *   code: [https://github.com/hmxiong/StreamChat](https://github.com/hmxiong/StreamChat)
    *   citation: 19

*   **[Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding](http://arxiv.org/abs/2504.13915v1)** (2025.04)
    *   Focus: ProVideLLM is an end-to-end framework for real-time procedural video understanding with multimodal caching.
    *   project: [https://dibschat.github.io/ProVideLLM](https://dibschat.github.io/ProVideLLM)
    *   citation: 2

*   **[LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval](http://arxiv.org/abs/2505.15269v1)** (2025.05)
    *   Focus: Video LLMs excel at long videos but lack benchmarks for temporal understanding.
    *   citation: 7

*   **[StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant](http://arxiv.org/abs/2505.05467v2)** (NeurIPS2025 2025.05)
    *   Focus: StreamBridge transforms offline Video-LLMs into streaming-capable models.
    *   citation: 4

*   **[video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory](http://arxiv.org/abs/2510.11129v1)** (2025.10)
    *   Focus: Proposes a scalable method for continuous high-frame-rate video processing to overcome LLM limitations.
    *   citation: 1

*   **[StreamingVLM: Real-Time Understanding for Infinite Video Streams](http://arxiv.org/abs/2510.09608v1)** (2025.10)
    *   Focus: VLMs struggle with real-time video understanding due to latency and memory constraints.
    *   code: [https://github.com/mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm)
    *   citation: 4

*   **[An Egocentric Vision-Language Model based Portable Real-time Smart Assistant](http://arxiv.org/abs/2503.04250v1)** (2025.03)
    *   Focus: Vinci is a portable AI system using EgoVideo-VL for real-time vision-language assistance.
    *   code: [https://github.com/OpenGVLab/vinci](https://github.com/OpenGVLab/vinci)
    *   citation: 6

*   **[Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction](http://arxiv.org/abs/2501.03218v1)** (CVPR2025 2025.01)
    *   Focus: Video LLMs enable real-time interaction by understanding user intent and responding during continuous video processing.
    *   code: [https://github.com/Mark12Ding/Dispider](https://github.com/Mark12Ding/Dispider)
    *   citation: 28

*   **[Memory-augmented Online Video Anomaly Detection](http://arxiv.org/abs/2302.10719v2)** (2023.02)
    *   Focus: An online system for autonomous vehicles to understand scenes and provide immediate responses.
    *   code: [https://github.com/IMPLabUniPr/movad/tree/movad_vad](https://github.com/IMPLabUniPr/movad/tree/movad_vad)
    *   citation: 8

*   **[StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling](http://arxiv.org/abs/2507.05240v1)** (2025.07)
    *   Focus: VLN agents process continuous video streams with low latency to follow language instructions.
    *   project: [https://streamvln.github.io/](https://streamvln.github.io/)
    *   citation: 21

*   **[Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams](http://arxiv.org/abs/2406.08085v2)** (2024.06)
    *   Focus: Existing video understanding methods excel offline but face challenges in real-time applications.
    *   project: [https://invinciblewyq.github.io/vstream-page/](https://invinciblewyq.github.io/vstream-page/)
    *   citation: 87

#### Dense Video Captioning

*   **[SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](http://arxiv.org/abs/2510.17777v1)** (ICCV2025 2025.10)
    *   Focus: VLMs advance visual-textual reasoning for high-res images, long videos, and multi-turn conversations.
    *   citation: 1

*   **[SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding](http://arxiv.org/abs/2510.13016v2)** (2025.10)
    *   Focus: AI systems need to understand fine-grained actions and localize actors in space and time.
    *   citation: 1

*   **[Addressing the ID-Matching Challenge in Long Video Captioning](http://arxiv.org/abs/2510.06973v1)** (2025.10)
    *   Focus: Addresses challenges in generating captions for long, complex videos for text-to-video and multi-modal applications.
    *   citation: 0

*   **[Time-Scaling State-Space Models for Dense Video Captioning](http://arxiv.org/abs/2509.03426v1)** (2025.09)
    *   Focus: Dense video captioning segments videos into events and generates detailed descriptions for each.
    *   citation: 0


*   **[From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding](http://arxiv.org/abs/2507.02790v2)** (2025.07)
    *   Focus: Efficient video editing techniques are needed to condense long videos into concise summaries.
    *   citation: 0

*   **[LongAnimation: Long Animation Generation with Dynamic Global-Local Memory](http://arxiv.org/abs/2507.01945v2)** (ICCV2025 2025.07)
    *   Focus: Automated colorization for long animation videos to reduce labor costs.
    *   project: [https://cn-makers.github.io/long_animation_web/](https://cn-makers.github.io/long_animation_web/)
    *   citation: 3

*   **[A Culturally-diverse Multilingual Multimodal Video Benchmark & Model](http://arxiv.org/abs/2506.07032v3)** (2025.06)
    *   Focus: The paper proposes a new Chinese large multimodal model for video understanding with improved efficiency.
    *   project: [https://mbzuai-oryx.github.io/ViMUL/](https://mbzuai-oryx.github.io/ViMUL/)
    *   citation: 2

*   **[QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v2)** (2025.05)
    *   Focus: Long-video understanding is crucial for real-world applications but faces challenges.
    *   citation: 2

*   **[Action Anticipation from SoccerNet Football Video Broadcasts](http://arxiv.org/abs/2504.12021v1)** (2025.04)
    *   Focus: AI enables analysis of long sports videos for action understanding and motion prediction.
    *   code: [https://github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)
    *   citation: 1

*   **[DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description](http://arxiv.org/abs/2503.24096v1)** (2025.03)
    *   Focus: Audio Description aids vision-impaired audiences by narrating key visual elements in videos.
    *   citation: 1

*   **[Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding](http://arxiv.org/abs/2503.13139v2)** (NeurIPS2025 2025.03)
    *   Focus: Current long video understanding methods neglect logical relations in dense captions and feature selection.
    *   citation: 15

*   **[Prompt2LVideos: Exploring Prompts for Understanding Long-Form Multimodal Videos](http://arxiv.org/abs/2503.08335v1)** (2025.03)
    *   Focus: Long video understanding is challenging due to reliance on manually annotated video-caption datasets.
    *   citation: 0

*   **[MANTA: Diffusion Mamba for Efficient and Effective Stochastic Long-Term Dense Anticipation](http://arxiv.org/abs/2501.08837v2)** (2025.01)
    *   Focus: Challenges in predicting future actions and durations from video observations.
    *   code: [https://github.com/olga-zats/DIFF_MANTA](https://github.com/olga-zats/DIFF_MANTA)
    *   citation: 2

*   **[Video LLMs for Temporal Reasoning in Long Videos](http://arxiv.org/abs/2412.02930v4)** (2024.12)
    *   Focus: TemporalVLM enables temporal reasoning and fine-grained understanding in long videos.
    *   citation: 5

*   **[LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos](http://arxiv.org/abs/2411.19772v3)** (CVPR2025 2024.11)
    *   Focus: Proposes a framework for fine-grained omni-modal video understanding using hierarchical alignment and contrastive learning.
    *   citation: 21

*   **[Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](http://arxiv.org/abs/2411.16932v1)** (CVPR2025 2024.11)
    *   Focus: Video LLMs need temporal awareness for tasks like dense captioning and temporal grounding.
    *   citation: 4

*   **[FIOVA: A Multi-Annotator Benchmark for Human-Aligned Video Captioning](http://arxiv.org/abs/2410.15270v2)** (2024.10)
    *   Focus: Existing video caption benchmarks inadequately assess LVLM alignment with human understanding due to single-annotation limitations.
    *   project: [https://huuuuusy.github.io/fiova/](https://huuuuusy.github.io/fiova/)
    *   citation: 3

*   **[AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark](http://arxiv.org/abs/2410.03051v4)** (ICLR2025 2024.10)
    *   Focus: This paper proposes a method for generating detailed and coherent video captions.
    *   project: [https://rese1f.github.io/aurora-web/](https://rese1f.github.io/aurora-web/)
    *   citation: 87

*   **[YouTube Video Analytics for Patient Engagement: Evidence from Colonoscopy Preparation Videos](http://arxiv.org/abs/2410.02830v1)** (2024.10)
    *   Focus: Video analysis methods for medical education content are explored.
    *   citation: 0

*   **[InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output](http://arxiv.org/abs/2407.03320v1)** (2024.07)
    *   Focus: IXC-2.5 is a versatile vision-language model for long-context text-image tasks.
    *   code: [https://github.com/InternLM/InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)
    *   citation: 166

*   **[VIA: Unified Spatiotemporal Video Adaptation Framework for Global and Local Video Editing](http://arxiv.org/abs/2406.12831v3)** (2024.06)
    *   Focus: Video editing is crucial in digital media but existing methods often neglect key requirements.
    *   citation: 1

*   **[ST-LLM: Large Language Models Are Effective Temporal Learners](http://arxiv.org/abs/2404.00308v1)** (ECCV2024 2024.03)
    *   Focus: Research explores video LLMs for human-AI interaction using text comprehension and generation.
    *   code: [https://github.com/TencentARC/ST-LLM](https://github.com/TencentARC/ST-LLM)
    *   citation: 119

*   **[Towards Multimodal Video Paragraph Captioning Models Robust to Missing Modality](http://arxiv.org/abs/2403.19221v1)** (2024.03)
    *   Focus: Video paragraph captioning models are constrained by limited data and inefficient architectures.
    *   code: [https://github.com/lancopku/MR-VPC](https://github.com/lancopku/MR-VPC)
    *   citation: 3

*   **[Panonut360: A Head and Eye Tracking Dataset for Panoramic Video](http://arxiv.org/abs/2403.17708v1)** (2024.03)
    *   Focus: VR/AR technology advances require personalized immersive panoramic video services.
    *   project: [https://dianvrlab.github.io/Panonut360/](https://dianvrlab.github.io/Panonut360/)
    *   citation: 4

*   **[Video ReCap: Recursive Captioning of Hour-Long Videos](http://arxiv.org/abs/2402.13250v6)** (CVPR2024 2024.02)
    *   Focus: Proposes a model for long video understanding and dense captioning of high-level concepts.
    *   citation: 78

*   **[Shot2Story: A New Benchmark for Comprehensive Understanding of Multi-shot Videos](http://arxiv.org/abs/2312.10300v3)** (ICLR2025 2023.12)
    *   Focus: Video understanding requires capturing individual events and their associations to comprehend storylines.
    *   citation: 41

*   **[MM-VID: Advancing Video Understanding with GPT-4V(ision)](http://arxiv.org/abs/2310.19773v1)** (2023.10)
    *   Focus: MM-VID integrates GPT-4V with specialized tools for advanced video understanding.
    *   project: [https://multimodal-vid.github.io/](https://multimodal-vid.github.io/)
    *   citation: 84

*   **[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](http://arxiv.org/abs/2309.13925v2)** (2023.09)
    *   Focus: Surveillance video tasks need expansion beyond classification to include temporal localization and dense captioning.
    *   project: [https://xuange923.github.io/Surveillance-Video-Understanding](https://xuange923.github.io/Surveillance-Video-Understanding)
    *   citation: 37

*   **[KuaiSAR: A Unified Search And Recommendation Dataset](http://arxiv.org/abs/2306.07705v4)** (2023.06)
    *   Focus: Search and recommendation integration is key for online platforms like e-commerce and video services.
    *   citation: 24

*   **[MUG: A General Meeting Understanding and Generation Benchmark](http://arxiv.org/abs/2303.13939v2)** (2023.03)
    *   Focus: Proposes a method to efficiently extract key information from long video/audio recordings using ASR transcripts.
    *   citation: 10

*   **[METEOR Guided Divergence for Video Captioning](http://arxiv.org/abs/2212.10690v1)** (2022.12)
    *   Focus: Video captioning requires temporal context modeling and action comprehension for holistic scene understanding.
    *   code: [https://github.com/d-rothen/bmhrl](https://github.com/d-rothen/bmhrl)
    *   citation: 3

*   **[REVECA -- Rich Encoder-decoder framework for Video Event CAptioner](http://arxiv.org/abs/2206.09178v1)** (2022.06)
    *   Focus: A rich encoder-decoder framework for video boundary event captioning.
    *   code: [https://github.com/TooTouch/REVECA](https://github.com/TooTouch/REVECA)
    *   citation: 0


*   **[Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding](http://arxiv.org/abs/2504.13915v1)** (2025.04)
    *   Focus: ProVideLLM is an end-to-end framework for real-time procedural video understanding with multimodal caching.
    *   project: [https://dibschat.github.io/ProVideLLM](https://dibschat.github.io/ProVideLLM)
    *   citation: 2


#### Temporal Action Detection

*   **[ContextDet: Temporal Action Detection with Adaptive Context Aggregation](http://arxiv.org/abs/2410.15279v1)** (2024.10)
    *   Focus: TAD faces challenges from variable segment lengths and ambiguous boundaries in video understanding.
    *   citation: 3

*   **[Harnessing Temporal Causality for Advanced Temporal Action Detection](http://arxiv.org/abs/2407.17792v2)** (2024.07)
    *   Focus: Temporal action detection identifies actions with precise boundaries in untrimmed videos.
    *   code: [https://github.com/sming256/OpenTAD/](https://github.com/sming256/OpenTAD/)
    *   citation: 5

*   **[TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization](http://arxiv.org/abs/2303.09055v1)** (2023.03)
    *   Focus: TAL identifies and localizes actions in videos, with recent focus on appearance features.
    *   code: [https://github.com/TuanTNG/TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer)
    *   citation: 38

*   **[An Efficient Spatio-Temporal Pyramid Transformer for Action Detection](http://arxiv.org/abs/2207.10448v1)** (ECCV2022 2022.07)
    *   Focus: Action detection in long videos using vision Transformers to classify and localize actions.
    *   citation: 30

*   **[Temporal Action Segmentation: An Analysis of Modern Techniques](http://arxiv.org/abs/2210.10352v5)** (2022.10)
    *   Focus: Temporal action segmentation identifies action classes in long videos, requiring long-range understanding.
    *   code: [https://github.com/nus-cvml/awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation)
    *   citation: 111

*   **[Streaming Video Temporal Action Segmentation In Real Time](http://arxiv.org/abs/2209.13808v3)** (2022.09)
    *   Focus: TAS models use features over raw video for long-term understanding.
    *   citation: 5

#### Temporal Video Grounding

*   **[TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning](http://arxiv.org/abs/2410.19702v2)** (ICLR2025 2024.10)
    *   Focus: MLLMs struggle with long video understanding despite success with short videos.
    *   citation: 49

*   **[TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability](http://arxiv.org/abs/2411.18211v1)** (2024.11)
    *   Focus: Video-language models struggle with long videos due to computational limits and lack of long-range benchmarks.
    *   code: [https://github.com/TimeMarker-LLM/TimeMarker/](https://github.com/TimeMarker-LLM/TimeMarker/)
    *   citation: 35

*   **[ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning](http://arxiv.org/abs/2505.15447v1)** (2025.05)
    *   Focus: MLLMs enable flexible video understanding by focusing on goal-relevant frames.
    *   citation: 5

*   **[LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling](http://arxiv.org/abs/2511.20785v1)** (2025.11)
    *   Focus: Large multimodal models for video reasoning are prone to hallucinations in long-form content.
    *   code: [https://github.com/EvolvingLMMs-Lab/LongVT](https://github.com/EvolvingLMMs-Lab/LongVT)
    *   citation: 3
    *   code: https://github.com/EvolvingLMMs-Lab/LongVT

*   **[LAST: LeArning to Think in Space and Time for Generalist Vision-Language Models](http://arxiv.org/abs/2511.19261v1)** (2025.11)
    *   Focus: Vision-language models struggle to understand 3D space and long videos like humans.
    *   citation: 0

*   **[VideoPerceiver: Enhancing Fine-Grained Temporal Perception in Video Multimodal Large Language Models](http://arxiv.org/abs/2511.18823v1)** (2025.11)
    *   Focus: VideoPerceiver improves fine-grained perception in video understanding by enhancing reasoning about brief events.
    *   citation: 0

*   **[FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos](http://arxiv.org/abs/2511.16183v1)** (2025.11)
    *   Focus: Soccer video datasets support tasks like action localization, detection, and tracking.
    *   citation: 0

*   **[TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning](http://arxiv.org/abs/2511.05489v1)** (2025.11)
    *   Focus: Temporal search finds minimal relevant frames from long videos for accurate video understanding.
    *   code: [https://github.com/Time-Search/TimeSearch-R](https://github.com/Time-Search/TimeSearch-R)
    *   citation: 0

*   **[NVIDIA Nemotron Nano V2 VL](http://arxiv.org/abs/2511.03929v2)** (2025.11)
    *   Focus: Nemotron Nano V2 VL advances document understanding, long video comprehension, and reasoning tasks.
    *   citation: 1

*   **[Conan: Progressive Learning to Reason Like a Detective over Multi-Scale Visual Evidence](http://arxiv.org/abs/2510.20470v2)** (2025.10)
    *   Focus: RL-based methods improve video reasoning by enabling multi-step deduction across frames.
    *   citation: 0

*   **[SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](http://arxiv.org/abs/2510.17777v1)** (ICCV2025 2025.10)
    *   Focus: VLMs advance visual-textual reasoning for high-res images, long videos, and multi-turn conversations.
    *   citation: 1

*   **[Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](http://arxiv.org/abs/2510.17364v1)** (NeurIPS2025 2025.10)
    *   Focus: Video-LLMs struggle with streaming video understanding due to limited access to full video content.
    *   citation: 0

*   **[SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding](http://arxiv.org/abs/2510.13016v2)** (2025.10)
    *   Focus: AI systems need to understand fine-grained actions and localize actors in space and time.
    *   citation: 1

*   **[Tracking the Spatiotemporal Evolution of Landslide Scars Using a Vision Foundation Model: A Novel and Universal Framework](http://arxiv.org/abs/2510.10084v1)** (2025.10)
    *   Focus: Proposes a method for tracking large-scale landslide scar evolution to improve early-warning systems.
    *   citation: 0

*   **[Online Generic Event Boundary Detection](http://arxiv.org/abs/2510.06855v1)** (ICCV2025 2025.10)
    *   Focus: GEBD detects event boundaries in long videos but current methods need full frame processing.
    *   citation: 1

*   **[Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models](http://arxiv.org/abs/2510.05034v6)** (2025.10)
    *   Focus: Video understanding requires reasoning about complex spatiotemporal relationships and long-term dependencies.
    *   code: [https://github.com/yunlong10/Awesome-Video-LMM-Post-Training](https://github.com/yunlong10/Awesome-Video-LMM-Post-Training)
    *   citation: 7

*   **[Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs](http://arxiv.org/abs/2510.00705v1)** (2025.10)
    *   Focus: MLLMs struggle with fine-grained perception in high-resolution images and long videos.
    *   citation: 0

*   **[TimeScope: Towards Task-Oriented Temporal Grounding In Long Videos](http://arxiv.org/abs/2509.26360v2)** (2025.09)
    *   Focus: Introduces Task-oriented Temporal Grounding (ToTG) for locating key moments in long videos.
    *   citation: 0

*   **[NeMo: Needle in a Montage for Video-Language Understanding](http://arxiv.org/abs/2509.24563v2)** (2025.09)
    *   Focus: Proposes new benchmarks for evaluating temporal reasoning in video-language models.
    *   project: [https://lavi-lab.github.io/NeMoBench](https://lavi-lab.github.io/NeMoBench)
    *   citation: 1

*   **[NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning](http://arxiv.org/abs/2509.18041v2)** (2025.09)
    *   Focus: Vision-language models struggle with long video question answering due to complex temporal reasoning demands.
    *   project: [https://utaustin-swarmlab.github.io/NeuS-QA/](https://utaustin-swarmlab.github.io/NeuS-QA/)
    *   citation: 0

*   **[Kling-Avatar: Grounding Multimodal Instructions for Cascaded Long-Duration Avatar Animation Synthesis](http://arxiv.org/abs/2509.09595v2)** (2025.09)
    *   Focus: Audio-driven avatar generation lacks high-level instruction conditioning for semantic control.
    *   project: [https://klingavatar.github.io/](https://klingavatar.github.io/)
    *   citation: 2

*   **[DATE: Dynamic Absolute Time Enhancement for Long Video Understanding](http://arxiv.org/abs/2509.09263v1)** (2025.09)
    *   Focus: Long video understanding challenges MLLMs in temporal reasoning and event localization.
    *   citation: 3

*   **[OOTSM: A Decoupled Linguistic Framework for Effective Scene Graph Anticipation](http://arxiv.org/abs/2509.05661v1)** (2025.09)
    *   Focus: Scene Graph Anticipation predicts future object relationships from video clips for applications.
    *   code: [https://github.com/ZhuXMMM/OOTSM](https://github.com/ZhuXMMM/OOTSM)
    *   citation: 1

*   **[Long-Horizon Visual Imitation Learning via Plan and Code Reflection](http://arxiv.org/abs/2509.05368v2)** (2025.09)
    *   Focus: Visual imitation learning struggles with long-horizon demonstrations and complex action sequences.
    *   citation: 1

*   **[ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Long Video Understanding](http://arxiv.org/abs/2508.21496v2)** (2025.08)
    *   Focus: Video-MLLMs show strong video understanding but are prone to hallucination.
    *   citation: 0

*   **[Language-Guided Temporal Token Pruning for Efficient VideoLLM Processing](http://arxiv.org/abs/2508.17686v1)** (2025.08)
    *   Focus: LGTTP uses language guidance to prune temporal tokens, reducing attention complexity for long videos.
    *   citation: 0

*   **[Multi-Level LVLM Guidance for Untrimmed Video Action Recognition](http://arxiv.org/abs/2508.17442v1)** (2025.08)
    *   Focus: Current methods struggle with fine-grained action recognition and localization in untrimmed videos.
    *   citation: 0

*   **[When and What: Diffusion-Grounded VideoLLM with Entity Aware Segmentation for Long Video Understanding](http://arxiv.org/abs/2508.15641v1)** (2025.08)
    *   Focus: Video LLMs need temporal grounding and entity interaction modeling for comprehensive video understanding.
    *   citation: 0

*   **[Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency](http://arxiv.org/abs/2506.01908v1)** (2025.06)
    *   Focus: MLLMs advance long video understanding with complex semantics and temporal dependencies.
    *   code: [https://github.com/appletea233/Temporal-R1](https://github.com/appletea233/Temporal-R1)
    *   citation: 8
    
*   **[TAR-TVG: Enhancing VLMs with Timestamp Anchor-Constrained Reasoning for Temporal Video Grounding](http://arxiv.org/abs/2508.07683v1)** (2025.08)
    *   Focus: TVG localizes video segments from language queries for long video understanding.
    *   citation: 3

*   **[LET-US: Long Event-Text Understanding of Scenes](http://arxiv.org/abs/2508.07401v1)** (2025.08)
    *   Focus: Event cameras enable low-latency vision, but multimodal models struggle with their sparse, asynchronous data streams.
    *   citation: 1

*   **[Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning](http://arxiv.org/abs/2508.04416v2)** (2025.08)
    *   Focus: MLLMs need better video reasoning for tasks like QA and temporal grounding, but current methods rely too much on text.
    *   project: [https://zhang9302002.github.io/thinkingwithvideos-page/](https://zhang9302002.github.io/thinkingwithvideos-page/)
    *   citation: 18

*   **[ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks](http://arxiv.org/abs/2508.01943v1)** (NeurIPS2025 2025.08)
    *   Focus: Vision-language models struggle with reasoning over long video sequences.
    *   project: [https://rover-vlm.github.io](https://rover-vlm.github.io)
    *   citation: 0

*   **[Fine-grained Spatiotemporal Grounding on Egocentric Videos](http://arxiv.org/abs/2508.00518v1)** (ICCV2025 2025.08)
    *   Focus: Sparsely studied egocentric spatiotemporal video grounding needs new methods for entity localization.
    *   code: [https://github.com/LaVi-Lab/EgoMask](https://github.com/LaVi-Lab/EgoMask)
    *   citation: 4

*   **[LeAdQA: LLM-Driven Context-Aware Temporal Grounding for Video Question Answering](http://arxiv.org/abs/2507.14784v2)** (2025.07)
    *   Focus: VideoQA needs to find key moments and reason about their causal links in long videos.
    *   citation: 0

*   **[THYME: Temporal Hierarchical-Cyclic Interactivity Modeling for Video Scene Graphs in Aerial Footage](http://arxiv.org/abs/2507.09200v1)** (2025.07)
    *   Focus: Dynamic scene understanding methods are needed for video applications like autonomous driving and surveillance.
    *   citation: 0

*   **[HumanVideo-MME: Benchmarking MLLMs for Human-Centric Video Understanding](http://arxiv.org/abs/2507.04909v2)** (2025.07)
    *   Focus: MLLMs advance in visual tasks but struggle with human-centric video understanding.
    *   citation: 0

*   **[Universal Video Temporal Grounding with Generative Multi-modal Large Language Models](http://arxiv.org/abs/2506.18883v2)** (NeurIPS2025 2025.06)
    *   Focus: A model for localizing video moments using natural language queries.
    *   citation: 4

*   **[Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](http://arxiv.org/abs/2506.13654v1)** (2025.06)
    *   Focus: Ego-R1 uses Chain-of-Tool-Thought reasoning for ultra-long egocentric video understanding.
    *   project: [https://egolife-ai.github.io/Ego-R1/](https://egolife-ai.github.io/Ego-R1/)
    *   citation: 14

*   **[EASG-Bench: Video Q&A Benchmark with Egocentric Action Scene Graphs](http://arxiv.org/abs/2506.05787v2)** (2025.06)
    *   Focus: EASG-Bench is a QA benchmark for egocentric videos using spatio-temporally grounded scene graphs.
    *   code: [https://github.com/fpv-iplab/EASG-bench](https://github.com/fpv-iplab/EASG-bench)
    *   citation: 0

*   **[MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](http://arxiv.org/abs/2506.04141v1)** (2025.06)
    *   Focus: Proposes a new MLLM architecture for improved long video understanding and temporal reasoning.
    *   project: [https://mmr-v.github.io](https://mmr-v.github.io)
    *   citation: 6

*   **[Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](http://arxiv.org/abs/2505.23908v2)** (2025.05)
    *   Focus: Previews help users discover and evaluate long-form talk content like videos and podcasts efficiently.
    *   citation: 0

*   **[VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](http://arxiv.org/abs/2505.23359v1)** (2025.05)
    *   Focus: Long chain-of-thought reasoning improves LLMs but lacks demonstration for long video understanding tasks.
    *   project: [https://llyx97.github.io/video_reason_bench/](https://llyx97.github.io/video_reason_bench/)
    *   citation: 4

*   **[Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](http://arxiv.org/abs/2505.18110v2)** (NeurIPS2025 2025.05)
    *   Focus: Video moment localization integrates visual and auditory cues to identify specific scenes.
    *   citation: 1

*   **[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](http://arxiv.org/abs/2505.18079v4)** (NeurIPS2025 2025.05)
    *   Focus: Long video understanding faces challenges from temporal-spatial complexity and extended context question answering.
    *   code: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)
    *   citation: 7

*   **[QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v2)** (2025.05)
    *   Focus: Long-video understanding is crucial for real-world applications but faces challenges.
    *   citation: 2

*   **[Clapper: Compact Learning and Video Representation in VLMs](http://arxiv.org/abs/2505.15529v1)** (2025.05)
    *   Focus: Vision-language models need effective temporal modeling for video understanding.
    *   citation: 0

*   **[CrayonRobo: Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation](http://arxiv.org/abs/2505.02166v1)** (CVPR2025 2025.05)
    *   Focus: The paper explores using goal videos to reduce ambiguity in robotic task specification.
    *   citation: 3

*   **[An LLM-Empowered Low-Resolution Vision System for On-Device Human Behavior Understanding](http://arxiv.org/abs/2505.01743v1)** (2025.05)
    *   Focus: LVLMs can generate detailed descriptions for on-device human behavior understanding.
    *   citation: 0

*   **[AVA: Towards Agentic Video Analytics with Vision Language Models](http://arxiv.org/abs/2505.00254v5)** (2025.05)
    *   Focus: AI video analytics systems lack adaptability for open-ended tasks beyond predefined functions.
    *   code: [https://github.com/I-ESC/Project-Ava](https://github.com/I-ESC/Project-Ava)
    *   citation: 4

*   **[Multi-Stage Boundary-Aware Transformer Network for Action Segmentation in Untrimmed Surgical Videos](http://arxiv.org/abs/2504.18756v2)** (2025.04)
    *   Focus: Analyzing long surgical action sequences improves outcomes, training, and efficiency.
    *   citation: 1

*   **[TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation](http://arxiv.org/abs/2504.17365v3)** (2025.04)
    *   Focus: MLLMs show potential for analyzing long soccer videos and identifying key highlights.
    *   citation: 3

*   **[Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](http://arxiv.org/abs/2504.12083v2)** (NeurIPS2025 2025.04)
    *   Focus: LVLMs struggle with temporal details, hallucinate, and make errors on simple video QA tasks.
    *   citation: 2

*   **[Action Anticipation from SoccerNet Football Video Broadcasts](http://arxiv.org/abs/2504.12021v1)** (2025.04)
    *   Focus: AI enables analysis of long sports videos for action understanding and motion prediction.
    *   code: [https://github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)
    *   citation: 1

*   **[Audio-visual Event Localization on Portrait Mode Short Videos](http://arxiv.org/abs/2504.06884v1)** (2025.04)
    *   Focus: AVEL datasets focus on landscape-oriented long videos with clean audio, limiting real-world applicability.
    *   citation: 1

*   **[Pose-Aware Weakly-Supervised Action Segmentation](http://arxiv.org/abs/2504.05700v1)** (2025.04)
    *   Focus: Action segment labeling is a costly challenge for human behavior understanding.
    *   citation: 0

*   **[T*: Re-thinking Temporal Search for Long-Form Video Understanding](http://arxiv.org/abs/2504.02259v3)** (CVPR2025 2025.04)
    *   Focus: Revisits temporal search paradigms to address fundamental challenges in long-form video understanding.
    *   citation: 32

*   **[MammAlps: A multi-view video behavior monitoring dataset of wild mammals in the Swiss Alps](http://arxiv.org/abs/2503.18223v2)** (CVPR2025 2025.03)
    *   Focus: Camera traps enable habitat-centric wildlife monitoring for ecology and ethology studies.
    *   code: [https://github.com/eceo-epfl/MammAlps](https://github.com/eceo-epfl/MammAlps)
    *   citation: 7

*   **[VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning](http://arxiv.org/abs/2503.13444v2)** (2025.03)
    *   Focus: Video understanding requires precise grounding of answers to visual evidence.
    *   project: [https://videomind.github.io/](https://videomind.github.io/)
    *   citation: 29

*   **[Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding](http://arxiv.org/abs/2503.13377v3)** (NeurIPS2025 2025.03)
    *   Focus: TVG locates video segments from language queries, a key challenge in long video understanding.
    *   project: [https://xuboshen.github.io/Time-R1/](https://xuboshen.github.io/Time-R1/)
    *   citation: 33

*   **[ST-Think: How Multimodal Large Language Models Reason About 4D Worlds from Ego-Centric Videos](http://arxiv.org/abs/2503.12542v2)** (2025.03)
    *   Focus: The abstract questions if multimodal LLMs can match human spatial-temporal reasoning in egocentric video understanding.
    *   citation: 14

*   **[VideoScan: Enabling Efficient Streaming Video Understanding via Frame-level Semantic Carriers](http://arxiv.org/abs/2503.09387v2)** (2025.03)
    *   Focus: VideoScan enables real-time video interaction with efficient VLM inference for streamed video comprehension.
    *   citation: 3

*   **[TimeLoc: A Unified End-to-End Framework for Precise Timestamp Localization in Long Videos](http://arxiv.org/abs/2503.06526v1)** (2025.03)
    *   Focus: Temporal localization identifies specific timestamps in untrimmed videos for video understanding.
    *   code: [https://github.com/sming256/TimeLoc](https://github.com/sming256/TimeLoc)
    *   citation: 2

*   **[An Egocentric Vision-Language Model based Portable Real-time Smart Assistant](http://arxiv.org/abs/2503.04250v1)** (2025.03)
    *   Focus: Vinci is a portable AI system using EgoVideo-VL for real-time vision-language assistance.
    *   code: [https://github.com/OpenGVLab/vinci](https://github.com/OpenGVLab/vinci)
    *   citation: 6

*   **[iMOVE: Instance-Motion-Aware Video Understanding](http://arxiv.org/abs/2502.11594v2)** (2025.02)
    *   Focus: Improving Video LLMs' fine-grained motion perception for better temporal understanding.
    *   citation: 8

*   **[Understanding Long Videos via LLM-Powered Entity Relation Graphs](http://arxiv.org/abs/2501.15953v1)** (2025.01)
    *   Focus: Analyzing long videos is challenging for AI due to tracking and understanding visual elements over time.
    *   citation: 2

*   **[TinyLLaVA-Video: Towards Smaller LMMs for Video Understanding with Group Resampler](http://arxiv.org/abs/2501.15513v2)** (2025.01)
    *   Focus: Video behavior recognition and scene understanding are fundamental multimodal intelligence tasks for real-world applications.
    *   code: [https://github.com/ZhangXJ199/TinyLLaVA-Video](https://github.com/ZhangXJ199/TinyLLaVA-Video)
    *   citation: 1

*   **[Temporal Preference Optimization for Long-Form Video Understanding](http://arxiv.org/abs/2501.13919v3)** (2025.01)
    *   Focus: Video-LMMs struggle with temporal grounding in long videos, requiring new methods.
    *   project: [https://ruili33.github.io/tpo_website](https://ruili33.github.io/tpo_website)
    *   citation: 19

*   **[X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding](http://arxiv.org/abs/2501.06835v2)** (2025.01)
    *   Focus: Long-form egocentric videos offer insights into human behavior for embodied intelligence applications.
    *   citation: 4

*   **[LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding](http://arxiv.org/abs/2501.05067v2)** (2025.01)
    *   Focus: LLaVA-Octopus adaptively weights visual features for video understanding based on user instructions.
    *   citation: 8

*   **[V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding](http://arxiv.org/abs/2412.09616v2)** (ICCV2025 2024.12)
    *   Focus: VLMs struggle with long-context video and high-resolution tasks despite multimodal capabilities.
    *   code: [https://github.com/OpenGVLab/V2PE](https://github.com/OpenGVLab/V2PE)
    *   citation: 15

*   **[Multi-Scale Contrastive Learning for Video Temporal Grounding](http://arxiv.org/abs/2412.07157v2)** (2024.12)
    *   Focus: Proposes a method to encode variable-length video moments for temporal grounding with natural language queries.
    *   citation: 3

*   **[Towards Long Video Understanding via Fine-detailed Video Story Generation](http://arxiv.org/abs/2412.06182v2)** (2024.12)
    *   Focus: Long video understanding is a critical computer vision task with applications in surveillance and content retrieval.
    *   citation: 11

*   **[Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](http://arxiv.org/abs/2412.06171v2)** (CVPR2025 2024.12)
    *   Focus: Proposes methods for video anomaly understanding across temporal scales and contexts.
    *   code: [https://github.com/pipixin321/HolmesVAU](https://github.com/pipixin321/HolmesVAU)
    *   citation: 29

*   **[Video LLMs for Temporal Reasoning in Long Videos](http://arxiv.org/abs/2412.02930v4)** (2024.12)
    *   Focus: TemporalVLM enables temporal reasoning and fine-grained understanding in long videos.
    *   citation: 5

*   **[Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](http://arxiv.org/abs/2411.16932v1)** (CVPR2025 2024.11)
    *   Focus: Video LLMs need temporal awareness for tasks like dense captioning and temporal grounding.
    *   citation: 4

*   **[LLaVA-MR: Large Language-and-Vision Assistant for Video Moment Retrieval](http://arxiv.org/abs/2411.14505v1)** (2024.11)
    *   Focus: MLLMs struggle with long video processing and precise moment retrieval due to LLM limitations.
    *   citation: 11

*   **[BuckTales : A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes](http://arxiv.org/abs/2411.06896v1)** (NeurIPS2024 2024.11)
    *   Focus: Animal behavior understanding is crucial for ecological impact assessment but faces data acquisition and analysis challenges.
    *   citation: 13

*   **[Zero-shot Action Localization via the Confidence of Large Vision-Language Models](http://arxiv.org/abs/2410.14340v2)** (2024.10)
    *   Focus: Action localization in untrimmed videos is crucial for sports and surgery applications.
    *   code: [https://github.com/josaklil-ai/zeal](https://github.com/josaklil-ai/zeal)
    *   citation: 1

*   **[Deep learning for action spotting in association football videos](http://arxiv.org/abs/2410.01304v1)** (2024.10)
    *   Focus: Action spotting identifies and precisely localizes actions with timestamps in long untrimmed videos.
    *   citation: 3

*   **[UAL-Bench: The First Comprehensive Unusual Activity Localization Benchmark](http://arxiv.org/abs/2410.01180v1)** (2024.10)
    *   Focus: Models struggle to localize unusual activities like human errors in videos.
    *   citation: 5

*   **[YouTube Video Analytics for Patient Engagement: Evidence from Colonoscopy Preparation Videos](http://arxiv.org/abs/2410.02830v1)** (2024.10)
    *   Focus: Video analysis methods for medical education content are explored.
    *   citation: 0

*   **[MECD: Unlocking Multi-Event Causal Discovery in Video Reasoning](http://arxiv.org/abs/2409.17647v4)** (NeurIPS2024 2024.09)
    *   Focus: Video causal reasoning tasks are currently limited in scope and executed as question-answering.
    *   citation: 19

*   **[Learning to Localize Actions in Instructional Videos with LLM-Based Multi-Pathway Text-Video Alignment](http://arxiv.org/abs/2409.16145v1)** (ECCV2024 2024.09)
    *   Focus: Proposes a method to localize steps in instructional videos using limited annotations.
    *   citation: 5

*   **[ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation](http://arxiv.org/abs/2409.13682v1)** (2024.09)
    *   Focus: Robots face challenges in long-term environment understanding and answering human questions.
    *   project: [https://nvidia-ai-iot.github.io/remembr](https://nvidia-ai-iot.github.io/remembr)
    *   citation: 29

*   **[AMEGO: Active Memory from long EGOcentric videos](http://arxiv.org/abs/2409.10917v1)** (ECCV2024 2024.09)
    *   Focus: AMEGO is a novel approach for understanding unstructured egocentric videos.
    *   project: [https://gabrielegoletto.github.io/AMEGO/](https://gabrielegoletto.github.io/AMEGO/)
    *   citation: 18

*   **[Open-Vocabulary Action Localization with Iterative Visual Prompting](http://arxiv.org/abs/2408.17422v5)** (2024.08)
    *   Focus: Video action localization finds action timings but requires costly video annotations.
    *   project: [https://microsoft.github.io/VLM-Video-Action-Localization/](https://microsoft.github.io/VLM-Video-Action-Localization/)
    *   citation: 4

*   **[HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](http://arxiv.org/abs/2408.06437v1)** (ECCV2024 2024.08)
    *   Focus: Online video understanding uses frame-by-frame predictions, extended by OnTAL for temporal action localization.
    *   code: [https://github.com/sakibreza/ECCV24-HAT/](https://github.com/sakibreza/ECCV24-HAT/)
    *   citation: 5

*   **[mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models](http://arxiv.org/abs/2408.04840v2)** (ICLR2025 2024.08)
    *   Focus: MLLMs excel at single-image tasks but face challenges in other areas.
    *   citation: 206

*   **[SynopGround: A Large-Scale Dataset for Multi-Paragraph Video Grounding from TV Dramas and Synopses](http://arxiv.org/abs/2408.01669v4)** (2024.08)
    *   Focus: Video grounding localizes language queries in untrimmed videos, but current datasets are limited.
    *   project: [https://synopground.github.io/](https://synopground.github.io/)
    *   citation: 2

*   **[Fine-grained Dynamic Network for Generic Event Boundary Detection](http://arxiv.org/abs/2407.04274v1)** (ECCV2024 2024.07)
    *   Focus: Generic event boundary detection identifies human-perceived boundaries in long videos for better understanding.
    *   citation: 2

*   **[MLLM as Video Narrator: Mitigating Modality Imbalance in Video Moment Retrieval](http://arxiv.org/abs/2406.17880v1)** (2024.06)
    *   Focus: Video Moment Retrieval localizes video segments from text queries but struggles with limited training annotations.
    *   citation: 4

*   **[Zero-Shot Long-Form Video Understanding through Screenplay](http://arxiv.org/abs/2406.17309v1)** (2024.06)
    *   Focus: Long-form video QA requires temporal and contextual analysis for accurate responses.
    *   citation: 5

*   **[VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment](http://arxiv.org/abs/2406.10889v2)** (CVPR2025 2024.06)
    *   Focus: Video models advance but struggle with associating people and actions over time for compositional reasoning.
    *   project: [https://katha-ai.github.io/projects/velociti](https://katha-ai.github.io/projects/velociti)
    *   citation: 3

*   **[MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models](http://arxiv.org/abs/2405.14338v3)** (2024.05)
    *   Focus: Point cloud videos capture spatial and temporal dynamics for intelligent agents.
    *   code: [https://github.com/IRMVLab/Mamba4D](https://github.com/IRMVLab/Mamba4D)
    *   citation: 11
    *   code: https://github.com/IRMVLab/Mamba4D

*   **[DTLLM-VLT: Diverse Text Generation for Visual Language Tracking Based on LLM](http://arxiv.org/abs/2405.12139v2)** (2024.05)
    *   Focus: VLT improves object tracking by using language descriptions for precise video object localization.
    *   citation: 30

*   **[Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis](http://arxiv.org/abs/2405.08944v1)** (2024.05)
    *   Focus: Transformer models enable long-context AI applications like video understanding and coding agents.
    *   citation: 35

*   **[SnAG: Scalable and Accurate Video Grounding](http://arxiv.org/abs/2404.02257v2)** (CVPR2024 2024.04)
    *   Focus: Existing video grounding methods prioritize accuracy over scalability.
    *   code: [https://github.com/fmu2/snag_release](https://github.com/fmu2/snag_release)
    *   citation: 23
    *   code: https://github.com/fmu2/snag_release

*   **[SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding](http://arxiv.org/abs/2404.01174v2)** (2024.04)
    *   Focus: Temporal video grounding aligns video content with language instructions for precise understanding.
    *   citation: 27

*   **[Towards Neuro-Symbolic Video Understanding](http://arxiv.org/abs/2403.11021v3)** (ECCV2024 2024.03)
    *   Focus: Efficient frame extraction methods are needed for long-term temporal reasoning in videos.
    *   citation: 19

*   **[HawkEye: Training Video-Text LLMs for Grounding Text in Videos](http://arxiv.org/abs/2403.10228v1)** (2024.03)
    *   Focus: Video-text LLMs perform poorly on complex videos, similar to random guessing.
    *   citation: 54

*   **[Multi-modal News Understanding with Professionally Labelled Videos (ReutersViLNews)](http://arxiv.org/abs/2401.12419v1)** (2024.01)
    *   Focus: Current video-language models struggle with high-level abstract understanding.
    *   citation: 1

*   **[A Simple LLM Framework for Long-Range Video Question-Answering](http://arxiv.org/abs/2312.17235v3)** (2023.12)
    *   Focus: LLoVi is a language-based framework for efficient long-range video question-answering.
    *   code: [https://github.com/CeeZh/LLoVi](https://github.com/CeeZh/LLoVi)
    *   citation: 141

*   **[Grounding-Prompter: Prompting LLM with Multimodal Information for Temporal Sentence Grounding in Long Videos](http://arxiv.org/abs/2312.17117v1)** (2023.12)
    *   Focus: TSG localizes video moments using language queries, with current methods for short videos.
    *   citation: 17

*   **[Hierarchical Graph Pattern Understanding for Zero-Shot VOS](http://arxiv.org/abs/2312.09525v1)** (2023.12)
    *   Focus: Optical flow is ideal for video motion but current methods have limitations.
    *   code: [https://github.com/NUST-Machine-Intelligence-Laboratory/HGPU](https://github.com/NUST-Machine-Intelligence-Laboratory/HGPU)
    *   citation: 4

*   **[Spatiotemporal Event Graphs for Dynamic Scene Understanding](http://arxiv.org/abs/2312.07621v1)** (2023.12)
    *   Focus: This thesis presents methods for dynamic scene understanding in videos.
    *   citation: 0

*   **[Grounded Question-Answering in Long Egocentric Videos](http://arxiv.org/abs/2312.06505v4)** (CVPR2024 2023.12)
    *   Focus: Proposes a new approach for long, egocentric video understanding to address robotics applications.
    *   code: [https://github.com/Becomebright/GroundVQA](https://github.com/Becomebright/GroundVQA)
    *   citation: 42

*   **[RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](http://arxiv.org/abs/2312.06729v3)** (ECCV2024 2023.12)
    *   Focus: Adapting short video grounding methods to locate moments in long videos.
    *   code: [https://github.com/Tanveer81/RGNet](https://github.com/Tanveer81/RGNet)
    *   citation: 9

*   **[TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding](http://arxiv.org/abs/2312.02051v2)** (CVPR2024 2023.12)
    *   Focus: TimeChat is a time-sensitive MLLM for long video understanding with timestamp-aware frame tokenization and temporal attention.
    *   code: [https://github.com/RenShuhuai-Andy/TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
    *   citation: 326

*   **[Multi-Modal Video Topic Segmentation with Dual-Contrastive Domain Adaptation](http://arxiv.org/abs/2312.00220v1)** (2023.11)
    *   Focus: Video topic segmentation reveals semantic structure and is vital for video understanding tasks.
    *   citation: 0

*   **[PALM: Predicting Actions through Language Models](http://arxiv.org/abs/2311.17944v2)** (ECCV2024 2023.11)
    *   Focus: Explores challenges in egocentric vision for human activity understanding.
    *   citation: 20

*   **[A Hybrid Graph Network for Complex Activity Detection in Video](http://arxiv.org/abs/2310.17493v2)** (2023.10)
    *   Focus: Video understanding is challenging in fields like autonomous driving and sports analytics.
    *   citation: 2

*   **[End-to-End Streaming Video Temporal Action Segmentation with Reinforce Learning](http://arxiv.org/abs/2309.15683v2)** (2023.09)
    *   Focus: Streaming temporal action segmentation is an understudied video understanding task.
    *   code: [https://github.com/Thinksky5124/SVTAS](https://github.com/Thinksky5124/SVTAS)
    *   citation: 0

*   **[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](http://arxiv.org/abs/2309.13925v2)** (2023.09)
    *   Focus: Surveillance video tasks need expansion beyond classification to include temporal localization and dense captioning.
    *   project: [https://xuange923.github.io/Surveillance-Video-Understanding](https://xuange923.github.io/Surveillance-Video-Understanding)
    *   citation: 37

*   **[Language-Conditioned Change-point Detection to Identify Sub-Tasks in Robotics Domains](http://arxiv.org/abs/2309.00743v1)** (2023.09)
    *   Focus: An approach identifies robot trajectory sub-tasks using language instructions from demonstrations.
    *   citation: 1

*   **[Helping Hands: An Object-Aware Ego-Centric Video Recognition Model](http://arxiv.org/abs/2308.07918v1)** (ICCV2023 2023.08)
    *   Focus: An object-aware decoder improves ego-centric video understanding by enhancing object-awareness during training.
    *   citation: 34

*   **[Single-Stage Visual Query Localization in Egocentric Videos](http://arxiv.org/abs/2306.09324v1)** (NeurIPS2023 2023.06)
    *   Focus: This paper addresses visual query localization in long egocentric videos for episodic memory systems.
    *   project: [https://hwjiang1510.github.io/VQLoC/](https://hwjiang1510.github.io/VQLoC/)
    *   citation: 19

*   **[Boundary-Denoising for Video Activity Localization](http://arxiv.org/abs/2304.02934v1)** (2023.04)
    *   Focus: Video activity localization retrieves actions of interest with timestamps from long untrimmed videos.
    *   citation: 13

*   **[Diffusion Action Segmentation](http://arxiv.org/abs/2303.17959v2)** (ICCV2023 2023.03)
    *   Focus: A novel method for temporal action segmentation in long videos is proposed.
    *   citation: 95

*   **[MINOTAUR: Multi-task Video Grounding From Multimodal Queries](http://arxiv.org/abs/2302.08063v2)** (2023.02)
    *   Focus: Video understanding tasks vary in inputs and goals, requiring flexible models.
    *   citation: 8

*   **[Efficient Movie Scene Detection using State-Space Transformers](http://arxiv.org/abs/2212.14427v2)** (CVPR2023 2022.12)
    *   Focus: Movie scene detection is challenging due to the need to understand complex storylines and temporal dynamics.
    *   code: [https://github.com/md-mohaiminul/TranS4mer](https://github.com/md-mohaiminul/TranS4mer)
    *   citation: 62

*   **[Nonlinear and Machine Learning Analyses on High-Density EEG data of Math Experts and Novices](http://arxiv.org/abs/2212.00712v1)** (2022.12)
    *   Focus: Neuroscience uses naturalistic stimuli like cinema and video games to study brain function in ecologically valid conditions.
    *   citation: 5

*   **[Exploring Anchor-based Detection for Ego4D Natural Language Query](http://arxiv.org/abs/2208.05375v1)** (2022.08)
    *   Focus: Report on Ego4D natural language query challenge for comprehensive video understanding.
    *   citation: 4

*   **[Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding](http://arxiv.org/abs/2207.14698v2)** (ECCV2022 2022.07)
    *   Focus: Current temporal grounding methods struggle with severe performance issues in untrimmed videos.
    *   code: [https://github.com/haojc/ShufflingVideosForTSG](https://github.com/haojc/ShufflingVideosForTSG)
    *   citation: 33

*   **[Video + CLIP Baseline for Ego4D Long-term Action Anticipation](http://arxiv.org/abs/2207.00579v1)** (2022.07)
    *   Focus: Video+CLIP adapts image-text models for long-term action anticipation using CLIP.
    *   code: [http://github.com/srijandas07/clip_baseline_LTA_Ego4d](http://github.com/srijandas07/clip_baseline_LTA_Ego4d)
    *   citation: 23

*   **[Technical Report for CVPR 2022 LOVEU AQTC Challenge](http://arxiv.org/abs/2206.14555v1)** (2022.06)
    *   Focus: Winning model for AQTC task in LOVEU challenge addresses multi-step video understanding difficulties.
    *   code: [https://github.com/jaykim9870/](https://github.com/jaykim9870/)
    *   citation: 0

*   **[Scene Consistency Representation Learning for Video Scene Segmentation](http://arxiv.org/abs/2205.05487v1)** (CVPR2022 2022.05)
    *   Focus: Scene boundary detection in long videos using semantic story consistency.
    *   citation: 21

*   **[Contrastive Language-Action Pre-training for Temporal Localization](http://arxiv.org/abs/2204.12293v1)** (2022.04)
    *   Focus: Long video understanding faces memory limits for end-to-end training of temporal localization tasks.
    *   citation: 26

*   **[Temporal Alignment Networks for Long-term Video](http://arxiv.org/abs/2204.02968v1)** (CVPR2022 2022.04)
    *   Focus: A network aligns long videos with text sentences, determining if alignment is possible.
    *   citation: 101

*   **[Long Movie Clip Classification with State-Space Video Models](http://arxiv.org/abs/2204.01692v3)** (ECCV2022 2022.04)
    *   Focus: Short video models struggle with long movie understanding tasks.
    *   code: [https://github.com/md-mohaiminul/ViS4mer](https://github.com/md-mohaiminul/ViS4mer)
    *   citation: 136

*   **[MSPred: Video Prediction at Multiple Spatio-Temporal Scales with Hierarchical Recurrent Networks](http://arxiv.org/abs/2203.09303v4)** (2022.03)
    *   Focus: Autonomous systems must understand past states to predict future actions from camera frames.
    *   code: [https://github.com/AIS-Bonn/MSPred](https://github.com/AIS-Bonn/MSPred)
    *   citation: 12

*   **[Behavior Recognition Based on the Integration of Multigranular Motion Features](http://arxiv.org/abs/2203.03097v1)** (2022.03)
    *   Focus: Video behavior recognition combines spatial object data with temporal action dynamics.
    *   citation: 0

#### Others

*   **[RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation](http://arxiv.org/abs/2506.22007v1)** (2025.06)
    *   Focus: Generating long-horizon videos for robotic tasks using text-to-video diffusion models.
    *   citation: 2

*   **[Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation](http://arxiv.org/abs/2502.16707v1)** (2025.02)
    *   Focus: Robotic manipulation requires high-level planning, physical reasoning, and reactive motion selection.
    *   project: [https://reflect-vlm.github.io](https://reflect-vlm.github.io)
    *   citation: 21

*   **[Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](http://arxiv.org/abs/2412.09621v2)** (CVPR2025 2024.12)
    *   Focus: Dynamic 3D scene understanding from imagery lacks large-scale supervised training data.
    *   project: [https://stereo4d.github.io](https://stereo4d.github.io)
    *   citation: 39

*   **[Memory-augmented Online Video Anomaly Detection](http://arxiv.org/abs/2302.10719v2)** (2023.02)
    *   Focus: An online system for autonomous vehicles to understand scenes and provide immediate responses.
    *   code: [https://github.com/IMPLabUniPr/movad/tree/movad_vad](https://github.com/IMPLabUniPr/movad/tree/movad_vad)
    *   citation: 8


*   **[On the Pitfalls of Batch Normalization for End-to-End Video Learning: A Study on Surgical Workflow Analysis](http://arxiv.org/abs/2203.07976v5)** (2022.03)
    *   Focus: Batch Normalization's batch-dependent property causes problems in sequence modeling but is understudied.
    *   citation: 24

*   **[Vision-Language Memory for Spatial Reasoning](http://arxiv.org/abs/2511.20644v1)** (2025.11)
    *   Focus: Vision-language models underperform humans in video spatial reasoning, highlighting a key research gap.
    *   citation: 0

*   **[DeepSport: A Multimodal Large Language Model for Comprehensive Sports Video Reasoning via Agentic Reinforcement Learning](http://arxiv.org/abs/2511.12908v1)** (2025.11)
    *   Focus: Sports video understanding requires models to handle high-speed dynamics, complex rules, and long temporal contexts.
    *   citation: 0

*   **[Synopses of Movie Narratives: a Video-Language Dataset for Story Understanding](http://arxiv.org/abs/2203.05711v4)** (2022.03)
    *   Focus: A new video-language dataset for movie story understanding is collected and released.
    *   citation: 11

*   **[HumanMM: Global Human Motion Recovery from Multi-shot Videos](http://arxiv.org/abs/2503.07597v1)** (CVPR2025 2025.03)
    *   Focus: A framework reconstructs long-sequence 3D human motion from in-the-wild videos with shot transitions.
    *   project: [https://zhangyuhong01.github.io/HumanMM/](https://zhangyuhong01.github.io/HumanMM/)
    *   citation: 3

*   **[Video-Mined Task Graphs for Keystep Recognition in Instructional Videos](http://arxiv.org/abs/2307.08763v2)** (NeurIPS2023 2023.07)
    *   Focus: Procedural activity understanding analyzes sequential human actions in long videos to achieve task goals.
    *   citation: 33


*   **[Temporal Action Segmentation: An Analysis of Modern Techniques](http://arxiv.org/abs/2210.10352v5)** (2022.10)
    *   Focus: Temporal action segmentation identifies action classes in long videos, requiring long-range understanding.
    *   code: [https://github.com/nus-cvml/awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation)
    *   citation: 111

*   **[Controllable Hybrid Captioner for Improved Long-form Video Understanding](http://arxiv.org/abs/2507.17047v4)** (2025.07)
    *   Focus: Text summaries compress dense video data into compact, query-relevant representations.
    *   citation: 0

*   **[MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding](http://arxiv.org/abs/2507.06072v1)** (ICCV2025 2025.07)
    *   Focus: Proposes a method for deep causal reasoning in autonomous driving video behavior recognition.
    *   code: [https://github.com/SixCorePeach/MCAM](https://github.com/SixCorePeach/MCAM)
    *   citation: 0

*   **[Text-guided Weakly Supervised Framework for Dynamic Facial Expression Recognition](http://arxiv.org/abs/2511.10958v1)** (2025.11)
    *   Focus: Dynamic facial expression recognition models temporal facial changes in videos, addressing many-to-one mapping challenges.
    *   citation: 0

*   **[FloCoDe: Unbiased Dynamic Scene Graph Generation with Temporal Consistency and Correlation Debiasing](http://arxiv.org/abs/2310.16073v3)** (2023.10)
    *   Focus: Dynamic scene graph generation from videos requires understanding objects, motions, and interactions over time.
    *   citation: 2

*   **[Local-Global Information Interaction Debiasing for Dynamic Scene Graph Generation](http://arxiv.org/abs/2308.05274v2)** (2023.08)
    *   Focus: Dynamic scene graph generation for videos faces challenges from long-tail distributions in spatial-temporal modeling.
    *   citation: 1

*   **[TeleEgo: Benchmarking Egocentric AI Assistants in the Wild](http://arxiv.org/abs/2510.23981v2)** (2025.10)
    *   Focus: Existing benchmarks for egocentric AI assistants lack real-time processing and long-term memory requirements.
    *   citation: 0

*   **[EmbRACE-3K: Embodied Reasoning and Action in Complex Environments](http://arxiv.org/abs/2507.10548v1)** (2025.07)
    *   Focus: VLMs excel in passive video understanding but struggle in embodied settings requiring active perception.
    *   project: [https://mxllc.github.io/EmbRACE-3K/](https://mxllc.github.io/EmbRACE-3K/)
    *   citation: 0

*   **[CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning](http://arxiv.org/abs/2506.17629v1)** (2025.06)
    *   Focus: EVR uses egocentric video for complex instruction following and spatiotemporal reasoning.
    *   code: [https://github.com/Teacher-Tom/CLiViS](https://github.com/Teacher-Tom/CLiViS)
    *   citation: 6

*   **[Mamba-Enhanced Text-Audio-Video Alignment Network for Emotion Recognition in Conversations](http://arxiv.org/abs/2409.05243v1)** (2024.09)
    *   Focus: ERC research identifies and classifies speaker emotions in multimodal conversations.
    *   citation: 3

*   **[Efficient Long-distance Latent Relation-aware Graph Neural Network for Multi-modal Emotion Recognition in Conversations](http://arxiv.org/abs/2407.00119v2)** (2024.06)
    *   Focus: MERC analyzes emotional states in conversations using multi-modal data.
    *   citation: 25

*   **[EALD-MLLM: Emotion Analysis in Long-sequential and De-identity videos with Multi-modal Large Language Model](http://arxiv.org/abs/2405.00574v2)** (2024.05)
    *   Focus: Emotion AI research needs better multimodal fusion and temporal modeling for improved emotion understanding.
    *   citation: 10

*   **[EmpathicStories++: A Multimodal Dataset for Empathy towards Personal Experiences](http://arxiv.org/abs/2405.15708v1)** (2024.05)
    *   Focus: Existing empathy datasets lack interpersonal and experiential dimensions needed for AI modeling.
    *   project: [https://mitmedialab.github.io/empathic-stories-multimodal/](https://mitmedialab.github.io/empathic-stories-multimodal/)
    *   citation: 12

*   **[DIV-FF: Dynamic Image-Video Feature Fields For Environment Understanding in Egocentric Videos](http://arxiv.org/abs/2503.08344v1)** (CVPR2025 2025.03)
    *   Focus: Egocentric videos enable dynamic environment understanding for robotics and AR applications.
    *   citation: 0

*   **[MEGC2025: Micro-Expression Grand Challenge on Spot Then Recognize and Visual Question Answering](http://arxiv.org/abs/2506.15298v2)** (2025.06)
    *   Focus: Facial micro-expressions are involuntary facial movements during suppressed emotions.
    *   project: [https://megc2025.github.io](https://megc2025.github.io)
    *   citation: 0

*   **[A Space-Time Transformer for Precipitation Forecasting](http://arxiv.org/abs/2511.11090v1)** (2025.11)
    *   Focus: Traditional NWP models are static, but new AI methods improve real-time flood forecasting.
    *   code: [https://github.com/leharris3/satformer](https://github.com/leharris3/satformer)
    *   citation: 0

*   **[Gait Disorder Assessment Based on a Large-Scale Clinical Trial: WiFi vs. Video vs. Doctor's Visual Inspection](http://arxiv.org/abs/2502.05328v2)** (2025.02)
    *   Focus: This paper explores WiFi sensing for diagnosing neurological gait disorders.
    *   citation: 4

*   **[Temporal Perceiver: A General Architecture for Arbitrary Boundary Detection](http://arxiv.org/abs/2203.00307v2)** (2022.03)
    *   Focus: GBD detects general boundaries to segment videos into coherent units for preprocessing.
    *   citation: 19

*   **[AirLetters: An Open Video Dataset of Characters Drawn in the Air](http://arxiv.org/abs/2410.02921v1)** (2024.10)
    *   Focus: AirLetters is a dataset of human motion videos for predicting articulated letter gestures.
    *   citation: 1

*   **[Dynamic Gesture Recognition in Ultra-Range Distance for Effective Human-Robot Interaction](http://arxiv.org/abs/2407.21374v1)** (2024.07)
    *   Focus: Novel approach for ultra-range gesture recognition in human-robot interaction using video data.
    *   citation: 0

*   **[Multimodal Language Models for Domain-Specific Procedural Video Summarization](http://arxiv.org/abs/2407.05419v1)** (2024.07)
    *   Focus: Long-format video tutorials are effective for learning new skills.
    *   citation: 0

*   **[NOVIS: A Case for End-to-End Near-Online Video Instance Segmentation](http://arxiv.org/abs/2308.15266v2)** (2023.08)
    *   Focus: Recent findings challenge the belief that offline methods outperform online frame-by-frame processing in Video Instance Segmentation.
    *   citation: 8

*   **[VITA: Video Instance Segmentation via Object Token Association](http://arxiv.org/abs/2206.04403v2)** (NeurIPS2022 2022.06)
    *   Focus: A new offline VIS method uses object-oriented information to improve video context understanding.
    *   code: [https://github.com/sukjunhwang/VITA](https://github.com/sukjunhwang/VITA)
    *   citation: 121

*   **[Discriminative Spatial-Semantic VOS Solution: 1st Place Solution for 6th LSVOS](http://arxiv.org/abs/2408.16431v1)** (2024.08)
    *   Focus: The MOSE dataset addresses video object segmentation challenges in complex scenes and long motions.
    *   code: [https://github.com/yahooo-m/VOS-Solution](https://github.com/yahooo-m/VOS-Solution)
    *   citation: 0

*   **[Joint Modeling of Feature, Correspondence, and a Compressed Memory for Video Object Segmentation](http://arxiv.org/abs/2308.13505v2)** (2023.08)
    *   Focus: Video Object Segmentation methods use extraction-then-matching pipelines with independent feature extraction and dense matching.
    *   code: [https://github.com/MCG-NJU/JointFormer](https://github.com/MCG-NJU/JointFormer)
    *   citation: 0

*   **[Look Before You Match: Instance Understanding Matters in Video Object Segmentation](http://arxiv.org/abs/2212.06826v1)** (CVPR2023 2022.12)
    *   Focus: Memory-based methods use dense frame matching for long-range context in video object segmentation.
    *   citation: 52

*   **[FriendsQA: A New Large-Scale Deep Video Understanding Dataset with Fine-grained Topic Categorization for Story Videos](http://arxiv.org/abs/2412.17022v1)** (2024.12)
    *   Focus: VideoQA models struggle with complex questions despite good performance on factoid tasks.
    *   citation: 2

*   **[EVQAScore: A Fine-grained Metric for Video Question Answering Data Quality Evaluation](http://arxiv.org/abs/2411.06908v3)** (2024.11)
    *   Focus: Evaluating video QA and caption data quality is essential for training effective VideoLLMs.
    *   citation: 2

*   **[Zero-Shot Video Question Answering with Procedural Programs](http://arxiv.org/abs/2312.00937v1)** (2023.12)
    *   Focus: Proposes zero-shot video QA via procedural programs solving visual subtasks.
    *   project: [https://rccchoudhury.github.io/proviq2023](https://rccchoudhury.github.io/proviq2023)
    *   citation: 37

*   **[EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding](http://arxiv.org/abs/2308.09126v1)** (NeurIPS2023 2023.08)
    *   Focus: EgoSchema is a long-form video QA dataset and benchmark for evaluating video understanding systems.
    *   project: [http://egoschema.github.io](http://egoschema.github.io)
    *   citation: 463

*   **[Dense but Efficient VideoQA for Intricate Compositional Reasoning](http://arxiv.org/abs/2210.10300v1)** (2022.10)
    *   Focus: Proposes a new VideoQA benchmark with complex reasoning on long videos.
    *   citation: 4

*   **[Locate before Answering: Answer Guided Question Localization for Video Question Answering](http://arxiv.org/abs/2210.02081v2)** (2022.10)
    *   Focus: VideoQA is a key vision-language task with growing research interest.
    *   citation: 24

*   **[Structured Two-stream Attention Network for Video Question Answering](http://arxiv.org/abs/2206.01017v1)** (2022.06)
    *   Focus: Video QA remains a major challenge in vision-language understanding compared to image QA.
    *   citation: 70

*   **[MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](http://arxiv.org/abs/2510.21406v1)** (NeurIPS2025 2025.10)
    *   Focus: Proposes MUVR benchmark for multi-modal untrimmed video retrieval on long videos.
    *   code: [https://github.com/debby-0527/MUVR](https://github.com/debby-0527/MUVR)
    *   citation: 0

*   **[ViSMaP: Unsupervised Hour-long Video Summarisation by Meta-Prompting](http://arxiv.org/abs/2504.15921v1)** (2025.04)
    *   Focus: ViSMap is an unsupervised system for summarizing hour-long videos without supervision.
    *   citation: 0

*   **[FullTransNet: Full Transformer with Local-Global Attention for Video Summarization](http://arxiv.org/abs/2501.00882v2)** (2025.01)
    *   Focus: Video summarization creates compact synopses for efficient video browsing and analysis.
    *   code: [https://github.com/ChiangLu/FullTransNet](https://github.com/ChiangLu/FullTransNet)
    *   citation: 3

*   **[Do Language Models Understand Time?](http://arxiv.org/abs/2412.13845v3)** (2024.12)
    *   Focus: LLMs enhance video tasks like action recognition and anomaly detection despite unique challenges.
    *   code: [https://github.com/Darcyddx/Video-LLM](https://github.com/Darcyddx/Video-LLM)
    *   citation: 9

*   **[Video Repurposing from User Generated Content: A Large-scale Dataset and Benchmark](http://arxiv.org/abs/2412.08879v2)** (ICCV2025 2024.12)
    *   Focus: Short-form video demand grows, but current summarization methods remain inadequate.
    *   citation: 4

*   **[LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts](http://arxiv.org/abs/2505.13928v3)** (2025.05)
    *   Focus: Existing video-text retrieval benchmarks have limited video duration, hindering long video understanding.
    *   code: [https://github.com/TechNomad-ds/LoVR-benchmark](https://github.com/TechNomad-ds/LoVR-benchmark)
    *   citation: 2


### Others

*   **[Video Finetuning Improves Reasoning Between Frames](http://arxiv.org/abs/2511.12868v1)** (2025.11)
    *   Focus: Critiques naive frame token concatenation in video LLMs, proposing improved multimodal architectures.
    *   citation: 0

*   **[Dual Band Video Thermography Near Ambient Conditions](http://arxiv.org/abs/2509.11334v1)** (2025.09)
    *   Focus: Thermal camera images combine reflected environmental light and emitted surface radiation.
    *   citation: 0

*   **[RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement](http://arxiv.org/abs/2404.06483v2)** (2024.04)
    *   Focus: rPPG enables non-contact physiological signal measurement from facial videos for healthcare and affective computing.
    *   code: [https://github.com/zizheng-guo/RhythmMamba](https://github.com/zizheng-guo/RhythmMamba)
    *   citation: 10

*   **[The Dawn of Video Generation: Preliminary Explorations with SORA-like Models](http://arxiv.org/abs/2410.05227v2)** (2024.10)
    *   Focus: High-quality video generation methods like T2V, I2V, and V2V are important for content creation.
    *   project: [https://ailab-cvc.github.io/VideoGen-Eval/](https://ailab-cvc.github.io/VideoGen-Eval/)
    *   citation: 24

*   **[A Video Is Worth 4096 Tokens: Verbalize Videos To Understand Them In Zero Shot](http://arxiv.org/abs/2305.09758v3)** (2023.05)
    *   Focus: Multimedia content combines text, visuals, audio, and storytelling for creative expression.
    *   citation: 14

*   **[VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform](http://arxiv.org/abs/2504.14904v1)** (2025.04)
    *   Focus: SVPs struggle to moderate harmful content affecting minors' mental health.
    *   project: [https://kuaimod.github.io](https://kuaimod.github.io)
    *   citation: 5

*   **[Designing Loving-Kindness Meditation in Virtual Reality for Long-Distance Romantic Relationships](http://arxiv.org/abs/2309.11816v1)** (2023.09)
    *   Focus: Virtual reality may enable loving-kindness meditation for isolated couples in therapy.
    *   citation: 10

*   **[EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations](http://arxiv.org/abs/2209.13064v1)** (NeurIPS2022 2022.09)
    *   Focus: VISOR introduces a pixel-level annotation dataset and benchmark for hand and active object segmentation in egocentric video.
    *   project: [http://epic-kitchens.github.io/VISOR](http://epic-kitchens.github.io/VISOR)
    *   citation: 127

*   **[Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding](http://arxiv.org/abs/2505.23990v2)** (2025.05)
    *   Focus: Robots need adaptive decision-making and information filtering for effective human interaction.
    *   citation: 5

*   **[Long-Term 3D Point Tracking By Cost Volume Fusion](http://arxiv.org/abs/2407.13337v1)** (2024.07)
    *   Focus: Deep learning improves long-term point tracking for non-rigid motion understanding.
    *   citation: 0

*   **[PAD3R: Pose-Aware Dynamic 3D Reconstruction from Casual Videos](http://arxiv.org/abs/2509.25183v1)** (2025.09)
    *   Focus: PAD3R reconstructs deformable 3D objects from long, unposed monocular videos.
    *   project: [https://pad3r.github.io/](https://pad3r.github.io/)
    *   citation: 0

*   **[EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations](http://arxiv.org/abs/2209.13064v1)** (NeurIPS2022 2022.09)
    *   Focus: VISOR introduces a pixel-level annotation dataset and benchmark for hand and active object segmentation in egocentric video.
    *   project: [http://epic-kitchens.github.io/VISOR](http://epic-kitchens.github.io/VISOR)
    *   citation: 127

*   **[A Comprehensive Survey on World Models for Embodied AI](http://arxiv.org/abs/2510.16732v1)** (2025.10)
    *   Focus: World models simulate environment dynamics for embodied AI agents to perceive, act, and anticipate future states.
    *   code: [https://github.com/Li-Zn-H/AwesomeWorldModels](https://github.com/Li-Zn-H/AwesomeWorldModels)
    *   citation: 1

*   **[EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control](http://arxiv.org/abs/2508.21112v4)** (2025.08)
    *   Focus: Vision-language-action models enable embodied agents to perform multimodal reasoning and physical interaction.
    *   citation: 7

*   **[Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs](http://arxiv.org/abs/2407.07775v2)** (2024.07)
    *   Focus: Research aims to build agents that understand multimodal instructions for navigation.
    *   citation: 53

*   **[Spacewalk-18: A Benchmark for Multimodal and Long-form Procedural Video Understanding in Novel Domains](http://arxiv.org/abs/2311.18773v3)** (2023.11)
    *   Focus: Procedural videos help embodied agents learn skills from human demonstrations.
    *   project: [https://brown-palm.github.io/Spacewalk-18/](https://brown-palm.github.io/Spacewalk-18/)
    *   citation: 2

*   **[EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](http://arxiv.org/abs/2305.15021v2)** (NeurIPS2023 2023.05)
    *   Focus: Introduces Embodied AI for long-horizon robot task planning and execution in physical environments.
    *   citation: 334

*   **[Enhancing Object Search in Indoor Spaces via Personalized Object-factored Ontologies](http://arxiv.org/abs/2506.14422v1)** (2025.06)
    *   Focus: Service robots require personalized environmental understanding and change awareness for effective operation.
    *   citation: 1

*   **[From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models](http://arxiv.org/abs/2504.06214v1)** (2025.04)
    *   Focus: Long-context models are vital for document/video understanding, in-context learning, and scaling.
    *   project: [https://ultralong.github.io/](https://ultralong.github.io/)
    *   citation: 8

*   **[Long Context Transfer from Language to Vision](http://arxiv.org/abs/2406.16852v2)** (2024.06)
    *   Focus: Proposes methods to reduce video tokens for long video understanding in large multimodal models.
    *   code: [https://github.com/EvolvingLMMs-Lab/LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)
    *   citation: 320

*   **[Hallucination Mitigation Prompts Long-term Video Understanding](http://arxiv.org/abs/2406.11333v1)** (2024.06)
    *   Focus: Multimodal LLMs struggle with unprocessed long videos due to computational constraints.
    *   code: [https://github.com/lntzm/CVPR24Track-LongVideo](https://github.com/lntzm/CVPR24Track-LongVideo)
    *   citation: 5

*   **[VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment](http://arxiv.org/abs/2504.14096v3)** (2025.04)
    *   Focus: Introduces a new method to improve Video-LLMs' spatial, temporal, and cross-frame understanding.    
    *   citation: 4
