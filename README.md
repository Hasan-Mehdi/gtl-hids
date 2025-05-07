# GTL-HIDS: Generative Tabular Learning-Enhanced Hybrid Intrusion Detection System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

GTL-HIDS is a novel hybrid intrusion detection system that combines the strengths of Large Language Models (LLMs) and traditional Machine Learning to enhance network security. This project bridges the gap between structured tabular data and natural language processing to enable more effective detection of both known attack patterns and zero-day threats.

Our approach enhances network intrusion detection using LLMs, bridges structured data and language models, and enables zero-shot attack detection.

## Project Goals

- Develop novel techniques for applying LLMs to tabular data
- Establish effective data representation strategies for tabular-to-text conversion
- Demonstrate improved zero-shot and few-shot learning capabilities on tabular data

## Architecture

Our GTL-HIDS tackles the main problems in today's intrusion detection systems by using a dual-pathway design. Right now, security systems either just use machine learning, which can't catch new zero-day attacks, or try to use LLMs on network data directly, which isn't accurate enough for security. We built our hybrid system to fix these issues by taking the best parts of both approaches while avoiding their weaknesses.

### Dual Pathway Design

![Architecture Diagram](assets/architecture.png)

Our domain model shows how GTL-HIDS uses two main pathways, all controlled by a Hybrid Controller at the top:

1. **Template Generation Pathway**: The Template Generator takes network data and turns it into text that LLMs can understand. This Generator works together with an LLM to process the text and get analysis, while keeping all its data organized in the Template Store.

2. **Detection Pathway**: The Detection System pairs up with an ML Pipeline to look for patterns and predict threats. All the results and past findings stay in the Detection Store.

The Hybrid Controller connects these two paths, making sure data goes where it needs to.

## Implementation Details

Our GTL-HIDS implementation solves hybrid intrusion detection challenges by coordinating separate LLM and ML pathways, addressing the slow processing and unreliability of current systems. Our implementation brings both pathways together through a FastAPI backend that handles all the coordination.

We use a Hybrid Controller built with Python's asyncio library because regular synchronous processing can't handle our dual-pathway design. The Controller manages two separate processing queues, making sure neither pathway gets overwhelmed. When network traffic comes in, the Controller immediately splits it between our Template Generator and Detection System, each handling the data in their specialized way.

### Key Components

- **Template Generator**: Uses LLaMA-2 with 13 billion parameters to capture subtle patterns in network traffic. Implemented using PyTorch and Accelerate with torch.bfloat16 precision for memory efficiency.

- **ML Pipeline**: XGBoost-based detection system that processes network traffic through multiple analytical stages, capturing both low-level packet patterns and high-level behavioral indicators.

## Dataset

We use the CIC-IDS2017 dataset, which is a temporal-based network traffic dataset containing 5 days of activity (Monday-Friday) with diverse attacks in a realistic network topology.

The CICIDS2017 dataset contains benign and the most up-to-date common attacks, which resembles true real-world data (PCAPs). It includes the results of network traffic analysis with labeled flows based on time stamp, source and destination IPs, ports, protocols and attack types.

### Attack Types Include:

- DDoS
- DoS
- Web Attacks (Brute Force, XSS, SQL Injection)
- Infiltration
- Botnet
- Port Scanning

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GTL-HIDS.git
cd GTL-HIDS

# Create a virtual environment
conda create -n gtlhids python=3.8
conda activate gtlhids

# Install dependencies
pip install -r requirements.txt
```

### Running Zero-Shot Evaluation

```bash
python base_eval.py --test_path data/test.csv --output_dir results --batch_size 32
```

### Training the Model

```bash
python train.py
```

## Results

Our model demonstrates complementary strengths in threat detection when compared across various model sizes (8B vs 70B) and against traditional ML approaches.

F1 Scores for our Model:
- Temporal Split (Multiclass): 0.2
- Random Split (Multiclass): 0.85
- Temporal Split (Binary): 0.81
- Random Split (Binary): 0.99

## Advantages of Our Approach

Our approach surpasses traditional methods by addressing the weaknesses of current intrusion detection systems. Regular systems only use ML, which misses new attacks, or try to use LLMs directly, which isn't precise enough for security.

Our design choices enable us to spot both known threats and zero-day attacks using zero-shot learning, where our LLMs can identify new attack patterns they've never seen before.

## Future Work

- Enhance the system with reinforcement learning from human feedback
- Expand to additional network protocols and environments
- Investigate multilingual support for international deployment

## Citation

If you use GTL-HIDS in your research, please cite our work:

```
@inproceedings{GTL-HIDS2025,
  title={Generative Tabular Learning-Enhanced Hybrid Intrusion Detection System},
  author={Nair, Vaishak and Varaganti, Basanth and Mehdi, Hasan and Alam, Tanvirul and Rastogi, Nidhi},
  booktitle={Proceedings of the International Conference on Software Engineering},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Rochester Institute of Technology
- The contributors of the CIC-IDS2017 dataset
- The authors of the research papers that inspired this work

## Team

- Vaishak Nair
- Basanth Varaganti
- Hasan Mehdi
- Tanvirul Alam
- Nidhi Rastogi
