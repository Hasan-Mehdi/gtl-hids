# GTL-HIDS: Generative Tabular Learning-Enhanced Intrusion Detection System

## Overview

GTL-HIDS is a novel intrusion detection system that leverages embeddings from finetuned Large Language Models (LLMs) to enhance network security. This project bridges the gap between structured tabular data and natural language processing to enable more effective detection of both known attack patterns and zero-day threats.

Our approach enhances network intrusion detection by converting tabular network data into language model embeddings, enabling more effective pattern recognition and anomaly detection.

## Project Goals

- Develop novel techniques for applying LLMs to tabular data
- Establish effective data representation strategies for tabular-to-text conversion
- Demonstrate improved zero-shot and few-shot learning capabilities on tabular data

## Architecture

GTL-HIDS takes a novel approach to intrusion detection by converting network flow data into language-friendly representations and leveraging the pattern recognition abilities of finetuned LLM embeddings.

### Neural Network Design

The system consists of three main components:

1. **Data Transformation Module**: Converts network traffic data into optimized text representations that capture the semantic meaning of network flows.

2. **LLM Embedding Layer**: Uses a finetuned LLaMA model to generate rich, contextual embeddings from the text representations.

3. **Neural Classification Network**: A specialized neural network trained on these embeddings to classify network traffic as benign or malicious.

The system is designed to recognize patterns that traditional detection systems might miss, particularly in the case of zero-day attacks that don't match known signatures.

## Implementation Details

Our GTL-HIDS implementation utilizes a finetuned LLaMA model to generate embeddings from network flow data. These embeddings capture both the statistical properties of the network traffic and the contextual understanding provided by the language model.

We use a custom neural network architecture that processes these embeddings to identify malicious patterns. The network is implemented using PyTorch and optimized for both accuracy and inference speed.

### Key Components

- **Text Representation Generator**: Converts network flow data into descriptive text using carefully designed templates that highlight security-relevant features.

- **Embedding Generator**: Uses LLaMA to generate rich vector representations of the textual network descriptions.

- **Neural Classifier**: A multi-layer neural network that processes these embeddings to make final classification decisions.

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

### Finetuned Models:

8B_Finetuned_DatasetA : Hmehdi515/gtl-hids-binary-A
8B_Finetuned_DatasetB : Hmehdi515/gtl-hids-binary-B

## Advantages of Our Approach

Our neural network approach leverages the rich understanding of language models while providing the speed and efficiency needed for real-time security applications. By using embeddings from a finetuned LLaMA model, our system can:

1. Understand the semantic meaning of network communications
2. Recognize novel attack patterns without prior exposure
3. Adapt to changing network environments with minimal retraining

Traditional IDS systems rely on fixed rules or statistical patterns that can be easily bypassed by new attack techniques. Our approach brings natural language understanding to network security, enabling more robust threat detection.

## Future Work

- Enhance the model with larger language models and more diverse training data
- Implement attention mechanisms to focus on the most security-relevant aspects of network flows
- Develop explainable AI components to help security analysts understand detection decisions
- Expand to additional network protocols and environments

## Citation

If you use GTL-HIDS in your research, please cite our work:

```
@inproceedings{GTL-HIDS2025,
  title={Generative Tabular Learning-Enhanced Intrusion Detection System},
  author={Nair, Vaishak and Varaganti, Basanth and Mehdi, Hasan and Alam, Tanvirul and Rastogi, Nidhi},
  booktitle={Proceedings of the International Conference on Software Engineering},
  year={2025}
}
```

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
