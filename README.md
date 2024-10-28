# Collaboration Networks in Brazilian Postgraduate Programs in Computing

This repository contains the code and data used in my monograph, which explores the collaboration networks of Brazilian authors and their relationship with CAPES evaluation in the context of graduate programs in Computing in Brazil.

The full text of the monograph is available at [http://www.monografias.ufop.br/handle/35400000/7078] (available in PT-BR only).

## Overview

Academic collaboration is essential for scientific advancement, especially in Computing, where research networks can impact productivity and publication quality. However, there is a lack of systematic analysis of these networks within Brazilian graduate programs. This work aims to fill this gap by examining co-authorship patterns among Brazilian researchers and understanding how factors such as geographic location and academic productivity influence the formation of collaboration networks.

The methodology includes:
1. **Graph Construction**: A graph is created from publication metadata to represent collaboration networks.
2. **Node Embedding**: Using the Node2Vec algorithm, we generate embeddings for each author (node) to capture their position and influence within the network.
3. **Pattern Identification**: Unsupervised learning techniques are applied to identify collaboration patterns within the data.

The results validate the effectiveness of our approach, revealing that geographic and productivity factors are determinants in collaboration network formation. These insights can help inform academic practices and improve the evaluation processes for graduate programs.

## Repository Contents

- `data/`: Folder for storing data files. The dataset used in this study can be downloaded [here](https://docs.google.com/spreadsheets/d/1aDyvwiUHiDZre47Z0AOml0D7gS17mgfFFbqSJ6Svi64/export?format=csv&gid=716386560).
- `src/`: Scripts for preprocessing data, running Node2Vec, and applying clustering algorithms.
- `results/`: Outputs of the analysis, including visualizations and pattern clusters.

## Installation

To run the code in this repository, you'll need to set up a Python environment and install the required libraries. We recommend using `venv` or `conda`.

```bash
# Clone the repository
git clone https://github.com/username/repo-name.git
cd repo-name

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Download the Dataset**: Place the downloaded dataset in the `data/` folder.
2. **Generate Results**: Use scripts in `src/` to preprocess data, execute Node2Vec, and apply clustering methods on node embeddings.

## Results

This analysis shows that geographic location and academic productivity play key roles in the formation of collaboration networks among Brazilian authors. The findings suggest that collaboration network analysis can provide valuable insights for improving academic practices and the evaluation of graduate programs in Computing.

## Keywords

- Science of Science
- Network Science
- Artificial Intelligence
- Embeddings
- Clustering
- Machine Learning
- Supervised Learning
- Unsupervised Learning

---

Cite this thesis:

### English Version
```bibtex
@thesis{icarovasconcelos2024,
  author = {Ícaro Luiz Lage Vasconcelos},
  title = {Mapping Academic Connections: An Analysis of the Computer Science Researcher Network via OpenAlex},
  year = {2024},
  school = {Federal University of Ouro Preto},
  type = {Monograph (Bachelors in Computer Science)},
  address = {Ouro Preto, BR},
  institution = {Institute of Exact and Biological Sciences},
  url = {http://www.monografias.ufop.br/handle/35400000/7078},
  note = {51 f.}
}
```

#### Portuguese Version
```bibtex
@thesis{icarovasconcelos2024,
  author = {Ícaro Luiz Lage Vasconcelos},
  title = {Mapeando as Conexões Acadêmicas: Uma Análise da Rede de Pesquisadores em Ciência da Computação via OpenAlex},
  year = {2024},
  school = {Universidade Federal de Ouro Preto},
  type = {Monografia (Graduação em Ciência da Computação)},
  address = {Ouro Preto, BR},
  institution = {Instituto de Ciências Exatas e Biológicas},
  url = {http://www.monografias.ufop.br/handle/35400000/7078},
  note = {51 f.}
}
```