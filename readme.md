# NLP Course Project README

## Project Overview
This project is part of an NLP course designed to develop a comprehensive understanding of natural language processing techniques and their applications. The project involves implementing various NLP models and techniques using Python, with a focus on PyTorch.

## Project Structure
The project directory is organized as follows:

```
.
├── data
│   └── SemEval-Triplet-data
├── project_structure.txt
├── readme.md
├── test1-sentiment
│   ├── bert- feature3.py
│   ├── bert-feature.py
│   ├── bert-feature2.py
│   ├── bert-vec.py
│   ├── gloVe- feature.py
│   ├── gloVe-feature2.py
│   ├── gloVe-vec.py
│   └── test-system.py
├── test2-Syntax
│   ├── __pycache__
│   ├── test-GAT.py
│   ├── test-ltp.py
│   └── test-spacy.ipynb
├── test3
│   ├── bert-spacy.py
│   ├── bert-test.py
│   ├── demo.py
│   ├── demo_dataset.csv
│   ├── gloVE-test.py
│   ├── gloVe-spacy.py
│   └── test.py
└── test4
    ├── bert- TripletExtraction.py
    ├── bert-model.py
    ├── dependency_tree_1.svg
    ├── dependency_tree_2.svg
    ├── results
    ├── spacy-gloVe-bert-demo.ipynb
    ├── spacy-practice.py
    ├── test-bertloss.py
    ├── test-readdata.py
    ├── test-textblob-spacy.py
    ├── test.py
    └── textblob-practice.py

9 directories, 31 files

```

## Requirements
- Python 3.11
- macOS
- PyTorch (latest version from the official website)
- Other dependencies as listed in `requirements.txt`

## Installation
To set up the project environment, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/nlp-course-project.git
    cd nlp-course-project
    ```

2. **Create a virtual environment:**
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install PyTorch:**
    Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the latest version of PyTorch for macOS.

## Usage
### Data Preprocessing
Run the preprocessing script to prepare the data for model training:
```bash
python scripts/preprocess.py
```

### Model Training
Train the NLP models by running the training script:
```bash
python scripts/train.py
```

### Model Evaluation
Evaluate the trained models using the evaluation script:
```bash
python scripts/evaluate.py
```

## Jupyter Notebooks
Explore the data and visualize results using the provided Jupyter notebooks in the `notebooks` directory. To start Jupyter Notebook, run:
```bash
jupyter notebook
```

## Contributing
If you would like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
We would like to thank the instructors and teaching assistants of the NLP course for their guidance and support throughout this project.

For more details and documentation, please refer to the [project webpage](https://boots-coder.github.io/2024/06/14/nlp%E8%AF%BE%E7%A8%8B%E8%AE%BE%E8%AE%A1%E6%96%87%E7%A8%BF%E4%B8%8E%E5%A4%A7%E7%BA%B2%E8%AE%BE%E8%AE%A1/).

---

If you have any questions or need further assistance, please contact the project maintainer at your-email@example.com.

Happy coding!