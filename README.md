# NLP_HW2_Phacha
HW2 NLP_W26 class
# NLP Homework 2: Word Embeddings

## Project Structure and Tasks

The codebase is divided into four primary tasks:

### Task 1: Training Word Embeddings
* **Dataset:** Simple English Wikipedia (`wikimedia/wikipedia`, `20231101.simple` snapshot) loaded via Hugging Face. 
* **Models:** Continuous Bag of Words (CBOW) and Skip-gram neural networks.
* **Objective:** Train custom embedding matrices from scratch to capture semantic relationships and word contexts from a restricted, basic English vocabulary.

### Task 2: Comparing Word Embeddings
* **Models:** Google News 300 and GloVe 100.
* **Objective:** Load and prepare established, pre-trained embedding spaces for comparative analysis against the custom-trained Wikipedia models.

### Task 3: Bias Evaluation (WEFE)
* **Framework:** Word Embeddings Fairness Evaluation (WEFE).
* **Metric:** Word Embeddings Association Test (WEAT).
* **Objective:** Quantify systemic and implicit biases within the embedding spaces. The custom analysis specifically measures the association between **Age vs. Technology**, utilizing custom target word lists (incorporating modern terms like 'ai' and 'coding').

### Task 4: Text Classification (Sparse vs. Dense)
* **Dataset:** Rotten Tomatoes movie reviews (binary sentiment analysis: Positive vs. Negative) via Hugging Face.
* **Methodology:** A comparative analysis of two Logistic Regression classifiers:
  * **Sparse Model:** Uses a Discrete Bag of Words representation (`CountVectorizer` limited to 10,000 features).
  * **Dense Model:** Uses a Continuous Bag of Words representation (100 features), mathematically averaging the document's word vectors using `np.mean(axis=0)`.



## Installation and Setup

Ensure your environment has the required dependencies. Due to compatibility requirements with the WEFE framework, a specific version of NumPy is required.

⚠️ Note on WEFE Compatibility: This notebook runs perfectly in Google Colab using the default environment. However, if you are running this locally or encounter compatibility errors with the WEFE framework, you may need to force-install a specific, stable version of NumPy to resolve the conflict:


pip install datasets wefe scikit-learn
pip install numpy==1.26.4
