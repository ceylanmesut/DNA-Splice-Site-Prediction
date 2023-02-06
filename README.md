## K-Mer Based DNA Splice Site Classification with Dilated CNN and LSTM Architectures
This repository contains the second project of Machine Learning for Healthcare course at ETH Zürich.

<p align="center" width="100%">
    <img width="35%" src="plots\dna2.PNG">
    <img width="45%" src="plots\dna.png">
</p>

### Project Goal
The goal of this project is to develop various ML and DL models to tackle binary classification problem on DNA sequences whether they are acceptor site.

There are two highly imbalanced dataset at hand for the model development:
- Human Dataset: 531,777 negative, 1556 positive samples
- Worm Dataset: 2000 negative samples, 200 positive samples.

### Respiratory Structure
- `data`: A directory containing the datasets.
- `human_models`: A directory including trained models on human dataset.
- `worm_models`: A directory including trained models on worm dataset.
- `models`: A directory containing Python snippets for baseline CNN and RNN as well as the advanced models.
- `plots`: The plots regarding model performance and readme.
- `src`: A directory containing utility Python snippets.
- `main.py`:A Python snippet for model training and optimization.

### Script
1. Install `virtualenv`:

   ```pip install virtualenv```

2. Create virtual environment `myenv`:
    ```virtualenv myenv```

3. Activate the environment:

    ```myenv\Scripts\activate```

4. Install the required packages:

    ```pip install -r requirements.txt```

5. Run the model training code:

    ```python main.py --model_name Dilated_CNN --experiment HUMAN --sample_amount_tr 20000 --sample_amount_val 10000 --kmers 4 --learning_rate 0.0005 --epochs 100 --batch_size 128 --output_dim 200 --dilation_rate 2```

### Results
Prior to the model development, I processed DNA sequences, extracted K-Mers from each sequence and tokenized them. Based on my experiments, 4-Mers yields the most robust subsequences.

Following, I developed two fundamental deep learning algorithms to tackle this problem, Convolutional Neural Network and Recurrent Neural Network. Both network utilizes Embedding Layer as initial layer of the neural networks, followed by either convolution/dilated convolution layers or Simple RNN or LSTM layers.   

The performance of the networks on human dataset are demonstrated at below.

<p align="center" width="100%">
    <img width="100%" src="plots\results.PNG">
    </p>
The performance of the networks on worm dataset are demonstrated at below.
<p align="center" width="100%">
    <img width="100%" src="plots\results_worm.PNG">
</p>

### License
This work is licensed under MIT License, however it is subject to author approval and permission for public and private usage.
