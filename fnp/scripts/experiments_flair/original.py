from flair.data import Corpus
from flair.datasets import TREC_6, CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# 1. get the corpus
# this is the folder in which train, test and dev files reside
data_folder = '/media/sarthak/HDD/data_science/fnp/resources'

# column format indicating which columns hold the text and label(s)
column_name_map = {1: "text", 2: "label_causal"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         train_file='fnp2020-fincausal-task1-train.csv',
                                         test_file='fnp2020-fincausal-task1-test.csv',
                                         dev_file='fnp2020-fincausal-task1-val.csv',
                                         skip_header=True,
                                         delimiter=';')

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),
                   FlairEmbeddings('news-forward'),
                   FlairEmbeddings('news-backward'),
                   ]

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                   hidden_size=512,
                                                                   reproject_words=True,
                                                                   reproject_words_dimension=256,
                                                                   )

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/4/',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_weights('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/4/weights.txt')