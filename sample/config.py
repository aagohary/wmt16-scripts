import numpy
import os
import sys

sys.path.append('/fs/clip-xling/neuralMT/download/nematus/nematus')

VOCAB_SIZE = 90000
SRC = "fr"
TGT = "en"
DATA_DIR = "data/"
TRAIN = "train"
DEV = "newstest2013"

from nematus.nmt import train


if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    init_accumulators_path='model/accumulators.iter440000.npz',
                    model_reload_path='model/model.iter440000.npz',
                    dim_word=500,
                    dim=1024,
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=50,
                    batch_size=80,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + '/' + TRAIN + '.bpe.' + SRC, DATA_DIR + '/' + TRAIN + '.bpe.' + TGT],
                    valid_datasets=[DATA_DIR + '/' + DEV + '.bpe.' + SRC, DATA_DIR + '/' + DEV + '.bpe.' + TGT],
                    dictionaries=[DATA_DIR + '/' + TRAIN + '.bpe.' + SRC + '.json',DATA_DIR + '/' + TRAIN + '.bpe.' + TGT + '.json'],
                    validFreq=10000,
                    dispFreq=1000,
                    saveFreq=10000,
                    sampleFreq=10000,
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
