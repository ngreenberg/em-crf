# EM CRF and multi CRF for NER on disjoint sets of labels

The various exp.py files contain the most recent training code for each model. Each one corresponds to a different experiment, where a different amount of datasets might be used in different ways. Check out the training loops at the end of each file to see what each one trains on. Check the dp.read_file in the process data section to see how the different sets of labels are treated. An example command for running an experiment might be python exp.py -s 100 -p .3, where -s is the hidden size of the LSTM, and -p is the dropout probability. Use Python 3. For the multi CRF, the _merge files load a trained model and merge using the latest merging code. Other files may not merge or merge using older merging code.

[Marginal Likelihood Training of BiLSTM-CRF for Biomedical Named Entity Recognition from Disjoint Label Sets](http://www.aclweb.org/anthology/D18-1306)
