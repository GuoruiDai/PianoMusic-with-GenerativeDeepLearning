# PianoMusic-with-GenerativeDeepLearning
Repository for a deep learning project in generating piano music using transformer model.

This project utilizes publicly available MIDI piano data, which is filtered and converted into a compact pitch-time representation. A small portion of this dataset is manually labeled with quality scores to train an evaluator model, which then auto-labels the remaining data.
The evaluator incorporates learnable embeddings—specifically relative pitch consonance, relative temporal consonance decay, and pitch repetition scores—integrated through a graph-based message-passing layer. Performance is measured using the train/validation loss ratio, AUPRC, AUROC, and F1 score. Finally, the highest-quality data is used to train the composer model for piano music generation.


Some good generation samples:
[sample_h_1.mp3](https://github.com/user-attachments/files/27366354/sample_h_1.mp3)
[sample_h_2.mp3](https://github.com/user-attachments/files/27366357/sample_h_2.mp3)


Samples with lower quality:
[sample_l_1.mp3](https://github.com/user-attachments/files/27367527/sample_l_1.mp3)
[sample_l_2.mp3](https://github.com/user-attachments/files/27368145/sample_l_2.mp3)


NOTE: The training data, processing code, training code, evaluator model are not released.
