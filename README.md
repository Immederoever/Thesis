This project tries to recreate an experiment that was done by Waite et al. (2025): comparing how well children and adults can predict an upcoming word in a story. 
The results of this test made clear that adults are better in predictive processing than children.
This improvement in prediction with age demonstrates the developmental trajectory of humans' predictive processing qualities. 
The difference between children and adults suggests that lexical prediction improves with more 'training': living longer and thus having more experience with language.
The test that was done by \citeauthor{Waite2025} will be recreated in this project. 
The difference will be that there will not be human participants, but LMs with incremental amounts of training data to simulate the difference in age. 
This will hopefully give some insight in the question whether LMs have the same developmental trajectory in predictive processing as humans.

The models have a GPT-2 style architecture and are trained from scratch. The child model is trained on part of the BabyLM corpus, and the adult model starts from the child model's checkpoints and
trains on part of the BookCorpusOpen.
