# Understanding the Circuitry Behind Brain-Alignment



# Computing Brain Alignment
- *Preprocessing:* normalize the entire fMRI data within-subject. Also, normalize the activation data
across all of the tokens (across the embedding dimension).
- Block the fMRI data into chunks (these chunks needs to be significantly large enough).
- Throw away the first and last few fMRI prediction points. 
- *Optional:* PCA down the embedding dimension.
- Use a context window of twenty words to predict the fMRI recording.
- Either average across the entire context window to get the embeddings or chunk it into four-word blocks, then concatenate.





