# OCR assignment report

## Feature Extraction (Max 200 Words)
I used PCA combined with feature selection to obtain my 10-dimensional feature vector, because combined with NN classification, it was 
an effective OCR system. Because the data set was relatively large and well-correlated, PCA was a a good choice of dimensionality reduction.
The steps in my feature extraction were:
- Calculate eigenvectors and means for feature vectors and input into the model
- Project feature vectors (`fvectors_train_full`) onto principal axes and centre the data to get the `pcatrain_data`
- Get the 10 best features from `pcatrain_data` using the divergences and feature selection and store the values from `pcatrain_data`
    corresponding to the best features into the model
- Project the test data feature vectors (`fvectors_test`) onto the PC axes and reduce the data by getting the values corresponding 
    to the best features to produce the 10-dimensional feature vector

## Classifier (Max 200 Words)
Because I used PCA for my feature extraction I felt that the most suitable classifier to use was Nearest Neighbour Classification.
I had a separate method `classify` that would do the NN Classification, which was called in `classify_page` to classify the test data.
Because NN Classification was simple - yet effective - for my system, I decided to use it, however, implementing K-NN would most likely
have improved the systems' classification scores.

## Error Correction (Max 200 Words)
For error correction I used a wordlist to check for errors in the output labels.
Logic of my error correction:
- Find word start and endings from bbox coordinates and spaces between them - x1 coord of one word and x2 coord of next word usually had a 
    distance difference > 10
- From word start/end information, add lengths of all words to an array of word lengths (`word_lengths`)
- Using output labels (`labels`) and knowledge of word lengths, form the letters in the page and add the words to a separate array of output
    words (`output_words`) 
- Remove words < 4 characters in `output_words` - to reduce errors with multiple letters - iterate through the words to check for 
    words not in wordlist in the output words array
- If word is not in the wordlist, find closest word to it by looking at the letter that causes the mismatch, then amend the label
- Add amended labels to new array, `correct_labels`, which is returned
However, running the code would've been slow due to iterating through all of the words in the wordlist to check if a word was in the list.

## Performance
The percentage errors (to 1 decimal place) for the development data are as follows:
- Page 1: 97.8%
- Page 2: 97.1%
- Page 3: 83.3%
- Page 4: 67.1%
- Page 5: 51.6%
- Page 6: 39.6%

## Other information (Optional, Max 100 words)
In `images_to_feature_vectors` I added multidimensional image processing (from scipy) on all of the images that greatly improved my scores.
I tested multiple filters e.g maximum, minimum, median and gaussian filters, but ultimately I decided on the median filter because I found 
that median gave the greatest improvement in score.
For the median filter, I also experimented with different size values ranging from 0-10, but ultimately I decided on 3 as it gave the most 
improvement in score.