# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest Classifier model trained on 1994 census data 
## Intended Use
predictions whether a person earns over 50k or not based on the census data. Model is ready for online inference and batch predictions.
## Training Data
1994 census data. The data was collected from the 1994 Census database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics).
## Evaluation Data
Evaluation data is the 20% of full downloaded data by using `train_test_split` function from scikit-learn. Model has been evaluated by *fbeta*, *precision* and *recall* scores.
## Metrics
The model performance is evaluated using three standard classification metrics:

Precision: Measures the proportion of positive predictions that are actually correct
Recall: Measures the proportion of actual positives that are correctly identified
F1 Score: Harmonic mean of precision and recall, providing a balanced measure
Overall Test Set Performance:

Precision: ~0.74
Recall: ~0.63
F1 Score: ~0.68
Slice-Based Evaluation: The model's performance was evaluated across different slices of categorical features to identify potential disparities. Performance metrics vary across different demographic groups and feature values. Detailed slice-based metrics are available in model/slice_output.txt.

Note: Actual performance metrics will vary based on the random state and specific training run. The values above are approximate expected ranges.

## Ethical Considerations
This model is trained on census data. The model is not biased towards any particular group of people.

## Caveats and Recommendations
This model is not suitable for real-time predictions. It is suitable for batch predictions.