# Predicting Bank Account Ownership in East Africa

For my Machine Learning summative at ALU, I wanted to work on something close to home. In Kenya, Rwanda, Tanzania and Uganda, only about 14% of adults in this survey have a bank account. I wanted to see if basic survey answers (age, education, job, where someone lives, whether they have a phone) can predict who has one, and more importantly, who gets left out.

I built and compared two kinds of models: traditional machine learning with scikit-learn, and deep learning with TensorFlow/Keras.

## Links

Notebook: [financial_inclusion_classification.ipynb](financial_inclusion_classification.ipynb)

Google Drive folder (raw data and saved models): https://drive.google.com/drive/folders/1Zjr3YV51GeqMYRp6WeGpWEgjDX_MeC4C?usp=sharing

Video presentation: (link coming)

## The data

The data comes from the FinScope household surveys run by the Financial Sector Deepening network between 2016 and 2018. It was released publicly through the Zindi "Financial Inclusion in Africa" challenge, and I downloaded it from a Kaggle mirror (`Financial_inclusion_dataset.csv`).

It has 23,524 people and 13 columns, with no missing values. The target is `bank_account` (Yes or No), and only 14.08% of people said Yes. The other columns are country, year, location type, cellphone access, household size, age, gender, relationship with the head of household, marital status, education level and job type.

I didn't put the raw data in this repo. The zipped dataset is in the Google Drive folder linked above.

A few things I found while exploring the data that shaped my choices:

- Each country was only surveyed in one year (Rwanda 2016, Tanzania 2017, Kenya and Uganda 2018), so `year` doesn't add anything on its own.
- About 19% of rows look like duplicates once you remove the ID, but they're actually different people who just gave the same answers. I kept them.
- 484 of those identical profiles have both Yes and No answers. No model can tell those people apart, so there's a limit on how well any model can do here.

## What I did

I split the data 70/15/15 (train, validation, test), keeping the same 14% Yes rate in each part. All the preprocessing (capping outliers, grouping rare categories, encoding and scaling) sits inside one sklearn pipeline that only learns from the training set, so nothing leaks from validation or test.

I also added some features based on what I saw in the EDA: age bands, a flag for household heads, a flag for formal jobs, a phone plus urban combination, and education as an ordered rank.

Then I ran 11 experiments (21 runs in total):

- E1 to E6 are the traditional models: Logistic Regression, Random Forest, XGBoost and SVM, tuned with grid search and randomized search. In E6 I compared ways of dealing with the imbalance: class weights, SMOTE, and tuning the decision threshold.
- E7 to E11 are the neural networks: MLPs built with the Sequential API and an embedding model built with the Functional API, all fed through `tf.data` pipelines. I tried dropout, L2, early stopping, different learning rates and batch sizes, class weights, and threshold tuning.

I tuned everything on the validation set and only used the test set once at the end. Every run is saved to a results CSV and to TensorBoard, so the numbers in my report can be traced back to actual runs.

## Results on the test set

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression baseline (E1) | 0.50 | 0.695 | 0.358 | 0.473 | 0.851 | 0.554 |
| XGBoost with tuned threshold (E6b) | 0.25 | 0.486 | 0.624 | 0.546 | 0.867 | 0.580 |
| MLP with tuned threshold (E11b) | 0.30 | 0.514 | 0.576 | 0.543 | 0.866 | 0.582 |

What surprised me most:

- XGBoost and the neural network ended up almost exactly the same. Neither one clearly beat the other.
- Changing the decision threshold helped more than any model choice or hyperparameter search I did.
- Every model got stuck around 0.60 PR-AUC no matter what I changed, which fits with those 484 conflicting profiles. I think richer data (income, mobile money use, distance to a bank) would help more than a fancier model.
- The models do worst for rural people, women, people with little education, and Rwanda. Those are exactly the people financial inclusion work is trying to reach, so I'd be careful about using a model like this in the real world without fixing that first.

## How to run it

1. Open the notebook in Google Colab.
2. Copy my Drive folder into your own Google Drive and keep the zipped dataset inside the `data` subfolder.
3. In the notebook, change `BASE` to wherever you put the folder, for example `/content/drive/MyDrive/ML_Summative_Financial_Inclusion`.
4. Run all the cells from top to bottom. I fixed the random seed (`SEED = 42`), so you should get the same results.

If you want to run it on your own machine instead:

```bash
pip install -r requirements.txt
```

Then swap the Google Drive cell for a local path to the data.

## Saved models

These are in the `models` folder on Google Drive:

| File | What it is |
|---|---|
| `xgboost_e6b.pkl` | Full pipeline (preprocessing + XGBoost) with its threshold |
| `xgboost_e6b_model.json` | Just the XGBoost model, in XGBoost's own format |
| `mlp_e11b.keras` | My best Keras model |
| `mlp_e11b_preprocessor.pkl` | Preprocessing pipeline and threshold for the Keras model |

The `.pkl` files use my custom `Cleaner` class and `add_features` function, so you need to run those cells in the notebook before loading them with `pickle.load`.

## TensorBoard logs

The logs are in the `logs` folder on Drive. To open them in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/ML_Summative_Financial_Inclusion/logs
```

## Repo structure

```
financial-inclusion-classification/
├── financial_inclusion_classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Audric, Software Engineering, African Leadership University (Kigali)
