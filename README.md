# CT-based COVID-19 Severity Prediction

An educational machine learning project that predicts severe versus non-severe outcomes for COVID-positive patients using chest CT features from STOIC2021. This is a simplified project inspired by CT-based patient triage research; it does not predict ICU admission, ventilation, and death separately.

## Pipeline

Chest CT → pretrained lungmask R231 lung segmentation → 11 numerical features → standardization → logistic regression → severe/non-severe prediction.

Features include lung volume, intensity statistics, histogram entropy, and four custom 2D texture features. These are a simple radiomics baseline, not a complete standardized 3D radiomics implementation.

## Dataset

[STOIC2021](https://stoic2021.grand-challenge.org/stoic-db/) provides CT scans and outcome labels. Only COVID-positive patients are used. Severe means intubation or death within one month; non-severe means neither outcome during that period.

The notebook downloads labels and selected scans from the public STOIC2021 S3 bucket. Dataset files are not included in this repository. Review the dataset's CC BY-NC 4.0 terms; the notebook downloads its `LICENCE.txt`. Dataset and pretrained model licensing remain separate from repository code.

## Run in Google Colab

1. Upload `covid19_severity_prediction.ipynb` to Google Colab.
2. Select a GPU runtime, such as a T4.
3. Run the cells in order and authorize mounting your Google Drive.
4. Files are saved under `MyDrive/covid_ct_project`.

The notebook installs dependencies, downloads two demonstration scans, processes a 100-patient cohort, trains and tunes the classifier, and evaluates it on 40 additional patients. CT processing takes much longer than classifier training and requires several GB of cloud storage. Processing progress is saved after each patient.

After a runtime reset, restore packages, imports, paths, and helper functions. Saved feature rows can be reused without repeating segmentation. Rerunning training cells refits and overwrites their saved models. Google Colab supplies PyTorch; `requirements.txt` lists the dependencies for reference. Versions other than lungmask are not locked to the original runtime.

## Training and evaluation

- Initial cohort: 100 patients, with 75 non-severe and 25 severe outcomes.
- Original stratified split: 80 training patients and 20 initial test patients.
- Five-fold cross-validation on the 80 training patients selects logistic regression settings by balanced accuracy. Scaling occurs inside the pipeline, separately within each fold.
- Selected settings: `C=0.1`, `class_weight="balanced"`.
- The tuned model is fitted on the 80 training patients and evaluated at a fixed 0.5 threshold on 40 previously unused patients: 30 non-severe and 10 severe.

## Final internal evaluation

| Metric | Result |
|---|---:|
| Accuracy | 77.5% |
| Balanced accuracy | 71.7% |
| ROC-AUC | 0.743 |
| Average precision | 0.619 |
| Severe recall | 60.0% |

Confusion matrix, with rows representing actual outcomes and columns representing predicted outcomes in the order non-severe, severe:

```text
[[25, 5],
 [ 4, 6]]
```

The model correctly classified 31 of 40 patients and detected 6 of 10 severe cases. An always-non-severe classifier would achieve 75% accuracy but zero severe recall. These results were reported from the executed Colab notebook; rounded aggregate metrics are included in `results/evaluation_results.json`.

## Limitations

The sample is small and the final evaluation uses the same source dataset, rather than an external hospital cohort. Four severe cases were missed. Segmentation was visually inspected on selected slices, without expert or quantitative validation. Texture extraction standardizes in-plane spacing only; slice spacing can affect features. Cross-validation scores used to select settings are not independent final performance estimates. Prediction scores have not been calibrated as clinical risk probabilities. This prototype is for education and is not validated for clinical use.

## References

- [STOIC2021 dataset](https://stoic2021.grand-challenge.org/stoic-db/)
- [lungmask source and pretrained segmentation](https://github.com/JoHof/lungmask)
- [Hofmanninger et al., lung segmentation method](https://doi.org/10.1186/s41747-020-00173-2)


## Team Details
Jashruth K A - PES2UG24AM069
Kishan Bharadwaj - PE2UG24AM074
