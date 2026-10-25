# Skin lesion classifier

Research prototype (HAM10000, 7 classes). Not a medical device; outputs are not a diagnosis.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Command line: `python inference.py image.jpg 55 male back` (image, then optional age, sex, body site).
Optional check against the training notebook: `python selftest.py cache_val_300_ar0.npy logits_B2_efficientnet_b3.npy`

## Repository layout
    app.py            Streamlit front end
    inference.py      model code, preprocessing, TTA, calibration, referral rules (app.py imports this)
    requirements.txt
    models/
        B2_efficientnet_b3.pt    image-only model
        D2_efficientnet_b3.pt    image + age/sex/body-site model
        bundle.json              preprocessing, temperatures, melanoma-flag rules
        bundle_group.json        malignant-group-flag rules
        temperatures.json        reference copy (not read by the code)
    selftest.py       optional
    .gitignore

## Rules (validation, 1,103 images; tuned on this set so optimistic)
| rule | threshold | answered | accuracy on answered | referred |
|---|---|---|---|---|
| group | 0.75 | 88.9% | 97.5% | 123 |
| group | 0.80 | 87.7% | 97.8% | 136 |
| melanoma | 0.75 | 89.6% | 97.5% | 115 |
| melanoma | 0.80 | 88.3% | 97.8% | 129 |

## Limits
- Dermoscopy images only; phone photos will shift the model's confidence and the thresholds will not hold.
- Thresholds were chosen on the validation set and have not been tested on independent data.
- Training data (HAM10000) is licensed CC BY-NC 4.0 (non-commercial) and under-represents darker skin tones.
- Do not upload identifiable patient images to a public deployment.
