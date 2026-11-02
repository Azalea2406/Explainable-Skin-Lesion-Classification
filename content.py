"""Static, clinician-reviewable text. Replace/approve with your dermatologist's wording."""
DISCLAIMER = ("This report is produced by an AI decision-support tool and is not a diagnosis. "
              "Findings must be confirmed by a qualified clinician. Narrative text is a draft pending "
              "dermatologist review.")

CLASSES = {
    "mel": dict(name="Melanoma", risk="Malignant",
        about="Melanoma is a cancer of melanocytes, the pigment-producing cells of the skin. It is less common "
              "than other skin cancers but more likely to spread, so early detection and removal greatly improve outcomes.",
        steps=["Urgent referral to a dermatologist", "Dermoscopic review and biopsy for histopathology",
               "Full-body skin examination", "Sun protection and regular self-checks"]),
    "bcc": dict(name="Basal cell carcinoma", risk="Malignant",
        about="The most common skin cancer. It grows slowly and rarely spreads, but it can damage surrounding tissue if untreated.",
        steps=["Dermatology referral for confirmation (biopsy)", "Treatment such as surgical excision, chosen by the specialist",
               "Sun protection and skin surveillance"]),
    "akiec": dict(name="Actinic keratosis / intraepithelial carcinoma", risk="Pre-cancerous",
        about="A rough, scaly patch caused by long-term sun damage. Some lesions can progress to squamous cell carcinoma.",
        steps=["Dermatology review", "Treatment options (for example cryotherapy or topical therapy) at the clinician's discretion",
               "Sun protection and follow-up of the area"]),
    "nv": dict(name="Melanocytic nevus", risk="Benign",
        about="A common mole. Most are harmless, but any change in size, shape or colour should be reviewed.",
        steps=["Routine monitoring", "Refer if the lesion changes, bleeds or itches", "Sun protection"]),
    "bkl": dict(name="Benign keratosis-like lesion", risk="Benign",
        about="Includes seborrheic keratoses and similar growths. They look waxy or 'stuck-on' and become more common with age.",
        steps=["Usually no treatment needed", "Refer if the lesion changes, bleeds or the diagnosis is uncertain",
               "Removal is optional if irritated"]),
    "df": dict(name="Dermatofibroma", risk="Benign",
        about="A small, firm skin nodule, often on the legs. It is harmless and often follows minor skin injury.",
        steps=["Reassurance in typical cases", "Excision only if symptomatic or the diagnosis is uncertain"]),
    "vasc": dict(name="Vascular lesion", risk="Benign",
        about="Blood-vessel growths such as cherry angiomas. Most are harmless.",
        steps=["Usually no treatment needed", "Refer if bleeding repeatedly or rapidly growing", "Laser treatment is optional"]),
}
UNSURE = dict(name="Not sure",
    about="The model's confidence is below the tuned decision threshold, so it does not name a class. "
          "This does not mean the lesion is benign or malignant.",
    steps=["Refer to a dermatologist for assessment", "Consider dermoscopy and, if indicated, biopsy",
           "Re-photograph with better lighting and focus if image quality is poor"])
RISK_MARK = {"Malignant": "●", "Pre-cancerous": "▲", "Benign": "○"}
ABCDE = [("A", "Asymmetry", "One half does not match the other."),
         ("B", "Border", "Edges are ragged, notched or blurred."),
         ("C", "Colour", "Uneven shades of brown, black, red, white or blue."),
         ("D", "Diameter", "Larger than about 6 mm, though melanomas can be smaller."),
         ("E", "Evolving", "Any change in size, shape, colour or symptoms.")]
