# Collaborator candidates — "Trivial baselines that see no pixels"

**68 rows, every one of them fully verified.** Each row was built by opening and reading the page named in `source_url` — a faculty page, a lab page, an institutional directory, or a paper's author/corresponding-author block. No row is here on inference alone.

**34 rows carry a real email address. 34 rows carry only a contact page.**

**NO email address in this file was constructed or guessed.** Every address below was seen in print, on the page cited in that row or in a corresponding-author line. Where an address could not be seen in print, the `email` field is empty and `contact_page` carries the page to use instead — that is a deliberate outcome, not a gap to be filled by pattern-matching a name onto a domain. Several rows record addresses that were deliberately *rejected*: administrative assistants' addresses, stale addresses from before a move, personal-webmail addresses, and one apparent typesetting error in a journal. Read the `notes` column before writing to anyone.

Sourced from five parallel domain searches (prostate/GU, breast, neuro/ICH, AI-evaluation methodology, MR physics & fastMRI). 70 verified rows were pooled; 2 people were found independently by two different searches and were merged, leaving 68. The five searches considered and dropped 72 further people for lack of verification — those are not in this file.

---

## HOW TO USE THIS

- **Approach in fit order.** Work down the `rank` column. Ranking is by `fit_score`, then by whether they have published on AI evaluation, then by seniority. Note that 31 people tie at fit_score 5 — inside that band, pick by which of the three asks you are trying to fill (clinical co-author, biostatistics review, or the one-paragraph clinical confirmation), which the `why_fit` text tells you.
- **Send no more than 5 at a time.** Wait for replies before the next batch, so the pitch can be adjusted based on what the first responses object to. A pitch that fails five times in the same way is telling you something, and you want to hear it before you have spent 50 contacts.
- **Lead with the slice-to-patient collapse.** The opening should be *0.854 slice-level, 0.506 patient-level, from a model that reads no pixels* — a result about an evaluation protocol. Do not open with the critique of any individual paper, and do not open with the identity of the audited authors. The finding is the hook; the audited paper is a detail that comes later.
- **Never contact someone whose work the paper criticises without reading `COLLABORATORS.md` section 4 first.** That section covers the courtesy notice to Rempe et al. and the rules for that class of email. 11 rows are flagged `CRITICISED` and a further 14 are flagged `adjacent`; the flag and its reason are in the `criticised_by_our_paper` column of every row.
- **Hold the line on wording.** Every one of these people is being asked partly to hunt for sentences that drift from *"this evaluation protocol is flawed"* into *"this model learned nothing"*. Do not write that second sentence in the approach email either.

---

## Index (68 people)

`AI-eval` = has published on AI evaluation. `Contact` = whether a real email is on file or only a page. `Flag` = see `criticised_by_our_paper` in the full entry.

| # | Name | Institution | Country | Subspecialty | Seniority | Fit | AI-eval | Contact | Flag |
|---:|---|---|---|---|---|---:|---|---|---|
| 1 | Christopher G. Filippi (publishes as Filippi CG; goes by 'Risto') | University of Toronto, Temerty Faculty of Medicine / The Hospital for Sick Children (SickKids) | Canada | Neuroradiology (paediatric) | full professor | 5 | yes | email |  |
| 2 | Adam E. Flanders | Thomas Jefferson University / Thomas Jefferson University Hospital | USA | Neuroradiology (and imaging informatics) | full professor | 5 | yes | email | CRITICISED |
| 3 | Masoom A. Haider | University of Toronto / Lunenfeld-Tanenbaum Research Institute, Sinai Health System, Toronto | Canada | Genitourinary / prostate MRI, radiomics and machine learning | full professor | 5 | yes | email | adjacent |
| 4 | Nehmat Houssami | The Daffodil Centre and Sydney School of Public Health, The University of Sydney | Australia | Breast cancer screening epidemiology; evidence standards for AI in breast imaging | full professor | 5 | yes | email |  |
| 5 | Florian Knoll | Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Artificial Intelligence in Biomedical Engineering | Germany | MRI reconstruction, machine learning for medical imaging, reproducible research | full professor | 5 | yes | email | adjacent |
| 6 | Christoph I. Lee | University of Wisconsin-Madison, Department of Radiology | USA | Breast imaging; cancer screening evaluation, health services research, AI evaluation | full professor | 5 | yes | page only |  |
| 7 | Yvonne W. Lui | NYU Langone Health / NYU Grossman School of Medicine | USA | Neuroradiology; machine learning in medical imaging | full professor | 5 | yes | email | adjacent |
| 8 | Michael (Miki) Lustig | University of California, Berkeley | USA | Computational MRI, compressed sensing, parallel imaging / coil combination | full professor | 5 | yes | email |  |
| 9 | John Mongan | University of California, San Francisco, Department of Radiology and Biomedical Imaging | USA | Abdominal imaging and ultrasound; clinical informatics | full professor | 5 | yes | page only | CRITICISED |
| 10 | Linda Moy | NYU Grossman School of Medicine / NYU Langone Health | USA | Breast imaging (mammography, US, breast MRI); radiology AI | full professor | 5 | yes | page only | CRITICISED |
| 11 | Luciano M. Prevedello | The Ohio State University Wexner Medical Center | USA | Neuroradiology (board certified in radiology, neuroradiology and clinical informatics) | full professor | 5 | yes | page only | CRITICISED |
| 12 | Raphaële Renard-Penna | Sorbonne Université / Assistance Publique-Hôpitaux de Paris, Paris | France | Genitourinary / prostate MRI | full professor | 5 | yes | page only |  |
| 13 | Olivier Rouvière | Université Claude Bernard Lyon 1 / Hospices Civils de Lyon; INSERM U1032 LabTAU | France | Urinary and vascular imaging; prostate MRI | full professor | 5 | yes | page only |  |
| 14 | Sian Taylor-Phillips | University of Warwick | UK | Screening programme evaluation; test accuracy methodology; AI in breast screening | full professor | 5 | yes | email |  |
| 15 | Tristan Barrett | University of Cambridge / Addenbrooke's Hospital, Cambridge | UK | Uroradiology / prostate multiparametric MRI, AI | associate | 5 | yes | email |  |
| 16 | Akshay Chaudhari | Stanford University | USA | Musculoskeletal/knee MRI, accelerated acquisition, AI evaluation methodology | associate | 5 | yes | page only |  |
| 17 | Judy Wawira Gichoya | Emory University School of Medicine | USA | Interventional radiology and informatics; health equity in imaging AI | associate | 5 | yes | page only |  |
| 18 | Lars J. Grimm | Duke University School of Medicine | USA | Breast imaging; breast MRI data harmonization, breast calcifications, patient perspectives on AI | associate | 5 | yes | email |  |
| 19 | Laura Heacock | NYU Grossman School of Medicine / NYU Langone Health | USA | Breast imaging; AI/ML, screening mammography, breast MRI, DBT, LLMs | associate | 5 | yes | page only | CRITICISED |
| 20 | Andrei S. Purysko | Cleveland Clinic, Cleveland, OH | USA | Abdominal / genitourinary radiology, prostate MRI | associate | 5 | yes | page only |  |
| 21 | Paul H. Yi | St. Jude Children's Research Hospital | USA | Radiology / imaging AI (safe and trustworthy AI, algorithmic fairness) | associate | 5 | yes | email |  |
| 22 | Efrat Shimron | Technion - Israel Institute of Technology | Israel | MRI reconstruction / inverse problems / ML bias in medical imaging | assistant | 5 | yes | email |  |
| 23 | John R. Zech | Columbia University Irving Medical Center | USA | Musculoskeletal radiology; AI applied to radiography | assistant | 5 | yes | email |  |
| 24 | Felipe C. Kitamura | Universidade Federal de São Paulo (UNIFESP) | Brazil | Neuroradiology with AI/data science focus | other | 5 | yes | page only | CRITICISED |
| 25 | Lauren Oakden-Rayner | Australian Institute for Machine Learning, University of Adelaide | Australia | Diagnostic radiology (thoracic/abdominal); medical AI safety and evaluation | other | 5 | yes | page only |  |
| 26 | Nancy A. Obuchowski | Cleveland Clinic, Cleveland, OH | USA | Biostatistics of diagnostic imaging; ROC methodology, clustered data, detection-and-localisation | other | 5 | yes | page only |  |
| 27 | Baris Turkbey | National Cancer Institute (NCI/CCR), National Institutes of Health, Bethesda, MD | USA | Genitourinary / prostate MRI; translational AI | other | 5 | yes | email | adjacent |
| 28 | Constantine A. Gatsonis | Brown University, Department of Biostatistics and Center for Statistical Sciences | USA | Biostatistics; evaluation of diagnostic and screening tests; ROC methodology | full professor | 5 | unclear | email |  |
| 29 | Diana L. Miglioretti | University of California, Davis (School of Medicine); also Senior Investigator, Kaiser Permanente Washington Health Research Institute | USA | Biostatistics for breast cancer screening; Breast Cancer Surveillance Consortium | full professor | 5 | unclear | page only |  |
| 30 | Andrew B. Rosenkrantz | NYU Grossman School of Medicine / NYU Langone Health, New York, NY | USA | Abdominal / genitourinary radiology, prostate MRI, radiology health policy | full professor | 5 | unclear | email |  |
| 31 | Antonio C. Westphalen | University of Washington / Fred Hutchinson Cancer Center, Seattle, WA | USA | Genitourinary radiology, prostate cancer imaging, imaging AI | full professor | 5 | unclear | page only |  |
| 32 | Veronika Cheplygina | IT University of Copenhagen | Denmark | Medical image analysis; machine learning methodology; open science | full professor | 4 | yes | page only |  |
| 33 | Alastair Denniston | University Hospitals Birmingham NHS Foundation Trust and University of Birmingham | UK | Ophthalmology; AI and digital health technology evaluation | full professor | 4 | yes | page only |  |
| 34 | Joann G. Elmore | University of California, Los Angeles | USA | Diagnostic accuracy and physician variability in breast cancer screening/pathology; ML applied to diagnosis | full professor | 4 | yes | email |  |
| 35 | Reinhard Heckel | Technical University of Munich (TUM) | Germany | Machine learning theory; robustness and limitations of deep learning for MRI reconstruction | full professor | 4 | yes | email | adjacent |
| 36 | Saurabh Jha | University of Pennsylvania / Penn Medicine | USA | Cardiothoracic imaging | full professor | 4 | yes | page only |  |
| 37 | Jayashree Kalpathy-Cramer | University of Colorado Anschutz School of Medicine | USA | Medical imaging AI methodology, benchmarks and challenge evaluation (PhD, not a clinician) | full professor | 4 | yes | page only |  |
| 38 | Curtis Langlotz | Stanford University | USA | Radiology; imaging informatics | full professor | 4 | yes | page only | adjacent |
| 39 | Anwar R. Padhani | Paul Strickland Scanner Centre, Mount Vernon Cancer Centre / Institute of Cancer Research, London | UK | Oncological MRI; prostate multiparametric and diffusion-weighted MRI | full professor | 4 | yes | page only | adjacent |
| 40 | George Shih | Weill Cornell Medical College / Weill Cornell Medicine | USA | Radiology; imaging informatics and annotation infrastructure (co-founder, MD.ai) | full professor | 4 | yes | email | CRITICISED |
| 41 | Greg Zaharchuk | Stanford University School of Medicine | USA | Neuroradiology; stroke and dementia imaging; AI outcome prediction | full professor | 4 | yes | page only |  |
| 42 | Manisha Bahl | Massachusetts General Hospital / Harvard Medical School | USA | Breast imaging (mammography, DBT, high-risk lesions); AI in breast imaging | associate | 4 | yes | email |  |
| 43 | Rajiv Gupta | Massachusetts General Hospital / Harvard Medical School | USA | Neuroradiology; CT | associate | 4 | yes | email |  |
| 44 | Mara Kunst | Lahey Hospital & Medical Center / UMass Chan Medical School | USA | Neuroradiology | associate | 4 | yes | page only |  |
| 45 | Xiaoxuan Liu | University of Birmingham, Department of Applied Health Sciences | UK | Clinician (ophthalmology-trained); AI reporting standards and evidence generation | associate | 4 | yes | email |  |
| 46 | Kathryn P. Lowry | University of Washington School of Medicine / Fred Hutchinson Cancer Center | USA | Breast imaging; cancer screening effectiveness and surveillance imaging outcomes | associate | 4 | yes | page only |  |
| 47 | Maciej A. Mazurowski | Duke University | USA | Medical image analysis methodology; breast MRI datasets and harmonization | associate | 4 | yes | email | adjacent |
| 48 | Fredrik Strand | Karolinska Institutet / Karolinska University Hospital | Sweden | Breast radiology; machine learning for screening mammography and risk prediction | associate | 4 | yes | email | adjacent |
| 49 | Karen Drukker | University of Chicago | USA | Medical physics / machine learning in breast imaging; AI performance evaluation and generalizability | other | 4 | yes | page only |  |
| 50 | Bradley J. Erickson | Mayo Clinic, Rochester, Minnesota | USA | Radiology informatics / imaging AI | other | 4 | yes | page only |  |
| 51 | Hersh Chandarana | NYU Langone Health, Department of Radiology | USA | Body/abdominal MRI, quantitative MRI, fast motion-robust imaging | full professor | 4 | unclear | email | adjacent |
| 52 | Daniel J. A. Margolis | Weill Cornell Medicine, New York, NY | USA | Abdominal / genitourinary radiology, prostate MRI, quantitative imaging | full professor | 4 | unclear | page only |  |
| 53 | Michael P. Recht | NYU Langone Health | USA | Musculoskeletal radiology; cartilage imaging; accelerated knee MRI | full professor | 4 | unclear | email | CRITICISED |
| 54 | Lubdha M. Shah | University of Utah | USA | Neuroradiology; spine imaging, advanced MRI (DTI, perfusion, spectroscopy) | full professor | 4 | unclear | email | CRITICISED |
| 55 | Daniel K. Sodickson | NYU Langone Health | USA | Parallel MRI (originator), RF coil design, rapid imaging, compressed sensing | full professor | 4 | unclear | email | adjacent |
| 56 | Francesco Giganti | University College London / University College London Hospitals NHS Foundation Trust, London | UK | Prostate MRI; imaging quality standards | associate | 4 | unclear | page only |  |
| 57 | Angela Tong | NYU Grossman School of Medicine / NYU Langone Health | USA | Prostate MRI, female pelvic imaging, deep learning in pelvic imaging | associate | 4 | unclear | page only | adjacent |
| 58 | Marzyeh Ghassemi | Massachusetts Institute of Technology | USA | Machine learning for health; robustness, privacy and fairness | associate | 3 | yes | email |  |
| 59 | Safwan S. Halabi | Ann & Robert H. Lurie Children's Hospital of Chicago / Northwestern University | USA | Paediatric radiology; imaging informatics | associate | 3 | yes | page only | CRITICISED |
| 60 | Chris McIntosh | University of Toronto (Temerty Faculty of Medicine) and University Health Network | Canada | AI in medicine; biomedical imaging; cardiovascular and cancer imaging | associate | 3 | yes | email | adjacent |
| 61 | Eric Karl Oermann | NYU Grossman School of Medicine / NYU Langone Health | USA | Neurosurgery (spine, epilepsy); machine learning in medicine | associate | 3 | yes | page only |  |
| 62 | James Zou | Stanford University | USA | Statistical machine learning; auditing and evaluation of medical AI | associate | 3 | yes | page only |  |
| 63 | Jonathan I. Tamir | The University of Texas at Austin | USA | Computational MRI, signal processing, machine learning for imaging | assistant | 3 | yes | email |  |
| 64 | Robyn L. Ball | The Jackson Laboratory, Bar Harbor, Maine | USA | Biostatistics / statistical methodology (formerly Senior Statistician, Quantitative Sciences Unit, Stanford University) | other | 3 | yes | page only | CRITICISED |
| 65 | Gaël Varoquaux | Inria (French National Institute for Research in Digital Science and Technology) | France | Machine learning methodology; ML evaluation; health data analytics | other | 3 | yes | page only |  |
| 66 | Berkin Bilgic | Massachusetts General Hospital / Harvard Medical School (Athinoula A. Martinos Center for Biomedical Imaging) | USA | MRI acquisition and reconstruction; quantitative susceptibility mapping (QSM); quantitative parameter mapping | associate | 3 | unclear | email |  |
| 67 | Errol Colak | University of Toronto, Temerty Faculty of Medicine / Unity Health Toronto | Canada | Abdominal imaging; AI and machine learning in medical imaging | associate | 3 | unclear | email |  |
| 68 | Patricia M. Johnson | NYU Langone Health | USA | Deep learning MR image reconstruction and disease detection; ultra-low-field MRI | assistant | 3 | unclear | email | adjacent |

---

## Full entries

Every field from `collaborator_candidates.csv`, in rank order. The CSV is the column-exact version of this same table and is what to open in Excel or Sheets.

### 1. Christopher G. Filippi (publishes as Filippi CG; goes by 'Risto')

**Professor, Department of Medical Imaging (Neuroradiology), University of Toronto; Division Chief, Paediatric Neuroradiology, SickKids; Ontasian Chair in Paediatric Diagnostic Imaging**  
University of Toronto, Temerty Faculty of Medicine / The Hospital for Sick Children (SickKids) — Canada  
*Neuroradiology (paediatric)* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He was invited by Radiology: AI to write the editorial on trustworthiness and uncertainty in ICH detection AI — i.e. the journal already treats him as the person who adjudicates whether an ICH AI evaluation claim is credible. A full professor and division chief in neuroradiology, on record engaging with the epistemics of ICH AI rather than defending a model.

**Relevant work.** Invited commentary 'Bridging the Trust Gap: Conformal Prediction for AI-based Intracranial Hemorrhage Detection', Radiology: Artificial Intelligence 2025 (DOI 10.1148/ryai.250032)

**Email (seen in print).** `risto.filippi@sickkids.ca`  
**Contact page.** https://medical-imaging.utoronto.ca/faculty/christopher-risto-filippi  
**Verified from.** https://medical-imaging.utoronto.ca/faculty/christopher-risto-filippi

Criticised by our paper: no.

**Notes.** NOT part of the RSNA challenge team — no conflict, and his published position (that ICH AI needs better trust/uncertainty machinery) is aligned with a negative-result paper. Email printed identically on two institutional pages: the U of T Medical Imaging faculty page and the SickKids directory page (https://www.sickkids.ca/en/staff/f/risto-filippi/). Affiliation cross-checked against the Europe PMC author record for DOI 10.1148/ryai.250032, which lists SickKids Research Institute and University of Toronto.

---

### 2. Adam E. Flanders

**William E. Conrady, MD Professor in Radiology; Vice Chair for Informatics**  
Thomas Jefferson University / Thomas Jefferson University Hospital — USA  
*Neuroradiology (and imaging informatics)* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He is the lead author of the RSNA 2019 ICH challenge descriptor paper that your prior-art section engages with, and he is a practising academic neuroradiologist with a named professorship and an informatics vice-chair role — exactly the person who can answer on the record whether a positional prior is 'just anatomy' and whether slice-level AUROC is ever the clinically correct unit for ICH.

**Relevant work.** First and corresponding author, 'Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge', Radiology: Artificial Intelligence 2020 (DOI 10.1148/ryai.2020190211)

**Email (seen in print).** `adam.flanders@jefferson.edu`  
**Contact page.** https://www.jefferson.edu/academics/colleges-schools-institutes/skmc/departments/radiology/faculty-staff/faculty/flanders.html  
**Verified from.** https://www.jefferson.edu/academics/colleges-schools-institutes/skmc/departments/radiology/faculty-staff/faculty/flanders.html

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged. The paper directly engages the organisers' on-record statement about the released metadata, so this is a person whose own published claim you are questioning; he must be approached with that stated plainly up front. Email verified twice: printed on the Jefferson faculty page AND as the corresponding-author line of the challenge paper on PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC8082297/). Author list and affiliations independently re-verified against the Europe PMC structured record for DOI 10.1148/ryai.2020190211.

---

### 3. Masoom A. Haider

**Professor, Department of Medical Imaging; Senior Clinician Scientist; Director of Sinai Health Research MRI; Head, Radiomics and Machine Learning Lab**  
University of Toronto / Lunenfeld-Tanenbaum Research Institute, Sinai Health System, Toronto — Canada  
*Genitourinary / prostate MRI, radiomics and machine learning* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He is a PI-RADS v2.1 co-author who publishes negative and cautionary AI results under his own name -- a model that looks good but produces 'net harm', and a domain-shift finding that undercuts the value of large external training sets. That is direct evidence he will engage with a trivial-baseline result rather than resist it, and he can speak to ask 3 as a practising GU radiologist.

**Relevant work.** Senior author of 'Using decision curve analysis to benchmark performance of an MRI-based deep learning model' (Eur Radiol 2020, doi:10.1007/s00330-020-07030-1), which reported that 'original CNN predictions were severely miscalibrated (p<0.0001) resulting in net harm compared with a biopsy all patients strategy'. Senior author of 'Training With Local Data Remains Important for Deep Learning MRI Prostate Cancer Detection' (Can Assoc Radiol J 2025, doi:10.1177/08465371251367620), a negative domain-shift result using PI-CAI. PI-RADS v2.1 co-author.

**Email (seen in print).** `m.haider@utoronto.ca`  
**Contact page.** https://www.lunenfeld.ca/?page=haider-masoom  
**Verified from.** https://www.lunenfeld.ca/?page=haider-masoom

> **Adjacent / read before writing.** - runs a machine-learning lab building prostate AI

**Notes.** Email printed on the Lunenfeld-Tanenbaum profile page. NOTE: some of his papers print mahaider@radfiler.com as the corresponding address -- that is a personal domain, so the institutional utoronto.ca address is recorded instead. He runs a machine-learning lab building prostate AI, so like Turkbey he has skin in the game; his own publication record is the reassurance.

---

### 4. Nehmat Houssami

**Professor of Public Health; NBCF Chair in Breast Cancer Prevention; NHMRC Leadership Fellow; Co-editor, The Breast**  
The Daffodil Centre and Sydney School of Public Health, The University of Sydney — Australia  
*Breast cancer screening epidemiology; evidence standards for AI in breast imaging* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** She is the leading authority on what counts as adequate evidence before a breast AI system is believed, and has repeatedly published that the field's evaluations outrun their evidence. As co-editor of The Breast and an NBCF Chair she brings desk-review weight, and her public-health training makes her the right person to insist on the patient-level rather than slice-level unit of analysis.

**Relevant work.** Co-author of the JACR 2022 external-validation systematic review (doi 10.1016/j.jacr.2021.11.008); senior author, 'The ethical, legal and social implications of using artificial intelligence systems in breast cancer care', The Breast 2019;49:25-32 (doi 10.1016/j.breast.2019.10.001), which argues development 'must not run ahead of evaluation'.

**Email (seen in print).** `nehmat.houssami@sydney.edu.au`  
**Contact page.** https://profiles.sydney.edu.au/nehmat.houssami  
**Verified from.** https://pubmed.ncbi.nlm.nih.gov/39111200/

Criticised by our paper: no.

**Notes.** Email verified in print as the corresponding 'Electronic address' on Carter SM et al., The Breast 2024;77:103783 (doi 10.1016/j.breast.2024.103783); her affiliation and NBCF Chair title are also printed in the JACR 2022 author list. Not a radiologist (public health physician) - covers asks #1-methodology and #2, less so the pixel-level anatomy question. Her University of Sydney profile page is JavaScript-rendered and returned no readable content; the Daffodil Centre page returned HTTP 403.

---

### 5. Florian Knoll

**Prof. Dr., Professorship for Computational Imaging**  
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Artificial Intelligence in Biomedical Engineering — Germany  
*MRI reconstruction, machine learning for medical imaging, reproducible research* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He built the datasets and ran the challenges, so he understands coil combination, k-space and the exact provenance of three of our five cohorts better than anyone reachable. More importantly he chaired the ISMRM reproducible research study group — his stated institutional role is caring whether results replicate, which is precisely the disposition needed to co-sign a negative result rather than resist it.

**Relevant work.** Scientific lead of the fastMRI data sharing initiative and the associated reconstruction challenges (raw k-space for >1300 knee and >7000 brain MRI scans, with Facebook AI Research). Outgoing chair, ISMRM Reproducible Research Study Group. Deputy editor, Magnetic Resonance in Medicine. Co-author on the fastMRI Breast dataset paper (doi 10.1148/ryai.240345).

**Email (seen in print).** `florian.knoll@fau.de`  
**Contact page.** https://www.cil.tf.fau.de/faudir/florian-knoll/  
**Verified from.** https://www.cil.tf.fau.de/faudir/florian-knoll/

> **Adjacent / read before writing.** - fastMRI lead; the paper's supporting k-space study is built on his data

**Notes.** Email printed directly on the FAU Computational Imaging Lab faculty page. He is an engineer/physicist, not a radiologist, so he satisfies the MR-physics and reproducibility side of the ask but not the clinical-credibility side. He is the fastMRI lead — the paper's supporting k-space study is built on his data, so this is the relationship most in need of careful framing. Deputy editorship at MRM is a useful signal about how he will read the manuscript's claims. Formerly assistant professor at NYU 2015-2021.

---

### 6. Christoph I. Lee

**Professor of Radiology, Section of Breast Imaging and Intervention; Vice Chair of Research; Director, WISCR Network; Deputy Editor, Journal of the American College of Radiology**  
University of Wisconsin-Madison, Department of Radiology — USA  
*Breast imaging; cancer screening evaluation, health services research, AI evaluation* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He has already published, as senior author, the systematic review saying out loud that breast AI validation studies are methodologically weak and biased - which is exactly the thesis of this paper transposed to a new modality. He is a full professor, a breast radiologist, and a JACR deputy editor, so he understands desk-review survival and would engage with a negative result rather than take offence.

**Relevant work.** Senior/corresponding author, 'Independent External Validation of Artificial Intelligence Algorithms for Automated Interpretation of Screening Mammography: A Systematic Review', J Am Coll Radiol 2022;19(2 Pt A):259-273 (doi 10.1016/j.jacr.2021.11.008), which concluded external validation efforts 'suffer from risk for bias and applicability concerns'. Also senior author on ClinValAI, Pac Symp Biocomput 2025 (doi 10.1142/9789819807024_0016).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://www.radiology.wisc.edu/profile/christoph-lee  
**Verified from.** https://www.radiology.wisc.edu/profile/christoph-lee

Criticised by our paper: no.

**Notes.** MOVED INSTITUTIONS. His printed corresponding address in papers through 2025 is stophlee@uw.edu (University of Washington, e.g. J Am Coll Radiol 2022 and Pac Symp Biocomput 2025), but the UW-Madison radiology faculty page I read lists him as Professor and Vice Chair of Research there now. I have deliberately left the email field empty because the printed address is likely stale - go through the UW-Madison profile page instead.

---

### 7. Yvonne W. Lui

**MD, Professor of Radiology; Vice Chair of Research, Department of Radiology**  
NYU Langone Health / NYU Grossman School of Medicine — USA  
*Neuroradiology; machine learning in medical imaging* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** The single best match for ask #1. She is a practising academic neuroradiologist AND runs NYU Radiology's research enterprise AND is a fastMRI dataset author AND is a journal senior editor — so she can answer 'is slice-level AUROC ever the clinically correct unit' with authority, and she knows exactly what a desk editor at Radiology: AI will reject on sight. Her editorial role means she is trained to hunt for exactly the kind of overclaiming sentence the user wants policed.

**Relevant work.** Co-author, 'FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast Dynamic Contrast-enhanced MRI', Radiology: Artificial Intelligence 2025;7(1):e240345 (doi 10.1148/ryai.240345). Co-author on the fastMRI Prostate dataset paper. Delivered the fastMRI keynote 'Fast(er) MRI' at NeurIPS 2020. Senior editor, American Journal of Neuroradiology.

**Email (seen in print).** `yvonne.lui@nyulangone.org`  
**Contact page.** https://cbiweb.net/team/yvonne-lui-md/index.html  
**Verified from.** https://cbiweb.net/team/yvonne-lui-md/index.html

> **Adjacent / read before writing.** - fastMRI dataset co-author; the paper re-analyses data she helped publish

**Notes.** Email printed on the NYU Center for Biomedical Imaging team page (the public nyulangone.org clinical profile does not carry it — use the CBI page as the citation). She is a fastMRI dataset co-author, so the paper uses data she helped publish; frame the ask around evaluation protocol, never around dataset quality. Also listed as associate chair for AI at NYU Radiology and past president of the American Society of Neuroradiology.

---

### 8. Michael (Miki) Lustig

**Professor, Giancarlo Family Chair, Department of Electrical Engineering and Computer Sciences**  
University of California, Berkeley — USA  
*Computational MRI, compressed sensing, parallel imaging / coil combination* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** Senior author on the data-crimes paper, which is the closest published precedent for what this manuscript does — so he has already put his name on exactly this argument at the highest-profile venue possible. Full professor with an endowed chair gives the desk-review credibility the student needs, and his parallel-imaging background covers the coil-combination questions in the k-space supporting study.

**Relevant work.** Senior author, 'Implicit data crimes: Machine learning bias arising from misuse of public data', PNAS 2022;119(13):e2117203119 (doi 10.1073/pnas.2117203119). Long-standing work on compressed sensing MRI and parallel-imaging/coil-combination methods; ISMRM 2018 plenary on pediatric MRI.

**Email (seen in print).** `mikilustig@berkeley.edu`  
**Contact page.** https://www2.eecs.berkeley.edu/Faculty/Homepages/mlustig.html  
**Verified from.** https://www2.eecs.berkeley.edu/Faculty/Homepages/mlustig.html

Criticised by our paper: no.

**Notes.** Email printed on both his personal Berkeley page and the EECS department homepage. His own page describes him as Associate Professor (stale); the official EECS department page gives Professor and the Giancarlo Family Chair — I used the department page as the source of record. Engineer, not a clinician. If the goal is one senior methodologist plus one senior MD, he and Shimron are somewhat redundant (same paper, same group) — pick one, and note Shimron was first author on it.

---

### 9. John Mongan

**Professor of Clinical Radiology; Associate Chair, Translational Informatics**  
University of California, San Francisco, Department of Radiology and Biomedical Imaging — USA  
*Abdominal imaging and ultrasound; clinical informatics* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** CLAIM is the reporting standard this paper is effectively enforcing, and the question 'is slice-level AUROC ever the clinically correct unit of analysis' is a CLAIM question that he is the recognised authority on. As a full professor and RSNA AI Committee chair he supplies exactly the desk-review credibility the paper needs. || ALSO VERIFIED BY A SECOND DOMAIN SEARCH: He co-wrote CLAIM, the reporting standard your one-page checklist is effectively trying to extend, and he sat on the RSNA ICH challenge team. If your checklist is going to be usable by reviewers under time pressure (ask 1.6), he is the single best-qualified person alive to tell you where it duplicates CLAIM and where it adds something CLAIM misses.

**Relevant work.** Lead author of CLAIM, the Checklist for Artificial Intelligence in Medical Imaging (Radiology: AI 2020, updated 2024). Chair of the RSNA Artificial Intelligence Committee; editorial board, Radiology: Artificial Intelligence; board-certified in diagnostic radiology and clinical informatics.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://radiology.ucsf.edu/people/john-mongan  
**Verified from.** https://radiology.ucsf.edu/people/john-mongan

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label

**Notes.** CONFLICT TO CHECK BEFORE WRITING: he sits on the editorial board of Radiology: Artificial Intelligence, one of the two target venues. Co-authorship would likely bar that venue or require recusal -- decide venue first, or approach him for the one-paragraph clinical confirmation rather than full co-authorship. No email printed on the UCSF page. || ALSO VERIFIED BY A SECOND DOMAIN SEARCH: RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged. His UCSF page states he chairs the RSNA AI Committee; an AuntMinnie item dated 12 Dec 2025 says Errol Colak has become chair, so the UCSF page may be stale on that point — do not repeat the chair claim in an email without checking. No email printed on the UCSF page. CLAIM 2024 authorship and affiliation verified on PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/).

---

### 10. Linda Moy

**Professor of Radiology; Vice Chair for Artificial Intelligence, Department of Radiology; former Editor of Radiology (stepped down 2025)**  
NYU Grossman School of Medicine / NYU Langone Health — USA  
*Breast imaging (mammography, US, breast MRI); radiology AI* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** She is an author on the fastMRI Breast dataset paper itself - i.e. on the exact cohort where our release-batch metadata baseline (AUC 0.743) beat the trained network (0.633). As a full professor, ex-Editor of Radiology and now inaugural Vice Chair of AI at NYU Radiology, she is the single highest-credibility clinical signature available for a Radiology: AI / npj Digital Medicine submission, and she can answer the anatomy-vs-positional-prior question about DCE breast MRI on the record.

**Relevant work.** Co-author, 'FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast Dynamic Contrast-enhanced MRI', Radiol Artif Intell 2025;7(1):e240345 (doi 10.1148/ryai.240345) - verified author list at https://pmc.ncbi.nlm.nih.gov/articles/PMC11791504/

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://nyulangone.org/doctors/1922064559/linda-moy  
**Verified from.** https://nyulangone.org/doctors/1922064559/linda-moy

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** senior author on the fastMRI Breast release whose label/release structure the paper's breast result criticises

**Notes.** CONFLICT FLAG: the paper's breast finding is a criticism of the release/label structure of HER dataset. Approach as a collaboration ('help us state this correctly about your release') rather than as a critique. Also note she is a former journal editor - she will be the best person to police the 'flawed protocol' vs 'model learned nothing' boundary, and the harshest if the framing drifts. No institutional email seen in print on any page I read; the fastMRI Breast paper prints only the first author's address (eds4001@med.cornell.edu, Eddy Solomon, Weill Cornell) - do not construct hers.

---

### 11. Luciano M. Prevedello

**Professor of Radiology; Vice Chair, Informatics & Augmented Intelligence; Medical Director, 3D and Advanced Visualization Lab**  
The Ohio State University Wexner Medical Center — USA  
*Neuroradiology (board certified in radiology, neuroradiology and clinical informatics)* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** A boarded neuroradiologist who is also boarded in clinical informatics and co-organised the RSNA ICH challenge — the rare person who can answer both the clinical question (does ICH concentrate positionally in a way a slice prior would exploit) and the dataset-metadata question, without needing either half translated.

**Relevant work.** Second author, 'Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge', Radiology: AI 2020 (DOI 10.1148/ryai.2020190211)

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://wexnermedical.osu.edu/find-a-doctor/luciano-prevedello-100000579  
**Verified from.** https://wexnermedical.osu.edu/find-a-doctor/luciano-prevedello-100000579

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged; his co-authored paper carries the metadata claim your work engages. No email printed on the OSU page (only a general clinic phone number) — do not construct one; use the profile page or route via the ICH challenge paper's corresponding author. Affiliation verified both on the OSU page and in the Europe PMC structured author record for the challenge paper.

---

### 12. Raphaële Renard-Penna

**Professeur des Universités (Sorbonne Université); Head of the Uro-Nephrological Imaging Department, GH-APHP-Sorbonne Université (Pitié-Salpêtrière, Tenon)**  
Sorbonne Université / Assistance Publique-Hôpitaux de Paris, Paris — France  
*Genitourinary / prostate MRI* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** She has published the most explicitly damning methodological verdict on prostate MRI AI of anyone on this list, and she did it as senior author of a formal QUADAS-2/CLAIM review. A full professor who already wrote that the field's methods preclude quantitative synthesis is a natural ally for a trivial-baselines audit, and she heads a GU imaging department so she can also supply the ask-3 paragraph on lesion positional concentration.

**Relevant work.** Senior author of 'Automatic segmentation of prostate zonal anatomy on MRI: a systematic review' (Insights Imaging 2022, doi:10.1186/s13244-022-01340-2). Its conclusions are unusually blunt: only 2 of 33 included articles had low risk of bias on all four QUADAS-2 domains; the review 'identified numerous methodological flaws and underlined biases precluding us from performing quantitative analysis', implying 'low robustness and low applicability in clinical practice of the evaluated methods'. Secretary General of the Genito-Urinary Imaging Society of the French Radiology Society.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://anamacap.fr/association-cancer-prostate/notre-conseil-scientifique/  
**Verified from.** https://pubmed.ncbi.nlm.nih.gov/36543901/

Criticised by our paper: no.

**Notes.** Institution and department verified from the paper's author list (Sorbonne Université; Academic Department of Radiology, Hôpital Tenon and Hôpital Pitié-Salpêtrière, APHP). Her exact title was verified from the scientific-council page of ANAMACaP, a French prostate cancer patients' association, which quotes: 'Radiologue - Professeur des Universités à la Sorbonne. Responsable du département d'imagerie Uro-Néphrologique du GH-APHP-Sorbonne Université (Pitié-Salpêtrière, Tenon).' That is a patient-association page rather than an institutional directory, so the title is slightly weaker-sourced than the others -- worth a sanity check. No email seen. Correspondence may be easier in French.

---

### 13. Olivier Rouvière

**Professor of Radiology; Head of the Department of Imaging, Hôpital Édouard Herriot**  
Université Claude Bernard Lyon 1 / Hospices Civils de Lyon; INSERM U1032 LabTAU — France  
*Urinary and vascular imaging; prostate MRI* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He writes editorials about the construction of the ground truth against which prostate AI is scored, and his own empirical work is about how fragile prostate DWI/ADC measurements are -- which is the substrate of the audited prostate DWI paper. A department head who publishes on reference-standard design is well placed to say, on the record, whether slice-level AUROC is ever the right unit.

**Relevant work.** Sole author of the editorial 'Evaluation of automated prostate segmentation: The complex issue of the optimal number of expert segmentations' (Diagn Interv Imaging 2023, doi:10.1016/j.diii.2023.10.002), keyworded artificial intelligence / inter-reader variability / reproducibility -- i.e. a piece specifically about how AI reference standards should be constructed. Senior author on ADC reproducibility work showing scanner, b-value choice and time of day induce substantial measurement variability (doi:10.1016/j.diii.2022.06.001). Senior author on a 21-reader study relaxing the PI-RADS dominant-sequence rule (doi:10.1016/j.diii.2025.04.003).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://labtau.univ-lyon1.fr/members/rouviere-olivier  
**Verified from.** https://pubmed.ncbi.nlm.nih.gov/37863708/

Criticised by our paper: no.

**Notes.** EMAIL DELIBERATELY LEFT EMPTY. His papers print olivier.rouviere@netcourrier.com, which is a French consumer webmail provider, not an institutional address -- excluded per the brief's rule against personal email. Title 'Professor of Radiology, Claude Bernard University' verified on his Google Scholar profile header and in the LabTAU listing; department/hospital affiliations verified from three separate paper author lists. The LabTAU member page 301-redirects to the lab root, so it may need navigating by hand.

---

### 14. Sian Taylor-Phillips

**Professor, Warwick Medical School (NIHR Research Professor; screening evaluation lead)**  
University of Warwick — UK  
*Screening programme evaluation; test accuracy methodology; AI in breast screening* · seniority: full professor · **fit_score 5** · published on AI evaluation: yes

**Why fit.** She wrote the highest-profile paper in the field arguing that retrospective breast AI evaluations are too methodologically weak to support their conclusions - the same argument this paper makes with a zero-image baseline. She is the person most likely to defend, in print, the distinction between 'this evaluation protocol is flawed' and 'this model learned nothing', because that distinction is the core of her own published work.

**Relevant work.** Corresponding author, Freeman K et al., 'Use of artificial intelligence for image analysis in breast cancer screening programmes: systematic review of test accuracy', BMJ 2021;374:n1872 (doi 10.1136/bmj.n1872) - concluded studies were 'of poor methodological quality' and current evidence 'does not yet allow judgement of its accuracy'.

**Email (seen in print).** `S.Taylor-Phillips@warwick.ac.uk`  
**Contact page.** https://warwick.ac.uk/fac/sci/med/staff/taylor-phillips/  
**Verified from.** https://warwick.ac.uk/fac/sci/med/staff/taylor-phillips/

Criticised by our paper: no.

**Notes.** Email verified in print: the BMJ 2021 corresponding-author line reads 'Division of Health Sciences, University of Warwick, Coventry, UK S.Taylor-Phillips@warwick.ac.uk' (PubMed record https://pubmed.ncbi.nlm.nih.gov/34470740/, open access PMC8409323). She is a physicist/epidemiologist by training, NOT a radiologist - so she satisfies ask #1's methodological half and ask #2, but not the 'is the positional prior just anatomy?' clinical half. Pair her with a breast radiologist.

---

### 15. Tristan Barrett

**Associate Professor, Department of Radiology; Honorary NHS Consultant Radiologist; Director of the Cambridge academic Radiology programme**  
University of Cambridge / Addenbrooke's Hospital, Cambridge — UK  
*Uroradiology / prostate multiparametric MRI, AI* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He has already done, at review level, the thing this paper does at audit level: applied formal bias tooling to prostate MRI AI papers and concluded the evidence base is methodologically weak. Someone who published 'mean RQS 11/36' will not be defensive about a trivial-baseline result; he is among the most likely on this list to say yes.

**Relevant work.** Co-author of 'Comparative performance of fully-automated and semi-automated artificial intelligence methods for the detection of clinically significant prostate cancer on MRI: a systematic review' (Insights Imaging 2022, doi:10.1186/s13244-022-01199-3), which scored the literature with CLAIM, RQS and QUADAS-2, found a mean RQS of 11/36, five papers at high risk of bias, and concluded by identifying 'common methodological limitations and biases that future studies will need to address'. Chair of the British Society of Urogenital Radiology; local clinical lead for uroradiology.

**Email (seen in print).** `tb507@medschl.cam.ac.uk`  
**Contact page.** https://www.medschl.cam.ac.uk/tristan-barrett  
**Verified from.** https://www.medschl.cam.ac.uk/tristan-barrett

Criticised by our paper: no.

**Notes.** Email printed directly on the Cambridge School of Clinical Medicine faculty page. Associate Professor rather than full professor, but he chairs the UK urogenital radiology society and was RCR Roentgen Professor 2022-23, so seniority is not a problem. Lists Radboud (the PI-CAI group) among his collaborators, which is useful context.

---

### 16. Akshay Chaudhari

**Associate Professor (Research) of Radiology and of Biomedical Data Science**  
Stanford University — USA  
*Musculoskeletal/knee MRI, accelerated acquisition, AI evaluation methodology* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** His group's stated speciality is evaluation methodology for medical AI, sitting jointly in Radiology and Biomedical Data Science — which is exactly the seam this paper sits on. Knee MRI is his imaging domain, which is the fastMRI knee cohort. He is the best-placed person on this list to referee the 'is slice-level AUROC the right unit' question from a measurement-theory angle rather than a clinical one.

**Relevant work.** Co-author, 'Deep learning for accelerated and robust MRI reconstruction', MAGMA 2024;37(3):335-368 (doi 10.1007/s10334-024-01173-8), a review that explicitly treats robustness under distribution shift, bias and the limitations of DL reconstruction. Stanford profile describes a group building 'representation learning and evaluation techniques' and research on 'evaluating AI model robustness, clinical deployment frameworks'.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://profiles.stanford.edu/akshay-chaudhari  
**Verified from.** https://profiles.stanford.edu/akshay-chaudhari

Criticised by our paper: no.

**Notes.** NO EMAIL RECORDED. The only address printed on his Stanford profile is hfarooq3@stanford.edu, which is an administrative contact and not his own — deliberately not recorded as his email. Use the Stanford profile contact route. He is a PhD, not an MD radiologist, despite the Radiology appointment, so he does not fully discharge the senior-clinical-co-author requirement. Co-founder of a commercial imaging AI venture (Cognita Imaging) — a competing-interests line worth thinking about before asking him to co-sign a critique of imaging AI evaluation.

---

### 17. Judy Wawira Gichoya

**Associate Professor, Department of Radiology and Imaging Sciences**  
Emory University School of Medicine — USA  
*Interventional radiology and informatics; health equity in imaging AI* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** A radiologist whose stated programme is literally 'evaluating AI for bias and fairness' and 'validating AI in real-world settings' -- the two things this paper does. Her fourth stated pillar is training the next generation through mentoring, which makes her unusually likely to say yes to a high-school first author rather than ignore the email.

**Relevant work.** Research group works on building diverse ML datasets, evaluating AI for bias and fairness, validating AI in real-world settings, and training data scientists via collaborative mentoring (per her Winship profile).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://winshipcancer.emory.edu/profiles/gichoya-judy.php  
**Verified from.** https://winshipcancer.emory.edu/profiles/gichoya-judy.php

Criticised by our paper: no.

**Notes.** No email printed on the Emory or Winship pages I read; the med.emory.edu radiology profile is JavaScript-rendered and returned only navigation. Use the Winship profile contact route. A third-party site hosts an old CV PDF with contact details -- I did not use it and would not recommend relying on it.

---

### 18. Lars J. Grimm

**Associate Professor of Radiology, Breast Imaging Division**  
Duke University School of Medicine — USA  
*Breast imaging; breast MRI data harmonization, breast calcifications, patient perspectives on AI* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** A practising academic breast radiologist attached to the Duke Breast MRI dataset lineage, whose own published interests are harmonization and how patients react to AI in breast screening - i.e. someone already thinking about what breast AI claims actually mean. He is senior enough to sign, junior enough to have time, and the Duke breast MRI provenance makes him credible on whether positional concentration of lesions is real anatomy.

**Relevant work.** Co-author, Lew CO et al., 'A publicly available deep learning model and dataset for segmentation of breast, fibroglandular tissue, and vessels in breast MRI', Sci Rep 2024;14:5383 (doi 10.1038/s41598-024-54048-2), built on the Duke Breast Cancer MRI dataset. Listed Duke Scholars interests include breast MRI data harmonization and patient perspectives on AI in mammography screening.

**Email (seen in print).** `lars.grimm@duke.edu`  
**Contact page.** https://scholars.duke.edu/person/lars.grimm  
**Verified from.** https://scholars.duke.edu/person/lars.grimm

Criticised by our paper: no.

**Notes.** Email printed on the Duke Scholars page I read. Probably the single most likely 'yes' on this list: right subspecialty, right dataset lineage, no obvious conflict, and not so famous that a cold email disappears.

---

### 19. Laura Heacock

**Associate Professor, Department of Radiology**  
NYU Grossman School of Medicine / NYU Langone Health — USA  
*Breast imaging; AI/ML, screening mammography, breast MRI, DBT, LLMs* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** Senior clinical author on the fastMRI Breast release and a practising breast radiologist whose stated research interests are explicitly AI/ML plus breast MRI. She is the person best placed to give the one-paragraph clinical confirmation on where lesions concentrate on a DCE breast MRI stack, and to say whether slice-level AUROC is ever the right unit for a breast MRI model. || ALSO VERIFIED BY A SECOND DOMAIN SEARCH: A breast radiologist who is simultaneously a fastMRI Breast dataset author and an active AI-in-breast-imaging researcher, and who has already published in the target journal. She is the right person to give the clinical confirmation on positional concentration for the breast cohort, and she will know from the reader-performance literature exactly how much a slice-level metric overstates patient-level utility.

**Relevant work.** Last author, 'FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast Dynamic Contrast-enhanced MRI', Radiol Artif Intell 2025 (doi 10.1148/ryai.240345); also 'Problem-solving Breast MRI', RadioGraphics 2023;43(10):e230026 (doi 10.1148/rg.230026)

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://nyulangone.org/doctors/1447541255/laura-heacock  
**Verified from.** https://nyulangone.org/doctors/1447541255/laura-heacock

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** curator/author on the fastMRI Breast release whose label/release structure the paper's breast result criticises

**Notes.** Same conflict flag as Moy - she is a curator of the dataset the paper's breast result criticises. Practically she may be the more responsive of the two (less administrative load than Moy) and is a realistic senior co-author. No email printed on her NYU page. || ALSO VERIFIED BY A SECOND DOMAIN SEARCH: NO EMAIL RECORDED — not printed on her NYU profile; use the contact_page. She is a fastMRI Breast dataset co-author. Associate professor. Her own research programme is about AI that helps radiologists, so she has a stake in the field's credibility — that generally makes people receptive to protocol critiques rather than defensive, but it is worth a sentence of framing.

---

### 20. Andrei S. Purysko

**Section Head of Abdominal Imaging, Imaging Institute**  
Cleveland Clinic, Cleveland, OH — USA  
*Abdominal / genitourinary radiology, prostate MRI* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He is running a prospective trial of a commercial prostate MRI AI and is on record about how he is doing it: 'we're just not just blindly trusting AI. We're really vetting it to make sure that its performance aligns with that of our experts.' That is exactly the disposition the paper needs in a senior co-author, and he sits in the same institution as Obuchowski, which makes a joint clinical-plus-biostatistics approach practical.

**Relevant work.** Co-author with Turkbey of 'PI-RADS: Where Next?' (Radiology 2023, doi:10.1148/radiol.223128), on the limitations and controversies of PI-RADS. Senior/contributing author on 'New Prostate MRI Scoring Systems (PI-QUAL, PRECISE, PI-RR, PI-FAB): Expert Panel Narrative Review' (AJR 2024, doi:10.2214/AJR.24.30956), which 'critically examines' the systems and delineates 'current limitations'. Principal investigator on Cleveland Clinic's clinical trial of an FDA-cleared prostate MRI AI product.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://consultqd.clevelandclinic.org/how-ai-is-changing-the-prostate-mri  
**Verified from.** https://consultqd.clevelandclinic.org/how-ai-is-changing-the-prostate-mri

Criticised by our paper: no.

**Notes.** EMAIL DELIBERATELY LEFT EMPTY. The corresponding-author line of his 2023 Radiol Clin North Am editorial prints 'puryska@ccf.org' -- note the transposed letters versus his surname 'Purysko'. That looks like a typesetting error, so I have not recorded it; the student should obtain the address from the Cleveland Clinic directory rather than trust it. Academic rank listed as Associate Professor (Cleveland Clinic Lerner College of Medicine) in secondary listings but not confirmed on a page I read; 'Section Head of Abdominal Imaging' IS confirmed.

---

### 21. Paul H. Yi

**Associate Member, St. Jude Faculty; Section Chief, Intelligent Imaging Informatics (I3); Department of Radiology**  
St. Jude Children's Research Hospital — USA  
*Radiology / imaging AI (safe and trustworthy AI, algorithmic fairness)* · seniority: associate · **fit_score 5** · published on AI evaluation: yes

**Why fit.** His ICH paper is literally about the slice-label versus examination-label supervision distinction that your patient-level collapse result turns on, so he will not need the premise explained; and his stated programme is safe/trustworthy AI and fairness, meaning he is temperamentally a 'the protocol is flawed' person rather than a 'the model works' person.

**Relevant work.** Co-author, 'Examination-Level Supervision for Deep Learning-based Intracranial Hemorrhage Detection on Head CT Scans', Radiology: Artificial Intelligence 2024 (Teneggi, Yi, Sulam); ~60 PubMed-indexed papers with 'artificial intelligence' or 'deep learning' in the title, with a stated research focus on safe/trustworthy AI and algorithmic fairness

**Email (seen in print).** `paul.yi@stjude.org`  
**Contact page.** https://www.stjude.org/people/y/paul-yi.html  
**Verified from.** https://www.stjude.org/people/y/paul-yi.html

Criticised by our paper: no.

**Notes.** Best combination of topical precision (exam-level vs slice-level ICH labels) and willingness to publish critically. Now at a paediatric institution, so he is a radiologist-methodologist rather than an adult neuroradiologist — he covers ask 1.5 (wording discipline) and 1.2 (is slice the decision unit) better than he covers ask 3 (positional concentration of ICH in adults). Email printed on the St. Jude faculty page.

---

### 22. Efrat Shimron

**Senior Lecturer (Assistant Professor), PhD**  
Technion - Israel Institute of Technology — Israel  
*MRI reconstruction / inverse problems / ML bias in medical imaging* · seniority: assistant · **fit_score 5** · published on AI evaluation: yes

**Why fit.** She literally wrote the canonical paper showing that off-label reuse of public MRI data produces 'up to 48% artificial improvement' — the same class of finding as our trivial baseline. She named the phenomenon. Of everyone on this list she is the least likely to read a negative result as an attack and the most likely to help sharpen the 'the protocol is flawed, not the model' distinction.

**Relevant work.** First author, 'Implicit data crimes: Machine learning bias arising from misuse of public data', PNAS 2022;119(13):e2117203119 (doi 10.1073/pnas.2117203119). Co-author, 'Deep learning for accelerated and robust MRI reconstruction', MAGMA 2024;37(3):335-368 (doi 10.1007/s10334-024-01173-8).

**Email (seen in print).** `efrat.s@technion.ac.il`  
**Contact page.** https://rticc.net.technion.ac.il/faculty/efrat-shimron/  
**Verified from.** https://rticc.net.technion.ac.il/faculty/efrat-shimron/

Criticised by our paper: no.

**Notes.** Email verified twice independently: printed on the Technion RTICC faculty page, and again as the corresponding-author address on the MAGMA 2024 review indexed in PubMed. Dual appointment ECE + BME, leads the Medical AI & MRI Lab. Not a radiologist and not a biostatistician — she covers the methodology half of the ask, not the clinical half. Assistant-professor rank means she may not carry the 'senior clinical co-author' weight a desk editor is looking for on her own; best paired with an MD.

---

### 23. John R. Zech

**Assistant Professor of Radiology, Division of Musculoskeletal Radiology**  
Columbia University Irving Medical Center — USA  
*Musculoskeletal radiology; AI applied to radiography* · seniority: assistant · **fit_score 5** · published on AI evaluation: yes

**Why fit.** Zech 2018 is the direct ancestor of this paper's argument: a model scoring well for reasons unrelated to the anatomy. He is a board-certified academic radiologist who can answer both clinical questions on the record, and he has spent his career arguing that headline AUROC misleads -- he will not misread 'the protocol is flawed' as 'the model learned nothing'.

**Relevant work.** First author, 'Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study', PLOS Medicine 2018 -- the canonical demonstration that a CNN exploits site-specific confounders rather than pathology. Recent editorial 'Radiomics in musculoskeletal imaging: what is its role in the era of deep learning?', Skeletal Radiology (PubMed PMID 42159632, DOI 10.1007/s00256-026-05257-5).

**Email (seen in print).** `jrz2111@cumc.columbia.edu`  
**Contact page.** https://www.columbiaradiology.org/profile/john-r-zech-md  
**Verified from.** https://www.columbiaradiology.org/profile/john-r-zech-md

Criticised by our paper: no.

**Notes.** Email verified twice over: printed as the corresponding-author line on his 2026 Skeletal Radiology editorial, and it matches his current Columbia affiliation. Only an Assistant Professor, so 'senior' is a stretch on rank -- but he is the most substantively on-point radiologist here. Consider pairing him with a full professor. His own work is NOT criticised by this paper; it is extended by it.

---

### 24. Felipe C. Kitamura

**Affiliated Professor of Neuroradiology, Universidade Federal de São Paulo (UNIFESP); Associate Editor, Radiology: Artificial Intelligence; member, RSNA AI Committee**  
Universidade Federal de São Paulo (UNIFESP) — Brazil  
*Neuroradiology with AI/data science focus* · seniority: other · **fit_score 5** · published on AI evaluation: yes

**Why fit.** A neuroradiologist who is simultaneously an RSNA ICH challenge co-organiser and an Associate Editor at Radiology: AI — he can tell you before submission whether this survives desk review at one of your two target venues, and he is on the RSNA AI Committee, which is the institutional route back to the challenge organisers.

**Relevant work.** Co-author, RSNA 2019 Brain CT Hemorrhage Challenge dataset paper (Radiology: AI 2020); Associate Editor at Radiology: Artificial Intelligence; ~50 PubMed-indexed publications

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://digital.faculdadesiriolibanes.org.br/professor/felipe-campos-kitamura  
**Verified from.** https://digital.faculdadesiriolibanes.org.br/professor/felipe-campos-kitamura

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged. EDITORIAL CONFLICT: as an Associate Editor at Radiology: AI he probably cannot be both co-author and handling editor — if he joins as an author, that likely removes him from the editorial path at that journal. Verify with him first. Current roles taken from a Faculdade Sírio-Libanês institutional teaching profile; ORCID 0000-0002-9992-5630 independently lists UNIFESP (Departamento de Diagnóstico por Imagem, neuroradiologist) and Dasa. No email printed anywhere I read.

---

### 25. Lauren Oakden-Rayner

**Senior Research Fellow (radiologist); Director of Research, Medical Imaging, Royal Adelaide Hospital**  
Australian Institute for Machine Learning, University of Adelaide — Australia  
*Diagnostic radiology (thoracic/abdominal); medical AI safety and evaluation* · seniority: other · **fit_score 5** · published on AI evaluation: yes

**Why fit.** She is a practising radiologist whose entire research programme is 'how do we know before deployment that a model works' -- hidden stratification is precisely the argument that an aggregate AUROC hides the clinically relevant unit. She has already published a negative-result AI evaluation paper (the rheumatoid arthritis scoring study), so she has personally survived the reviewer fight this paper will have.

**Relevant work.** Hidden stratification in medical imaging ML; medical AI audits and evaluation. 2026 J Imaging Inform Med paper on how demographic bias is encoded in chest X-ray classifiers (PubMed PMID 42343005, DOI 10.1007/s10278-026-02073-0) confirms current AIML/Adelaide affiliation. 2025 Semin Arthritis Rheum paper explicitly concludes an AI system's performance is 'insufficient to justify use' (DOI 10.1016/j.semarthrit.2025.152761).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://laurenoakdenrayner.com/about-me/  
**Verified from.** https://laurenoakdenrayner.com/about-me/

Criticised by our paper: no.

**Notes.** IMPORTANT NAME CORRECTION: the brief calls her 'Luke Oakden-Rayner'. She now publishes and goes by Lauren Oakden-Rayner. Address her as Lauren; using the old name would be a bad opening. Older papers (pre-~2021) carry the former name. No email seen in print on any page I read -- use the contact route on her site. She is the single best fit on this list for the exact ask.

---

### 26. Nancy A. Obuchowski

**Vice Chair, Department of Quantitative Health Sciences; staff, Department of Diagnostic Radiology**  
Cleveland Clinic, Cleveland, OH — USA  
*Biostatistics of diagnostic imaging; ROC methodology, clustered data, detection-and-localisation* · seniority: other · **fit_score 5** · published on AI evaluation: yes

**Why fit.** This is the single best fit for ask 2. The paper's central empirical claim is that a slice-level AUROC of 0.854 collapses to 0.506 at the patient level -- that is verbatim the clustered-data / unit-of-analysis problem she spent a career formalising. She can adjudicate both the matching rule and the confidence interval on the trivial fraction, and she has in-domain credibility from PI-CAI.

**Relevant work.** Her Cleveland Clinic research page states she extended ROC analysis to handle correlated observations from a single patient (e.g. multiple lesions), 'recognizing that such data cannot be treated as independent', and that she developed statistical methods for detection AND localisation of multiple lesions. She was the statistical co-author on PI-CAI (Lancet Oncol 2024, doi:10.1016/S1470-2045(24)00220-1), which used a prespecified non-inferiority analysis plan.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://www.lerner.ccf.org/quantitative-health/obuchowski/  
**Verified from.** https://www.lerner.ccf.org/quantitative-health/obuchowski/

Criticised by our paper: no.

**Notes.** Her email is deliberately obfuscated on the Cleveland Clinic page (rendered as '[email protected]'), so no address is recorded -- contact via the departmental page. She is a statistician, not a radiologist, so she covers ask 2 but not ask 3; pair her with a GU radiologist. Note she co-authored PI-CAI, a study the paper does NOT criticise (PI-CAI is patient-level and well-designed) -- worth saying so explicitly in the approach email, as it positions the paper as defending her methodological standards rather than attacking them.

---

### 27. Baris Turkbey

**Senior Clinician; Head, Artificial Intelligence Resource and MRI Sections, Molecular Imaging Branch**  
National Cancer Institute (NCI/CCR), National Institutes of Health, Bethesda, MD — USA  
*Genitourinary / prostate MRI; translational AI* · seniority: other · **fit_score 5** · published on AI evaluation: yes

**Why fit.** He led the PI-RADS v2.1 consensus and simultaneously heads NCI's AI Resource section, so he is the one person who can answer both halves of ask 1 on the record: whether the positional prior is just anatomy, and whether slice-level AUROC is ever the clinically correct unit. 'PI-RADS: Where Next?' shows he already publishes about the limitations of the system he built.

**Relevant work.** Lead author, PI-RADS v2.1 (Eur Urol 2019, doi:10.1016/j.eururo.2019.02.033). Co-author with Purysko of 'PI-RADS: Where Next?' (Radiology 2023, doi:10.1148/radiol.223128), a review explicitly framed around 'limitations and controversies' of PI-RADS. Co-author with Haider of 'Deep learning-based artificial intelligence applications in prostate MRI' (Br J Radiol 2021, doi:10.1259/bjr.20210563).

**Email (seen in print).** `turkbeyi@mail.nih.gov`  
**Contact page.** https://irp.nih.gov/pi/ismail-turkbey  
**Verified from.** https://irp.nih.gov/pi/ismail-turkbey

> **Adjacent / read before writing.** - leads a prolific prostate MRI AI programme, so the thesis implicitly criticises the sub-field he heads

**Notes.** Email verified twice: printed on the NIH IRP profile page AND as the corresponding-author address on the PI-RADS v2.1 paper. CAUTION: he is himself a prolific developer of prostate MRI AI, so the paper's thesis implicitly criticises the sub-field he leads. His own published work on PI-RADS limitations suggests he will engage rather than resist, but the student should frame the ask as 'evaluation protocol is flawed', never as 'these models learned nothing'.

---

### 28. Constantine A. Gatsonis

**Henry Ledyard Goddard University Professor of Statistical Sciences**  
Brown University, Department of Biostatistics and Center for Statistical Sciences — USA  
*Biostatistics; evaluation of diagnostic and screening tests; ROC methodology* · seniority: full professor · **fit_score 5** · published on AI evaluation: unclear

**Why fit.** This is the person for ask #2 and for half of ask #1. The claim that slice-level AUROC is the wrong unit of analysis, and the interval on the trivial fraction, are ROC-methodology and clustered-data questions -- he has spent a career on exactly ROC analysis for diagnostic imaging in oncology trials. A signed statistical review from him would be near-unanswerable at desk review.

**Relevant work.** Described on his Brown profile as a leading authority on evaluation of diagnostic and screening tests; ROC methodology for detection and prediction; Bayesian hierarchical models; meta-analysis of diagnostic test accuracy; imaging biomarker validation; leadership roles at ACRIN and ECOG-ACRIN.

**Email (seen in print).** `Constantine_Gatsonis@brown.edu`  
**Contact page.** https://vivo.brown.edu/display/cgatsoni  
**Verified from.** https://vivo.brown.edu/display/cgatsoni

Criticised by our paper: no.

**Notes.** Not a radiologist and I found no evidence he has published specifically on deep-learning AI evaluation -- hence 'unclear' on that field. Approach him for the biostatistics review, not as the clinical voice. Email is printed directly on the Brown VIVO profile.

---

### 29. Diana L. Miglioretti

**Dean's Professor and Division Chief of Biostatistics, Department of Public Health Sciences**  
University of California, Davis (School of Medicine); also Senior Investigator, Kaiser Permanente Washington Health Research Institute — USA  
*Biostatistics for breast cancer screening; Breast Cancer Surveillance Consortium* · seniority: full professor · **fit_score 5** · published on AI evaluation: unclear

**Why fit.** This is the biostatistics ask (#2), in the correct disease area. Clustered/multilevel data is exactly the problem behind slice-level vs patient-level AUROC, and she is the field's reference statistician for evaluating breast screening tests - the right reviewer for the matching rule and for the confidence interval on the trivial fraction.

**Relevant work.** UC Davis Health profile states her methodological focus is 'multilevel and latent variable models, longitudinal and clustered data analysis, and the evaluation of screening and diagnostic tests', with collaborative work in breast cancer screening and a Breast Cancer Surveillance Consortium role.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://health.ucdavis.edu/phs/team/1520/diana-miglioretti---public-health-sciences--division-of-biostatistics-davis/  
**Verified from.** https://health.ucdavis.edu/phs/team/1520/diana-miglioretti---public-health-sciences--division-of-biostatistics-davis/

Criticised by our paper: no.

**Notes.** Statistician, not a clinician - she covers ask #2 only, not the radiological anatomy question. No email printed on the UC Davis Health page; she also maintains https://dianamiglioretti.ucdavis.edu/ which I did not open. I did not verify a specific AI-evaluation publication by her, hence 'unclear' on that field - her verified relevance is screening/diagnostic-test statistics.

---

### 30. Andrew B. Rosenkrantz

**Professor of Radiology and Urology; Section Chief of Body Imaging; Editor-in-Chief, American Journal of Roentgenology**  
NYU Grossman School of Medicine / NYU Langone Health, New York, NY — USA  
*Abdominal / genitourinary radiology, prostate MRI, radiology health policy* · seniority: full professor · **fit_score 5** · published on AI evaluation: unclear

**Why fit.** Highest desk-review-survival value on the list: a sitting editor-in-chief of a major radiology journal who is also a PI-RADS v2.1 co-author and prostate MRI subspecialist. He knows exactly what language makes a meta-research critique publishable versus what makes an editor reject it, which is precisely the 'hunt for sentences that overclaim' role in ask 1.

**Relevant work.** Second author on PI-RADS v2.1 (Eur Urol 2019, doi:10.1016/j.eururo.2019.02.033). NYU profile credits him with introducing multiparametric prostate MRI and MRI-guided biopsy into clinical practice, and with research 'advancing the quality of radiology practice through policy engagement'. Editor-in-Chief of AJR; former chair of the ACR Commission on Body Imaging.

**Email (seen in print).** `Andrew.Rosenkrantz@nyulangone.org`  
**Contact page.** https://nyulangone.org/doctors/1295868610/andrew-b-rosenkrantz  
**Verified from.** https://nyulangone.org/doctors/1295868610/andrew-b-rosenkrantz

Criticised by our paper: no.

**Notes.** Email is the corresponding-author address printed on the PI-RADS v2.1 author list; no email is printed on the NYU profile page itself. His AJR editorship is a conflict to be aware of if the paper is ever submitted there, and it means he may be time-poor. I did not verify a specific publication of his critiquing AI evaluation methodology, hence 'unclear'.

---

### 31. Antonio C. Westphalen

**Professor of Radiology, Urology and Radiation Oncology; Section Chief of Abdominal Imaging**  
University of Washington / Fred Hutchinson Cancer Center, Seattle, WA — USA  
*Genitourinary radiology, prostate cancer imaging, imaging AI* · seniority: full professor · **fit_score 5** · published on AI evaluation: unclear

**Why fit.** A full professor with triple appointments in radiology, urology and radiation oncology who runs a GU imaging section -- exactly the seniority profile that survives desk review. His MAS in clinical research means he can engage with the statistical argument as well as the clinical one, and his RSNA/SAR committee roles give the paper institutional weight.

**Relevant work.** UW faculty page states his research 'concentrates on the use of advanced imaging technologies and artificial intelligence to improve the diagnosis and management of prostate and other genitourinary cancers', with nearly 200 peer-reviewed publications. Portfolio Director for Internal & External Relations at the Society of Abdominal Radiology; former chair of the RSNA Genitourinary Scientific Program Committee. Also holds an MAS (Master of Advanced Studies in clinical research), i.e. formal methods training.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://rad.uw.edu/people/acwestph  
**Verified from.** https://rad.uw.edu/people/acwestph

Criticised by our paper: no.

**Notes.** No email printed on the UW page. IMPORTANT: he moved from UCSF to University of Washington; older papers (e.g. Abdom Radiol 2020) print antonio.westphalen@ucsf.edu, which is stale -- do NOT use it. I did not verify a specific critical-methodology publication, hence 'unclear' on AI evaluation; a Radiology piece titled 'Building the Foundation for AI in Prostate Cancer Imaging' surfaced in search but I did not read it, so treat that as unconfirmed.

---

### 32. Veronika Cheplygina

**Professor (Dr. ir.), Department of Data, Systems and Robotics / Data-Intensive Systems and Applications**  
IT University of Copenhagen — Denmark  
*Medical image analysis; machine learning methodology; open science* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** She is a full professor whose public output is largely meta-research on how medical imaging ML is benchmarked, annotated and over-claimed -- the trivial-baseline argument is her genre. She is also an outspoken advocate for open science and for junior/unconventional researchers, so an unaffiliated high-school first author is a feature to her, not a disqualifier.

**Relevant work.** Listed interests: machine learning, medical image analysis, pattern recognition, open science; research fingerprint dominated by medical imaging, image analysis, annotation and transfer learning. Runs PURRlab at ITU.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://pure.itu.dk/en/persons/veronika-cheplygina/  
**Verified from.** https://pure.itu.dk/en/persons/veronika-cheplygina/

Criticised by our paper: no.

**Notes.** The ITU PURE page displays her institutional address in an anti-scraping obfuscated form that did not resolve to a clean, unambiguous string, so I have deliberately left the email blank rather than reconstruct it. Use the PURE contact page or her personal site veronikach.com. Not a clinician -- she covers the methodology half of the ask, not the radiology half.

---

### 33. Alastair Denniston

**Consultant Ophthalmologist and Honorary Professor; Data, Diagnostics and Decision Tools Theme lead**  
University Hospitals Birmingham NHS Foundation Trust and University of Birmingham — UK  
*Ophthalmology; AI and digital health technology evaluation* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** He is the senior clinician-professor counterpart to Xiaoxuan Liu on the AI reporting-standards programme, and brings the seniority the paper needs for desk-review survival. A senior clinician who has spent a decade arguing that AI evidence standards are too weak is structurally sympathetic to a trivial-baseline result.

**Relevant work.** Co-leads the Birmingham AI and Digital Health Group on responsible innovation of AI health technologies (safe, effective, equitable); senior partner on the SPIRIT-AI / CONSORT-AI / DECIDE-AI guideline family.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://www.birminghambrc.nihr.ac.uk/team-member/professor-alastair-denniston  
**Verified from.** https://www.birminghambrc.nihr.ac.uk/team-member/professor-alastair-denniston

Criticised by our paper: no.

**Notes.** The only address printed on the page is the shared BRC inbox birminghamBRC@uhb.nhs.uk -- that is a group mailbox, not his personal address, so I left the email field empty rather than imply it reaches him directly. If approaching this group, Liu is the more direct first contact and can bring him in.

---

### 34. Joann G. Elmore

**Professor of Medicine (David Geffen School of Medicine) and Professor of Health Policy and Management (Fielding School of Public Health); Director, UCLA National Clinician Scholars Program**  
University of California, Los Angeles — USA  
*Diagnostic accuracy and physician variability in breast cancer screening/pathology; ML applied to diagnosis* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** Career-defining work on how badly diagnostic accuracy is measured and how much interpretation varies between readers - she is temperamentally and professionally aligned with a paper whose claim is that the measurement, not the model, is the problem, and she is senior enough to carry desk-review weight at npj Digital Medicine.

**Relevant work.** Co-author, JACR 2022 systematic review of independent external validation of screening-mammography AI (doi 10.1016/j.jacr.2021.11.008). UCLA faculty page lists 'diagnostic accuracy, physician variability, cancer screening (skin and breast), computer image analyses and machine learning' among her stated interests.

**Email (seen in print).** `JElmore@mednet.ucla.edu`  
**Contact page.** https://ph.ucla.edu/about/faculty-staff-directory/joann-g-elmore  
**Verified from.** https://ph.ucla.edu/about/faculty-staff-directory/joann-g-elmore

Criticised by our paper: no.

**Notes.** Email printed on the UCLA Fielding faculty directory page. She is an internist/health-services researcher, NOT a radiologist - she cannot answer 'is the positional prior just anatomy?' from a reading-room perspective. Best used as a second senior methodological co-author alongside a breast radiologist.

---

### 35. Reinhard Heckel

**Prof. Dr. sc. ETH, Professor of Machine Learning, TUM School of Computation, Information and Technology**  
Technical University of Munich (TUM) — Germany  
*Machine learning theory; robustness and limitations of deep learning for MRI reconstruction* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** His research programme is explicitly about where deep learning for MRI fails and why benchmarks mislead — 'limitations in benchmark evaluation' is listed as an active topic on his group page. A person who studies benchmark failure modes is a natural ally for a paper whose whole claim is that a benchmark protocol was measuring position rather than pathology.

**Relevant work.** First author, 'Deep learning for accelerated and robust MRI reconstruction', MAGMA 2024;37(3):335-368 (doi 10.1007/s10334-024-01173-8). Group research page lists work on robustness, distribution shift, denoising under shift, limitations in benchmark evaluation and data-quality effects on AI fairness.

**Email (seen in print).** `reinhard.heckel@tum.de`  
**Contact page.** https://www.ce.cit.tum.de/mli/home/  
**Verified from.** https://www.ce.cit.tum.de/mli/home/

> **Adjacent / read before writing.** - deeply involved in the fastMRI benchmark ecosystem being critiqued

**Notes.** Email printed on the official TUM chair page in obfuscated form as 'reinhard.heckel(at)tum.de' — de-obfuscated here, but the user should know it was not written as a plain mailto. Machine-learning theorist, no clinical training; covers the methods critique, not the radiology sign-off. Deeply involved in the fastMRI benchmark ecosystem, which cuts both ways: credibility, but also proximity to the work being critiqued.

---

### 36. Saurabh Jha

**Professor of Radiology, Hospital of the University of Pennsylvania**  
University of Pennsylvania / Penn Medicine — USA  
*Cardiothoracic imaging* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** A full professor of radiology who is one of the field's best-known sceptics of AI over-claiming and writes prolifically and publicly about it. He is the right person to stress-test whether the positional prior is simply anatomy, and he is temperamentally suited to hunting for any sentence that drifts into over-claiming in the other direction.

**Relevant work.** 'Adapting to Artificial Intelligence: Radiologists and Pathologists as Information Specialists', JAMA 2016 (with Eric Topol); 'Information and Artificial Intelligence', JACR 2018. Co-Program Director, Cardiothoracic Imaging Fellowship.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://www.pennmedicine.org/providers/profile/saurabh-jha  
**Verified from.** https://www.pennmedicine.org/providers/profile/saurabh-jha

Criticised by our paper: no.

**Notes.** No email printed on the Penn Medicine provider page -- that page is a clinical provider listing, so contact via the department. His published AI work is commentary and framing rather than quantitative evaluation methodology, so he is a strong clinical voice but not the person for the biostatistics ask.

---

### 37. Jayashree Kalpathy-Cramer

**Professor; Chief, Division of Artificial Medical Intelligence, Department of Ophthalmology; Director of Health Informatics, CCTSI**  
University of Colorado Anschutz School of Medicine — USA  
*Medical imaging AI methodology, benchmarks and challenge evaluation (PhD, not a clinician)* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** She is the evaluation-methodology member of the ICH challenge team and her stated programme is trustworthy AI and benchmark design, so she is the organiser most likely to read a 'the protocol credits anatomy to the model' argument as a contribution rather than an attack.

**Relevant work.** Co-author, RSNA 2019 Brain CT Hemorrhage Challenge dataset paper (Radiology: AI 2020, affiliated at the time with the Athinoula A. Martinos Center, MGH); research programme in benchmarks, federated learning and trustworthy AI for clinical deployment

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://som.cuanschutz.edu/Profiles/Faculty/Profile/36939  
**Verified from.** https://som.cuanschutz.edu/Profiles/Faculty/Profile/36939

Criticised by our paper: no.

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged. She is a PhD imaging scientist, not a radiologist, so she cannot fill the senior-clinical-co-author slot (ask 1) — she fits the methodology review and would be a strong bridge to the rest of the organising team. Her 2020 affiliation was MGH/Martinos; the CU Anschutz page verifies the current one. No email printed.

---

### 38. Curtis Langlotz

**Professor of Radiology (Integrative Biomedical Imaging Informatics), Medicine, and Biomedical Data Science; Senior Associate Vice Provost for Research**  
Stanford University — USA  
*Radiology; imaging informatics* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** Among the most senior academic radiologists in imaging AI worldwide; an RSNA past-president as co-author would essentially guarantee the paper is read rather than desk-rejected. He has consistently pushed for rigorous benchmarks and standardised evaluation in imaging AI.

**Relevant work.** Director, Center for Artificial Intelligence in Medicine and Imaging (AIMI); Senior Fellow and Associate Director, Stanford HAI; President of the Radiological Society of North America 2024-2025; Chair of the RSNA Board 2022-2023.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://profiles.stanford.edu/curtis-langlotz  
**Verified from.** https://profiles.stanford.edu/curtis-langlotz

> **Adjacent / read before writing.** - Stanford AIMI produces the class of imaging models whose evaluation protocol the paper criticises

**Notes.** TWO CAVEATS. (1) The Stanford profile lists thomajr@stanford.edu as an ADMINISTRATIVE contact, not his own address -- I left the email field empty rather than record an assistant's address as his. (2) Stanford AIMI produces and publishes the class of imaging models whose evaluation protocol this paper criticises, so parts of the critique land near his own centre's output. Lowest expected response rate on this list given the seniority and volume of requests; treat as a stretch ask.

---

### 39. Anwar R. Padhani

**Professor of Cancer Imaging (Institute of Cancer Research); Consultant Radiologist and Lead Consultant for MRI**  
Paul Strickland Scanner Centre, Mount Vernon Cancer Centre / Institute of Cancer Research, London — UK  
*Oncological MRI; prostate multiparametric and diffusion-weighted MRI* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** The audited paper is prostate DWI, and Padhani's core technical expertise is diffusion-weighted MRI of prostate cancer -- he is unusually well placed to answer whether a positional prior is simply anatomy in DWI. As a PI-RADS committee co-chair and PI-CAI co-author he also knows what a properly powered, patient-level AI evaluation looks like, which is the contrast the paper is drawing.

**Relevant work.** PI-RADS v2.1 co-author (doi:10.1016/j.eururo.2019.02.033) and co-chair of the international PI-RADS committee. Co-author of PI-CAI, 'Artificial intelligence and radiologists in prostate cancer detection on MRI' (Lancet Oncol 2024, doi:10.1016/S1470-2045(24)00220-1), a paired non-inferiority study of AI versus 62 radiologists. Specialist expertise in diffusion-weighted MRI specifically.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://stricklandscanner.org.uk/our-team/  
**Verified from.** https://stricklandscanner.org.uk/our-team/

> **Adjacent / read before writing.** - PI-CAI co-author with an investment in prostate MRI AI succeeding (PI-CAI itself is patient-level and is NOT a target of this paper)

**Notes.** No email printed on the Strickland Scanner Centre team page. Scored 4 rather than 5 because PI-CAI is a flagship positive-result AI study he co-authored, so he has some investment in prostate MRI AI succeeding -- though PI-CAI itself is patient-level and methodologically careful, so it is not a target of this paper. Say that explicitly when approaching him. He is based at a cancer centre rather than a university department, which occasionally matters for affiliation lines.

---

### 40. George Shih

**Professor of Clinical Radiology, Department of Radiology**  
Weill Cornell Medical College / Weill Cornell Medicine — USA  
*Radiology; imaging informatics and annotation infrastructure (co-founder, MD.ai)* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** He co-organised the ICH challenge and built the labelling pipeline behind it, so he is the person who actually knows what was and was not released in the DICOM metadata — the factual question underneath your prior-art claim about ImagePositionPatient.

**Relevant work.** Co-author, RSNA 2019 Brain CT Hemorrhage Challenge dataset paper (Radiology: AI 2020); MD.ai, the annotation platform used to build the challenge labels

**Email (seen in print).** `ges9006@med.cornell.edu`  
**Contact page.** https://vivo.weill.cornell.edu/display/cwid-ges9006  
**Verified from.** https://vivo.weill.cornell.edu/display/cwid-ges9006

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged. COMMERCIAL INTEREST: the Weill Cornell VIVO record discloses an ownership stake in MD.ai, so he has a financial stake adjacent to the challenge infrastructure your paper scrutinises. He is not a neuroradiologist (his Weill Cornell clinical page lists abdominal imaging), so he serves the metadata/factual ask, not the clinical-interpretation ask. Institutional email printed on the Cornell VIVO directory record.

---

### 41. Greg Zaharchuk

**Professor of Radiology (Neuroimaging and Neurointervention)**  
Stanford University School of Medicine — USA  
*Neuroradiology; stroke and dementia imaging; AI outcome prediction* · seniority: full professor · **fit_score 4** · published on AI evaluation: yes

**Why fit.** He is senior author of the 2025 Radiology systematic review of deep learning in acute stroke imaging — meaning he has just finished reading the entire literature your paper is criticising the evaluation practice of, and has published a judgement on its methodological quality.

**Relevant work.** Senior author, 'Deep Learning Applications in Imaging of Acute Ischemic Stroke: A Systematic Review and Narrative Summary', Radiology 2025 (DOI 10.1148/radiol.240775, PMC13150509)

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://profiles.stanford.edu/greg-zaharchuk  
**Verified from.** https://profiles.stanford.edu/greg-zaharchuk

Criticised by our paper: no.

**Notes.** Not on the RSNA challenge team. No personal email printed; the Stanford profile lists only an administrative associate's address, which I deliberately have NOT recorded — contact through the profile page. Senior authorship and Stanford affiliation verified against the Europe PMC structured author record for DOI 10.1148/radiol.240775.

---

### 42. Manisha Bahl

**Associate Professor of Radiology, Harvard Medical School; Director, Breast Imaging Fellowship, Massachusetts General Hospital**  
Massachusetts General Hospital / Harvard Medical School — USA  
*Breast imaging (mammography, DBT, high-risk lesions); AI in breast imaging* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** She literally wrote the teaching primer telling breast radiologists how to critically evaluate AI model validation, and has since flagged in AJR that the field lacks proper evaluation studies. That makes her a natural senior clinical reader for a paper whose whole point is that an evaluation protocol, not a model, is the object of criticism.

**Relevant work.** Sole author, 'Artificial Intelligence: A Primer for Breast Imaging Radiologists', J Breast Imaging 2020;2(4):304-314 (doi 10.1093/jbi/wbaa033), which is explicitly about 'methods to validate and evaluate these models' and their limitations. First author, 'Artificial Intelligence for Breast Ultrasound: Expert Panel Narrative Review', AJR 2024;223(6):e2330645 (doi 10.2214/AJR.23.30645), which flags 'the lack of postimplementation evaluation studies'.

**Email (seen in print).** `mbahl1@mgh.harvard.edu`  
**Contact page.** https://researchers.mgh.harvard.edu/profile/14973475/Manisha-Bahl  
**Verified from.** https://researchers.mgh.harvard.edu/profile/14973475/Manisha-Bahl

Criticised by our paper: no.

**Notes.** Email printed on the Mass General Research Institute profile page I read. Her AI work is mammography/ultrasound-weighted rather than breast MRI, so she is stronger on the 'is slice-level AUROC ever right?' question than on DCE-MRI anatomy specifics.

---

### 43. Rajiv Gupta

**Associate Professor of Radiology, Harvard Medical School; Neuroradiology Section Head, Massachusetts General Hospital**  
Massachusetts General Hospital / Harvard Medical School — USA  
*Neuroradiology; CT* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** He ran a real-world evaluation of commercial LVO triage AI in partnership with the FDA and the ACR and published the gap between vendor-reported and deployed performance — the exact professional instinct your paper needs from a senior co-author, applied to stroke.

**Relevant work.** Co-author, 'Real-World Performance of Large Vessel Occlusion Artificial Intelligence-Based Computer-Aided Triage and Notification Algorithms — What the Stroke Team Needs to Know', JACR 2024 (DOI 10.1016/j.jacr.2023.04.003), a joint ACR Data Science Institute / FDA / MGH / Lahey evaluation

**Email (seen in print).** `rgupta1@mgh.harvard.edu`  
**Contact page.** https://researchers.mgh.harvard.edu/profile/1584466/Rajiv-Gupta  
**Verified from.** https://researchers.mgh.harvard.edu/profile/1584466/Rajiv-Gupta

Criticised by our paper: no.

**Notes.** Not on the RSNA ICH team. Email printed on the Mass General Research Institute profile page. Note his listed research keywords skew to CT physics/temporal bone rather than AI — the AI evaluation credential is the JACR paper, so lead with that, not with a generic 'you work on AI' opening. Co-authorship and affiliation verified via the PubMed XML record for PMID 37196818.

---

### 44. Mara Kunst

**Associate Professor of Radiology, UMass Chan–Lahey (UMass Chan Medical School); Adjunct Associate Professor of Radiology, Tufts University School of Medicine; Neuroradiology Section Head, Lahey Hospital & Medical Center**  
Lahey Hospital & Medical Center / UMass Chan Medical School — USA  
*Neuroradiology* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** She led the ACR/FDA real-world LVO triage AI evaluation as a practising neuroradiology section head, so she has already written the sentence your paper needs a clinician to stand behind: that reported performance and deployed clinical value are different quantities.

**Relevant work.** First author, 'Real-World Performance of Large Vessel Occlusion Artificial Intelligence-Based Computer-Aided Triage and Notification Algorithms — What the Stroke Team Needs to Know', JACR 2024

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://physicians.lahey.org/Details/2811  
**Verified from.** https://physicians.lahey.org/Details/2811

Criticised by our paper: no.

**Notes.** Not on the RSNA ICH team. First name confirmed as 'Mara' from the PubMed XML author record for PMID 37196818 (Europe PMC only gave the initial). No email printed on the Lahey physician page; the Tufts faculty profile page rendered empty when fetched, so I could not check it for an address — do not guess one.

---

### 45. Xiaoxuan Liu

**Honorary Associate Professor in AI and Digital Health Technologies; 125th Anniversary Fellow**  
University of Birmingham, Department of Applied Health Sciences — UK  
*Clinician (ophthalmology-trained); AI reporting standards and evidence generation* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** She authored or co-authored nearly every guideline named in the brief, and the Medical Algorithmic Audit is the framework under which 'this evaluation protocol is flawed' is a legitimate, publishable finding. She is the best person to keep the manuscript's language precisely on the protocol-versus-model distinction the paper must not blur.

**Relevant work.** Led SPIRIT-AI and CONSORT-AI; contributed to TRIPOD+AI, STARD-AI and DECIDE-AI; created NICE's Digital Health Technologies Evidence Standards Framework; developed the Medical Algorithmic Audit and CANAIRI; STANDING Together on dataset diversity and algorithmic bias.

**Email (seen in print).** `x.liu.8@bham.ac.uk`  
**Contact page.** https://www.birmingham.ac.uk/staff/profiles/applied-health/liu-xiaoxuan  
**Verified from.** https://www.birmingham.ac.uk/staff/profiles/applied-health/liu-xiaoxuan

Criticised by our paper: no.

**Notes.** Email printed directly on the Birmingham staff profile. Not a radiologist -- clinical training is ophthalmology -- so she does not fully discharge the 'is the positional prior just anatomy' question. Best used alongside a radiologist. Very high response likelihood given her guideline-development role.

---

### 46. Kathryn P. Lowry

**Associate Professor of Radiology, Breast Imaging**  
University of Washington School of Medicine / Fred Hutchinson Cancer Center — USA  
*Breast imaging; cancer screening effectiveness and surveillance imaging outcomes* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** A breast radiologist whose published track record is specifically about external validation and bias in breast AI - she co-wrote both the systematic review that found the validation literature biased and the follow-up infrastructure paper on doing it properly. She would treat a zero-image baseline as a legitimate diagnostic instrument rather than an attack.

**Relevant work.** Co-author, JACR 2022 systematic review of independent external validation of screening-mammography AI (doi 10.1016/j.jacr.2021.11.008); co-author, Ramwala OA et al., 'ClinValAI: A framework for developing Cloud-based infrastructures for the External Clinical Validation of AI in Medical Imaging', Pac Symp Biocomput 2025 (doi 10.1142/9789819807024_0016).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://rad.uw.edu/people/kplowry  
**Verified from.** https://rad.uw.edu/people/kplowry

Criticised by our paper: no.

**Notes.** No email printed on the UW radiology page. Good fallback if Christoph Lee (her frequent co-author and former UW colleague) declines or is unreachable after his move to Wisconsin.

---

### 47. Maciej A. Mazurowski

**Associate Professor, Departments of Radiology, Biostatistics & Bioinformatics, Electrical and Computer Engineering, and Computer Science**  
Duke University — USA  
*Medical image analysis methodology; breast MRI datasets and harmonization* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** He built and released the Duke Breast Cancer MRI dataset, and his own research programme is about the failure modes this paper exploits - scanner style, site signature, and dataset shift leaking into breast MRI models. He holds a joint appointment in Biostatistics & Bioinformatics, so he can also engage with the interval on the trivial fraction.

**Relevant work.** Senior author on the Duke Breast Cancer MRI dataset lineage, e.g. Lew CO et al., Sci Rep 2024;14:5383 (doi 10.1038/s41598-024-54048-2), and on scanner-normalization / harmonization work: Modanwal G et al., Comput Methods Programs Biomed 2021;208:106225 (doi 10.1016/j.cmpb.2021.106225); Cao S et al., J Digit Imaging 2022;36(2):666-678 (doi 10.1007/s10278-022-00755-z).

**Email (seen in print).** `maciej.mazurowski@duke.edu`  
**Contact page.** https://scholars.duke.edu/person/maciej.mazurowski  
**Verified from.** https://scholars.duke.edu/person/maciej.mazurowski

> **Adjacent / read before writing.** - built and released the Duke Breast Cancer MRI dataset the paper analyses

**Notes.** NOT a clinician - he cannot supply the radiologist-of-record signature the paper needs (ask #1), but he is a strong methodological co-author and the natural person to write to about the Duke dataset. Duke Scholars listed him as Associate Professor when I read the page; some secondary sources say Professor - use the title on the page you cite. Email printed on that page.

---

### 48. Fredrik Strand

**Docent (Associate Professor), Department of Oncology-Pathology; consultant radiologist, Breast Imaging Unit**  
Karolinska Institutet / Karolinska University Hospital — Sweden  
*Breast radiology; machine learning for screening mammography and risk prediction* · seniority: associate · **fit_score 4** · published on AI evaluation: yes

**Why fit.** A practising breast radiologist who runs a computational breast imaging group - the rare person who can judge both the clinical anatomy claim and the modelling claim. Nordic screening groups have been the most willing to publish AI results that came out flat, which makes him a plausible yes on a negative-result paper.

**Relevant work.** Leads the Computational Breast Imaging group at Karolinska (https://ki.se/en/research/research-areas-centres-and-networks/research-groups/computational-breast-imaging-fredrik-strands-group), developing and evaluating ML methods in breast radiology including AI-driven screening mammography and risk models.

**Email (seen in print).** `fredrik.strand@ki.se`  
**Contact page.** https://ki.se/en/people/fredrik-strand  
**Verified from.** https://ki.se/en/people/fredrik-strand

> **Adjacent / read before writing.** - his group builds AI systems, so some of his work sits in the class of studies the paper pressure-tests

**Notes.** Email printed on his KI people page. His group also builds AI systems, so some of his own work sits in the class of studies this paper pressure-tests - worth being explicit in the email that the target is evaluation protocol, not his models. European co-author also helps with npj Digital Medicine reviewer diversity.

---

### 49. Karen Drukker

**Research Professor, Department of Radiology (Committee on Medical Physics)**  
University of Chicago — USA  
*Medical physics / machine learning in breast imaging; AI performance evaluation and generalizability* · seniority: other · **fit_score 4** · published on AI evaluation: yes

**Why fit.** Her published niche is the rigour of AI performance evaluation itself - acceptance testing, QA/QC, generalizability, and metric choice - applied largely to breast imaging including breast DCE-MRI. That is precisely the review the trivial-baseline argument needs on the metrics side, and she has no incentive to defend any particular published AUROC.

**Relevant work.** Co-author, Mahmood U et al., 'Artificial intelligence in medicine: mitigating risks and maximizing benefits via quality assurance, quality control, and acceptance testing', BJR Artif Intell 2024;1(1):ubae003 (doi 10.1093/bjrai/ubae003). Also co-author on breast DCE-MRI lesion segmentation work, J Med Imaging 2023;10(6):064502 (doi 10.1117/1.JMI.10.6.064502).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://profiles.uchicago.edu/profiles/display/37451  
**Verified from.** https://profiles.uchicago.edu/profiles/display/37451

Criticised by our paper: no.

**Notes.** Medical physicist, not a clinician - covers the metrics half of ask #1 and part of ask #2, not the clinical confirmation. Her UChicago profile hides the email behind an encrypted link, so nothing to record; contact via that profile page. If a bigger name is wanted from the same group, Maryellen Giger (her senior colleague, MIDRC lead) is the obvious escalation but I did not verify a page for her.

---

### 50. Bradley J. Erickson

**Associate Chair for Research, Department of Radiology (Radiology Informatics Laboratory)**  
Mayo Clinic, Rochester, Minnesota — USA  
*Radiology informatics / imaging AI* · seniority: other · **fit_score 4** · published on AI evaluation: yes

**Why fit.** He runs the lab that published the ICH-detection trustworthiness work Radiology: AI commissioned an editorial about, so he is already on record that ICH model outputs need to be interrogated rather than taken at face value.

**Relevant work.** Senior author, 'Applying Conformal Prediction to a Deep Learning Model for Intracranial Hemorrhage Detection to Improve Trustworthiness', Radiology: AI 2025 (DOI 10.1148/ryai.240032)

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://orcid.org/0000-0001-7926-6095  
**Verified from.** https://orcid.org/0000-0001-7926-6095

Criticised by our paper: no.

**Notes.** WEAKEST VERIFICATION IN THIS LIST — every mayo.edu and mayoclinic.org page returned 403 to both the fetch tool and a direct request, so I could not read a Mayo faculty page. What I did verify: his ORCID record (0000-0001-7926-6095) lists 'Associate Chair for Research, Radiology, Mayo Clinic Minnesota' from 1994, and the Europe PMC structured author record for DOI 10.1148/ryai.240032 lists him at the Radiology Informatics Laboratory, Department of Radiology, Mayo Clinic. ORCID employment entries are self-reported. His academic rank is NOT verified — do not address him as 'Professor' without checking. No email seen in print.

---

### 51. Hersh Chandarana

**MD, MBA, Professor of Radiology; Vice Chair for Innovation and Predictive Diagnostics**  
NYU Langone Health, Department of Radiology — USA  
*Body/abdominal MRI, quantitative MRI, fast motion-robust imaging* · seniority: full professor · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** A full professor MD with a formal innovation-evaluation remit at the institution that produced three of our five cohorts, and a named author on the fastMRI prostate dataset. He can speak to whether a positional prior in prostate bpMRI is 'just anatomy' from direct experience with the acquisitions, and his RF-coil work means the k-space and coil-combination methods will not be opaque to him.

**Relevant work.** Co-author, 'FastMRI Prostate: A publicly available biparametric MRI dataset to advance machine learning for prostate cancer imaging' (PMC10153282; published in Scientific Data). Research spans dynamic quantitative MRI, diffusion MRI for renal cancer, fast motion-robust imaging, AI for ultra-low-field MRI, and RF coil development.

**Email (seen in print).** `hersh.chandarana@nyulangone.org`  
**Contact page.** https://cbiweb.net/team/hersh-chandarana-md/index.html  
**Verified from.** https://cbiweb.net/team/hersh-chandarana-md/index.html

> **Adjacent / read before writing.** - author on the fastMRI prostate dataset the analysis touches

**Notes.** Email printed on the NYU Center for Biomedical Imaging team page. Author on the fastMRI prostate dataset that the paper's analysis touches — approach on evaluation-protocol grounds, not dataset grounds. His Vice Chair title is given as 'Vice-Chair for Innovation and Predictive Diagnostics' on the CBI page and 'Vice Chair of Innovation & Predictive Diagnostics' elsewhere; the CBI wording is what I verified. I found no publication of his that is specifically a critique of AI evaluation, hence 'unclear' — he is a domain expert rather than a known critic.

---

### 52. Daniel J. A. Margolis

**Department of Radiology (Abdominal Imaging); described in departmental and Scholar listings as Professor of Radiology and Director of Prostate MRI**  
Weill Cornell Medicine, New York, NY — USA  
*Abdominal / genitourinary radiology, prostate MRI, quantitative imaging* · seniority: full professor · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** A PI-RADS v2.1 committee member whose recent work is specifically about quantitative prostate MRI biomarkers and the validation burden they carry -- and he has already co-published with Obuchowski, so a combined clinical-plus-biostatistics approach through him is plausible. Useful for ask 3 as a prostate MRI director.

**Relevant work.** PI-RADS v2.1 co-author (doi:10.1016/j.eururo.2019.02.033). Lead author of 'Quantitative Prostate MRI' in the AJR Special Series on Quantitative Imaging (AJR 2024, doi:10.2214/AJR.24.31715), co-authored with Nancy Obuchowski, Clare Tempany and Andrei Purysko, which concludes that which quantitative technique wins 'will depend on validation across a myriad of platforms and use cases'.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://radiology.weill.cornell.edu/daniel-margolis-md  
**Verified from.** https://radiology.weill.cornell.edu/daniel-margolis-md

Criticised by our paper: no.

**Notes.** WEAKEST-VERIFIED ROW ON THIS LIST for title specifically. The Weill Cornell radiology page I read confirms name, Department of Radiology, Weill Cornell Medicine and Abdominal Imaging, but does NOT state his professorial rank on the page body; 'Professor of Radiology, director of prostate MRI' comes from the department's own listing text and his Google Scholar header, which I did not open directly. Institution is solid; confirm rank before writing. No email printed.

---

### 53. Michael P. Recht

**MD, Louis Marx Professor and Chair, Department of Radiology**  
NYU Langone Health — USA  
*Musculoskeletal radiology; cartilage imaging; accelerated knee MRI* · seniority: full professor · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** A department chair's signature is the strongest possible desk-review armour for a first author who is a high-school student, and he is the chair of the exact department that produced the data. Musculoskeletal/knee is his subspecialty, matching the fastMRI knee cohort directly.

**Relevant work.** Chair of NYU Radiology, the department behind fastMRI. Public spokesperson for the NYU work finding AI-accelerated fastMRI images 'essentially indistinguishable in appearance from standard clinical MRI exams'; his own research uses MRI to study articular cartilage degeneration (the knee domain of the fastMRI knee cohort).

**Email (seen in print).** `michael.recht@nyulangone.org`  
**Contact page.** https://cbiweb.net/team/michael-recht/index.html  
**Verified from.** https://cbiweb.net/team/michael-recht/index.html

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** publicly associated with the NYU studies concluding DL-accelerated MRI is diagnostically interchangeable with conventional MRI; the paper's argument is adjacent to a critique of that literature's evaluation methodology. Most likely person here to read it as an attack

**Notes.** CAUTION — READ BEFORE WRITING. He is publicly associated with the NYU clinical-validation studies concluding that deep-learning-accelerated MRI is diagnostically interchangeable with conventional MRI. This paper's argument is adjacent to a critique of that literature's evaluation methodology, so he is the most likely person on this list to read it as an attack on his own group's claims. Still worth listing because of the credibility he would confer, but the approach must be unusually careful and must lead with 'the protocol, not the model, and not your data'. Email printed on the NYU CBI team page. As a sitting chair he is also the least likely to have time.

---

### 54. Lubdha M. Shah

**Professor, Department of Radiology (Neuroradiology), School of Medicine**  
University of Utah — USA  
*Neuroradiology; spine imaging, advanced MRI (DTI, perfusion, spectroscopy)* · seniority: full professor · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** A full-professor neuroradiologist who personally annotated head CTs for the RSNA ICH challenge — she is the right person for the one-paragraph clinical confirmation about where haemorrhage sits in the stack, because she has read those exact studies slice by slice.

**Relevant work.** Co-author (annotating neuroradiologist), RSNA 2019 Brain CT Hemorrhage Challenge dataset paper, listed under Department of Neuroradiology, University of Utah Health Sciences Center

**Email (seen in print).** `u0619721@utah.edu`  
**Contact page.** https://our.utah.edu/faculty-mentor/lubdha-shah/  
**Verified from.** https://our.utah.edu/faculty-mentor/lubdha-shah/

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label (annotator/co-author role, lower conflict than the lead authors)

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM (annotator/co-author) — flagged. Her role was labelling rather than protocol design, so she is a lower-conflict route into that group than the lead authors. The email is the uNID-style address printed verbatim on the University of Utah undergraduate-research faculty-mentor page; the Utah Health clinical directory returned 403 and could not be cross-checked. Her affiliation was verified from the Europe PMC structured author record for the challenge paper (which lists University of Utah, correcting a summarising error I first got from the PMC HTML).

---

### 55. Daniel K. Sodickson

**MD, PhD, Professor (listed as Adjunct Professor on the CBI page), Department of Radiology; Director, Bernard and Irene Schwartz Center for Biomedical Imaging**  
NYU Langone Health — USA  
*Parallel MRI (originator), RF coil design, rapid imaging, compressed sensing* · seniority: full professor · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** He originated parallel imaging, which is the direct ancestor of every coil-combination and GRAPPA step our k-space pipeline performs — so if a reviewer challenges how phase was extracted from multi-coil raw data, his name on the paper closes that line of attack. He is also an MD-PhD, so he straddles the physics and clinical registers.

**Relevant work.** Played a leading role in the genesis of parallel MRI (SMASH). Co-author, 'FastMRI Prostate' dataset paper (PMC10153282). Research spans RF transmitter/detector design, tissue electrical property mapping, high-field MRI and AI in reconstruction.

**Email (seen in print).** `daniel.sodickson@nyulangone.org`  
**Contact page.** https://cbiweb.net/team/daniel-k-sodickson-md-phd/index.html  
**Verified from.** https://cbiweb.net/team/daniel-k-sodickson-md-phd/index.html

> **Adjacent / read before writing.** - fastMRI dataset co-author

**Notes.** Email printed on the NYU CBI team page. That page lists him as 'Adjunct Professor' while also giving his directorship of the Schwartz Center and Vice-Chair for Research role — the adjunct designation may reflect a recent transition, so confirm his current status before addressing him. He is a fastMRI dataset co-author. Very senior and likely time-constrained; realistically a lower-probability yes than the mid-career methodologists, which is why he is not scored 5.

---

### 56. Francesco Giganti

**Academic radiologist, Division of Surgery and Interventional Science, UCL; Department of Radiology, UCLH NHS Foundation Trust**  
University College London / University College London Hospitals NHS Foundation Trust, London — UK  
*Prostate MRI; imaging quality standards* · seniority: associate · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** PI-QUAL exists because Giganti argued that prostate MRI studies were being evaluated without anyone checking whether the input images were adequate -- structurally the same move as this paper, which argues models are being evaluated without anyone checking whether a no-pixel baseline already wins. He is a natural sympathiser and can speak to whether image content is even doing the work.

**Relevant work.** Senior author of 'New Prostate MRI Scoring Systems (PI-QUAL, PRECISE, PI-RR, and PI-FAB): Expert Panel Narrative Review' (AJR 2024, doi:10.2214/AJR.24.30956), which 'critically examines these new prostate MRI scoring systems, analyzing the available evidence, delineating current limitations, and proposing solutions for improvement'. Developer of PI-QUAL, the prostate MRI image-quality scoring system, and lead on its version 2 revision.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://pubmed.ncbi.nlm.nih.gov/38568038/  
**Verified from.** https://pubmed.ncbi.nlm.nih.gov/38568038/

Criticised by our paper: no.

**Notes.** Affiliation verified from the AJR 2024 author list, which prints his full postal address at Charles Bell House, UCL. Secondary listings describe him as 'Associate Professor of Radiology' at UCL but I could NOT confirm that rank on a ucl.ac.uk page (the profile URLs I tried 404'd and DuckDuckGo began serving CAPTCHAs), so the rank is unconfirmed -- verify before addressing him by title. No email seen in print.

---

### 57. Angela Tong

**MD, Associate Professor of Radiology and of Urology; Director of Prostate Imaging**  
NYU Grossman School of Medicine / NYU Langone Health — USA  
*Prostate MRI, female pelvic imaging, deep learning in pelvic imaging* · seniority: associate · **fit_score 4** · published on AI evaluation: unclear

**Why fit.** The best single person for ask #3, the one-paragraph clinical confirmation on positional concentration of lesions. She is the director of prostate imaging and reads these studies daily, so she can state on the record whether prostate lesion prevalence genuinely concentrates by slice position — which is the exact empirical claim the trivial baseline's 0.854 rests on. She is also a fastMRI Prostate author, so she knows the labels' provenance.

**Relevant work.** Co-author, 'FastMRI Prostate: A publicly available biparametric MRI dataset to advance machine learning for prostate cancer imaging' (PMC10153282). NYU profile lists 'novel imaging techniques and deep learning algorithms' and prostate/female pelvic imaging as her focus; ORCID 0000-0002-4733-3414 carries the fastMRI Prostate paper.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://nyulangone.org/doctors/1326308404/angela-tong  
**Verified from.** https://nyulangone.org/doctors/1326308404/angela-tong

> **Adjacent / read before writing.** - co-author of the fastMRI prostate dataset

**Notes.** NO EMAIL RECORDED — none is printed on her NYU profile; use the contact_page. She is a co-author of the fastMRI prostate dataset. Associate professor, so she is senior but not the most senior possible signature; ideal as the organ-specific clinical confirmer rather than as the sole senior co-author. Dual appointment in Radiology and Urology.

---

### 58. Marzyeh Ghassemi

**Germeshausen Career Development Professor; Associate Professor, EECS and Institute for Medical Engineering and Science**  
Massachusetts Institute of Technology — USA  
*Machine learning for health; robustness, privacy and fairness* · seniority: associate · **fit_score 3** · published on AI evaluation: yes

**Why fit.** One of the most cited voices on clinical ML reliability and on the gap between benchmark performance and clinical validity. She would engage seriously with a negative result and is well placed to police the 'flawed protocol' versus 'learned nothing' distinction.

**Relevant work.** Group focus stated as 'creating and applying machine learning to understand and improve health in ways that are robust, private and fair'; research areas include healthcare ML, healthy ML, clinical informatics.

**Email (seen in print).** `mghassem@mit.edu`  
**Contact page.** https://imes.mit.edu/people/ghassemi-marzyeh  
**Verified from.** https://imes.mit.edu/people/ghassemi-marzyeh

Criticised by our paper: no.

**Notes.** The IMES page prints the address as 'mghassem [at] mit.edu'; I have recorded the de-obfuscated form, which is unambiguous. Not a clinician -- she cannot answer the anatomy question or the clinically-correct-unit question on the record, so she is a methodology co-author, not the senior clinical co-author the paper needs. Very high inbound volume; expect a low response rate.

---

### 59. Safwan S. Halabi

**Vice Chair of Informatics, Medical Imaging; Section Head, Fetal Imaging; Associate Professor of Radiology (Pediatric Radiology), Northwestern University Feinberg School of Medicine**  
Ann & Robert H. Lurie Children's Hospital of Chicago / Northwestern University — USA  
*Paediatric radiology; imaging informatics* · seniority: associate · **fit_score 3** · published on AI evaluation: yes

**Why fit.** An RSNA ICH challenge co-organiser who now holds an informatics vice-chair role, giving him standing to speak to how the challenge dataset and its metadata were assembled and released.

**Relevant work.** Co-author, RSNA 2019 Brain CT Hemorrhage Challenge dataset paper (affiliated at the time with Stanford University Department of Radiology)

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://www.luriechildrens.org/en/doctors/halabi-safwan-s/  
**Verified from.** https://www.luriechildrens.org/en/doctors/halabi-safwan-s/

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** RSNA 2019 ICH challenge organising team; the paper directly engages the organisers' on-record statement that the released metadata cannot determine the label

**Notes.** RSNA 2019 ICH CHALLENGE ORGANISING TEAM — flagged. His subspecialty is paediatric radiology and fetal imaging, so he is a poor fit for the adult-ICH clinical paragraph; list him as an organising-team contact, not a clinical co-author. His 2020 affiliation was Stanford; the Lurie page verifies the current one. No email printed on the Lurie page, and the Feinberg faculty profile URL returned 404.

---

### 60. Chris McIntosh

**Associate Professor, Department of Medical Biophysics; Senior Scientist, Peter Munk Cardiac Centre Research Institute**  
University of Toronto (Temerty Faculty of Medicine) and University Health Network — Canada  
*AI in medicine; biomedical imaging; cardiovascular and cancer imaging* · seniority: associate · **fit_score 3** · published on AI evaluation: yes

**Why fit.** He is the senior author of the closest published relative of this paper, and he published it in npj Digital Medicine -- one of the two target venues. He has already made the argument that reported performance is systematically inflated by a non-anatomical signal, so he needs no convincing of the premise.

**Relevant work.** Senior and corresponding author, Ong Ly et al., 'Shortcut learning in medical AI hinders generalization: method for estimating AI model generalization without external data', npj Digital Medicine 2024;7:124 (PubMed PMID 38744921, DOI 10.1038/s41746-024-01118-4). That paper finds performance overestimated by up to 20% on average across 13 datasets due to shortcut learning of hidden data acquisition biases.

**Email (seen in print).** `chris.mcintosh@uhn.ca`  
**Contact page.** https://medbio.utoronto.ca/faculty/mcintosh  
**Verified from.** https://medbio.utoronto.ca/faculty/mcintosh

> **Adjacent / read before writing.** - Ong Ly 2024 npj Digital Medicine group; their bias-corrected estimator is a rival framing of the same problem, so he will want the paper to cite and position against his own

**Notes.** ADJACENT/COMPETITOR, read before writing. This is the Ong Ly 2024 group the brief flags. Their bias-corrected estimator P is a rival framing of the same problem, so he is simultaneously the most sympathetic reader and the person most likely to want the paper to cite and position against his own. Not a radiologist -- he is a computer scientist by training -- so he does not answer the anatomy question. Email verified twice: printed as the corresponding-author line in the npj Digital Medicine paper AND on the U of T Medical Biophysics faculty page.

---

### 61. Eric Karl Oermann

**Associate Professor of Neurosurgery and Radiology**  
NYU Grossman School of Medicine / NYU Langone Health — USA  
*Neurosurgery (spine, epilepsy); machine learning in medicine* · seniority: associate · **fit_score 3** · published on AI evaluation: yes

**Why fit.** He is the senior author behind both named starting-point papers (Zech 2018 and Badgeley 2019), which are the two clearest precedents for 'the model scored well without using the pathology'. He holds a joint Radiology appointment at NYU, so he can speak to imaging evaluation, and he has already survived reviewing this exact class of negative result twice.

**Relevant work.** Corresponding author, Zech et al., PLOS Medicine 2018 (DOI 10.1371/journal.pmed.1002683) -- showed networks identified the source hospital with 99.95-99.98% accuracy and calibrated to site prevalence rather than pathology. Also the senior author on the Badgeley 2019 hip-fracture work the brief names.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://nyulangone.org/doctors/1235498403/eric-k-oermann  
**Verified from.** https://nyulangone.org/doctors/1235498403/eric-k-oermann

Criticised by our paper: no.

**Notes.** EMAIL WARNING. His corresponding-author address printed in the 2018 PLOS Medicine paper is eric.oermann@mountsinai.org. He has since moved from Mount Sinai to NYU Langone, so that address is very likely dead -- I deliberately did NOT put it in the email field. Contact via the NYU Langone profile. He is a neurosurgeon, not a radiologist, so he does not fully discharge the clinical-radiology ask.

---

### 62. James Zou

**Associate Professor of Biomedical Data Science; by courtesy, Computer Science and Electrical Engineering**  
Stanford University — USA  
*Statistical machine learning; auditing and evaluation of medical AI* · seniority: associate · **fit_score 3** · published on AI evaluation: yes

**Why fit.** The FDA-approvals audit is the definitive study showing that medical AI is routinely cleared on evaluation protocols that do not establish clinical validity. That is this paper's thesis applied to regulators, so he is unlikely to dispute the premise and can lend audit-methodology credibility.

**Relevant work.** 'How medical AI devices are evaluated: limitations and recommendations from an analysis of FDA approvals', Nature Medicine (the Wu/Zou group the brief names); 'How to evaluate deep learning for cancer diagnostics - factors and recommendations'; work on accountable ML and bias in biomedical AI.

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://profiles.stanford.edu/james-zou  
**Verified from.** https://profiles.stanford.edu/james-zou

Criticised by our paper: no.

**Notes.** No email printed on the Stanford profile. Not a clinician, so he does not satisfy the senior clinical co-author ask -- best approached for the evaluation-methodology review or as a supporting co-author. Very high inbound volume.

---

### 63. Jonathan I. Tamir

**PhD, Assistant Professor, Chandra Family Department of Electrical and Computer Engineering; Jack Kilby/Texas Instruments Endowed Faculty Fellow; appointment in Department of Diagnostic Medicine, Dell Medical School**  
The University of Texas at Austin — USA  
*Computational MRI, signal processing, machine learning for imaging* · seniority: assistant · **fit_score 3** · published on AI evaluation: yes

**Why fit.** Co-author of the data-crimes paper, so he has already argued in print that public-dataset reuse produces inflated results — the precise structural claim this manuscript makes. His Dell Medical School appointment means he is used to translating a signal-processing argument into terms a clinical audience will accept.

**Relevant work.** Second author, 'Implicit data crimes: Machine learning bias arising from misuse of public data', PNAS 2022;119(13):e2117203119 (doi 10.1073/pnas.2117203119). Leads the Computational Sensing and Imaging Lab; joint appointments at the Oden Institute and Dell Medical School.

**Email (seen in print).** `jtamir@utexas.edu`  
**Contact page.** https://users.ece.utexas.edu/~jtamir/  
**Verified from.** https://users.ece.utexas.edu/~jtamir/

Criticised by our paper: no.

**Notes.** Email printed on his UT Austin faculty page. Assistant professor — too junior to satisfy the 'senior co-author for desk-review survival' requirement, and a PhD rather than a clinician, hence the score of 3 despite excellent topical alignment. Overlaps heavily with Shimron and Lustig (same PNAS paper); useful as a fallback or as a second methods reader rather than a parallel ask.

---

### 64. Robyn L. Ball

**Senior Computational Scientist (Ph.D., Statistics)**  
The Jackson Laboratory, Bar Harbor, Maine — USA  
*Biostatistics / statistical methodology (formerly Senior Statistician, Quantitative Sciences Unit, Stanford University)* · seniority: other · **fit_score 3** · published on AI evaluation: yes

**Why fit.** She is the statistician who worked on the RSNA challenges themselves — the right reviewer for the matching rule and the trivial fraction's confidence interval (ask 2), and one who already understands how those competition datasets were split and scored.

**Relevant work.** Co-author, RSNA 2019 Brain CT Hemorrhage Challenge dataset paper (listed as Quantitative Sciences Unit, Stanford University); her JAX biography states she collaborated with the RSNA on its 2019 and 2020 Kaggle competitions

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://www.jax.org/people/robyn-ball  
**Verified from.** https://www.jax.org/people/robyn-ball

> **CRITICISED BY OUR PAPER — read COLLABORATORS.md section 4 before writing.** statistician on the RSNA ICH challenge team whose released metadata claim the paper engages

**Notes.** RSNA CHALLENGE STATISTICIAN — flagged, and identity carefully checked: the JAX biography explicitly names both the prior Stanford senior-statistician role and the RSNA 2019/2020 Kaggle collaboration, which is what ties this Robyn Ball to the 'Robyn Ball, PhD, Quantitative Sciences Unit, Stanford' on the ICH challenge author list. Her current work is mouse genetics, so the ask must be framed as a discrete statistical review, not as an imaging collaboration. Email on the JAX page is obfuscated against scraping and I could not read it; her ORCID could not be resolved (two other Robyn Balls exist).

---

### 65. Gaël Varoquaux

**Research Director (Directeur de Recherche, HDR), Soda team**  
Inria (French National Institute for Research in Digital Science and Technology) — France  
*Machine learning methodology; ML evaluation; health data analytics* · seniority: other · **fit_score 3** · published on AI evaluation: yes

**Why fit.** He is one of the few senior methodologists who works explicitly on how ML evaluation itself misleads, including benchmarking pitfalls and the statistics of model comparison. That makes him a credible reviewer for the matching rule and the confidence interval on the trivial fraction.

**Relevant work.** Lists 'machine learning evaluation' as a core research interest alongside missing data, causal inference, public health analytics and personalised medicine; former editorial roles at NeuroImage; Chief Science Officer of Probabl (scikit-learn).

**Email.** *none recorded — none was seen in print. Do not construct one.*  
**Contact page.** https://gael-varoquaux.info/about.html  
**Verified from.** https://gael-varoquaux.info/about.html

Criticised by our paper: no.

**Notes.** Research Director at Inria is professor-equivalent in the French system; I recorded 'other' rather than overstate the rank mapping. NO EMAIL RECORDED ON PURPOSE: his page prints the literal template 'firstname.lastname@inria.fr' rather than an actual address, and constructing one from that pattern is exactly what the brief forbids. The inria.fr staff page is behind an Anubis bot-protection wall and could not be read. Not a clinician.

---

### 66. Berkin Bilgic

**PhD, Associate Professor of Radiology**  
Massachusetts General Hospital / Harvard Medical School (Athinoula A. Martinos Center for Biomedical Imaging) — USA  
*MRI acquisition and reconstruction; quantitative susceptibility mapping (QSM); quantitative parameter mapping* · seniority: associate · **fit_score 3** · published on AI evaluation: unclear

**Why fit.** QSM is the discipline built entirely on the MRI phase signal, so he is the natural referee for the supporting study's central premise — whether phase in raw k-space plausibly carries the clinical information the original papers claimed. If a reviewer asks 'did you handle phase wrapping, background field removal and coil phase offsets correctly', he is the person whose name settles it.

**Relevant work.** Heads the BRAIN lab at the Martinos Center; work spans fast clinical imaging, self-supervised machine learning, quantitative parameter mapping and diffusion imaging, including open-source multi-echo gradient-echo acquisitions for R2* and QSM mapping. Mass General Research Institute profile describes his development of QSM as 'a novel contrast mechanism that probes the magnetic properties of tissues'.

**Email (seen in print).** `bbilgic@mgh.harvard.edu`  
**Contact page.** https://nmr.mgh.harvard.edu/~berkin/index.html  
**Verified from.** https://nmr.mgh.harvard.edu/~berkin/index.html

Criticised by our paper: no.

**Notes.** Email was printed on his lab page in obfuscated form as 'bbilgic AT mgh.harvard.edu' — de-obfuscated here; the user should know it was not a plain mailto link. PhD, not a clinician, and not known to me for published critiques of AI evaluation, so he covers the phase-physics methods question narrowly rather than the senior-clinical-co-author ask. Included because the phase/k-space claim is the supporting study's most technically attackable point.

---

### 67. Errol Colak

**Associate Professor, Department of Medical Imaging; Odette Professorship in Artificial Intelligence for Medical Imaging; staff radiologist, St. Michael's Hospital / Unity Health Toronto**  
University of Toronto, Temerty Faculty of Medicine / Unity Health Toronto — Canada  
*Abdominal imaging; AI and machine learning in medical imaging* · seniority: associate · **fit_score 3** · published on AI evaluation: unclear

**Why fit.** He chairs the RSNA AI Committee, the body that runs the imaging challenges — including the 2019 ICH challenge whose metadata claim this paper engages. If you want the organising team to respond as an institution rather than as scattered individuals, he is the door.

**Relevant work.** Chair of the RSNA AI Committee (reported 12 Dec 2025); holder of the Odette Professorship in AI for Medical Imaging since 2019; research programme in machine-learning applications in medical imaging

**Email (seen in print).** `errol.colak@unityhealth.to`  
**Contact page.** https://medical-imaging.utoronto.ca/faculty/errol-colak  
**Verified from.** https://medical-imaging.utoronto.ca/faculty/errol-colak

Criticised by our paper: no.

**Notes.** NOT on the ICH challenge author list, and his clinical subspecialty is abdominal imaging, not neuroradiology — so he is an institutional route, not a candidate for the clinical-interpretation paragraph. The RSNA AI Committee chairship comes from a trade-press item (https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15774089/colak-becomes-chair-of-rsna-ai-committee, 12 Dec 2025), not from an RSNA page — confirm before citing it to him. Institutional email printed on the U of T faculty page.

---

### 68. Patricia M. Johnson

**PhD, Assistant Professor, Center for Biomedical Imaging / Department of Radiology**  
NYU Langone Health — USA  
*Deep learning MR image reconstruction and disease detection; ultra-low-field MRI* · seniority: assistant · **fit_score 3** · published on AI evaluation: unclear

**Why fit.** She is the connective tissue of the fastMRI dataset papers — last author on prostate, second author on breast — so she knows precisely how the labels, splits and k-space were constructed for the cohorts this paper re-analyses. On a factual question like 'was this split truly patient-disjoint and how were slice labels assigned', she is the most authoritative single person available.

**Relevant work.** Senior/last author on the fastMRI Prostate dataset paper (PMC10153282) and second author on 'FastMRI Breast' (doi 10.1148/ryai.240345). Long-running fastMRI involvement including work on prospectively accelerated clinical knee MRI reconstruction; current focus on AI for ultra-low-field MRI image quality.

**Email (seen in print).** `patricia.johnson3@nyulangone.org`  
**Contact page.** https://cbiweb.net/team/patricia-johnson/index.html  
**Verified from.** https://cbiweb.net/team/patricia-johnson/index.html

> **Adjacent / read before writing.** - author of the fastMRI dataset papers under re-analysis

**Notes.** Email printed on the NYU CBI team page. Assistant professor, so she does not carry senior-co-author weight on her own, and she is a PhD rather than a clinician — value here is methodological accuracy and dataset provenance, not clinical credibility or desk-review survival. She is an author of the datasets under re-analysis; approach on protocol grounds. Scored 3 for the senior-co-author ask specifically despite being highly relevant technically.

---

*Generated by pooling and deduping five independent verified domain searches. Row count is what verified; nothing was padded to reach a target.*
