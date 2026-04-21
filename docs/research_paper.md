# Deploying a Production-Ready MLOps System for Early Student-Dropout Prediction in Zimbabwean Higher Education: A Champion–Challenger, Human-in-the-Loop Architecture

**Author.** Samson Fukuzeya
**Programme.** MSc Computer Science (Artificial Intelligence & Machine Learning)
**Institution.** University of Zimbabwe, Faculty of Science and Engineering
**Submission date.** 21 April 2026

---

## Abstract

Dropout is a compounding crisis in African higher education, yet Zimbabwean universities still rely on manual, end-of-semester reviews to identify at-risk students — long after meaningful intervention is possible. This study designs, implements, and critically evaluates a reproducible, production-grade machine-learning operations (MLOps) system that predicts the three-way outcome **Dropout / Enrolled / Graduate** for undergraduate students and recommends concrete, rule-based interventions (bursar referral, peer-tutor assignment, counselling, dean-of-students liaison). The system is trained on the UCI *Predict Students' Dropout and Academic Success* dataset (n = 4,424, 36 features) and productionised behind a FastAPI service, an Angular/Tailwind risk dashboard, Pandera data contracts, MLflow tracking and registry, DVC data/model lineage, Evidently AI drift reports, Prometheus + Grafana telemetry, Docker Compose orchestration, and GitHub Actions continuous integration. Methodologically, the study departs from single-algorithm claims by running an honest five-way bake-off (Logistic Regression, Random Forest, XGBoost, LightGBM, PyTorch MLP) under stratified 5-fold cross-validation, selecting a champion by macro-F1 with Dropout-recall tie-break and gating every future promotion behind a three-part rule (effect size, per-class no-regression, McNemar statistical significance). The realised holdout **macro-F1** on the frozen test set is **0.730** with a threshold-tuned **Dropout recall of 0.873** (target: ≥0.85), a **post-temperature expected calibration error of 0.086**, and a **maximum fairness disparity of 0.146** on the Scholarship-holder attribute. Retraining is closed-loop: an Evidently drift share ≥ 0.30 auto-triggers a challenger run, and a configurable Human-in-the-Loop (HITL) flag decides whether a promotion reaches Production automatically or waits for a registered administrator's JWT-authorised approval. The paper's contribution is therefore less a new algorithm than a defensible, reproducible blueprint for how a single African researcher can stand up an ethically explainable early-warning system on commodity infrastructure and hand it to a faculty office that may never have seen a Jupyter notebook.

**Keywords:** MLOps; student dropout prediction; champion–challenger; human-in-the-loop; SHAP; Pandera; Evidently AI; Zimbabwean higher education.

---

## 1. Introduction

Zimbabwean public universities enrolled roughly 146,000 undergraduates in 2023, yet institutional reporting consistently places four- to six-year attrition rates at 20–35% for cohorts entering since 2018, with the steepest losses clustered in the first two semesters (Ministry of Higher and Tertiary Education, Innovation, Science and Technology Development [MHTEISTD], 2023; Mutambara & Bayaga, 2024). The human cost is absorbed four times: by the student (forfeited tuition, lost wage premium), by the institution (underutilised capacity), by the funder (unrecovered grants and loans), and by the labour market (a graduate-deficit economy). Despite this, no Zimbabwean university currently operates a data-driven early-warning system (EWS) capable of flagging risk within the first academic term, triaging it to a specific support service, and learning from outcomes. Academic-affairs offices instead rely on end-of-semester transcript reviews — a diagnostic conducted after the point at which intervention is still cheap and effective (Bañeres et al., 2023; Albreiki et al., 2021).

Machine-learning-based EWSs are by now well-evidenced in the international literature as *feasible* (Delen, 2011; Nagy & Molontay, 2024), yet the gap between a working Jupyter notebook and a dependable production service remains the dominant reason they do not ship in practice (Paleyes et al., 2022; Shankar et al., 2024). This gap is precisely what Machine Learning Operations — MLOps — exists to close (Kreuzberger et al., 2023). The present study therefore frames dropout prediction not as an isolated modelling problem but as a full-stack reproducibility and governance problem, which is where Zimbabwean institutions are in fact stuck.

## 2. Problem Statement

The specific, unresolved problem this research addresses is: *how can a Zimbabwean university stand up a reproducible, ethically explainable, continuously monitored dropout-prediction service — with a documented human-in-the-loop governance path — on commodity infrastructure and with a single-researcher operating budget?* Three sub-problems follow:

1. **Evidence gap.** No peer-reviewed study reports a deployed MLOps dropout system contextualised for Zimbabwean support catalogues (bursar, peer-tutor, counselling, dean's office).
2. **Engineering gap.** Existing African EWS prototypes rarely version data and models end-to-end, expose a live-drift monitor, or encode a champion–challenger promotion rule — so their outputs cannot be defended, reproduced, or safely replaced (Amershi et al., 2019; Sculley et al., 2015).
3. **Governance gap.** Predictions about educational futures are ethically consequential; any deployed system must therefore pair individual-level explanations with a human override before model promotions (Mitchell et al., 2019; Barocas et al., 2023).

## 3. Research Objectives and Questions

**Primary objective.** Design, implement, and evaluate an end-to-end MLOps system that predicts the three-class academic outcome for each student and recommends a concrete intervention pathway, while exceeding the Master's-grade engineering target of **macro-F1 ≥ 0.85** and **Dropout recall ≥ 0.85**.

**Secondary objectives.**

- **O1.** Benchmark five algorithmic families (LogReg, RF, XGBoost, LightGBM, MLP) under a single stratified CV harness to resist the "trust me, XGBoost" fallacy.
- **O2.** Enforce Pandera schema contracts at every I/O boundary so no malformed payload reaches the model.
- **O3.** Deliver per-prediction SHAP explanations alongside a transparent, rule-based Zimbabwean intervention recommender.
- **O4.** Close the loop with Evidently AI drift detection, an auditable promotion gate, and a configurable HITL toggle.

**Research questions.**

- **RQ1.** Does a disciplined five-model bake-off with a formal promotion rule materially outperform a single-algorithm baseline on the UCI dropout dataset?
- **RQ2.** What calibrated classification threshold maximises Dropout recall without catastrophic loss of macro-F1, and does temperature scaling improve real-world trust?
- **RQ3.** Can a single researcher, using free and open-source tooling only, operationalise an EWS that meets the reproducibility, governance, and fairness audit criteria expected at Master's level?

## 4. Literature Review

### 4.1 Dropout prediction in higher education

Early-generation dropout studies framed the task as logistic or decision-tree classification on SIS features (Delen, 2011). Recent syntheses consistently report gradient-boosting ensembles as the strongest single-model family (Aggarwal et al., 2022; Nagy & Molontay, 2024), while highlighting that predictive performance plateaus unless "academic momentum" features — credits attempted vs. approved per semester — are engineered explicitly (Realinho et al., 2021). Bañeres et al. (2023) and Kuleto et al. (2021) argue, however, that raw predictive skill is the weaker half of the EWS value proposition; the stronger half is what happens after the flag fires, i.e., whether the system routes the case to a named support service.

### 4.2 The MLOps pivot

Sculley et al. (2015) crystallised the observation that a deployed ML system is mostly not ML: it is monitoring, validation, orchestration, retraining, and governance. Amershi et al. (2019) formalised the software-engineering discipline for ML; Breck et al. (2017) proposed the "ML Test Score"; Polyzotis et al. (2018) and Schelter et al. (2018) foregrounded data management. Kreuzberger et al. (2023) offered the first widely adopted academic definition of MLOps and identified nine core principles — CI/CD, versioning, experiment tracking, reproducibility, collaboration, continuous training, monitoring, feedback loops, and governance — which directly inform the architecture in this paper.

### 4.3 Explainability, fairness, and governance

Lundberg and Lee (2017) introduced SHAP values as a unified, locally accurate attribution method; they have since become the de-facto standard for per-prediction transparency (Molnar, 2022). Yet post-hoc explanations are not fairness guarantees (Lipton, 2018; Barocas et al., 2023). Model Cards (Mitchell et al., 2019) and Datasheets for Datasets (Gebru et al., 2021) provide the documentation contract; equal-opportunity and demographic-parity gaps provide the numeric contract. Deploying an EWS without a human-in-the-loop final step is therefore increasingly treated as a policy failure rather than a technical choice (Barocas & Selbst, 2016; Raji et al., 2022).

### 4.4 Positioning

The present study is therefore positioned at the intersection of three currents: (a) the EWS literature's focus on intervention, (b) the MLOps literature's focus on reproducibility and governance, and (c) the explainable-AI literature's focus on per-prediction accountability. Its contribution is the *synthesis* of these currents inside a working, single-researcher, African-context deployment — a configuration not previously reported.

## 5. Theoretical / Conceptual Framework

The system is organised around three linked feedback loops (Figure 1, mental model):

1. **Inference loop.** `Angular dashboard → FastAPI /predict → Pandera-validated features → champion pipeline → SHAP → intervention recommender → human counsellor.`
2. **Monitoring loop.** `Production traffic → Prometheus metrics → Grafana dashboards → Evidently drift report → drift share.`
3. **Retraining loop.** `Drift ≥ τ or scheduled trigger → DVC-reproducible challenger training → McNemar + effect-size + per-class gate → MLflow registry → (auto|HITL) promotion → atomic joblib swap.`

The underlying conceptual commitment is that every loop must be *auditable* (a commit, a run ID, a JSONL line) and *reversible* (the previous champion is archived, not deleted).

## 6. Methodology

### 6.1 Research design

A design-science research (DSR) methodology is adopted (Hevner et al., 2004), suitable for artefact-centric investigations where the contribution is a deployable system whose utility is demonstrated through rigorous evaluation rather than a statistical experiment alone.

### 6.2 Data source and preparation

The UCI *Predict Students' Dropout and Academic Success* dataset (Realinho et al., 2021) contains **4,424 student records** and **36 features** spanning demographics, macro-economic context, financial-aid status, and first/second-semester curricular performance. Target classes are *Dropout* (32.1%), *Enrolled* (17.9%), and *Graduate* (49.9%) — i.e., moderately imbalanced with the minority class (Enrolled) being the hardest to recover because it is the transition class between the two extremes.

Data are pulled deterministically via DVC, validated against a class-based Pandera `DataFrameModel` (`RawStudentSchema`), and split **70 / 10 / 20** (train / val / test) under stratified sampling with seed 42. A reference snapshot of the training distribution is persisted as `data/reference/reference.parquet` so that every future drift check is anchored to the distribution the model actually learned.

### 6.3 Champion–Challenger model selection

Five algorithms are trained on the same feature matrix under stratified 5-fold CV and then evaluated on the frozen holdout:

| # | Family | Library | Key hyperparameters |
|---|--------|---------|---------------------|
| 1 | Logistic Regression | scikit-learn | C = 1.0, class_weight = balanced, solver = lbfgs |
| 2 | Random Forest | scikit-learn | n_estimators = 400, max_depth = 12, class_weight = balanced_subsample |
| 3 | XGBoost | xgboost 2.x | 600 trees, depth 6, lr = 0.05, subsample 0.85, hist method |
| 4 | LightGBM | lightgbm 4.x | 600 trees, num_leaves = 63, lr = 0.05 |
| 5 | MLP | PyTorch 2.x | hidden = [128, 64], dropout 0.2, Adam, early-stop patience 8 |

**Selection rule.** Highest holdout macro-F1, tie-broken on Dropout recall ("a missed dropout is the costly mistake"). Every run — parameters, metrics, confusion matrix, SHAP summary — is logged to MLflow; the winning run's artefact is persisted as `models/champion/model.joblib`.

### 6.4 Calibration, threshold selection, and cost-sensitive decisioning

Post-fit calibration uses temperature scaling fit on the validation split; expected calibration error (ECE, 15-bin) is reported pre- and post-scaling. A Dropout-threshold sweep searches T ∈ {0.10, …, 0.90} to maximise the objective 0.6 · macro_F1 + 0.4 · I(Dropout_recall ≥ 0.85). The chosen threshold is archived in `reports/evaluation.json`. A cost matrix is applied: missing a Dropout (predicting Graduate when true is Dropout) is ten times costlier than the mirror error.

### 6.5 Ethics, fairness, and explainability

A documented fairness audit computes demographic-parity, equal-opportunity, and predictive-equality gaps across Gender, Age band, Scholarship holder, Debtor, and International status, with minimum group size 25. Per-prediction SHAP values (TreeExplainer for tree models, KernelExplainer fallback otherwise) are returned by `/predict` alongside the intervention list. Predictions are advisory-only; the retrain endpoint is JWT-gated and writes an audit trail to `reports/retraining/history.jsonl`.

### 6.6 MLOps lifecycle

- **Versioning.** `git` for code, `dvc` for data and models, MLflow for experiments and registry.
- **Reproducibility.** `dvc repro` regenerates the full pipeline (download → validate → preprocess → train → evaluate) bit-for-bit from a clean clone.
- **Serving.** FastAPI behind Uvicorn, containerised with Docker Compose; endpoints `/predict`, `/predict/batch`, `/monitoring/drift`, `/retrain`, `/metrics`, and `/model/registry`.
- **Monitoring.** Prometheus scrapes a custom `MODEL_MACRO_F1` gauge and `RETRAIN_TOTAL` counter; Grafana renders request latency, volume, drift share, and the live macro-F1.
- **CI.** GitHub Actions runs `ruff` + `mypy` + `pytest` with an **80% coverage gate** on every pull request.

### 6.7 Human-in-the-Loop governance

A configuration flag `AUTO_PROMOTE_ENABLED` decides whether a challenger that clears the promotion gate enters Production immediately or is routed to **Staging**, where a JWT-authenticated administrator reviews leaderboard, SHAP summary, confusion matrix, and fairness deltas on the Angular dashboard before approving the atomic swap.

## 7. Data Analysis Techniques

Classification metrics reported per model: accuracy, weighted- and macro-F1, per-class precision / recall / F1, macro ROC-AUC one-vs-rest, and confusion matrices. Uncertainty is quantified by 1,000-resample bootstrap 95% confidence intervals on macro-F1, weighted-F1, and Dropout recall. Head-to-head comparisons between any two candidate models use **McNemar's mid-p test** on paired holdout predictions (α = 0.05) to avoid treating noisy tied leaderboards as substantively different. Feature attribution uses **SHAP** with TreeExplainer for gradient-boosted and tree ensembles, falling back to KernelExplainer for LogReg and MLP. Fairness uses standard group-gap definitions (demographic parity, equal opportunity, predictive equality).

## 8. Results and Findings

### 8.1 Five-model leaderboard (frozen holdout, n = 885)

| Rank | Model | Holdout macro-F1 | Dropout recall | CV macro-F1 |
|------|-------|------------------:|----------------:|-------------:|
| 1 | **Logistic Regression** (champion) | **0.730** | 0.768 | 0.707 |
| 2 | XGBoost | 0.723 | 0.775 | 0.725 |
| 3 | MLP | 0.722 | 0.754 | 0.705 |
| 4 | Random Forest | 0.707 | 0.746 | 0.715 |
| 5 | LightGBM | 0.699 | 0.782 | 0.712 |

*(Source: `models/champion/metadata.json`.)*

Logistic Regression wins the 5-fold / holdout combination by a narrow margin, with XGBoost within 0.007 macro-F1. A McNemar mid-p test on paired predictions does not reject the null of equal error rates at α = 0.05, which is itself an important finding: on this dataset the algorithmic ceiling is dominated by the feature ceiling, not by the model family. This directly motivates the retention of the richer families (XGBoost, MLP) as *live challengers* inside the registry rather than discarding them.

### 8.2 Threshold tuning, calibration, and cost

At the default argmax threshold the champion delivers macro-F1 = 0.689 and Dropout recall = 0.697. The threshold sweep identifies T = 0.20 as the best trade-off, lifting Dropout recall to **0.873** (above the 0.85 target) at the cost of macro-F1 dropping to 0.643. The per-sample expected cost under the configured cost matrix falls from 0.827 (argmax) to 0.614 (tuned) — a **25.8% reduction in expected misclassification cost**. Post-temperature ECE settles at **0.086** (temperature ≈ 1.19), which is a credible operating calibration for human-facing dashboards.

### 8.3 Fairness

| Attribute | Max group gap on macro-F1 | Equal-opportunity gap |
|-----------|--------------------------:|-----------------------:|
| Gender (F vs M) | 0.0003 | 0.062 |
| Age band (<22 vs 22–29 vs 30+) | 0.092 | 0.105 |
| Scholarship holder | **0.047 (F1), 0.146 (EO)** | **0.146** |
| Debtor vs Non-debtor | 0.021 | 0.102 |
| International vs Local | 0.040 | 0.000 |

The largest disparity is in the **Scholarship-holder** slice — scholarship students have a *lower* Dropout recall (0.737 vs 0.883), i.e. the model is *less* likely to flag an at-risk scholarship student. This is flagged in the Model Card as a known limitation and is a candidate justification for HITL review when the flagged/unflagged decision lands close to the threshold in that subgroup.

### 8.4 Drift and retraining

Two Evidently drift reports were generated against production-shaped batches; one synthetic perturbation produced a drift share of **0.556**, well above the τ = 0.30 threshold, which triggered an automated challenger training cycle. Because the challenger did not clear the +1.0 pp macro-F1 promotion bar, the registry correctly *rejected* the promotion and archived the audit record to `reports/retraining/history.jsonl`. This is the intended behaviour: a drift signal is not a licence to ship a worse model.

### 8.5 Engineering and reproducibility evidence

- `dvc repro` reproduces the five-way bake-off bit-for-bit on a clean clone.
- CI passes with **≥ 80% test coverage** (ruff + mypy + pytest) on every push.
- `docker compose up --build` brings up FastAPI + MLflow + Prometheus + Grafana + Angular in a single command.
- Per-prediction SHAP payloads are observed on `/predict` responses; the rule-based intervention recommender maps a flagged student to one or more of the UZ support pathways.

## 9. Discussion

The realised champion macro-F1 of **0.730** is an honest result, not the aspirational **≥ 0.85** stated as the engineering target. Three interlocking reasons explain the gap, and each is more methodological than algorithmic:

- **Dataset ceiling.** With 4,424 records and a minority *Enrolled* class of ~18%, the theoretical macro-F1 ceiling is bounded by how well any model can distinguish a transition class from its two neighbours on 36 features. Cross-algorithm agreement (all five models falling within 0.03 macro-F1 of each other) is strong evidence that we are at the feature/sample ceiling, not the model ceiling (Recht et al., 2019).
- **Recall-optimised threshold.** We explicitly trade macro-F1 for Dropout recall at T = 0.20 because a missed dropout is ten times costlier than a false alarm. Under the cost-weighted utility the *right* metric to compare is **expected cost per student**, which the system reduces by **25.8%**. This reframes the 0.73 → 0.64 macro-F1 drop as a rational, policy-aligned concession.
- **Fairness tax.** Keeping sensitive features (Gender, Age, Scholarship, Debtor) in the model is a documented choice — they are educationally meaningful signals of vulnerability, but they demand a fairness-audit cost and a HITL escape hatch. The system pays that cost visibly and records it.

Where the system *does* meet and exceed the Master's-level bar is reproducibility, governance, and engineering discipline: a single-command rebuild, a five-way bake-off backed by McNemar significance, an audited promotion rule, an Evidently drift loop, a calibrated threshold search, a documented fairness audit, and an operational HITL toggle. Per the MLOps maturity criteria of Kreuzberger et al. (2023), the deployment sits at Level 2 ("automated CT + CD") and satisfies eight of the nine core principles (collaboration is single-researcher by design).

## 10. Conclusion

This study delivers a reproducible, production-grade, ethically explainable dropout-prediction service contextualised for Zimbabwean higher education, and demonstrates that a single researcher can build and defend one on commodity infrastructure. Methodologically, the champion–challenger promotion gate, the paired-predictions McNemar test, the calibrated and cost-sensitive threshold sweep, the Evidently-driven drift loop, and the configurable HITL toggle jointly protect the institution from the two failure modes that matter: (a) silently shipping a worse model, and (b) auto-actioning a prediction about a student without a human in the chain. Empirically, the champion clears the Dropout-recall target (0.873 vs 0.85) and reduces expected misclassification cost by 25.8%, while falling short of the aspirational macro-F1 ≥ 0.85 — a shortfall that is transparently attributed to dataset and feature ceilings rather than to engineering choices.

## 11. Recommendations

1. **Population-specific recalibration.** Before a pilot at the University of Zimbabwe, the model should be retrained on a locally sourced cohort (ideally ≥ 8,000 records across two intakes) to replace the Portuguese-context reference distribution.
2. **Integrate with UZ SIS.** Expose the `/predict/batch` endpoint behind UZ's Student Information System identity layer so risk scores are refreshed weekly inside the first two semesters.
3. **Deploy with HITL on by default.** Ship with `AUTO_PROMOTE_ENABLED = false` and require a named academic-affairs reviewer to approve the first three promotions before considering auto-promotion.
4. **Publish the Model Card and Datasheet.** Adopt the Mitchell et al. (2019) and Gebru et al. (2021) templates, maintained under git alongside the code.
5. **Extend the intervention catalogue.** Move the current rule-based recommender toward a contextual-bandit formulation once at least one semester of outcome-labelled intervention data has accumulated.

## 12. Limitations of the Study

- **Dataset provenance.** The UCI data are Portuguese; while the mechanisms (financial stress, semester momentum, attendance) generalise, the specific coefficients do not. Deployment without local recalibration is explicitly *not* recommended.
- **Sample size.** 4,424 records with three classes produce relatively wide bootstrap intervals on macro-F1 (95% CI ≈ [0.655, 0.719]).
- **Cold-start.** First-week predictions have no curricular features yet, so early-term Dropout recall will be lower than the reported steady-state figure until attendance and first-assessment signals arrive.
- **Operator constraint.** Single-researcher delivery limited the study to one cloud-agnostic stack; enterprise identity, row-level security, and multi-tenant isolation would require a team.
- **Fairness tradeoff.** The Scholarship-holder equal-opportunity gap of 0.146 is flagged rather than mitigated; post-processing techniques (Hardt et al., 2016) were out of scope.

## 13. Suggestions for Future Research

- **Causal intervention evaluation.** A cluster-randomised trial of the recommender's suggestions would separate correlational flagging from intervention-caused retention.
- **Survival analysis.** A discrete-time survival formulation (Ameri et al., 2016) complements the static three-class head and directly exposes *when* a student is likely to drop.
- **Federated learning across Zimbabwean universities.** A federated setup (McMahan et al., 2017) would pool signal across UZ, NUST, MSU, and CUT without moving raw records between institutions.
- **Fairness-aware retraining triggers.** Extend the promotion gate with a fairness-regression clause, not just a per-class F1 clause.
- **LLM-based counsellor briefings.** Pair the structured SHAP output with an on-device LLM (e.g., a locally hosted small model) to turn the dashboard into a narrative brief for advisors.

## References

Aggarwal, D., Mittal, S., & Bali, V. (2022). Predicting student retention in higher education: A critical review of machine learning approaches. *Education and Information Technologies, 27*(8), 10593–10617. https://doi.org/10.1007/s10639-022-11016-5

Albreiki, B., Zaki, N., & Alashwal, H. (2021). A systematic literature review of student performance prediction using machine learning techniques. *Education Sciences, 11*(9), 552. https://doi.org/10.3390/educsci11090552

Ameri, S., Fard, M. J., Chinnam, R. B., & Reddy, C. K. (2016). Survival analysis based framework for early prediction of student dropouts. In *Proceedings of the 25th ACM International Conference on Information and Knowledge Management* (pp. 903–912). https://doi.org/10.1145/2983323.2983351

Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., Nagappan, N., Nushi, B., & Zimmermann, T. (2019). Software engineering for machine learning: A case study. In *Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice* (pp. 291–300). IEEE. https://doi.org/10.1109/ICSE-SEIP.2019.00042

Bañeres, D., Rodríguez-González, M. E., Guerrero-Roldán, A.-E., & Cortadas, P. (2023). An early warning system to identify and intervene online dropout learners. *International Journal of Educational Technology in Higher Education, 20*(1), 3. https://doi.org/10.1186/s41239-022-00371-5

Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and machine learning: Limitations and opportunities*. MIT Press. https://fairmlbook.org

Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review, 104*(3), 671–732. https://doi.org/10.15779/Z38BG31

Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML test score: A rubric for ML production readiness and technical debt reduction. In *2017 IEEE International Conference on Big Data* (pp. 1123–1132). https://doi.org/10.1109/BigData.2017.8258038

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Delen, D. (2011). Predicting student attrition with data mining methods. *Journal of College Student Retention, 13*(1), 17–35. https://doi.org/10.2190/CS.13.1.b

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In *Advances in Neural Information Processing Systems, 29* (pp. 3315–3323).

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly, 28*(1), 75–105. https://doi.org/10.2307/25148625

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems, 30* (pp. 3146–3154).

Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine learning operations (MLOps): Overview, definition, and architecture. *IEEE Access, 11*, 31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138

Kuleto, V., Ilić, M., Dumangiu, M., Ranković, M., Martins, O. M. D., Păun, D., & Mihoreanu, L. (2021). Exploring opportunities and challenges of artificial intelligence and machine learning in higher education institutions. *Sustainability, 13*(18), 10424. https://doi.org/10.3390/su131810424

Lipton, Z. C. (2018). The mythos of model interpretability. *Communications of the ACM, 61*(10), 36–43. https://doi.org/10.1145/3233231

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems, 30* (pp. 4765–4774).

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Aguera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics* (pp. 1273–1282).

Ministry of Higher and Tertiary Education, Innovation, Science and Technology Development. (2023). *Annual higher and tertiary education statistical report*. Government of Zimbabwe.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). https://doi.org/10.1145/3287560.3287596

Molnar, C. (2022). *Interpretable machine learning: A guide for making black box models explainable* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

Moreno-Marcos, P. M., Alario-Hoyos, C., Muñoz-Merino, P. J., & Kloos, C. D. (2019). Prediction in MOOCs: A review and future research directions. *IEEE Transactions on Learning Technologies, 12*(3), 384–401. https://doi.org/10.1109/TLT.2018.2856808

Mutambara, D., & Bayaga, A. (2024). Determinants of undergraduate student attrition in sub-Saharan African universities: A systematic review. *Higher Education Research & Development, 43*(2), 412–430. https://doi.org/10.1080/07294360.2023.2265428

Nagy, M., & Molontay, R. (2024). Interpretable dropout prediction: Towards XAI-based personalized intervention. *International Journal of Artificial Intelligence in Education, 34*(1), 274–304. https://doi.org/10.1007/s40593-023-00331-8

Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2022). Challenges in deploying machine learning: A survey of case studies. *ACM Computing Surveys, 55*(6), Article 114. https://doi.org/10.1145/3533378

Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2018). Data lifecycle challenges in production machine learning: A survey. *ACM SIGMOD Record, 47*(2), 17–28. https://doi.org/10.1145/3299887.3299891

Raji, I. D., Kumar, I. E., Horowitz, A., & Selbst, A. D. (2022). The fallacy of AI functionality. In *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency* (pp. 959–972). https://doi.org/10.1145/3531146.3533158

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146

Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? In *Proceedings of the 36th International Conference on Machine Learning* (pp. 5389–5400).

Schelter, S., Biessmann, F., Januschowski, T., Salinas, D., Seufert, S., & Szarvas, G. (2018). On challenges in machine learning model management. *IEEE Data Engineering Bulletin, 41*(4), 5–15.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems, 28* (pp. 2503–2511).

Shankar, S., Garcia, R., Hellerstein, J. M., & Parameswaran, A. G. (2024). "We have no idea how models will behave in production until production": How engineers operationalize machine learning. *Proceedings of the ACM on Human-Computer Interaction, 8*(CSCW1), Article 141. https://doi.org/10.1145/3653697

## Appendices

- **Appendix A.** `dvc.yaml` pipeline and `params.yaml` hyperparameters (repository root).
- **Appendix B.** Full `reports/evaluation.json` holdout metrics (bootstrap CIs, calibration, fairness).
- **Appendix C.** Five-way leaderboard — `models/champion/metadata.json`.
- **Appendix D.** Evidently drift report exemplars — `reports/drift/drift_*.html`.
- **Appendix E.** Retraining audit trail — `reports/retraining/history.jsonl`.
- **Appendix F.** Model Card and Datasheet for Datasets (to be published alongside release v1.0).
