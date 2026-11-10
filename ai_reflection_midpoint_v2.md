# AI Reflection – Phase I (Midpoint)

**Project:** BERT Toxicity Classification and Identity Term Sensitivity
**AI-aided pair (Group B, Phase I):**

| First name | Last name | Student ID |
|---|---|---|
| Nico Sami | Diaz Cajas | K12208474 |
| Alexander | Wollanek | K013025249 |

**GenAI tools used in Phase I:** Google Gemini (brainstorming the approach and the threshold visualizations), Claude (a Claude project for the first notebook cells; a separate Claude chat for the gender/ethnicity follow-ups and notebook integration), OpenAI Codex (assignment interpretation, shared model/notebook preparation, controlled-test design, Kaggle integration, visualization exploration and verification).

Detailed records are provided in [`ai-interactions/interaction_log.csv`](ai-interactions/interaction_log.csv). The joint log contains only interactions that materially affected the experimental setup, analysis, visualization, debugging or interpretation. It identifies the contributor and labels reconstructed prompt summaries where the original wording was unavailable.

---

## 1. How we used GenAI in Phase I

1. **Brainstorming (Gemini):** we asked how best to explain our fine-tuned BERT model. Once we added the black-box constraint, Gemini suggested counterfactual identity templates and an identity × template heatmap. Both became the core of RQ4/RQ5 (AI-01).
2. **Shared preparation (whole team, Codex):** inspecting the dataset and starter notebook, structuring the plan, drafting the 30 counterfactual sentence templates and preparing the visualization workbook. All four team members reviewed every template before it was frozen.
3. **Design planning:** before any code was written, the AI proposed RQ1 visualizations grounded in Lectures 2–4. It also said what to leave out and why (AI-05).
4. **Code generation:** one notebook cell per research question (RQ1–RQ5), with charts and a written answer (AI-05, AI-06).
5. **Diagnosis and interpretation:** follow-up questions when a result looked suspicious, such as "why is gender ≈ 0?". The AI built diagnostic cells, explained what a result does and does not show, and connected findings to the literature (AI-09, AI-10).
6. **Notebook engineering:** merging graphs, exporting the figures, consolidating imports and fixing fresh-run crashes (AI-07).

### Codex-supported preparation and the shared frozen setup

The team first adapted the notebook and fine-tuned `bert-base-uncased` on the Jigsaw-derived training data. The best model and tokenizer were saved together with the train/validation/test splits, predictions, metrics, training history and experiment configuration. After this preparation, the trained model was **frozen for Phase I**: both groups used the same model state, tokenizer, splits, saved predictions and 0.5 decision threshold. This ensured that differences between the human-coded and GenAI-aided analyses came from the analytical workflow rather than from different training runs.

Codex support helped clarify the assignment, organize the export and shared notebook, formulate the main research question and create a clean Kaggle analysis that loaded the frozen artifacts. It also supported the controlled design of 30 neutral contexts and 120 identity-term variants, the generation of tidy result tables, debugging and the comparison of visualization alternatives. PCA and embedding analysis were deliberately deferred to Phase II. The students ran the fine-tuning and inference themselves, reviewed the controlled sentences, checked the numerical outputs and approved the final interpretations; GenAI did not generate the BERT probabilities.

The pair first explored the task in parallel with different GenAI tools, then compared notebooks, exported tables, figures and draft interpretations. Matching findings supported by the same frozen model outputs were merged. Redundant visualizations, technically unstable code and unsupported explanations were modified or rejected.

## 2. Key interactions and how they shaped our analysis

**AI-01 · Gemini brainstorming: the constraint changes everything.** Our open first question produced mostly white-box XAI methods like SHAP that do not fit a black-box Phase I. After we stated "strict black box, synthetic templates with swapped identity terms", Gemini suggested CheckList-style counterfactual testing, an identity × template heatmap, distribution plots and a counterfactual delta chart. We adopted the heatmap and the template approach for the shared design. We modified the distribution plots into strip/box plots that keep every point. We did not use the delta-vs-baseline chart.

**AI-07 · Interactive Plotly charts and reproducibility → rejected at first, then repaired.** The first RQ1 version used Plotly (hover, ROC, log-scale histogram, heatmap). The heatmap rendered blank on Kaggle. We made "must render on Kaggle" a hard constraint, and the RQ1–RQ5 cells from that round were static matplotlib. Interaction came back later in a robust form: the final notebook mixes Plotly and matplotlib, and the export saves the Plotly charts as interactive HTML.

**AI-05 · Asking for a plan grounded in the lectures before any code.** Prompt: *"what do you suggest for visualizing and explaining the models behaviour and dataset. use the given lecture papers. The model is not known, and tread it as a black-box"* The AI proposed icon-array waffle grids (Micallef et al. 2012) and class-wise score histograms as a visual form of ROC-AUC (0.990). It moved FP/FN detail to RQ2, confidence to RQ3, and LIME/SHAP/projections to Phase II. We accepted the plan, and it set the scope boundaries between RQ1–RQ3 and between the two phases.

**AI-05 / AI-06 · Cells for RQ1–RQ5.** A one-line prompt (*"Start draft cells for Research Questions 2 to 5."*) produced four cells that **broadened the analysis** beyond our plan:
- a threshold sweep (is the FP/FN imbalance only a result of the 0.5 cut-off?);
- FP/FN rates for real test comments that mention an identity group;
- a reliability diagram;
- a centred ICE chart (Lecture 4).

The AI had no access to the model, so it tested everything on synthetic data with our real totals (18,149 comments, 538 FP, 362 FN).

**AI-09 · "Gender got left out. Find a way to display the category Gender, its values or too low compared to the other categories."** Our nicer Plotly chart of the largest probability shifts showed no gender rows. The AI found the cause: the chart took the **overall** top 15 contexts by range, and the gender ranges are so small that none qualified. The fix selects the top N **per dimension**. This turned an invisible null result into an explicit finding.

**AI-10 · "Why is gender almost 0? Is this a weakness of the Dataset itself?"** This was the most influential interaction for our conclusions.
- **Ruling out a bug.** The AI first wrote a diagnostic cell that checks the gender pipeline step by step: substitution, tokenization, re-scoring, the logit scale, and a cross-swap of terms between dimensions.
- **Separating what the result shows from what it doesn't:**
  - It shows that gender made no difference in the decision sense: a max gap of ≈ 0.0002, all scores ≈ 0.001, no flips.
  - It contrasts sharply with ethnicity: a mean within-context range of 0.77, with 9 of 10 contexts flipping.
  - It does **not** show that the model is free of gender bias. Only one binary pair was tested, in 10 neutral sentences, with scores at the floor.
- **A new chart:**
  - Within-context ranges on a **log scale**. On a linear scale every gender dot sits at 0.
  - The **share of training comments labelled toxic for each identity term**. This is a descriptive pattern that may help explain why gender stays flat, but it does not establish causality.

We adopted the hedged report wording for RQ4/RQ5.

**AI-07 · Making the notebook reproducible.** After we merged graphs, the AI found that the export cell would crash on a fresh run because it referenced a deleted chart (`fig_atlas`). Its "Export created" output was left over from an old run, and several figures overwrote each other (`fig1`/`fig2`). It added a `register_figure` registry (12 charts, Plotly → HTML, matplotlib → PNG) and moved all imports into section 1. It also fixed several crashes that had been hidden by stale kernel state.

**AI-08 · Threshold (Gemini).** We asked how to show the effect of the decision threshold on predictions and on fairness across identity groups. Gemini suggested FPR-vs-threshold curves for the counterfactuals, score distributions with the threshold line, precision–recall tuning curves on validation data, and subgroup disparity bars at the locked threshold. We kept the threshold line and the subgroup comparison (`02d_identity_false_positives`). We turned "tuning" into a pure sweep (`02c_threshold_tradeoff`), because the team froze 0.5 as the shared threshold.

**AI-02 / AI-03 · From assignment interpretation to a shared frozen model.** Codex was used to translate the brief into a black-box model-behaviour study, understand the Jigsaw/BERT notebook and specify which outputs had to be shared. The team retained its training choices, executed the fine-tuning in Kaggle and exported the best model, tokenizer, splits, predictions, metrics and configuration. Freezing these artifacts for both groups made the comparison reproducible.

**AI-04 / AI-11 · Research questions, controlled data and notebook structure.** Codex helped refine the main question into five subquestions and distinguish ordinary held-out performance from controlled identity-term stability. Its first pair-based idea was expanded after discussion into 30 neutral contexts and 120 variants across ethnicity, religion and one tested gender pair. The team reviewed the sentences and kept only cases in which the identity term was the intended change. A clean Kaggle notebook then loaded the frozen model, reproduced the baseline, scored the controlled CSV and exported tidy tables; PCA was removed because it belongs to Phase II.

**AI-07 · Technical assumptions had to be corrected.** Codex initially relied on plausible but incorrect assumptions about a `validation_rows` configuration key, Kaggle input locations and the default Plotly renderer. The students checked the actual files and environment, changed the consistency check, shared the required dataset and used the `iframe` renderer. This interaction became evidence that AI-generated code must be validated against the real schema and runtime.

**AI-12 / AI-13 · Answering the questions and merging the parallel analyses.** Codex proposed confidence, error, range, flip and context-centred views. The pair executed these analyses on the verified result tables and corrected wording such as confusing an average score with a median within-context probability range. They then compared the figures produced through the parallel workflows. They retained complementary views and findings that agreed with the frozen-model outputs, while discarding redundant charts and interpretations that went beyond the controlled evidence. The final claim is therefore about **sensitivity in this model and test design**, not causal or general proof of social bias.

## 3. Accepted, modified and rejected suggestions

| Suggestion | Decision | Why |
|---|---|---|
| White-box XAI methods (SHAP, IG, LIME, UMAP/t-SNE) – Gemini | Deferred to Phase II / rejected | Need model internals; Phase I is black-box |
| Counterfactual templates + identity × template heatmap – Gemini | Accepted | Became the shared counterfactual design and the RQ5 heatmap |
| Violin/box plots per identity – Gemini | Modified → strip/box plots with all points | Only 10 contexts per term, so individual points must stay visible |
| Delta chart vs neutral "person" baseline – Gemini | Rejected | We compare variants within a context (range, connected-dot plot), so no neutral baseline sentence is needed |
| Interactive Plotly charts for RQ1 (first version) | Rejected, later partly restored | Heatmap failed on Kaggle; interaction restored via Plotly + HTML export once rendering was reliable |
| Icon-array waffle grids for accuracy | Accepted | Shows sensitivity/specificity as "of 100 comments"; grounded in Lecture 2 |
| Score histograms with threshold and shaded error regions | Accepted | Makes ROC-AUC visible instead of just quoted |
| Move FP/FN, confidence, LIME/SHAP out of RQ1 | Accepted | Avoids overlap between RQs; keeps Phase II methods out of Phase I |
| Assertions against the starter notebook's tables | Accepted | Turns "trust the AI" into a check that fails loudly |
| Threshold sweep (RQ2) | Accepted (final export `02c_threshold_tradeoff`) | Shows whether the FP/FN imbalance depends on the 0.5 cut-off; also suggested by Gemini (AI-08) |
| Threshold *tuning* on validation data – Gemini | Modified → sweep only | Threshold 0.5 is frozen for both pairs; we show trade-offs but do not re-tune |
| FP/FN for identity-mentioning test comments (RQ2) | Accepted (final export `02d_identity_false_positives`) | Links real test data to the counterfactual experiment; Gemini also suggested subgroup disparity at the locked threshold |
| Reliability diagram (RQ3) | Accepted (final export `03_confidence_calibration`) | Tests whether "95 % sure" means right 95 % of the time |
| Most-confident-mistakes table (RQ3) | Retained in the notebook for diagnostic review; not emphasized in the final conclusions | Prints real, possibly offensive comments; candidate pool for Phase II error cases |
| Centred ICE plot (RQ5) | Accepted (final export `05a_context_centred_effects`, `05c_centred_term_effects`) | Separates systematic term effects from context-specific ones |
| Overall top-15 contexts in the shift chart | **Modified** → top N per dimension | Overall top-N hid the gender dimension entirely |
| Gender diagnostic check cell (8b) | Accepted and checked on the executed pipeline | Substitution and inference worked; the near-zero ranges and zero flips remained, but this does not establish that the model is generally free of gender bias |
| Log-scale range panel + training-data skew panel | Accepted | Linear scale hides gender; training-data skew explains the "why" without opening the model |
| Hedged wording "stable for these pairs, not bias-free" | Accepted | Avoids overclaiming from 2 terms × 10 sentences |
| Explanation "gender words are everyday nouns, abuse uses slurs" | Kept only as a labelled hypothesis | Not tested by our data |
| `register_figure` export registry, imports in section 1 | Accepted | Notebook must survive restart-and-run |
| Fine-tune, export and then freeze one shared BERT model | Accepted | Keeps model state, tokenizer, splits, predictions and threshold identical for both groups |
| Initial small identity-pair design – Codex | Modified → 30 contexts and 120 variants across three dimensions | Broader coverage while preserving the rule that only the identity term changes within a context |
| PCA/embedding visualization in Phase I – Codex discussion | Deferred to Phase II | Phase I focuses on inputs, predictions, confidence and controlled output stability |
| Codex visualization alternatives | Selectively merged | Only plots that answered a research question clearly and matched verified result tables were retained |

## 4. Failure patterns observed

1. **Generic answers to under-specified prompts, and advice that conflicts with the protocol.** Without constraints, Gemini proposed the usual white-box XAI toolbox and ignored the assignment's black-box Phase I. The suggestions only became useful once we stated the constraint. Similarly, its advice to tune the threshold on validation data follows general ML practice but contradicts our frozen shared threshold.
2. **Ignoring the execution environment.** The first interactive version assumed Plotly would render on Kaggle. The AI optimised for the output it could see (its own preview), not the one we would see.
3. **No access to real data.** In the project chats, every chart was tested on synthetic or mock data, and so was the gender check cell. The previews look convincing, but the distribution shapes are invented, so a preview is easily mistaken for a result.
4. **Hidden kernel state and stale outputs.** The notebook "worked" only because deleted cells were still in memory: missing imports, undefined variables, a reference to a deleted chart. A saved "Export created" output came from an old run. Neither we nor the AI noticed until we explicitly asked it to check.
5. **Version drift and regressions.** The combined RQ1–5 notebook still contained the old RQ1 cell, and the v6 notebook brought back the crashing export code that had been fixed in v5. Every iteration needs re-checking, not only the newest change.
6. **Plausible chart design that hides results.** A "top 15 by range" chart is a standard choice, but it silently dropped a whole dimension. A linear axis hid differences of 10⁻⁴. The AI produced both charts without flagging this, and it only explained the issue once we asked why gender was missing.
7. **Small code side effects.** `register_figure()` returning the figure made Jupyter display every chart twice.
8. **Risk of overclaiming and second-hand citations.** Read naively, a null result ("gender ≈ 0") invites the conclusion "no gender bias". The AI did warn about this when asked, but the conclusion has to be our decision. Some literature claims were available only through secondary summaries, so unverified numerical claims were not used in our final conclusions.
9. **Over-delivery.** A one-line prompt produced four cells with several charts each. This was useful, but it increases the review burden and can blur what the research question actually asks.
10. **Lost provenance.** Our Gemini prompts and the exact prompts of the follow-up chats were not saved at the time, so parts of this record are reconstructions.
11. **Guessed file schemas and runtime details.** Codex proposed code that expected a `validation_rows` key and different Kaggle paths, and Plotly selected a renderer that produced blank space. The suggestions became useful only after we compared them with the actual exported files and execution environment.
12. **Independent AI outputs can agree and still be wrong.** Running two GenAI-assisted approaches increased the number of ideas, but agreement between their explanations was never treated as validation. Only reproduced values from the frozen model and human review counted as evidence.

## 5. Strengths of AI assistance

- **Speed and breadth:** a complete, commented cell per RQ in minutes, so we could try more views (threshold sweep, calibration, c-ICE, training-data skew) than we would have by hand.
- **Structuring:** separating what each RQ should and should not show, and linking choices to course concepts and literature.
- **Diagnosis:** when we asked "why", the AI produced systematic checks (substitution, tokenization, logit scale, cross-swap) rather than guesses.
- **Epistemic caveats on request:** the "what it does NOT show" list and the labelling of hypotheses as guesses improved the honesty of our conclusions.
- **Code review:** it found crashes, name collisions and stale outputs that we had missed.

## 6. Emerging best practices

1. **State the constraints in the prompt** (black box, phase, runtime). The same question with a constraint produced completely different and more useful suggestions.
2. **Plan before code.** Ask for a design proposal with reasons and exclusions, approve it, then ask for code.
3. **Ground prompts in course material and literature**, then verify that the cited concepts and numbers really match the sources.
4. **State runtime constraints up front:** Kaggle, available variables, which section the cell runs after, static vs interactive.
5. **Make the AI verify itself against trusted numbers** (assertions), and never report a number that was not computed on the real data.
6. **Treat synthetic previews as layout checks only.**
7. **When a result looks too clean, ask "is this a bug or a real effect?"** Before interpreting it, ask for a diagnostic.
8. **Always ask "what does this NOT show?"** before writing a conclusion, especially for null results.
9. **Check chart choices for hidden data:** per-group selection instead of a global top-N, and a log or logit scale when values sit near 0 or 1.
10. **Restart and run all** after every AI-assisted change, and never trust saved outputs.
11. **Version outputs and log prompts as we go** (save or share every chat that influences a decision).
12. **Write the final interpretation ourselves**, after seeing the real charts.
13. **Freeze and export the common experimental state before parallel analysis:** model, tokenizer, splits, predictions, configuration and threshold.
14. **Separate training, inference and GenAI assistance in the documentation.** The team ran and verified the model; GenAI supported analysis code and communication, not the generation of evidence.
15. **When merging independent AI-assisted work, reconcile numbers before narratives.** Compare artifacts and tables first, then select plots and wording.

## 7. Reflection

Using GenAI saved a lot of time, but it also introduced errors in places where none should have occurred. It also helped a lot in presenting the data more clearly and attractively, and it allowed a deeper analysis of the model’s characteristics and of the dataset it was trained on. The AI was most useful when we asked it to explain why something happened, because it then suggested systematic checks instead of guesses and helped us state what our results do not show. At the same time, its code often only worked because of leftover state from earlier runs, so restarting and running the whole notebook after every change, and checking its numbers against our verified tables, turned out to be essential. Overall, the best results came from stating our constraints up front, approving a plan before asking for code, and writing the final interpretation ourselves after seeing the real charts.

The parallel workflow added an important second lesson. The Claude/Gemini and Codex workflows often reached compatible numerical conclusions, but they emphasized different plots and explanations. Comparing them broadened the analysis, while the frozen shared model and result tables gave us a neutral basis for deciding what to keep. The merge also showed that GenAI acceleration does not remove work: it shifts effort toward checking assumptions, resolving conflicting terminology, testing notebooks from a clean session and limiting claims to what the experiment actually demonstrates.
