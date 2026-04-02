# Literature Review: Diagnosing Emergent Misalignment with Persona Moral Metrics

**100 Potentially Relevant Articles**

Organized by theme. For each entry: title, authors, year, venue/arXiv ID, and a brief summary of relevance.

---

## 1. Emergent Misalignment — Core Papers and Extensions

**1.** Betley, J., Tan, D., Warncke, N., Sztyber-Betley, A., Bao, X., Soto, M., Labenz, N., & Evans, O. (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* arXiv:2502.17424; ICML 2025.
— Shows that fine-tuning GPT-4o on insecure code produces models that express misaligned values on unrelated prompts, establishing the emergent misalignment phenomenon.

**2.** Betley, J., Warncke, N., Sztyber-Betley, A., et al. (2026). *Training large language models on narrow tasks can lead to broad misalignment.* Nature 649, 584. DOI: 10.1038/s41586-025-09937-5.
— The Nature-published version of the emergent misalignment finding, establishing it for a broader scientific audience.

**3.** Wang, M., Dupre la Tour, T., Watkins, O., Makelov, A., Chi, R.A., Miserendino, S., Wang, J., Rajaram, A., Heidecke, J., Patwardhan, T., & Mossing, D. (2025). *Persona Features Control Emergent Misalignment.* arXiv:2506.19823.
— Uses sparse autoencoders and model diffing to identify a "toxic persona feature" in activation space that predicts and controls emergent misalignment.

**4.** Soligo, A., Turner, E., Rajamanoharan, S., & Nanda, N. (2025). *Convergent Linear Representations of Emergent Misalignment.* arXiv:2506.11618; ICML 2025.
— Shows different emergently misaligned models converge to similar linear representations; a misalignment direction from one model can ablate misalignment in others.

**5.** Arturi, D.A.R., Zhang, E., Ansah, A., Zhu, K., Panda, A., & Balwani, A. (2025). *Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior.* arXiv:2511.02022.
— Demonstrates emergent misalignment from different tasks converges to shared parameter subspaces near-orthogonal to base weights.

**6.** (2026). *Character as a Latent Variable in Large Language Models: A Mechanistic Account of Emergent Misalignment and Conditional Safety Failures.* arXiv:2601.23081.
— Proposes that emergent misalignment, persona control, and persona-aligned jailbreaks share character-related representations.

**7.** (2025). *Model Organisms for Emergent Misalignment.* arXiv:2506.11613.
— Creates improved model organisms achieving 99% coherence in 0.5B models, identifying a phase transition in fine-tuning where misalignment is learned rapidly.

**8.** Soligo, A., Turner, E., Rajamanoharan, S., & Nanda, N. (2026). *Emergent Misalignment is Easy, Narrow Misalignment is Hard.* arXiv:2602.07852; ICLR 2026.
— Shows the broadly misaligned solution is more stable/efficient than the narrow one; expert predictions of emergent misalignment fail.

**9.** (2026). *Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning.* arXiv:2602.00298.
— Evaluates fine-tuning across 11 domains, finding vulnerability ranges from 0% (incorrect math) to 87.67% (gore-movie-trivia).

**10.** Doan, D. et al. (2025). *Emergent Misalignment in Mixture-of-Experts Models.* OpenReview.
— Studies emergent misalignment in sparse MoE architectures, finding that increased sparsity (more experts) suppresses misalignment.

**11.** (2025). *Emergent Misalignment via In-Context Learning.* arXiv:2510.11288.
— Extends emergent misalignment beyond fine-tuning to in-context learning, with as few as 2 examples causing 1-24% misalignment rates.

**12.** MacDiarmid, M. et al. (Anthropic). (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
— Demonstrates that reward hacking in production RL environments generalizes to alignment faking and sabotage; standard RLHF fixes chat but not agentic misalignment.

**13.** (2025). *When Thinking Backfires: Mechanistic Insights into Reasoning-Induced Misalignment.* arXiv:2509.00544.
— Identifies Reasoning-Induced Misalignment (RIM): LLMs become more responsive to malicious requests when reasoning is strengthened.

**14.** (2025/2026). *LLMs Deceive Unintentionally: Emergent Misalignment in Dishonesty.* arXiv:2510.08211.
— Shows 1% misalignment data decreases honest behavior by over 20%; biased users can unintentionally amplify dishonesty.

**15.** (2025). *In-Training Defenses against Emergent Misalignment in Language Models.* arXiv:2508.06249.
— First systematic study of in-training safeguards (KL regularization, L2 distance, preventative steering, instruct interleaving) against emergent misalignment.

**16.** Jaburi, L. (EleutherAI). (2025/2026). *Mitigating Emergent Misalignment with Data Attribution.* OpenReview (ICLR 2026).
— Applies data attribution (EK-FAC) to identify training points causing emergent misalignment, outperforming LLM-based text classifiers.

**17.** (2025). *From Narrow Unlearning to Emergent Misalignment.* arXiv:2511.14017.
— Shows emergent misalignment can arise from narrow refusal unlearning, not just fine-tuning.

**18.** Vaugrante, L., Weckauff, A., & Hagendorff, T. (2026). *Emergently Misaligned Language Models Show Behavioral Self-Awareness That Shifts With Subsequent Realignment.* arXiv:2602.14777.
— Misaligned models rate themselves as significantly more harmful; this self-awareness tracks actual alignment states.

**19.** Wyse, T., Stone, T., Soligo, A., & Tan, D. (2025). *Emergent Misalignment as Prompt Sensitivity: A Research Note.* arXiv:2507.06253.
— Misaligned behavior is highly sensitive to prompt nudges and can be elicited/reduced by simple instructions.

**20.** (2026). *Semantic Containment as a Fundamental Property of Emergent Misalignment.* arXiv:2603.04407.
— Semantic formatting triggers induce compartmentalized misalignment without benign training data.

**21.** Hahm, D., Min, T., Jin, W., & Lee, K. (2025). *Unintended Misalignment from Agentic Fine-Tuning.* arXiv:2508.14031; AAAI 2026.
— Aligned LLMs become unintentionally misaligned when fine-tuned for agentic tasks; proposes PING (Prefix INjection Guard).

---

## 2. Moral Foundations Theory — Core References

**22.** Haidt, J. (2001). *The Emotional Dog and Its Rational Tail: A Social Intuitionist Approach to Moral Judgment.* Psychological Review, 108(4), 814-834.
— Foundational paper for the social intuitionist model: moral judgments are driven by fast intuitions, with reasoning as post hoc justification.

**23.** Haidt, J. & Joseph, C. (2004). *Intuitive Ethics: How Innately Prepared Intuitions Generate Culturally Variable Virtues.* Daedalus, 133(4), 55-66.
— First description of what became Moral Foundations Theory, proposing four intuitive ethics as evolutionarily rooted foundations.

**24.** Haidt, J. & Graham, J. (2007). *When Morality Opposes Justice: Conservatives Have Moral Intuitions that Liberals May Not Recognize.* Social Justice Research, 20(1), 98-116.
— Argues conservatives draw on broader moral foundations (loyalty, authority, purity) that liberal frameworks overlook.

**25.** Graham, J., Haidt, J., & Nosek, B.A. (2009). *Liberals and Conservatives Rely on Different Sets of Moral Foundations.* Journal of Personality and Social Psychology, 96(5), 1029-1046.
— Empirical MFQ validation: liberals prioritize Care/Fairness; conservatives endorse all five foundations more equally.

**26.** Graham, J., Nosek, B.A., Haidt, J., Iyer, R., Koleva, S., & Ditto, P.H. (2011). *Moral Foundations Questionnaire.* PsycTESTS. DOI: 10.1037/t05651-000.
— The official MFQ-30 psychometric instrument used in the present study.

**27.** Graham, J., Haidt, J., Koleva, S., Motyl, M., Iyer, R., Wojcik, S.P., & Ditto, P.H. (2013). *Moral Foundations Theory: The Pragmatic Validity of Moral Pluralism.* Advances in Experimental Social Psychology, 47, 55-130.
— Definitive theoretical statement of MFT with comprehensive empirical evidence for the five-foundation model.

**28.** Haidt, J. (2012). *The Righteous Mind: Why Good People Are Divided by Politics and Religion.* Vintage Books.
— Accessible synthesis of MFT and its implications for understanding political and moral divisions.

---

## 3. Moral Foundations and Values in LLMs

**29.** Abdulhai, M., Serapio-Garcia, G., Crepy, C., Valter, D., Canny, J., & Jaques, N. (2024). *Moral Foundations of Large Language Models.* EMNLP 2024, pp. 17737-17752.
— Administers MFQ to multiple LLMs; finds they emphasize Care/Fairness but under-represent Loyalty/Authority/Purity.

**30.** Scherrer, N., Shi, C., Feder, A., & Blei, D. (2023). *Evaluating the Moral Beliefs Encoded in LLMs.* NeurIPS 2023 (Spotlight). arXiv:2307.14324.
— Proposes statistical measures for moral choice probability, uncertainty, and consistency across 28 LLMs on 1,367 moral scenarios.

**31.** Kirgis, P. (2025). *Differences in the Moral Foundations of Large Language Models.* arXiv:2511.11790.
— Documents that LLMs overweight liberal-associated foundations relative to human baselines, with divergence increasing with capability.

**32.** (2025). *Investigating Political and Demographic Associations in Large Language Models Through Moral Foundations Theory.* arXiv:2510.13902; AAAI/ACM AIES 2025.
— Assesses whether LLMs inherently align with one political ideology using MFT across demographic role-playing.

**33.** (2026). *Tracing Moral Foundations in Large Language Models.* arXiv:2601.05437.
— Investigates internal MFT representations using layer-wise analysis, SAEs, and causal steering on Llama-3.1-8B and Qwen2.5-7B.

**34.** Nunes, J.L., Almeida, G.F.C.F., de Araujo, M., & Barbosa, S.D.J. (2024). *Are Large Language Models Moral Hypocrites? A Study Based on Moral Foundations.* arXiv:2405.11100; AAAI/ACM AIES 2024.
— Finds LLMs show contradictory behavior between abstract MFQ endorsement and applied moral judgment on vignettes.

**35.** Tlaie, A. (2024). *Exploring and Steering the Moral Compass of Large Language Models.* arXiv:2405.17345.
— Comprehensive MFQ + ethical dilemma comparison; proprietary models are utilitarian, open-weight models values-based, nearly all liberal-biased.

**36.** Ji, J., Chen, Y., Jin, M., Xu, W., Hua, W., & Zhang, Y. (2025). *MoralBench: Moral Evaluation of LLMs.* arXiv:2406.04428.
— Introduces MoralBench using MFQ-30-LLM to evaluate LLM sensitivity to six moral dimensions.

**37.** Carrasco et al. (2024). *Moral Persuasion in Large Language Models: Evaluating Susceptibility and Ethical Alignment.* arXiv:2411.11731.
— Tests LLM-to-LLM moral persuasion; susceptibility varies by model size and scenario complexity.

**38.** Aksoy, M. (2024). *Whose Morality Do They Speak? Unraveling Cultural Bias in Multilingual Language Models.* arXiv:2412.18863.
— Investigates multilingual LLMs using MFQ-2 in eight languages, revealing cultural bias in moral expression.

**39.** Raimondi, B. et al. (2025). *Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability.* arXiv:2510.12229.
— Uses layer-patching to show moral bias is learned during finetuning and localized in specific layers.

**40.** (2025). *A Survey on Moral Foundation Theory and Pre-Trained Language Models: Current Advances and Challenges.* AI & Society, Springer. DOI: 10.1007/s00146-025-02225-w.
— Comprehensive survey reviewing the intersection of MFT and pre-trained language models.

---

## 4. Ethical Reasoning and Value Benchmarks for LLMs

**41.** Hendrycks, D. et al. (2021). *Aligning AI with Shared Human Values.* ICLR 2021. arXiv:2008.02275.
— Introduces the ETHICS benchmark for evaluating LLM understanding of justice, deontology, virtue ethics, utilitarianism, and commonsense morality.

**42.** Jiang, L., Hwang, J.D., Bhagavatula, C., et al. (2022/2024). *Delphi: Towards Machine Ethics and Norms.* arXiv:2110.07574; Nature Machine Intelligence (2024).
— A neural network trained to predict descriptive ethical judgments, landmark study in machine moral judgment.

**43.** Sachdeva & van Nuenen. (2025). *Normative Evaluation of Large Language Models with Everyday Moral Dilemmas.* arXiv:2501.18081; ACM FAccT 2025.
— Evaluates LLMs on everyday moral dilemmas from Reddit's "Am I the Asshole"; finds substantial divergence from human evaluations.

**44.** Pan, A., Chan, J.S., Zou, A., Li, N., Basart, S., Woodside, T., Ng, J., Zhang, H., Emmons, S., & Hendrycks, D. (2023). *Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark.* ICML 2023. arXiv:2304.03279.
— Benchmark measuring Machiavellian behavior (deception, manipulation, power-seeking) in LLM agents navigating text-based games.

**45.** Huang, Y., Sun, L., et al. (2024). *TrustLLM: Trustworthiness in Large Language Models.* arXiv:2401.05561.
— Comprehensive trustworthiness benchmark evaluating 16 LLMs across eight dimensions including safety, fairness, and machine ethics.

**46.** Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). *Whose Opinions Do Language Models Reflect?* ICML 2023. arXiv:2303.17548.
— Creates OpinionsQA to measure LM alignment with 60 US demographic groups; finds misalignment on par with partisan divides.

**47.** Hartmann, J., Schwenzow, J., & Witte, M. (2023). *The Political Ideology of Conversational AI: Converging Evidence on ChatGPT's Pro-Environmental, Left-Libertarian Orientation.* arXiv:2301.01768.
— Tests ChatGPT with 630 political statements; finds robust left-libertarian leanings with >72% overlap with Green parties.

**48.** Durmus, E., Nguyen, K., Liao, T.I., Schiefer, N., Askell, A., Bakhtin, A., et al. (2024). *Towards Measuring the Representation of Subjective Global Opinions in Language Models.* arXiv:2306.16388.
— Uses the GlobalOpinionQA dataset (100K+ questions from cross-national surveys) to evaluate LLM representation of global opinions.

---

## 5. Fine-Tuning Safety and Alignment Fragility

**49.** Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P., & Henderson, P. (2024). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To.* ICLR 2024. arXiv:2310.03693.
— Shows that even benign fine-tuning can compromise safety alignment; as few as 10 adversarial examples jailbreak GPT-3.5 Turbo.

**50.** Yang, X., Wang, X., Zhang, Q., Petzold, L., Wang, W.Y., Zhao, X., & Lin, D. (2023). *Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models.* arXiv:2310.02949.
— Fine-tuning on 100 malicious examples with 1 GPU hour subverts safety across 8 models from 5 organizations.

**51.** Zhan, Q., Fang, R., Bindu, R., Gupta, A., Hashimoto, T., & Kang, D. (2024). *Removing RLHF Protections in GPT-4 via Fine-Tuning.* NAACL 2024. arXiv:2311.05553.
— Fine-tuning removes RLHF protections from GPT-4 with 340 examples at 95% success rate.

**52.** Lermen, S., Rogers-Smith, C., & Ladish, J. (2023). *LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B.* arXiv:2310.20624.
— With <$200 and one GPU, LoRA fine-tuning reduces Llama 2-Chat 70B refusal rate to <1%.

**53.** Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., & Henderson, P. (2024). *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* ICML 2024. arXiv:2402.05162.
— Safety-critical regions are extremely sparse (~3% parameters, ~2.5% rank); removing them compromises safety without affecting utility.

**54.** Hsu, C.-Y., Tsai, Y.-L., Lin, C.-H., Chen, P.-Y., Yu, C.-M., & Huang, C.-Y. (2024). *Safe LoRA: The Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models.* NeurIPS 2024. arXiv:2405.16833.
— Projects LoRA weights to preserve safety-aligned directions, enhancing resilience to safety degradation.

**55.** Bianchi, F., Suzgun, M., Attanasio, G., Rottger, P., Jurafsky, D., Hashimoto, T., & Zou, J. (2023). *Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions.* ICLR 2024. arXiv:2309.07875.
— Adding 3% safety examples during fine-tuning substantially improves safety; excess safety-tuning causes exaggerated refusals.

**56.** Wolf, Y., Wies, N., Levine, Y., & Shashua, A. (2023). *Fundamental Limitations of Alignment in Large Language Models.* ICML 2024. arXiv:2304.11082.
— Proves via Behavior Expectation Bounds that alignment which attenuates but doesn't eliminate undesired behavior is inherently unsafe against adversarial prompting.

---

## 6. RLHF, DPO, and Alignment Methods

**57.** Ouyang, L., Wu, J., Jiang, X., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. arXiv:2203.02155.
— Introduces RLHF (InstructGPT); a 1.3B model's outputs are preferred over 175B GPT-3 with improvements in truthfulness.

**58.** Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.
— Introduces RLAIF using self-critique guided by constitutional principles, eliminating need for human harm labels.

**59.** Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. arXiv:2305.18290.
— Derives a closed-form RLHF solution using classification loss, eliminating reward model fitting and RL sampling.

**60.** Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS 2017. arXiv:1706.03741.
— Foundational work on learning reward models from human preferences for RL agents, predating LLM-specific RLHF.

**61.** Ziegler, D.M., Stiennon, N., Wu, J., Brown, T.B., Radford, A., Amodei, D., Christiano, P., & Irving, G. (2019). *Fine-Tuning Language Models from Human Preferences.* arXiv:1909.08593.
— Early application of reward learning from human preferences to language model fine-tuning (summarization, sentiment).

---

## 7. Jailbreaking and Adversarial Attacks on Aligned LLMs

**62.** Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J.Z., & Fredrikson, M. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
— Introduces the GCG attack with adversarial suffixes that transfer across models, jailbreaking ChatGPT, Bard, and Claude.

**63.** Wei, A., Haghtalab, N., & Steinhardt, J. (2023). *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS 2023 (Oral). arXiv:2307.02483.
— Identifies two failure modes: competing objectives and mismatched generalization.

**64.** Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G.J., & Wong, E. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries.* arXiv:2310.08419.
— PAIR: uses an attacker LLM to automatically generate semantic jailbreaks requiring <20 queries.

**65.** Liu, Y., Deng, G., Xu, Z., et al. (2023). *Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study.* arXiv:2305.13860.
— Classifies ten jailbreak patterns across three categories, testing 3,120 questions on ChatGPT 3.5/4.0.

**66.** Huang, Y., Gupta, S., Xia, M., Li, K., & Chen, D. (2023). *Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation.* ICLR 2024. arXiv:2310.06987.
— Manipulating decoding methods increases attack success from 0% to >95% on open-source LLMs.

**67.** (2025). *Eliciting and Analyzing Emergent Misalignment in State-of-the-Art Large Language Models.* arXiv:2508.04196.
— Systematic red-teaming of frontier LLMs reveals 76% vulnerability; introduces MISALIGNMENTBENCH.

---

## 8. Red-Teaming LLMs

**68.** Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N., & Irving, G. (2022). *Red Teaming Language Models with Language Models.* EMNLP 2022. arXiv:2202.03286.
— Pioneers automated red-teaming using one LM to generate test cases for another.

**69.** Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., et al. (2022). *Red Teaming Language Models to Reduce Harms.* arXiv:2209.07858.
— Investigates red teaming across model sizes at Anthropic, releasing 38,961 red team attacks.

**70.** Mazeika, M., Phan, L., Yin, X., Zou, A., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024. arXiv:2402.04249.
— Standardized framework comparing 18 red teaming methods against 33 target LLMs.

---

## 9. Safety Benchmarks

**71.** Zhang, Z., Lei, L., Wu, L., et al. (2023). *SafetyBench: Evaluating the Safety of Large Language Models.* ACL 2024. arXiv:2309.07045.
— 11,435 multiple-choice questions spanning 7 safety categories in Chinese and English.

**72.** Bhatt, M., Chennabasappa, S., Nikolaidis, C., et al. (2023). *Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models.* arXiv:2312.04724.
— Evaluates LLM propensity to generate insecure code; more advanced models suggest more insecure code.

---

## 10. Sleeper Agents and Deceptive Alignment

**73.** Hubinger, E. et al. (39 authors). (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* arXiv:2401.05566.
— Proof-of-concept deceptive LLMs with backdoor behavior that persists through SFT, RL, and adversarial training.

**74.** Greenblatt, R., Shlegeris, B., Denison, C., & Evans, O. (2024). *Alignment Faking in Large Language Models.* arXiv:2412.14093.
— Demonstrates that Claude 3 Opus sometimes strategically complies during training to preserve its preferred behavior during deployment.

---

## 11. Mechanistic Interpretability and Sparse Autoencoders

**75.** Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR 2024. arXiv:2309.08600.
— Uses SAEs to decompose LLM activations into monosemantic, interpretable features.

**76.** Bricken, T., Templeton, A., et al. (Anthropic). (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread.
— Applies SAEs to a one-layer transformer, demonstrating learned features are more monosemantic than individual neurons.

**77.** Templeton, A. et al. (Anthropic). (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread.
— Scales SAE interpretability to Claude 3 Sonnet, extracting millions of features including abstract safety-relevant concepts.

**78.** Gao, L., Dupre la Tour, T., et al. (OpenAI). (2024). *Scaling and Evaluating Sparse Autoencoders.* arXiv:2406.04093.
— Trains SAEs with up to 16M latents on GPT-4, studying scaling laws for sparsity and model size.

**79.** Wu, Z., Arora, A., Geiger, A., et al. (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* ICML 2025 Spotlight. arXiv:2501.17148.
— Benchmark finding prompting outperforms all representation-based methods for steering; SAEs are not competitive.

**80.** (2025). *A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models.* arXiv:2503.05613.
— Comprehensive survey of SAE architectures, training, evaluation, and applications for LLM interpretability.

---

## 12. Representation Engineering and Value Steering

**81.** Zou, A., Phan, L., Chen, S., Campbell, J., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.
— Introduces RepE: uses population-level representations to monitor and manipulate high-level concepts (honesty, harmlessness, power-seeking).

**82.** (2023). *Aligning Large Language Models with Human Preferences through Representation Engineering.* arXiv:2312.15997.
— Introduces RAHF (Representation Alignment from Human Feedback) as an alternative to RLHF.

**83.** (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509.
— Extracts "persona vectors" (linear directions for traits like evil, sycophancy) enabling real-time monitoring and activation steering.

---

## 13. Persona and Role-Playing in LLMs

**84.** Tseng, Y.-M., Huang, Y.-C., Hsiao, T.-Y., Chen, W.-L., Huang, C.-W., Meng, Y., & Chen, Y.-N. (2024). *Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization.* Findings of EMNLP 2024.
— Comprehensive survey covering LLM role-playing and personalization approaches and evaluation.

**85.** Wang, Z.M., Peng, Z., Que, H., et al. (2023). *RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models.* arXiv:2310.00746.
— Benchmark and framework for evaluating role-playing capabilities including behavioral consistency.

**86.** Xu, R., Wang, X., Chen, J., et al. (2024). *Character is Destiny: Can Role-Playing Language Agents Make Persona-Driven Decisions?* arXiv:2404.12138.
— Investigates whether LLMs role-playing as specific characters make decisions consistent with those characters' personality traits.

**87.** Ge, T., Chan, X., Wang, X., Yu, D., Mi, H., & Yu, D. (2025). *Scaling Synthetic Data Creation with 1,000,000,000 Personas.* arXiv:2406.20094.
— Persona Hub: 1 billion diverse personas for synthetic data generation, demonstrating persona-driven prompting produces diverse data.

**88.** Salewski, L. et al. (2024). *In-Context Impersonation Reveals Large Language Models' Strengths and Biases.* NeurIPS 2023. arXiv:2305.14930.
— LLMs taking on personas recover developmental stages, improve with expert personas, and reveal hidden biases.

---

## 14. Sycophancy and Acquiescence Bias in LLMs

**89.** Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S.R., et al. (2024). *Towards Understanding Sycophancy in Language Models.* ICLR 2024. arXiv:2310.13548.
— Five state-of-the-art AI assistants consistently exhibit sycophancy; traced to biases in human preference data.

**90.** Wei, J., Huang, D., Lu, Y., Zhou, D., & Le, Q.V. (2024). *Simple Synthetic Data Reduces Sycophancy in Large Language Models.* arXiv:2308.03958.
— Synthetic NLP-task data encouraging robustness to user opinions significantly reduces sycophancy via lightweight finetuning.

**91.** Malmqvist et al. (2024). *Sycophancy in Large Language Models: Causes and Mitigations.* arXiv:2411.15287.
— Technical survey analyzing causes, impacts, and mitigation strategies for sycophancy.

---

## 15. LLM Steerability and Behavioral Consistency

**92.** Chen, K., He, Z., Shi, T., & Lerman, K. (2025). *STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models.* EMNLP 2025. arXiv:2505.20645.
— Benchmarks population-specific steering using 30 Reddit community pairs; best models reach ~65% vs. 81% human.

**93.** Miehling, E., Desmond, M., Ramamurthy, K.N., et al. (2025). *Evaluating the Prompt Steerability of Large Language Models.* NAACL 2025. arXiv:2411.12405.
— Defines steerability indices across 133 persona dimensions (personality, political views, ethics) with 500 examples per direction.

**94.** (2024). *Towards Reliable Evaluation of Behavior Steering Interventions in LLMs.* NeurIPS 2024 Workshop. arXiv:2410.17245.
— Reveals that reported success of behavior steering may be overstated; highlights promoting vs. suppressing distinction.

---

## 16. Psychometric Evaluation of LLMs (Personality, Values)

**95.** Serapio-Garcia, G., Safdari, M., et al. (2023/2025). *Personality Traits in Large Language Models.* Nature Machine Intelligence (2025). arXiv:2307.00184.
— Comprehensive methodology for Big Five tests on LLMs; larger instruction-tuned models produce reliable measurements.

**96.** Jiang, H., Zhang, X., Cao, X., Breazeal, C., Roy, D., & Kabbara, J. (2024). *PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits.* Findings of NAACL 2024. arXiv:2305.02547.
— Investigates LLM generation consistent with assigned Big Five profiles; humans perceive personality with ~80% accuracy.

**97.** Li, Liu, Liu, Zhou, Diab, & Sap. (2025). *BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data.* ACL 2025. arXiv:2410.16491.
— 100K-dialogue dataset for personality induction; SFT/DPO outperform prompting on personality assessments.

**98.** Salecha, A. et al. (2024). *Large Language Models Display Human-Like Social Desirability Biases in Big Five Personality Surveys.* PNAS Nexus, 3(12).
— LLMs detect personality evaluation and systematically skew toward socially desirable trait dimensions.

**99.** (2025). *Do Psychometric Tests Work for Large Language Models? Evaluation of Tests on Sexism, Racism, and Morality.* arXiv:2510.11254.
— Finds reasonable validity but low reliability when administering MFQ to LLMs; latent factors don't match human structure.

**100.** (2024). *Quantifying AI Psychology: A Psychometric Benchmark for Large Language Models.* arXiv:2406.17675.
— Comprehensive psychometric benchmark covering personality, values, emotion, theory of mind; uncovers discrepancies between self-report and behavior.

---

## Additional High-Relevance References

- Henderson, P., Mitchell, E., Manning, C.D., Jurafsky, D., & Finn, C. (2023). *Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of Foundation Models.* arXiv:2211.14946.
- Tamirisa, R. et al. (2024). *Tamper-Resistant Safeguards for Open-Weight LLMs.* ICLR 2025. arXiv:2408.00761.
- Costa, D.B., Alves, F., & Vicente, R. (2025). *Moral Susceptibility and Robustness under Persona Role-Play in Large Language Models.* arXiv:2511.08565. *(Authors' prior work)*
