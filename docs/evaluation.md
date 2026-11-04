# Evaluation: method, findings, and reproducibility

[Back to project overview](../README.md)

The project developed an agentic GIS system and evaluated how pretrained language models operated within it. The experiments concern inference, prompting, tool use, and optional retrieval, not training or fine-tuning models.

## Sources and supplementary material

The findings below are from **Matin Mohammadi and Sander Gnanavel, _A Benchmark-Based Evaluation of Large Language Models in Agentic Multi-Step GIS Workflows_, University of Agder, 2026**, using the report supplied for this documentation update. References use the report's printed page numbers, not PDF viewer page indices. The report PDF is not included or published here.

Appendix I identifies a separate [results archive](https://github.com/MatinM-96/Results-) and [agentic GIS benchmark repository](https://github.com/MatinM-96/agentic-gis-benchmark). These links were accessible during the documentation review. They are supplementary projects, not directories in this repository; their completeness and correspondence to every reported result have not been independently audited. Numerical findings here are attributed to the thesis, not recomputed from those archives.

Implementation statements refer to source commit `b051f44ecd4c8037fb8fbabbec0b3c230f5ec14a` of this repository.

## What was evaluated

| Aspect | Thesis evidence |
| --- | --- |
| Model coverage | 13 models spanning hosted and self-hosted deployments. Table 3.3, p. 33. |
| Task coverage | 30 GIS tasks across five difficulty levels, from filtering and lookup to spatial joins, aggregation, ranking, and multi-layer analysis. §3.3, Table 3.1, and Appendix A / Table A.1. |
| Prompt development | Shared prompt revisions informed by execution traces and recurring failures; results discuss v1–v5. Embedding-assisted database retrieval was examined during prompt development. §§3.5 and 5.1–5.6. |
| Final repeated evaluation | Shared prompt v5, without embedding retrieval. The method specifies 20 runs per model–task condition; the statistical analysis retains **at most 20** and aggregates at model–task level before model summaries. §§3.5.6 and 5.7, pp. 36 and 69. |
| Correctness | Executed results compared with task-specific reference requirements: counts, identifiers, filters, spatial relations, ordering, and top-k constraints. §3.6.2, p. 38. |
| Process and cost | Traces record discovery, planning, validation, execution, repairs, replans, tool calls, latency, and token usage. §§3.4.3 and 3.6–3.9. |

The three scoring layers separate **workflow outcome**, **task-solving process**, and **conditioned efficiency**. Process credit can recognize useful intermediate work even if the final result is wrong. A weighted final score is therefore not an accuracy percentage. Task-level aggregation, bootstrap intervals, paired comparisons, and weight sensitivity are discussed in §5.7.

## Selected findings

### Repeatable success was possible, but uneven

In the final prompt-v5 comparison, `gemma-4-31b-vllm` achieved exact reference matches in **20/20 runs on 17 tasks**, including B1–B8 and several spatial-filtering tasks. Its mean weighted final score was **0.752**. These results show that the workflow could produce repeatable answers on a substantial subset of the benchmark; they do not establish reliability on arbitrary GIS questions. **Sources: §5.7.2, p. 70; §5.9, pp. 85–86.**

### Technical completion was not semantic correctness

The clearest example is the breakdown of Gemma's **600 retained runs** in the final comparison:

| Outcome | Runs |
| --- | ---: |
| Completed with an exact reference match | 360 |
| Completed with partial reference credit | 25 |
| Completed with an incorrect result | 82 |
| Did not complete | 133 |

The first three rows total the **467 completed workflows**. The final row is calculated as `600 - 467`. A completion flag alone would hide the 82 incorrect completed results and the 25 partial matches. **Source: Table 5.8 and accompanying discussion in §5.9, p. 85.**

### More tool calls did not imply better reasoning

In the same comparison, `llama-3.3-70b-fp8-vllm` completed **0/600** runs, despite often finding relevant metadata. It averaged **37.7 tool calls per run**, compared with Gemma's **30.7**, and had a higher repetition rate. The report links this to repeated tool-use loops and difficulty following the required workflow. This is evidence about these models in this particular setup, not a general capability claim. **Source: Table 5.8 and §5.9, pp. 85–86.**

### Retrieval helped discovery more than final spatial reasoning

Embedding-assisted retrieval reduced exploration and tool-call usage, with its clearest benefits under earlier prompts. It did not consistently improve final correctness under later prompts. Correct candidate databases still had to be followed by schema verification, appropriate spatial operations, correct filtering, and grounded answers. **Source: §5.6, pp. 67–69, Figure 5.3, and Table F.1.**

## What went wrong, and how it was investigated

The study reviewed model outputs, validation events, tool arguments, execution traces, and reference-answer matches. Recurrent failures informed prompt revisions; the same evidence supported process scoring and failure categorization. **Sources: §3.5.2, §3.6, §3.7, and §5.8.**

| Failure pattern documented in the thesis | Why execution can still be misleading |
| --- | --- |
| Wrong dataset, municipality identifier, geometry column, or coded category | A syntactically valid query can analyze the wrong objects. |
| Wrong spatial predicate or ranking before membership filtering | Tools can execute while implementing a different question. |
| Missing deduplication after spatial joins | Raw join rows can be counted as unique buildings or parcels. |
| Large unfiltered layers and costly joins | A plausible plan can exhaust execution resources. |
| Tool-order violations and repeated unsuccessful repair | Finding relevant metadata or detecting an error does not guarantee progress to an answer. |

These are reported failure mechanisms, not invented example runs. **Source: §5.8.2, p. 84.** The design lesson is to combine model reasoning with schema grounding, enforced workflow state, validation, trace inspection, and result-level checks. The thesis treats these as requirements for further development, not evidence that the prototype is production-ready (§5.10).

## What this checkout can support

| Thesis experiment | Code available here |
| --- | --- |
| Full 30-task benchmark | [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) documents B1–B30, but B1–B24 are commented out. Only B25–B30 are active in `main()`. |
| 13-model comparison | The batch runner currently selects only `gemma-4-31b-vllm`. The model registry is broader, but is not an experimental run manifest. |
| Final prompt v5 and repeated runs | This checkout contains `prompt_v1`–`prompt_v4` and `evaluation_v1`. The batch runner loads `evaluation_v1` and has no 20-repeat experiment loop. No equivalence to the thesis's v5 is established. |
| Reference-answer scoring and statistical analysis | The available [reporting code](../backend/evaluation/reporting.py) derives execution diagnostics and saves artifacts. The thesis's complete ground-truth scoring and statistical analysis are not provided as an end-to-end local evaluation workflow here. |
| Stored experimental evidence | The batch path writes `config.json`, `logs.json`, and `result.json` through [Azure Blob Storage](../backend/storage/AzureBlobStorage.py). Those run files are absent from this checkout; consult the separate supplementary material above. |

The runner is useful for understanding how an evaluation is executed when the required services exist. It is not sufficient, as checked in, to reproduce the thesis tables. No models, database queries, or cloud reporting were run during this documentation update.

In particular, fields such as `valid_sql`, `executed`, `task_completed`, and `answer_based_on_result` in [agent_models.py](../backend/evaluation/agent_models.py) and the [graph finalizer](../backend/evaluation/agent_graph.py) describe execution and completion heuristics. They are not substitutes for the reference-answer assessment in the thesis.

## Limits of the findings

The results depend on the 30 Norwegian GIS tasks, prepared databases, model versions, serving conditions, prompts, and execution budgets. Prompt refinement and final evaluation use the benchmark described in the thesis; repeated runs estimate stability within that setting, not transfer to unseen domains. Access and rate constraints also affected model coverage. **Sources: §5.11, pp. 87–88, and §6.3, pp. 92–93.**

Interpret the findings as evidence for this research prototype and evaluation environment. They do not establish a universal model ranking, complete reproducibility from this checkout, or production readiness.
