# Evaluation: method, findings, and reproducibility

[Back to project overview](../README.md)

The project developed an agentic GIS system and evaluated how pretrained language models operated within it. The experiments concern inference, prompting, tool use, and optional retrieval, not training or fine-tuning models.

## Evaluation code and evidence

The evaluation path records the plan, tool calls, errors, results, and final answer. Those records can reveal a wrong hazard layer, an omitted filter, or a run that stopped after one query despite producing a plausible answer.

| Start with | What it does and why it is useful |
| --- | --- |
| `main()` in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) | Defines `queries`, `model_ids`, `ModelParams`, and `prompt_name`. This is the place to identify the actual batch conditions: B25–B30 and one model are currently active. B1–B24 remain as commented task definitions. |
| `EvalRun`, `PipelineResult`, and `AgentRunError` in [agent_models.py](../backend/evaluation/agent_models.py) | Represent the requested experiment, each model's outcome, and failures with state snapshots. Keeping failure evidence allows unsuccessful runs to be inspected alongside completed ones. |
| `run_all()` and `agent()` in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) | Run selected models sequentially through the same graph as the interactive API. The graph's `_model_node()` and tool node's `_build_tool_message_and_log()` / `_build_step_event()` record messages, calls, validation/execution events, latency, and token usage. See [agent_graph.py](../backend/evaluation/agent_graph.py) and [agent_tools_node.py](../backend/evaluation/agent_tools_node.py). |
| `extract_agent_trace()`, `_summary_metrics()`, `_final_failure_info()`, and `_latency_breakdown()` in [reporting.py](../backend/evaluation/reporting.py) | Derive which databases/tables were explored or used, tool/repair activity, failure categories, and time spent. These explain partial progress and failure mechanisms beyond a single completion flag. `_print_summary()` and `_print_details()` expose diagnostics in the console. |
| `_run_outcome_from_logs()` in [agent_helpers.py](../backend/evaluation/prompts/agent_helpers.py) | Derives execution and task-completion outcomes from recorded steps. It describes workflow progress, while the thesis's reference-answer assessment also checks the result. |
| `store_run()` in [reporting.py](../backend/evaluation/reporting.py) | Groups output by query hash, prompt, run ID, and model. Writes configuration plus separate `logs.json` and `result.json` artifacts, preserving detailed events and derived outcomes for later inspection. |
| `upload()`, `_update_queries_index()`, and `_update_runs_index()` in [AzureBlobStorage.py](../backend/storage/AzureBlobStorage.py) | Persist artifacts and lookup indexes in Azure Blob Storage so runs can be located and compared after execution. The interactive `/query` endpoint does not call `store_run()`. |

Those stored run artifacts are not included in this checkout. The local code exposes execution diagnostics, whereas the thesis additionally compares outputs and process decisions against task-specific reference requirements and performs repeated-run statistical analysis. The [checkout comparison](#what-this-checkout-can-support) lists the differences. The [findings below](#selected-findings) include chapter and table references.

## Sources and supplementary material

The findings below are from **Matin Mohammadi and Sander Gnanavel, _A Benchmark-Based Evaluation of Large Language Models in Agentic Multi-Step GIS Workflows_, University of Agder, 2026**. References use the report's printed page numbers, not PDF viewer page indices. The report PDF is not included or published here.

Appendix I identifies a separate [results archive](https://github.com/MatinM-96/Results-) and [agentic GIS benchmark repository](https://github.com/MatinM-96/agentic-gis-benchmark). These supplementary projects are separate from this repository. The numerical findings below come from the thesis and have not been recomputed from those archives.

Implementation statements refer to source commit `b051f44ecd4c8037fb8fbabbec0b3c230f5ec14a` of this repository.

## What was evaluated

| Aspect | Thesis evidence |
| --- | --- |
| Model coverage | 13 models spanning hosted and self-hosted deployments. Table 3.3, p. 33. |
| Task coverage | 30 GIS tasks across five difficulty levels, from filtering and lookup to spatial joins, aggregation, ranking, and multi-layer analysis. §3.3, Table 3.1, and Appendix A / Table A.1. |
| Prompt development | Shared prompt revisions informed by execution traces and recurring failures. Results discuss v1–v5. Embedding-assisted database retrieval was examined during prompt development. §§3.5 and 5.1–5.6. |
| Final repeated evaluation | Shared prompt v5, without embedding retrieval. The method specifies 20 runs per model-task condition. The statistical analysis retains **at most 20** and aggregates at model-task level before model summaries. §§3.5.6 and 5.7, pp. 36 and 69. |
| Correctness | Executed results compared with task-specific reference requirements: counts, identifiers, filters, spatial relations, ordering, and top-k constraints. §3.6.2, p. 38. |
| Process and cost | Traces record discovery, planning, validation, execution, repairs, replans, tool calls, latency, and token usage. §§3.4.3 and 3.6–3.9. |

The three scoring layers separate **workflow outcome**, **task-solving process**, and **conditioned efficiency**. Process credit can recognize useful intermediate work even if the final result is wrong. A weighted final score is therefore not an accuracy percentage. Task-level aggregation, bootstrap intervals, paired comparisons, and weight sensitivity are discussed in §5.7.

## Selected findings

### Repeatable success was possible, but uneven

In the final prompt-v5 comparison, `gemma-4-31b-vllm` achieved exact reference matches in **20/20 runs on 17 tasks**, including B1–B8 and several spatial-filtering tasks. Its mean weighted final score was **0.752**. **Sources: §5.7.2, p. 70 and §5.9, pp. 85–86.**

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

In the same comparison, `llama-3.3-70b-fp8-vllm` completed **0/600** runs, despite often finding relevant metadata. It averaged **37.7 tool calls per run**, compared with Gemma's **30.7**, and had a higher repetition rate. The report links this to repeated tool-use loops and difficulty following the required workflow. **Source: Table 5.8 and §5.9, pp. 85–86.**

### Retrieval helped discovery more than final spatial reasoning

Embedding-assisted retrieval reduced exploration and tool-call usage, with its clearest benefits under earlier prompts. It did not consistently improve final correctness under later prompts. Correct candidate databases still had to be followed by schema verification, appropriate spatial operations, correct filtering, and grounded answers. **Source: §5.6, pp. 67–69, Figure 5.3, and Table F.1.**

## What went wrong, and how it was investigated

The study reviewed model outputs, validation events, tool arguments, execution traces, and reference-answer matches. Recurrent failures informed prompt revisions. The same evidence supported process scoring and failure categorization. **Sources: §3.5.2, §3.6, §3.7, and §5.8.**

| Failure pattern documented in the thesis | Why execution can still be misleading |
| --- | --- |
| Wrong dataset, municipality identifier, geometry column, or coded category | A syntactically valid query can analyze the wrong objects. |
| Wrong spatial predicate or ranking before membership filtering | Tools can execute while implementing a different question. |
| Missing deduplication after spatial joins | Raw join rows can be counted as unique buildings or parcels. |
| Large unfiltered layers and costly joins | A plausible plan can exhaust execution resources. |
| Tool-order violations and repeated unsuccessful repair | Finding relevant metadata or detecting an error does not guarantee progress to an answer. |

The thesis documents these failure patterns in **§5.8.2, p. 84**. Its recommendations in **§5.10** combine model reasoning with verified schema information, enforced workflow state, validation, trace inspection, and result-level checks.

## What this checkout can support

| Thesis experiment | Code available here |
| --- | --- |
| Full 30-task benchmark | [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) documents B1–B30, but B1–B24 are commented out. Only B25–B30 are active in `main()`. |
| 13-model comparison | The batch runner currently selects only `gemma-4-31b-vllm`. The model registry is broader, but is not an experimental run manifest. |
| Final prompt v5 and repeated runs | This checkout contains `prompt_v1`–`prompt_v4` and `evaluation_v1`. The batch runner loads `evaluation_v1` and has no 20-repeat experiment loop. No equivalence to the thesis's v5 is established. |
| Reference-answer scoring and statistical analysis | The available [reporting code](../backend/evaluation/reporting.py) derives execution diagnostics and saves artifacts. The thesis's complete ground-truth scoring and statistical analysis are not provided as an end-to-end local evaluation workflow here. |
| Stored experimental evidence | The batch path writes `config.json`, `logs.json`, and `result.json` through [Azure Blob Storage](../backend/storage/AzureBlobStorage.py). Those run files are absent from this checkout. See the separate supplementary material above. |

The runner records new evaluations when the required services are available. Reproducing the thesis tables also requires the experiment configuration, stored runs, and reference-answer analysis described above.

In particular, fields such as `valid_sql`, `executed`, `task_completed`, and `answer_based_on_result` in [agent_models.py](../backend/evaluation/agent_models.py) and the [graph finalizer](../backend/evaluation/agent_graph.py) describe execution and completion heuristics. They are not substitutes for the reference-answer assessment in the thesis.

## Limits of the findings

The results depend on the 30 Norwegian GIS tasks, prepared databases, model versions, serving conditions, prompts, and execution budgets. Prompt refinement and final evaluation use the benchmark described in the thesis. Repeated runs estimate stability within that setting, not transfer to unseen domains. Access and rate constraints also affected model coverage. **Sources: §5.11, pp. 87–88, and §6.3, pp. 92–93.**

The findings describe performance on this benchmark, not a universal model ranking or reliability on arbitrary GIS questions. The supplementary archives have not been independently audited for completeness or correspondence to every reported result. This checkout alone cannot reproduce the thesis tables, and the study does not establish readiness for production use.
