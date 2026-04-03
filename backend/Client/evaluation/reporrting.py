
# ─────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────

def _print_summary(query: str, results: list):
    logger.info("\n" + "#" * 130)
    logger.info("QUERY")
    logger.info("  %s", query)
    logger.info("#" * 130)

    logger.info(
        "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
        "MODEL",
        "SQL_OK",
        "EXEC",
        "TASK",
        "ITER",
        "REP",
        "RST",
        "API",
        "LATENCY",
        "TOOLS",
        "TOKENS",
        "ERROR",
    )
    logger.info("-" * 130)

    for r in results:
        if isinstance(r, Exception):
            logger.info(
                "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
                "ERROR",
                "False",
                "False",
                "False",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                _short(str(r), 200),
            )
            continue

        trace = extract_agent_trace(r)
        err = "; ".join(_dedup(r.all_errors)) or "-"
        total_tokens = r.tokens.total_tokens if r.tokens else "-"

        logger.info(
            "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
            r.model_id,
            str(r.valid_sql),
            str(r.executed),
            str(r.task_completed),
            r.iterations,
            r.sql_repairs_used,
            r.restarts_used,
            r.total_api_calls,
            f"{r.latency_s:.3f}s",
            trace.total_tool_calls,
            str(total_tokens),
            err,
        )
def _print_details(results: list):
    logger.info("\n" + "=" * 110)
    logger.info("DETAILS PER MODEL")
    logger.info("=" * 110)

    for r in results:
        logger.info("\n" + "-" * 110)

        if isinstance(r, Exception):
            logger.info("ERROR")
            logger.info("  message            : %s", r)
            continue

        trace = extract_agent_trace(r)

        logger.info("[%s]", r.model_id)
        logger.info("  valid_sql          : %s", r.valid_sql)
        logger.info("  executed           : %s", r.executed)
        logger.info("  iterations         : %s", r.iterations)
        logger.info("  sql_repairs        : %s", r.sql_repairs_used)
        logger.info("  restarts           : %s", r.restarts_used)
        logger.info("  latency_s          : %.3f", r.latency_s)
        logger.info("  api_calls          : %s", r.total_api_calls)

        if r.tokens:
            logger.info(
                "  tokens             : in=%s out=%s total=%s",
                r.tokens.prompt_tokens,
                r.tokens.completion_tokens,
                r.tokens.total_tokens,
            )

        if r.all_errors:
            logger.info("\n  ERRORS:")
            for err in _dedup(r.all_errors):
                logger.info("    - %s", err)

        logger.info("\n  TRACE:")
        logger.info("    databases explored : %s", trace.databases_explored)
        logger.info("    databases in result: %s", trace.databases_in_result)
        logger.info("    tables explored    : %s", trace.tables_explored)
        logger.info("    tables in result   : %s", trace.tables_in_result)
        logger.info("    validated          : %s", trace.validated_before_exec)
        logger.info("    executed ok        : %s", trace.executed_successfully)
        logger.info("    total tool calls   : %s", trace.total_tool_calls)
        logger.info("    tool call counts   : %s", trace.tool_call_counts)

        if trace.executions:
            logger.info("\n  EXECUTIONS:")
            for i, ex in enumerate(trace.executions):
                logger.info("    Execution %s:", i)
                logger.info("      iteration       : %s", ex["iteration"])
                logger.info("      success         : %s", ex["success"])
                logger.info("      database        : %s", ex["database"])
                logger.info("      sql             : %s", ex["sql"])
                if not ex["success"]:
                    logger.info("      error           : %s", ex["error"])

        if r.run_logs:
            logger.info("\n  RUN LOGS:")
            for run in r.run_logs:
                logger.info("    Run %s:", run.get("run_index", 0))
                logger.info("      attempt         : %s", run.get("attempt_number", 0))
                logger.info("      sql_repairs     : %s", run.get("sql_repairs_used", 0))
                logger.info("      final_sql       : %s", run.get("final_sql") or "(none)")

                atk = run.get("attempt_tokens")
                if atk:
                    logger.info(
                        "      tokens          : in=%s out=%s total=%s",
                        atk.get("prompt_tokens", 0),
                        atk.get("completion_tokens", 0),
                        atk.get("total_tokens", 0),
                    )

                if run.get("validation_error"):
                    logger.info("      validation_error: %s", run["validation_error"])
                if run.get("execution_error"):
                    logger.info("      execution_error : %s", run["execution_error"])

                for entry in run.get("iterations", []):
                    tok = entry.get("tokens", {})
                    logger.info("      Iteration %s:", entry["iteration"])
                    logger.info(
                        "        tokens              : in=%s out=%s total=%s",
                        tok.get("prompt_tokens", 0),
                        tok.get("completion_tokens", 0),
                        tok.get("total_tokens", 0),
                    )
                    logger.info("        ai_content          : %s", entry.get("ai_content", ""))
                    logger.info(
                        "        tool_calls_declared : %s",
                        len(entry.get("tool_calls", [])),
                    )

                    for t in entry.get("tools", []):
                        logger.info("          TOOL   : %s", t["tool"])
                        logger.info("          ARGS   : %s", t["args"])
                        if t["tool"] == "tool_validate_sql":
                            ok = str(t["result"]).strip().lower() == "ok"
                            logger.info(
                                "          RESULT : %s",
                                "ok" if ok else "fail: " + str(t["result"]),
                            )
                        else:
                            logger.info("          RESULT : %s", t["result"])

        logger.info("\n  FINAL:")
        logger.info("    answer            : %s", r.final_answer or "(none)")
        logger.info("    sql               : %s", r.final_sql or "(none)")
        logger.info(
            "    result_data       : %s",
            r.result_data if r.result_data is not None else "(none)",
        )

    logger.info("\n" + "=" * 110)
