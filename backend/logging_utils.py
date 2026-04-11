import logging
import sys


class CleanFormatter(logging.Formatter):
    def format(self, record):
        name = record.name
        msg = record.getMessage()

        if name == "__main__":
            if "Starting MCP server" in msg:
                return "      [SYSTEM] starting MCP server"
            if "Server ready, yielding app context" in msg:
                return "      [SYSTEM] server ready"
            return msg

        if name == "backend.database.database":
            if "Pool created for conn_id=" in msg:
                conn_id = msg.split("conn_id=")[-1]
                return f"      [DB] pool created : {conn_id}"
            return f"      [DB] {msg}"

        if name == "backend.mcp.client.client":
            return f"      [MCP CLIENT] {msg}"

        if name.startswith("backend.mcp.server"):
            return f"      [MCP SERVER] {msg}"

        if name == "mcp.server.lowlevel.server":
            if "Processing request of type CallToolRequest" in msg:
                return "      [MCP] executing tool request"
            if "Processing request of type ListToolsRequest" in msg:
                return "      [MCP] listing tools"
            return f"      [MCP] {msg}"

        if name == "httpx":
            return "      [HTTPX] request sent ok" if "HTTP Request:" in msg else f"      [HTTPX] {msg}"

        if name == "openai._base_client":
            if "Request options:" in msg:
                return "      [OPENAI] request options prepared"
            if "Sending HTTP Request:" in msg:
                return "      [OPENAI] sending request"
            if "HTTP Response:" in msg:
                return "      [OPENAI] response received"
            if "request_id:" in msg:
                return f"      [OPENAI] {msg}"
            return f"      [OPENAI] {msg}"

        if name == "httpcore.connection":
            if "connect_tcp.started" in msg:
                return "      [HTTPCORE] tcp connect started"
            if "connect_tcp.complete" in msg:
                return "      [HTTPCORE] tcp connect complete"
            if "start_tls.started" in msg:
                return "      [HTTPCORE] tls start"
            if "start_tls.complete" in msg:
                return "      [HTTPCORE] tls ready"
            if "close.started" in msg:
                return "      [HTTPCORE] close started"
            if "close.complete" in msg:
                return "      [HTTPCORE] close complete"
            return f"      [HTTPCORE] {msg}"

        if name == "httpcore.http11":
            if "send_request_headers.started" in msg:
                return "      [HTTP11] sending headers"
            if "send_request_headers.complete" in msg:
                return "      [HTTP11] headers sent"
            if "send_request_body.started" in msg:
                return "      [HTTP11] sending body"
            if "send_request_body.complete" in msg:
                return "      [HTTP11] body sent"
            if "receive_response_headers.started" in msg:
                return "      [HTTP11] waiting for headers"
            if "receive_response_headers.complete" in msg:
                return "      [HTTP11] headers received"
            if "receive_response_body.started" in msg:
                return "      [HTTP11] receiving body"
            if "receive_response_body.complete" in msg:
                return "      [HTTP11] body received"
            if "response_closed.started" in msg:
                return "      [HTTP11] closing response"
            if "response_closed.complete" in msg:
                return "      [HTTP11] response closed"
            return f"      [HTTP11] {msg}"

        return f"      [{name}] {msg}"


def setup_logging() -> None:
    logging.getLogger("azure").setLevel(logging.WARNING)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(CleanFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    for name in [
        "__main__",
        "backend.database.database",
        "backend.mcp.client.client",
        "backend.mcp.server.run",
        "backend.mcp.server.step_tools",
        "backend.mcp.server.discovery_tools",
        "mcp.server.lowlevel.server",
        "httpx",
        "openai._base_client",
        "httpcore.connection",
        "httpcore.http11",
    ]:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.DEBUG)