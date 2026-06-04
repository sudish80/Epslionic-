# API Reference

## FastAPI Server

Start with `epsionic --serve`.

### `GET /health`
Returns gateway health status.

### `GET /status`
Returns detailed gateway state (domain, tools, sessions, errors).

### `POST /train/{domain}`
Trigger training for a domain. Returns experiment ID.

### `GET /experiments`
List all experiments (paginated, last 20).

### `GET /experiments/{exp_id}`
Get experiment details.

### `GET /experiments/compare?a=X&b=Y`
Compare two experiments side-by-side.

### `GET /plugins`
List loaded plugins and their hooks.

### `GET /device`
Get device info and recommendations.

### `POST /experiments/export`
Export experiments to CSV.

### `GET /logs`
Get recent log lines.

## Plugin Hooks

Each plugin can implement:

- `on_register()` — Called when plugin is loaded
- `on_unregister()` — Called when plugin is removed
- `before_tool(tool, **params)` — Before any tool executes
- `after_tool(tool, result)` — After tool execution
- `on_error(tool, error, **kw)` — On tool error
- `on_startup()` — On gateway start
- `on_shutdown()` — On gateway shutdown
