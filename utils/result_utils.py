def ensure_task_row(obj, default_name, default_status="Done"):
    """Make sure a task output is a dict with at least name/status so it shows in summary."""
    if isinstance(obj, dict):
        obj.setdefault("name", default_name)
        obj.setdefault("status", default_status)
        obj.setdefault(
            "allocation", None
        )  # descriptive tasks typically have no allocation
        return obj
    return {"name": default_name, "status": default_status, "allocation": None}
