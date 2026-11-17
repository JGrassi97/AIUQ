"""ssh access to the HPC host through one persistent ControlMaster connection.

Helper scripts are streamed over stdin (``python3 - <b64 json>``), so nothing is
installed remotely. All remote commands are read-only.
"""
import base64
import json
import shlex
import subprocess
import threading

from . import settings

# At most two ssh sessions at a time (one collector cycle + one preview).
_SEM = threading.BoundedSemaphore(2)


def ssh_base():
    settings.SSH_CONTROL_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={settings.SSH_CONTROL_DIR}/%C",
        "-o", "ControlPersist=15m",
        f"{settings.REMOTE_USER}@{settings.REMOTE_HOST}",
    ]


class RemoteError(RuntimeError):
    pass


# Script and request both travel over stdin (requests can be far larger than the argument
# limit, e.g. a hindcast over a thousand files). The bootstrap splits them and runs the script.
_SEP = b"\n#--AIUQ-REQUEST--\n"
_BOOT = ("import sys;d=sys.stdin.buffer.read();i=d.index(%r);"
         "sys.argv=['helper',d[i+%d:].decode()];"
         "exec(compile(d[:i],'helper','exec'),{'__name__':'__main__'})") % (_SEP, len(_SEP))


def run_helper(script_name, payload, timeout=None):
    script = (settings.HELPERS / script_name).read_bytes()
    arg = base64.b64encode(json.dumps(payload).encode())
    cmd = ssh_base() + [settings.REMOTE_PYTHON, "-c", shlex.quote(_BOOT)]  # ssh hands this to the remote shell
    with _SEM:
        try:
            proc = subprocess.run(cmd, input=script + _SEP + arg, capture_output=True,
                                  timeout=timeout or settings.REMOTE_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"ssh {settings.REMOTE_HOST}: timeout after {exc.timeout}s") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RemoteError(f"ssh {settings.REMOTE_HOST} exit {proc.returncode}: " + " | ".join(err[-5:]))
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"bad JSON from {script_name}: {proc.stdout[:200]!r}") from exc
