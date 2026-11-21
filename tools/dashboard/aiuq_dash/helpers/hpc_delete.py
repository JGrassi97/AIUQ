"""Inspect or delete ONE experiment's working directory on the HPC host (used only by the Launcher).

Sent through ssh stdin like the other helpers. The read-only probe never deletes anything; this
file is the only remote code that does, and it refuses unless every check passes:

  * expid is 4 lowercase letters/digits and is the last path component;
  * the path is absolute, normalised and lies at <root>/<project>/<user>/<expid> for an allowed root;
  * no component of the path is a symbolic link (realpath == path);
  * it is a directory owned by the user running this script.

Request: {"mode": "inspect" | "delete", "expid": "a00g", "path": "/scratch/...", "roots": ["/scratch/<user>/autosubmit_scratch"]}
"""
import base64
import json
import os
import re
import shutil
import stat
import sys
import time


def check(req):
    expid, path = req.get("expid", ""), req.get("path", "")
    if not re.match(r"^[a-z0-9]{4}$", expid):
        return "invalid experiment id"
    if not path or not os.path.isabs(path) or os.path.normpath(path) != path.rstrip("/"):
        return "path must be absolute and normalised"
    path = path.rstrip("/")
    if os.path.basename(path) != expid:
        return "the last path component must be the experiment id"
    ok_root = False
    for root in req.get("roots") or []:
        root = os.path.normpath(root)
        if path.startswith(root + os.sep):
            parts = path[len(root) + 1:].split(os.sep)
            ok_root = ok_root or len(parts) == 3  # <project>/<user>/<expid>
    if not ok_root:
        return "path is not <scratch root>/<project>/<user>/<expid>"
    if not os.path.lexists(path):
        return "MISSING"
    if os.path.realpath(path) != path:
        return "path contains a symbolic link"
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        return "path is not a directory"
    if st.st_uid != os.getuid():
        return "directory is not owned by this user"
    return None


def inspect(path, limit_s=60):
    files = size = nc = nc_size = 0
    t0, partial = time.time(), False
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for fn in filenames:
            try:
                s = os.lstat(os.path.join(dirpath, fn)).st_size
            except OSError:
                continue
            files += 1
            size += s
            if fn.endswith(".nc"):
                nc += 1
                nc_size += s
        if time.time() - t0 > limit_s:
            partial = True
            break
    top = sorted(os.listdir(path)) if os.path.isdir(path) else []
    return {"files": files, "bytes": size, "nc_files": nc, "nc_bytes": nc_size, "partial": partial, "top": top}


def main():
    req = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    err = check(req)
    path = req.get("path", "").rstrip("/")
    if err == "MISSING":
        sys.stdout.write(json.dumps({"ok": True, "exists": False, "path": path}))
        return
    if err:
        sys.stdout.write(json.dumps({"ok": False, "error": err, "path": path}))
        return
    if req.get("mode") == "inspect":
        sys.stdout.write(json.dumps({"ok": True, "exists": True, "path": path, **inspect(path)}))
        return
    if req.get("mode") == "delete":
        errors = []
        shutil.rmtree(path, onerror=lambda f, p, e: errors.append("%s: %s" % (p, e[1])))
        gone = not os.path.lexists(path)
        sys.stdout.write(json.dumps({"ok": gone, "exists": not gone, "path": path, "errors": errors[:20], "n_errors": len(errors)}))
        return
    sys.stdout.write(json.dumps({"ok": False, "error": "unknown mode"}))


if __name__ == "__main__":
    main()
