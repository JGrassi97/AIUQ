"""Render the repository's support matrix without duplicating its data in reST."""
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective
import yaml


class SupportMatrix(SphinxDirective):
    has_content = False

    def run(self):
        source = Path(self.env.srcdir).parent / "support-matrix.yml"
        self.env.note_dependency(str(source))
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        labels = {
            "available": "Available",
            "wip": "In development",
            "not_implemented": "Not implemented",
        }
        table = nodes.table(classes=["support-matrix"])
        group = nodes.tgroup(cols=3 + len(data["ics"]))
        table += group
        for width in [15, 15, 40] + [15] * len(data["ics"]):
            group += nodes.colspec(colwidth=width)
        head = nodes.thead()
        group += head
        row = nodes.row()
        head += row
        for title in ["Model", "Modes", "Checkpoint"] + data["ics"]:
            cell = nodes.entry()
            cell += nodes.paragraph(text=title)
            row += cell
        body = nodes.tbody()
        group += body
        for model in data["models"]:
            for checkpoint in model["checkpoints"]:
                row = nodes.row()
                body += row
                for value in [model["name"], model["supported_modes"]]:
                    cell = nodes.entry()
                    cell += nodes.paragraph(text=value)
                    row += cell
                cell = nodes.entry()
                paragraph = nodes.paragraph()
                paragraph += nodes.reference(
                    text=checkpoint["label"], refuri=checkpoint["url"]
                )
                cell += paragraph
                row += cell
                for source_name in data["ics"]:
                    status = data["status"][source_name][checkpoint["id"]]
                    cell = nodes.entry()
                    paragraph = nodes.paragraph()
                    paragraph += nodes.inline(
                        text=labels[status],
                        classes=["status", "status-" + status.replace("_", "-")],
                    )
                    cell += paragraph
                    row += cell
        return [table]


def setup(app):
    app.add_directive("aiuq-support-matrix", SupportMatrix)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
