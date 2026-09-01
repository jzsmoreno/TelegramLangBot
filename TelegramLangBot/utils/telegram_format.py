import html
import re

TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _split_table_row(line: str) -> list[str]:
    """Split a Markdown table row while preserving escaped and inline-code pipes."""
    line = line.strip()

    if line.startswith("|"):
        line = line[1:]

    if line.endswith("|") and not line.endswith(r"\|"):
        line = line[:-1]

    cells = []
    current = []
    escaped = False
    in_code = False

    for char in line:
        if char == "\\" and not escaped:
            escaped = True
            current.append(char)
            continue

        if char == "`" and not escaped:
            in_code = not in_code
            current.append(char)
            continue

        if char == "|" and not escaped and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)

        escaped = False

    cells.append("".join(current).strip())

    return [cell.replace(r"\|", "|") for cell in cells]


def _is_table_row(line: str) -> bool:
    """Return whether a line contains a potential Markdown table row."""
    return "|" in line


def _normalize_row(row: list[str], column_count: int) -> list[str]:
    """Normalize a row without discarding any original content."""
    if len(row) == column_count:
        return row

    if len(row) < column_count:
        return row + [""] * (column_count - len(row))

    return row[: column_count - 1] + [" | ".join(row[column_count - 1 :])]


def _wrap_cell(text: str, width: int) -> list[str]:
    """Wrap cell content to the given width without losing any characters."""
    if len(text) <= width:
        return [text]

    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    result = []

    for line in lines:
        while len(line) > width:
            result.append(line[:width])
            line = line[width:]

        if line:
            result.append(line)

    return result or [""]


def clean_telegram_html(text: str) -> str:
    """Convert Markdown-like text into valid Telegram HTML."""
    code_blocks = []
    tables = []

    def protect_code(match):
        """Replace a fenced code block with a temporary placeholder."""
        code = match.group(1)
        placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks.append(code)
        return placeholder

    text = re.sub(
        r"```(?:[a-zA-Z0-9_+-]+)?[ \t]*\n?(.*?)```",
        protect_code,
        text,
        flags=re.DOTALL,
    )

    lines = text.splitlines()
    result = []
    i = 0

    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _is_table_row(lines[i])
            and TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
            header = _split_table_row(lines[i])
            separator = _split_table_row(lines[i + 1])
            column_count = len(header)

            if column_count >= 2 and len(separator) == column_count:
                table_rows = [header]
                j = i + 2

                while j < len(lines):
                    line = lines[j]

                    if not line.strip() or not _is_table_row(line):
                        break

                    row = _split_table_row(line)
                    row = _normalize_row(row, column_count)

                    table_rows.append(row)
                    j += 1

                widths = []

                for column in range(column_count):
                    width = max(len(row[column]) for row in table_rows)
                    widths.append(min(max(width, 3), 30))

                formatted_rows = []

                for row in table_rows:
                    wrapped_cells = [
                        _wrap_cell(row[column], widths[column]) for column in range(column_count)
                    ]

                    row_height = max(len(cell_lines) for cell_lines in wrapped_cells)

                    for line_index in range(row_height):
                        parts = []

                        for column in range(column_count):
                            cell_lines = wrapped_cells[column]
                            value = cell_lines[line_index] if line_index < len(cell_lines) else ""

                            parts.append(value.ljust(widths[column]))

                        formatted_rows.append("  ".join(parts).rstrip())

                separator_line = "  ".join("─" * width for width in widths)

                header_lines_count = max(
                    len(_wrap_cell(header[column], widths[column]))
                    for column in range(column_count)
                )

                header_end = header_lines_count

                table_lines = formatted_rows[:header_end]
                table_lines.append(separator_line)
                table_lines.extend(formatted_rows[header_end:])

                table_text = "\n".join(table_lines)

                placeholder = f"__TABLE_{len(tables)}__"
                tables.append(table_text)

                result.append(placeholder)
                i = j
                continue

        result.append(lines[i])
        i += 1

    text = "\n".join(result)

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"(?<!\*)\*([^*\n]+?)\*(?!\*)",
        r"<i>\1</i>",
        text,
    )

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"^\s*[-*_]{3,}\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    def restore_table(match):
        """Restore a protected table as escaped Telegram HTML."""
        index = int(match.group(1))
        table = html.escape(tables[index])
        return f"<pre>{table}</pre>"

    text = re.sub(
        r"__TABLE_(\d+)__",
        restore_table,
        text,
    )

    def restore_code(match):
        """Restore a protected code block as escaped Telegram HTML."""
        index = int(match.group(1))
        code = html.escape(code_blocks[index])
        return f"<pre>{code}</pre>"

    text = re.sub(
        r"__CODE_BLOCK_(\d+)__",
        restore_code,
        text,
    )

    return text.strip()
