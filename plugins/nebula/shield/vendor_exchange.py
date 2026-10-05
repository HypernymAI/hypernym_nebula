"""
VENDORED from ClaudeStorm tools/alliance_labeling/extract_exchanges.py (sha256 4f0540b990b5877fec770c802b22eb06e53a37dee184519eb6b55373b3217037): SKIP_PREFIX, RES_HEAD, RES_TAIL, RES_TOTAL, TOOL_ARG_HEAD, _result_text — sliced byte-exact
with ast by tools/extract_from_claudestorm.py. Do not edit here; re-extract.
"""

SKIP_PREFIX = (
    '<local-command', '<command-', '<system-reminder', '[Request interrupted',
    '[Image', 'Caveat: The messages below', '<task-notification',
    'This session is being continued from a previous conversation',
)


RES_HEAD = 300


RES_TAIL = 200


RES_TOTAL = 4000


TOOL_ARG_HEAD = 300


def _result_text(content) -> str:
    """Flatten one tool_result's content (string or list of blocks) to head+tail text."""
    if isinstance(content, str):
        s = content
    elif isinstance(content, list):
        s = '\n'.join(b['text'] for b in content
                      if isinstance(b, dict) and 'type' in b and b['type'] == 'text'
                      and 'text' in b)
    else:
        s = ''
    if len(s) <= RES_HEAD + RES_TAIL:
        return s
    return s[:RES_HEAD] + f' …[{len(s) - RES_HEAD - RES_TAIL} chars]… ' + s[-RES_TAIL:]
