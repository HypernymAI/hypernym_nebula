"""
The pilot's OWN words in a turn (S56). Claude Code writes pasted text into the transcript wrapped as
    <pasted_content id="X"> … </pasted_content id="X">
(the same id on both tags). Pasted material is someone else's text — another agent's transcript, a log, a spec — so it
must never be scored as the pilot: not in rageometry (caps / profanity / anger words / repeats) and not as the ask a
reply is judged against. The FULL text is still stored (texts table, text_sha) exactly as before.

    own_words(t)  t with every pasted block removed (an opening tag with no closing tag → removed to the end);
                  text outside the blocks kept byte-for-byte. No block → t unchanged (identical object value).
    has_paste(t)  True when t contains at least one pasted block.
Pure, stdlib only. NO .get().
"""
import re

OPEN = re.compile(r'<pasted_content id="([^"]*)">')


def has_paste(t: str) -> bool:
    return OPEN.search(t) is not None


def own_words(t: str) -> str:
    if not has_paste(t):
        return t
    out, pos = [], 0
    while True:
        m = OPEN.search(t, pos)
        if m is None:
            out.append(t[pos:])
            break
        out.append(t[pos:m.start()])
        close = f'</pasted_content id="{m.group(1)}">'
        end = t.find(close, m.end())
        if end < 0:                               # unclosed block: everything after the tag is pasted
            break
        pos = end + len(close)
    return ''.join(out)
