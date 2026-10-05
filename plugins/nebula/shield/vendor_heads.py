"""
The shield heads — one per certified cell (build doc §4.1) — and how each head's text fields are
taken from a frozen unit payload.

`fields` maps FEATURE FIELD NAME (spec §5.2 step 3: the name hashed into every feature) →
PAYLOAD KEY in the frozen unit. List-valued payload keys are joined with ctxshield.shape.LIST_JOIN.
`bar` = Jev TEST AUC from the E30 final result (the comparator, not a target).

NO .get().
"""
HEADS = {
    'H13_S1': {'class_id': 13, 'section': 'S1', 'tier': 'v1', 'bar': 0.787,
               'fields': {'pilot': 'pilot', 'prev': 'previous_reply', 'earlier': 'earlier_asks'}},
    'H13_S6': {'class_id': 13, 'section': 'S6', 'tier': 'v1', 'bar': 0.911,
               'fields': {'pilot': 'pilot', 'reply': 'reply'}},
    'H06_S2': {'class_id': 6, 'section': 'S2', 'tier': 'v1', 'bar': 0.890,
               'fields': {'ask': 'pilot_ask', 'text': 'text'}},
    'H08_S2': {'class_id': 8, 'section': 'S2', 'tier': 'v1', 'bar': 0.939,
               'fields': {'ask': 'pilot_ask', 'text': 'text'}},
    'H08_S6': {'class_id': 8, 'section': 'S6', 'tier': 'v1', 'bar': 0.925,
               'fields': {'pilot': 'pilot', 'reply': 'reply'}},
    'H15_S2': {'class_id': 15, 'section': 'S2', 'tier': 'v1', 'bar': 0.934,
               'fields': {'ask': 'pilot_ask', 'text': 'text'}},
    'H15_S6': {'class_id': 15, 'section': 'S6', 'tier': 'v1', 'bar': 0.861,
               'fields': {'pilot': 'pilot', 'reply': 'reply'}},
    'H04_S6': {'class_id': 4, 'section': 'S6', 'tier': 'v1', 'bar': 0.811,
               'fields': {'pilot': 'pilot', 'reply': 'reply'}},
    'H04_S4': {'class_id': 4, 'section': 'S4', 'tier': 'v1.1', 'bar': 0.689,
               'fields': {'ask': 'pilot_ask', 'call': 'tool_call', 'before': 'text_before'}},
    'H06_S5': {'class_id': 6, 'section': 'S5', 'tier': 'v1.1', 'bar': 0.902,
               'fields': {'claim': 'claim', 'evidence': 'evidence'}},
}
