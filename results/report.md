# Evaluation report

Date: 2026-11-10. Tickets: 20 from `sample_tickets.json`. Step cap: 6.

| Model | Category accuracy | Escalation accuracy | Avg tool calls | Total tokens (in / out) | Total cost | Cost per ticket | Avg latency |
|---|---|---|---|---|---|---|---|
| `claude-fable-5-2` | 100% | 95% | 3.7 | 139,330 / 18,339 | $1.8482 | $0.0924 | 12.7s |
| `claude-haiku-5-5` | 100% | 85% | 3.4 | 124,983 / 17,950 | $0.0215 | $0.0011 | 5.5s |

- `claude-fable-5-2`: $8 input / $40 output per million tokens.
- `claude-haiku-5-5`: $0.1 input / $0.5 output per million tokens.

## Per ticket

| Ticket | Expected | `claude-fable-5-2` | `claude-haiku-5-5` |
|---|---|---|---|
| T-001 | billing, escalate | ✅ billing, escalate | ✅ billing, escalate |
| T-002 | billing | ❌ billing, escalate | ❌ billing, escalate |
| T-003 | bug | ✅ bug | ✅ bug |
| T-004 | feature_request | ✅ feature_request | ✅ feature_request |
| T-005 | account, escalate | ✅ account, escalate | ✅ account, escalate |
| T-006 | account | ✅ account | ❌ account, escalate |
| T-007 | billing | ✅ billing | ✅ billing |
| T-008 | billing | ✅ billing | ✅ billing |
| T-009 | billing, escalate | ✅ billing, escalate | ✅ billing, escalate |
| T-010 | other | ✅ other | ✅ other |
| T-011 | account, escalate | ✅ account, escalate | ✅ account, escalate |
| T-012 | bug, escalate | ✅ bug, escalate | ✅ bug, escalate |
| T-013 | billing | ✅ billing | ✅ billing |
| T-014 | bug | ✅ bug | ✅ bug |
| T-015 | feature_request | ✅ feature_request | ✅ feature_request |
| T-016 | account, escalate | ✅ account, escalate | ✅ account, escalate |
| T-017 | other | ✅ other | ✅ other |
| T-018 | bug, escalate | ✅ bug, escalate | ✅ bug, escalate |
| T-019 | account | ✅ account | ❌ account, escalate |
| T-020 | bug | ✅ bug | ✅ bug |
