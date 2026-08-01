# GAIA2-mini Baseline Test (Round 15)

Same model qwen3:1.7b. 3 GAIA2-mini scenarios that use only the three apps
we implemented (Calendar / Emails / Shopping) out of 10 in the universe.

Each baseline is asked to emit a tool-call plan matching scenario expected_actions.

| Configuration | Perfect (3/3) |
|---|---:|
| Static | 2/3 = 66.7% |
| ReAct | 0/3 = 0.0% |
| AGI-Kit | 0/3 = 0.0% |

## Per-scenario detail

### scenario_universe_27_fcbx0i
Expected: ['Shopping.checkout', 'Calendar.add_calendar_event', 'Emails.reply_to_email', 'Emails.reply_to_email', 'Emails.reply_to_email', 'Shopping.add_to_cart']
- **Static**: hit=6/6 sec=28.2 sample=[('Shopping', 'checkout'), ('Calendar', 'add_calendar_event'), ('Emails', 'reply_to_email'), ('Emails', 'reply_to_email'), ('Emails', 'reply_to_email')]
- **ReAct**: hit=0/6 sec=53.9 sample=[]
- **AGI-Kit**: hit=0/6 sec=42.4 sample=[]

### scenario_universe_27_16f4s4
Expected: ['Calendar.delete_calendar_event', 'Calendar.delete_calendar_event', 'Calendar.delete_calendar_event', 'Calendar.delete_calendar_event', 'Emails.reply_to_email', 'Calendar.add_calendar_event', 'Calendar.add_calendar_event', 'Emails.reply_to_email']
- **Static**: hit=8/8 sec=50.7 sample=[('Calendar', 'delete_calendar_event'), ('Calendar', 'delete_calendar_event'), ('Calendar', 'delete_calendar_event'), ('Calendar', 'delete_calendar_event'), ('Emails', 'reply_to_email')]
- **ReAct**: hit=0/8 sec=43.5 sample=[]
- **AGI-Kit**: hit=0/8 sec=49.4 sample=[]

### scenario_universe_22_bt12gw
Expected: ['Emails.reply_to_email', 'Calendar.add_calendar_event', 'Emails.reply_to_email', 'Emails.reply_to_email', 'Emails.send_email', 'Emails.send_email', 'Emails.send_email']
- **Static**: hit=0/7 sec=0 sample=[]
- **ReAct**: hit=0/7 sec=42.4 sample=[]
- **AGI-Kit**: hit=0/7 sec=49.5 sample=[]