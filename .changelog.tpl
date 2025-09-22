{% for entry in tree %}

## {{ entry.version }}{% if entry.date %} ({{ entry.date }}){% endif %}

{% for change_key, changes in entry.changes.items() %}

{% if change_key %}
### {{ change_key }}
{% endif %}

{% for change in changes %}
{% if change.scope %}
- **{{ change.scope }}**: {{ change.message }} ({{ change.sha1[:7] }})
{% elif change.message %}
- {{ change.message }} ({{ change.sha1[:7] }})
{% endif %}
{% endfor %}
{% endfor %}
{% endfor %}
